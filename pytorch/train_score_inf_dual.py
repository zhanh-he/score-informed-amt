from __future__ import annotations

import os
import sys
import time
import logging
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from torch.optim import Adam, AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from hydra import initialize, compose
from omegaconf import OmegaConf
import wandb

from utilities import create_folder, create_logging, get_model_name
from losses import get_loss_func
from evaluate import SegmentEvaluator
from model import build_adapter
from score_inf import build_score_inf
from score_inf.wrapper import ScoreInfWrapper
from pytorch_utils import log_velocity_rolls

from train_score_inf import (
    init_wandb,
    _select_velocity_metrics,
    _model_param_sizes,
    _write_training_stats,
    _resolve_score_inf_conditioning,
    _required_target_rolls,
    _resolve_train_schedule,
    _phase_at_iteration,
    _apply_train_phase,
    build_dataloaders,
    _prepare_batch,
)

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
        FullOptimStateDictConfig,
    )
except Exception:
    FSDP = None
    StateDictType = None
    FullStateDictConfig = None
    FullOptimStateDictConfig = None


def _method_name(method) -> str:
    return str(method or "direct").strip()


def _is_direct_method(method) -> bool:
    return _method_name(method) == "direct"


def _setup_distributed() -> Tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA, but CUDA is not available.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, world_size, local_rank


def _cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _shard_train_loader(train_loader, rank: int, world_size: int, per_rank_batch_size: int) -> None:
    if world_size <= 1:
        return

    sampler = train_loader.batch_sampler
    if not hasattr(sampler, "segment_indexes"):
        raise RuntimeError("Unsupported batch sampler: expected custom sampler with segment_indexes.")

    rank_indexes = sampler.segment_indexes[rank::world_size]
    if len(rank_indexes) == 0:
        raise RuntimeError(
            f"Rank {rank} got zero training segments after sharding. "
            "Increase dataset size or reduce WORLD_SIZE."
        )

    sampler.segment_indexes = rank_indexes
    sampler.pointer = 0
    sampler.batch_size = per_rank_batch_size


def _save_checkpoint(model: torch.nn.Module, checkpoint_path: str, use_fsdp: bool, is_main: bool) -> None:
    if use_fsdp:
        if FSDP is None or StateDictType is None or FullStateDictConfig is None:
            raise RuntimeError("FSDP requested but unavailable in this PyTorch build.")
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            state = model.state_dict()
        if is_main:
            torch.save(state, checkpoint_path)
    else:
        if is_main:
            state = _unwrap_model(model).state_dict()
            torch.save(state, checkpoint_path)


def _build_train_model(cfg, device: torch.device):
    model_cfg = {"type": cfg.model.type, "params": cfg.model.params}
    adapter = build_adapter(model_cfg, model=None, cfg=cfg).to(device)

    score_cfg = getattr(cfg, "score_informed", None)
    if score_cfg is None:
        method = "direct"
        score_params = {}
    else:
        if OmegaConf.is_config(score_cfg):
            score_cfg = OmegaConf.to_container(score_cfg, resolve=True)
        if isinstance(score_cfg, dict):
            method = _method_name(score_cfg.get("method", "direct") or "direct")
            score_params = score_cfg.get("params", {}) or {}
        else:
            method = _method_name(getattr(score_cfg, "method", "direct") or "direct")
            score_params = getattr(score_cfg, "params", {}) or {}

    score_params, cond_keys = _resolve_score_inf_conditioning(cfg, method, score_params)
    target_rolls = _required_target_rolls(cfg.loss.loss_type)
    train_mode, switch_iteration = _resolve_train_schedule(cfg, score_cfg)
    post = build_score_inf(method, score_params).to(device)
    model = ScoreInfWrapper(adapter, post, freeze_base=False).to(device)
    return model, method, cond_keys, target_rolls, train_mode, switch_iteration, score_cfg


def train(cfg):
    dist_enabled, rank, world_size, local_rank = _setup_distributed()
    is_main = rank == 0
    use_fsdp = bool(getattr(cfg.exp, "use_fsdp", True)) and dist_enabled

    try:
        if dist_enabled:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")

        if dist_enabled and cfg.exp.batch_size % world_size != 0:
            raise ValueError(
                f"exp.batch_size ({cfg.exp.batch_size}) must be divisible by WORLD_SIZE ({world_size}) "
                "for dual-GPU training."
            )
        per_rank_batch_size = cfg.exp.batch_size // world_size if dist_enabled else cfg.exp.batch_size

        model, method, cond_keys, target_rolls, train_mode, switch_iteration, score_cfg = _build_train_model(cfg, device)

        if dist_enabled:
            if use_fsdp:
                if FSDP is None:
                    raise RuntimeError("exp.use_fsdp=true but FSDP is unavailable in this environment.")
                model = FSDP(model, device_id=torch.cuda.current_device(), use_orig_params=True)
            else:
                model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        model_name = get_model_name(cfg)
        if not _is_direct_method(method):
            model_name = f"{model_name}+score_{method}"
        checkpoints_dir = os.path.join(cfg.exp.workspace, "checkpoints", model_name)
        logs_dir = os.path.join(cfg.exp.workspace, "logs", model_name)
        base_model = _unwrap_model(model)
        params_count, params_k, params_m = _model_param_sizes(base_model)

        if is_main:
            create_folder(checkpoints_dir)
            create_folder(logs_dir)
            _write_training_stats(cfg, checkpoints_dir, model_name, params_count=params_count)
            create_logging(logs_dir, filemode="w")
            logging.info(cfg)
            logging.info(f"Using {device}.")
            logging.info(f"Model Params: {params_count} ({params_k:.3f} K, {params_m:.3f} M)")
            logging.info(
                f"Distributed={dist_enabled}, rank={rank}/{world_size}, "
                f"parallelism={'FSDP' if use_fsdp else ('DDP' if dist_enabled else 'single')}, "
                f"global_batch={cfg.exp.batch_size}, per_rank_batch={per_rank_batch_size}"
            )
            if not dist_enabled:
                logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        else:
            logging.basicConfig(level=logging.ERROR)

        if dist_enabled:
            dist.barrier()

        start_iteration = 0
        init_phase = _phase_at_iteration(train_mode, start_iteration, switch_iteration)
        _apply_train_phase(_unwrap_model(model), init_phase)

        train_loader, eval_loaders = build_dataloaders(cfg)
        _shard_train_loader(train_loader, rank, world_size, per_rank_batch_size)

        opt_name = str(cfg.exp.optim).lower()
        if opt_name == "adamw":
            optimizer = AdamW(model.parameters(), lr=cfg.exp.learnrate, weight_decay=cfg.exp.weight_decay)
        elif opt_name == "adam":
            optimizer = Adam(model.parameters(), lr=cfg.exp.learnrate, weight_decay=cfg.exp.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.exp.optim}")

        if is_main:
            init_wandb(cfg)
        else:
            os.environ["WANDB_MODE"] = "disabled"

        iteration = start_iteration
        train_bgn_time = time.time()
        train_loss = 0.0
        train_loss_steps = 0
        current_phase = None

        early_phase = 0
        early_step = int(early_phase * 0.1) if early_phase > 0 else 0

        evaluator_model = model if use_fsdp else _unwrap_model(model)
        evaluator = SegmentEvaluator(evaluator_model, cfg)
        loss_fn = get_loss_func(cfg=cfg)
        optimizer.zero_grad(set_to_none=True)

        for batch_data_dict in train_loader:
            phase = _phase_at_iteration(train_mode, iteration, switch_iteration)
            if phase != current_phase:
                _apply_train_phase(_unwrap_model(model), phase)
                current_phase = phase
                if is_main:
                    logging.info(f"Train phase switched to: {phase} at iteration {iteration}")

            if cfg.exp.decay and iteration % cfg.exp.reduce_iteration == 0 and iteration != 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.9

            model.train()
            audio, cond, batch_torch = _prepare_batch(batch_data_dict, device, cond_keys, target_rolls)
            out = model(audio, cond)
            loss = loss_fn(cfg, out, batch_torch, cond_dict=cond)

            train_loss += float(loss.item())
            train_loss_steps += 1

            if is_main:
                # Avoid CUDA tensor repr path in print(), which can mask the real kernel error.
                print(f"iter {iteration} loss {float(loss.detach().item()):.6f}")
                log_velocity_rolls(cfg, iteration, {"velocity_output": out["vel_corr"]}, batch_torch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            should_eval = (
                (iteration < early_phase and early_step > 0 and iteration % early_step == 0)
                or (iteration >= early_phase and iteration % cfg.exp.eval_iteration == 0)
            )
            if should_eval:
                if dist_enabled:
                    dist.barrier()

                avg_train_loss = None
                if train_loss_steps > 0 and iteration != 0:
                    loss_steps = torch.tensor([train_loss, float(train_loss_steps)], device=device)
                    if dist_enabled:
                        dist.all_reduce(loss_steps, op=dist.ReduceOp.SUM)
                    if float(loss_steps[1].item()) > 0:
                        avg_train_loss = float((loss_steps[0] / loss_steps[1]).item())

                do_eval_here = use_fsdp or is_main
                if do_eval_here:
                    train_stats = _select_velocity_metrics(evaluator.evaluate(eval_loaders["train"]))
                    maestro_stats = _select_velocity_metrics(evaluator.evaluate(eval_loaders["maestro"]))
                    smd_stats = _select_velocity_metrics(evaluator.evaluate(eval_loaders["smd"]))
                    maps_stats = _select_velocity_metrics(evaluator.evaluate(eval_loaders["maps"]))

                if is_main:
                    logging.info("------------------------------------")
                    logging.info(f"Iteration: {iteration}/{cfg.exp.total_iteration}")
                    train_fin_time = time.time()
                    if avg_train_loss is not None:
                        logging.info(f"    Train Loss: {avg_train_loss:.4f}")
                    logging.info(f"    Train Stat: {train_stats}")
                    logging.info(f"    Valid Maestro Stat: {maestro_stats}")
                    logging.info(f"    Valid SMD Stat: {smd_stats}")
                    logging.info(f"    Valid MAPS Stat: {maps_stats}")

                    log_payload = {
                        "iteration": iteration,
                        "train_stat": train_stats,
                        "valid_maestro_stat": maestro_stats,
                        "valid_smd_stat": smd_stats,
                        "valid_maps_stat": maps_stats,
                    }
                    if avg_train_loss is not None:
                        log_payload["train_loss"] = avg_train_loss
                    wandb.log(log_payload)

                    train_time = train_fin_time - train_bgn_time
                    validate_time = time.time() - train_fin_time
                    logging.info(
                        "Train time: {:.3f} s, validate time: {:.3f} s".format(train_time, validate_time)
                    )

                checkpoint_path = os.path.join(checkpoints_dir, f"{iteration}_iterations.pth")
                _save_checkpoint(model, checkpoint_path, use_fsdp=use_fsdp, is_main=is_main)
                if is_main:
                    logging.info(f"Model saved to {checkpoint_path}")

                train_loss = 0.0
                train_loss_steps = 0
                train_bgn_time = time.time()

                if dist_enabled:
                    dist.barrier()

            if iteration == cfg.exp.total_iteration:
                break

            iteration += 1

        if is_main:
            wandb.finish()

    finally:
        _cleanup_distributed(dist_enabled)


if __name__ == "__main__":
    initialize(config_path="./config", job_name="train_score_inf_dual", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])
    train(cfg)
