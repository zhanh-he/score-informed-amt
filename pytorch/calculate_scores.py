import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import wandb
from inference import VeloTranscription, resolve_checkpoint
from utilities import (
    TargetProcessor,
    create_folder,
    get_model_name,
    int16_to_float32,
    resolve_hdf5_dir,
    traverse_folder,
)


_FILM_TYPES = {"filmunet_pretrained", "filmunet"}
_TRANSKUN_TYPES = {"transkun_pretrained"}


def _method_name(method) -> str:
    return str(method or "direct").strip()


def _is_direct_method(method) -> bool:
    return _method_name(method) == "direct"


def _clean_name_part(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return ""
    return text


def _cond_suffix(cfg, method: str) -> str:
    if _is_direct_method(method):
        return ""
    conds = []
    for value in (cfg.model.input2, cfg.model.input3):
        text = _clean_name_part(value)
        if text and text not in conds:
            conds.append(text)
    return f"+{'_'.join(conds)}" if conds else ""


def _eval_wandb_name(cfg) -> str:
    explicit_name = _clean_name_part(getattr(cfg.wandb, "name", ""))
    if explicit_name:
        return explicit_name
    method = _method_name(cfg.score_informed.method)
    return f"eval-{cfg.dataset.test_set}-{cfg.model.type}-{method}{_cond_suffix(cfg, method)}"


def _mean_metrics(stats_dict: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        key: float(np.mean(values))
        for key, values in stats_dict.items()
        if values
    }


def _iteration_sweep(cfg) -> List[int]:
    start = int(getattr(cfg.exp, "eval_start_iteration", 0))
    end = int(getattr(cfg.exp, "eval_end_iteration", cfg.exp.total_iteration))
    step = int(
        getattr(
            cfg.exp,
            "eval_step_iteration",
            getattr(cfg.exp, "save_iteration", cfg.exp.eval_iteration),
        )
    )
    return list(range(start, end + 1, step))


def note_level_l1_per_window(
    output_segment: Dict[str, np.ndarray],
    target_segment: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Kim et al. note-level L1 error for a single segment."""
    error_rows: List[np.ndarray] = []
    num_notes = 0
    frames = min(
        output_segment["velocity_output"].shape[0],
        target_segment["velocity_roll"].shape[0],
    )
    for nth_frame in range(frames):
        gt_onset_frame = target_segment["onset_roll"][nth_frame]
        if np.count_nonzero(gt_onset_frame) == 0:
            continue
        pred_frame = output_segment["velocity_output"][nth_frame]
        gt_frame = target_segment["velocity_roll"][nth_frame]
        pred_onset = np.multiply(pred_frame, gt_onset_frame) * 128.0
        gt_onset = np.multiply(gt_frame, gt_onset_frame)
        note_error = np.abs(pred_onset - gt_onset)
        num_notes += int(np.count_nonzero(gt_onset_frame))
        error_rows.append(note_error[np.newaxis, :])
    if error_rows:
        segment_error = np.concatenate(error_rows, axis=0)
    else:
        segment_error = np.empty((0, 88), dtype=float)
    return segment_error, num_notes


def classification_error(score: np.ndarray, estimation: np.ndarray) -> Tuple[float, float, float]:
    """Binary frame classification metrics on flattened frame/key pairs."""
    score_bin = score.copy()
    estimation_bin = estimation.copy()

    score_bin[score_bin > 0] = 1
    estimation_bin[estimation_bin > 0.0001] = 1
    estimation_bin[estimation_bin <= 0.0001] = 0

    flat_score = score_bin.flatten()
    flat_est = estimation_bin.flatten()
    f1 = f1_score(flat_score, flat_est, average="macro", zero_division=0)
    precision = precision_score(flat_score, flat_est, average="macro", zero_division=0)
    recall = recall_score(flat_score, flat_est, average="macro", zero_division=0)
    return f1, precision, recall


def classification_with_mask(
    score: np.ndarray, estimation: np.ndarray, mask: np.ndarray
) -> Tuple[float, float, float]:
    """Frame-wise metrics restricted to positions where mask > 0."""
    mask_flat = mask.flatten() > 0
    if not np.any(mask_flat):
        return 0.0, 0.0, 0.0
    flat_score = (score > 0).astype(np.int32).flatten()[mask_flat]
    flat_est = (estimation > 0.0001).astype(np.int32).flatten()[mask_flat]
    f1 = f1_score(flat_score, flat_est, average="macro", zero_division=0)
    precision = precision_score(flat_score, flat_est, average="macro", zero_division=0)
    recall = recall_score(flat_score, flat_est, average="macro", zero_division=0)
    return f1, precision, recall


def num_simultaneous_notes(note_profile: Dict[str, np.ndarray], score: np.ndarray) -> int:
    start, end = note_profile["duration"]
    sim_note_count = 0
    for key in range(score.shape[0]):
        sim_note = score[key][start:end]
        if np.sum(sim_note) > 0:
            sim_note_count += 1
    return sim_note_count


def pedal_check(note_profile: Dict[str, np.ndarray], pedal: np.ndarray) -> bool:
    start, end = note_profile["duration"]
    return bool(np.sum(pedal[start:end]) > 0)


def get_midi_sound_profile(midi_vel_roll: np.ndarray) -> List[Dict[str, np.ndarray]]:
    sound_profile: List[Dict[str, np.ndarray]] = []
    for pitch, key in enumerate(midi_vel_roll):
        iszero = np.concatenate(([0], np.equal(key, 0).astype(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0]
        if ranges.size <= 2:
            continue
        temp = np.delete(ranges, [0, -1])
        sound_durations = temp.reshape(-1, 2)
        for duration in sound_durations:
            vel = midi_vel_roll[pitch, duration[0]]
            sound_profile.append({"pitch": pitch, "velocity": vel, "duration": duration})
    return sound_profile


def align_prediction_to_gt_intervals(
    predicted_roll: np.ndarray,
    gt_velocity_roll: np.ndarray,
) -> np.ndarray:
    """Project prediction onto GT note intervals (velocity-only evaluation mode)."""
    frames = min(predicted_roll.shape[0], gt_velocity_roll.shape[0])
    pred = predicted_roll[:frames]
    gt = gt_velocity_roll[:frames]
    aligned = np.zeros_like(pred, dtype=np.float32)

    gt_t = np.transpose(gt)
    pred_t = np.transpose(pred)
    gt_profile = get_midi_sound_profile(gt_t)

    for note_profile in gt_profile:
        pitch = int(note_profile["pitch"])
        start, end = note_profile["duration"]
        if end <= start:
            continue
        pred_note = pred_t[pitch][start:end].copy()
        pred_note[pred_note <= 0.0001] = 0
        picked = float(np.max(pred_note)) if pred_note.size else 0.0
        aligned[start:end, pitch] = picked

    return aligned


def _collect_eval_arrays(output_dict_list: Sequence[Dict[str, np.ndarray]], target_list: Sequence[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect frame-aligned matrices used by Kim-style metrics."""
    score_rows: List[np.ndarray] = []
    pedal_list: List[np.ndarray] = []
    estimation_rows: List[np.ndarray] = []
    frame_mask_rows: List[np.ndarray] = []

    for target_segment, output_segment in zip(target_list, output_dict_list):
        frames = target_segment["velocity_roll"].shape[0]
        for nth_frame in range(frames):
            gt_velframe = target_segment["velocity_roll"][nth_frame]
            gt_pedal = target_segment["pedal_frame_roll"][nth_frame]
            output_vel_frame = output_segment["velocity_output"][nth_frame]

            score_rows.append(gt_velframe[np.newaxis, :])
            pedal_list.append(np.asarray(gt_pedal).reshape(1))
            estimation_rows.append(output_vel_frame[np.newaxis, :])
            frame_mask_rows.append(target_segment["frame_roll"][nth_frame][np.newaxis, :])

    if not score_rows:
        empty_2d = np.empty((0, 88), dtype=float)
        empty_1d = np.empty((0,), dtype=float)
        return empty_2d, empty_2d, empty_1d, empty_2d

    score = np.concatenate(score_rows, axis=0)
    estimation = np.concatenate(estimation_rows, axis=0)
    pedal = np.concatenate(pedal_list, axis=0)
    frame_mask = np.concatenate(frame_mask_rows, axis=0)
    return score, estimation, pedal, frame_mask


def frame_max_metrics_from_list(output_dict_list: Sequence[Dict[str, np.ndarray]], target_list: Sequence[Dict[str, np.ndarray]]) -> Tuple[float, float]:
    """Compute only frame-max MAE/STD (lightweight path for train-time evaluation)."""
    score, estimation, _, _ = _collect_eval_arrays(output_dict_list, target_list)
    if score.size == 0:
        return 0.0, 0.0

    score_t = np.transpose(score)
    estimation_t = np.transpose(estimation)
    score_sound_profile = get_midi_sound_profile(score_t)
    accum_error: List[float] = []

    for note_profile in score_sound_profile:
        start, end = note_profile["duration"]
        vel_est = estimation_t[note_profile["pitch"]][start:end].copy()
        vel_est[vel_est <= 0.0001] = 0
        max_estimation = float(np.max(vel_est) * 128.0) if vel_est.size else 0.0
        notelevel_error = abs(max_estimation - float(note_profile["velocity"]))
        accum_error.append(notelevel_error)

    frame_max_error = float(np.mean(accum_error)) if accum_error else 0.0
    std_max_error = float(np.std(accum_error)) if accum_error else 0.0
    return frame_max_error, std_max_error


def detailed_f1_metrics_from_list(output_dict_list: Sequence[Dict[str, np.ndarray]], target_list: Sequence[Dict[str, np.ndarray]]) -> Tuple[List[Dict[str, Any]], float, float, float, float, float, float]:
    """Compute detailed Kim-style metrics and per-note error profile."""
    score, estimation, pedal, frame_mask = _collect_eval_arrays(output_dict_list, target_list)
    if score.size == 0:
        return [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    f1, precision, recall = classification_error(score.copy(), estimation.copy())
    frame_f1, frame_precision, frame_recall = classification_with_mask(score.copy(), estimation.copy(), frame_mask.copy())

    score_t = np.transpose(score)
    estimation_t = np.transpose(estimation)

    score_sound_profile = get_midi_sound_profile(score_t)
    error_profile: List[Dict[str, Any]] = []

    for note_profile in score_sound_profile:
        start, end = note_profile["duration"]
        vel_est = estimation_t[note_profile["pitch"]][start:end].copy()
        vel_est[vel_est <= 0.0001] = 0

        classification_check = bool(np.sum(vel_est) > 0)
        max_estimation = float(np.max(vel_est) * 128.0) if vel_est.size else 0.0
        notelevel_error = abs(max_estimation - float(note_profile["velocity"]))
        sim_note_count = num_simultaneous_notes(note_profile, score_t)
        pedal_onoff = pedal_check(note_profile, pedal)

        error_profile.append(
            {"pitch": note_profile["pitch"],
             "duration": (int(start), int(end)),
             "note_error": notelevel_error,
             "ground_truth": float(note_profile["velocity"]),
             "estimation": max_estimation,
             "pedal_check": pedal_onoff,
             "simultaneous_notes": sim_note_count,
             "classification_check": classification_check,
            } # type: ignore
        )
    return error_profile, f1, precision, recall, frame_f1, frame_precision, frame_recall


def onset_pick_metrics_from_list(output_dict_list: Sequence[Dict[str, np.ndarray]], target_dict_list: Sequence[Dict[str, np.ndarray]]) -> Tuple[float, float]:
    """Kim et al. onset-only evaluation."""
    score_error_rows: List[np.ndarray] = []
    num_note = 0
    for output_dict_segmentseconds, target_dict_segmentseconds in zip(
        output_dict_list, target_dict_list
    ):
        segment_error, num_onset = note_level_l1_per_window(
            output_dict_segmentseconds, target_dict_segmentseconds
        )
        if segment_error.size:
            score_error_rows.append(segment_error)
        num_note += num_onset
    if num_note == 0:
        return 0.0, 0.0

    score_error = np.concatenate(score_error_rows, axis=0) if score_error_rows else np.empty((0, 88))
    mean_error = float(np.sum(score_error) / num_note)
    non_zero = score_error[score_error != 0]
    std_error = float(non_zero.std()) if non_zero.size else 0.0
    return mean_error, std_error


class KimStyleEvaluator:
    """Run HPT inference and compute Kim et al. evaluation metrics."""

    CSV_FIELDS = [
        "test_h5",
        "frame_max_error",
        "frame_max_std",
        "onset_masked_error",
        "onset_masked_std",
    ]

    def __init__(
        self,
        cfg,
        checkpoint_path: Optional[str] = None,
        results_subdir: str = "kim_eval",
    ):
        self.cfg = cfg
        default_model_name = get_model_name(cfg)
        score_method = _method_name(cfg.score_informed.method)
        if not _is_direct_method(score_method):
            default_model_name = f"{default_model_name}+score_{score_method}"
        self.model_name = default_model_name

        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
            self.ckpt_iteration = self.checkpoint_path.stem.replace("_iterations", "")
            if self.checkpoint_path.parent.name:
                self.model_name = self.checkpoint_path.parent.name
        else:
            self.checkpoint_path = resolve_checkpoint(cfg, explicit_path=None)
            self.ckpt_iteration = self.checkpoint_path.stem.replace("_iterations", "")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.transcriptor = VeloTranscription(str(self.checkpoint_path), cfg)
        self.params_count = int(
            sum(p.numel() for p in self.transcriptor.model.parameters())
        )
        self.params_count_k = float(self.params_count / 1e3)
        self.params_count_m = float(self.params_count / 1e6)

        hdf5_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
        _, self.hdf5_paths = traverse_folder(hdf5_dir)

        self.results_dir = (
            Path(cfg.exp.workspace)
            / results_subdir
            / cfg.dataset.test_set
            / self.model_name
            / f"{self.ckpt_iteration}_iterations"
        )
        self.error_dict_dir = self.results_dir / "error_dict"
        create_folder(str(self.error_dict_dir))

    def _prepare_inputs(self, target_dict: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        target_dict["exframe_roll"] = target_dict["frame_roll"] * (1 - target_dict["onset_roll"])
        model_type = str(self.cfg.model.type)
        if model_type in _FILM_TYPES:
            input2 = target_dict["frame_roll"] if self.cfg.model.kim_condition == "frame" else None
            input3 = None
        elif model_type in _TRANSKUN_TYPES:
            input2 = None
            input3 = None
        else:
            input2 = target_dict.get(f"{self.cfg.model.input2}_roll") if self.cfg.model.input2 else None
            input3 = target_dict.get(f"{self.cfg.model.input3}_roll") if self.cfg.model.input3 else None
        return input2, input3

    def _process_file(self, hdf5_path: str) -> Optional[Dict[str, np.ndarray]]:
        with h5py.File(hdf5_path, "r") as hf:
            if hf.attrs["split"].decode() != "test":
                return None
            audio = int16_to_float32(hf["waveform"][:])
            midi_events = [e.decode() for e in hf["midi_event"][:]]
            midi_events_time = hf["midi_event_time"][:]

        segment_seconds = len(audio) / self.cfg.feature.sample_rate
        target_processor = TargetProcessor(segment_seconds=segment_seconds, cfg=self.cfg)
        target_dict, _, _ = target_processor.process(
            start_time=0, midi_events_time=midi_events_time, midi_events=midi_events, extend_pedal=True
        )

        input2, input3 = self._prepare_inputs(target_dict)
        transcribed = self.transcriptor.transcribe(audio, input2, input3, midi_path=None)
        output_dict = transcribed["output_dict"]

        predicted_roll = output_dict["velocity_output"]
        if str(self.cfg.model.type) in _TRANSKUN_TYPES:
            predicted_roll = align_prediction_to_gt_intervals(
                predicted_roll=predicted_roll,
                gt_velocity_roll=target_dict["velocity_roll"],
            )

        align_len = min(predicted_roll.shape[0], target_dict["velocity_roll"].shape[0])

        output_entry = {
            "velocity_output": predicted_roll[:align_len],
        }
        target_entry = {
            "velocity_roll": target_dict["velocity_roll"][:align_len],
            "frame_roll": target_dict["frame_roll"][:align_len],
            "onset_roll": target_dict["onset_roll"][:align_len],
            "pedal_frame_roll": target_dict["pedal_frame_roll"][:align_len],
        }

        output_dict_list = [output_entry]
        target_dict_list = [target_entry]

        frame_max_error, frame_max_std = frame_max_metrics_from_list(output_dict_list, target_dict_list)
        # Commented out F1-related matrix computation for faster/lightweight eval.
        # error_profile, f1, precision, recall, frame_mask_f1, frame_mask_precision, frame_mask_recall = detailed_f1_metrics_from_list(output_dict_list, target_dict_list)
        error_profile = []
        onset_masked_error, onset_masked_std = onset_pick_metrics_from_list(output_dict_list, target_dict_list)

        return {
            "audio_name": Path(hdf5_path).name,
            "frame_max_error": frame_max_error,
            "frame_max_std": frame_max_std,
            "onset_masked_error": onset_masked_error,
            "onset_masked_std": onset_masked_std,
            "error_profile": np.array(error_profile, dtype=object),
        } # type: ignore

    def run(self) -> Dict[str, List[float]]:
        csv_path = self.results_dir / f"{self.model_name}_{self.cfg.dataset.test_set}_kim.csv"
        create_folder(str(self.results_dir))

        aggregated: Dict[str, List[float]] = {field: [] for field in self.CSV_FIELDS if field != "test_h5"}
        processed = 0

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.CSV_FIELDS)

            progress = tqdm(sorted(self.hdf5_paths), desc="Kim Eval", unit="file", ncols=80)
            for hdf5_path in progress:
                metrics = self._process_file(hdf5_path)
                if not metrics:
                    continue

                error_profile = metrics.pop("error_profile")
                audio_name = metrics["audio_name"]
                error_dict_path = self.error_dict_dir / f"{Path(audio_name).stem}_aligned.npy"
                np.save(error_dict_path, error_profile, allow_pickle=True)

                row = [audio_name] + [metrics[field] for field in self.CSV_FIELDS[1:]]
                writer.writerow(row)

                for field in aggregated.keys():
                    aggregated[field].append(float(metrics[field]))

                processed += 1
                avg_frame_err = np.mean(aggregated["frame_max_error"])
                progress.set_postfix({"frame_err": f"{avg_frame_err:.2f}"}, refresh=False)

        if not processed:
            return {}

        return aggregated


def _run_single_mode(cfg) -> None:
    evaluator = KimStyleEvaluator(cfg)
    print("=" * 80)
    print("Evaluation Mode : Kim et al. (single checkpoint)")
    print(f"Model Name      : {evaluator.model_name}")
    print(f"Test Set        : {cfg.dataset.test_set}")
    print(f"Checkpoint      : {evaluator.checkpoint_path}")
    print(
        f"Params          : {evaluator.params_count} "
        f"({evaluator.params_count_k:.3f} K, {evaluator.params_count_m:.3f} M)"
    )
    print("=" * 80)

    stats_dict = evaluator.run()
    if not stats_dict:
        print("No test files processed.")
        return

    mean_stats = _mean_metrics(stats_dict)
    print("\n===== Kim-style Average Metrics =====")
    for key, value in mean_stats.items():
        print(f"{key}: {value:.4f}")


def _run_multi_mode(cfg) -> None:
    model_name = get_model_name(cfg)
    score_method = _method_name(cfg.score_informed.method)
    if not _is_direct_method(score_method):
        model_name = f"{model_name}+score_{score_method}"
    iterations = _iteration_sweep(cfg)
    ckpt_root = Path(cfg.exp.workspace) / "checkpoints" / model_name

    summary_dir = Path(cfg.exp.workspace) / "kim_eval_summary" / cfg.dataset.test_set / model_name
    create_folder(str(summary_dir))
    summary_csv = summary_dir / "iter_summary.csv"

    print("=" * 80)
    print("Evaluation Mode : Kim et al. (multi-checkpoint)")
    print(f"Model Name      : {model_name}")
    print(f"Test Set        : {cfg.dataset.test_set}")
    print(f"Checkpoint Dir  : {ckpt_root}")
    print(
        f"Iterations      : {iterations[0]} -> {iterations[-1]} "
        f"(step={iterations[1] - iterations[0] if len(iterations) > 1 else 0})"
    )
    print("=" * 80)

    run_name = _eval_wandb_name(cfg)
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )

    summary_fields = [
        "iteration",
        "checkpoint_path",
        "params_count",
        "params_count_k",
        "params_count_m",
        "frame_max_error",
        "frame_max_std",
        "onset_masked_error",
        "onset_masked_std",
    ]

    with open(summary_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
        writer.writeheader()

        for iteration in iterations:
            ckpt_path = ckpt_root / f"{iteration}_iterations.pth"
            evaluator = KimStyleEvaluator(cfg, checkpoint_path=str(ckpt_path))
            print(
                f"[eval] iter={iteration} | params={evaluator.params_count_k:.3f} K / "
                f"{evaluator.params_count_m:.3f} M"
            )
            stats_dict = evaluator.run()

            mean_stats = _mean_metrics(stats_dict)
            row = {
                "iteration": iteration,
                "checkpoint_path": str(ckpt_path),
                "params_count": evaluator.params_count,
                "params_count_k": round(evaluator.params_count_k, 6),
                "params_count_m": round(evaluator.params_count_m, 6),
            }
            row.update({k: round(v, 6) for k, v in mean_stats.items()})
            writer.writerow(row)

            payload = {
                "iteration": iteration,
                "eval/params_count": evaluator.params_count,
                "eval/params_count_k": evaluator.params_count_k,
                "eval/params_count_m": evaluator.params_count_m,
            }
            payload.update({f"eval/{k}": v for k, v in mean_stats.items()})
            wandb.log(payload)

    wandb.finish()

    print(f"\n[done] Wrote evaluation summary: {summary_csv}")


def main() -> None:
    initialize(config_path="./config", job_name="kim_eval", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])
    mode = str(getattr(cfg.exp, "run_infer", "single")).lower()
    if mode == "multi":
        _run_multi_mode(cfg)
    else:
        _run_single_mode(cfg)


if __name__ == "__main__":
    main()
