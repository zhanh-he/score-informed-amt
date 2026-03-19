import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

from calculate_scores import (
    _FILM_TYPES,
    _TRANSKUN_TYPES,
    _consume_flag,
    align_prediction_to_gt_intervals,
    frame_max_metrics_from_list,
    mir_eval_metrics,
    onset_pick_metrics_from_list,
)
from inference import VeloTranscription, resolve_checkpoint
from utilities import (
    TargetProcessor,
    create_folder,
    get_model_name,
    int16_to_float32,
    resolve_hdf5_dir,
    traverse_folder,
)

SHIFT_PROFILES: Dict[str, List[float]] = {
    "kim_full": [
        0.0,
        -0.1, -0.2, -0.3, -0.4, -0.5, -1.0, -1.5, -2.0, -2.5,
        0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5,
    ],
    "fast": [0.0, -0.1, 0.1],
    "coarse": [0.0, -0.1, -0.2, -0.5, 0.1, 0.2, 0.5],
}


def _shifts(profile: str, mode: str) -> List[float]:
    shifts_sec = SHIFT_PROFILES[profile]
    if mode == "shifted":
        return list(shifts_sec)
    seen = set()
    dedup: List[float] = []
    for shift in shifts_sec:
        value = abs(float(shift))
        key = round(value, 6)
        if key not in seen:
            seen.add(key)
            dedup.append(value)
    return dedup


def _help() -> None:
    print("Usage:")
    print("  python pytorch/calculate_scores_unaligned.py (--fast|--coarse|--kim_full) [--enable_mireval] (--fix_shifted|--random_shifted) <hydra overrides>")


def _shift_to_frames(shift_sec: float, fps: int) -> int:
    return int(round(float(shift_sec) * float(fps)))


def _random_shift(bound_sec: float, rng: np.random.Generator) -> float:
    bound = abs(float(bound_sec))
    if bound == 0.0:
        return 0.0
    steps = int(round(bound / 0.1))
    mags = np.arange(1, steps + 1, dtype=float) * 0.1
    mag = float(rng.choice(mags))
    sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    return sign * mag


def _shift_roll(arr: np.ndarray, shift_frames: int) -> np.ndarray:
    if arr.ndim == 0:
        return arr.copy()
    n = arr.shape[0]
    if shift_frames == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if abs(shift_frames) >= n:
        return out
    if shift_frames > 0:
        out[: n - shift_frames] = arr[shift_frames:]
    else:
        s = -shift_frames
        out[s:] = arr[: n - s]
    return out


def _shift_target(base_target: Dict[str, np.ndarray], shift_frames: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in base_target.items():
        if isinstance(v, np.ndarray) and k.endswith("_roll"):
            out[k] = _shift_roll(v, shift_frames)
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    if "onset_roll" in out and "frame_roll" in out:
        out["exframe_roll"] = out["frame_roll"] * (1 - out["onset_roll"])
    return out


def _shift_notes(
    note_events: Sequence[Dict[str, float]],
    shift_sec: float,
    duration_sec: float,
) -> List[Dict[str, float]]:
    shifted: List[Dict[str, float]] = []
    for event in note_events:
        onset = float(event["onset_time"]) - float(shift_sec)
        offset = float(event["offset_time"]) - float(shift_sec)
        onset = max(0.0, min(onset, duration_sec))
        offset = max(0.0, min(offset, duration_sec))
        if offset <= onset:
            continue
        shifted.append(
            {
                "midi_note": event["midi_note"],
                "onset_time": onset,
                "offset_time": offset,
                "velocity": event["velocity"],
            }
        )
    return shifted


def _mean_dict(data: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in data.items() if v}


def _plot_path(custom_plot_path: str, output_dir: Path, base_name: str) -> Path:
    if custom_plot_path:
        custom = Path(custom_plot_path)
        if custom.suffix:
            parent, stem = custom.parent, custom.stem
        else:
            parent = custom.parent if str(custom.parent) != "." else output_dir
            stem = custom.name
    else:
        parent, stem = output_dir, base_name
    return parent / f"{stem}_together.png"


def _fig_size(size: Union[str, Sequence[float], Tuple[float, float]]) -> Tuple[float, float]:
    if isinstance(size, str):
        text = size.strip().strip("[]")
        w, h = text.split(",")
        return float(w), float(h)
    return float(size[0]), float(size[1])


def plot_unaligned_summary_csv(
    csv_path: Union[str, Path],
    metric: str = "onset_masked",
    color: Optional[str] = None,
    size: Union[str, Sequence[float], Tuple[float, float]] = (9, 5),
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    y_max_axis: Optional[float] = None,
    dpi: int = 220,
    show_plot: bool = True,
) -> Path:
    metric = str(metric).strip().lower()
    metric_specs = {
        "onset_masked": ("onset_masked_error_avg", "onset_masked_std_avg", "#1f77b4", "Onset Masked MAE", "MAE"),
        "frame_max": ("frame_max_error_avg", "frame_max_std_avg", "#d62728", "Frame Max MAE", "MAE"),
        "mir_eval_precision": ("mir_eval_precision_avg", None, "#2ca02c", "mir_eval Precision", "Precision"),
        "mir_eval_recall": ("mir_eval_recall_avg", None, "#ff7f0e", r"Recall$_{10\%}$", r"Recall$_{10\%}$"),
        "mir_eval_f1": ("mir_eval_f1_avg", None, "#9467bd", "mir_eval F1", "F1"),
    }
    metric_key, std_key, default_color, metric_label, legend_label = metric_specs.get(
        metric,
        metric_specs["frame_max"],
    )

    fig_w, fig_h = _fig_size(size)
    line_color = color or default_color

    csv_path = Path(csv_path).expanduser().resolve()
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows_sorted = sorted(rows, key=lambda x: float(x["unaligned_shift"]))
    x = np.asarray([float(r["unaligned_shift"]) for r in rows_sorted], dtype=float)
    y = np.asarray([float(r[metric_key]) for r in rows_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(x, y, marker="o", linewidth=2.2, color=line_color, label=legend_label)
    if std_key is not None:
        s = np.asarray([float(r[std_key]) for r in rows_sorted], dtype=float)
        ax.fill_between(x, y - s, y + s, alpha=0.18, color=line_color, label="STD")
    ax.axvline(0.0, linestyle="--", linewidth=1.4, alpha=0.85, color="black", label="Aligned")
    ax.set_xlabel("Score Shift (seconds)")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} vs Shift")
    ax.grid(alpha=0.25)
    if y_max_axis is not None:
        ax.set_ylim(0.0, y_max_axis)
    elif std_key is None:
        ax.set_ylim(0.0, 1.0)
    ax.legend(framealpha=0.6, loc="lower right")
    fig.tight_layout()

    output_png = None
    if out_path is not None:
        output_png = Path(out_path).expanduser().resolve()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close(fig)
    return output_png


def _plot_together(
    rows: Sequence[Dict[str, object]],
    output_png: Path,
    title: str,
    y_max_axis: Optional[float],
) -> None:
    rows_sorted = sorted(rows, key=lambda x: float(x["unaligned_shift"]))
    x = np.asarray([float(r["unaligned_shift"]) for r in rows_sorted], dtype=float)
    y_onset = np.asarray([float(r["onset_masked_error_avg"]) for r in rows_sorted], dtype=float)
    y_frame = np.asarray([float(r["frame_max_error_avg"]) for r in rows_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y_onset, marker="o", linewidth=2.2, color="#1f77b4")
    ax.plot(x, y_frame, marker="s", linewidth=2.2, color="#d62728")
    ax.axvline(0.0, linestyle="--", linewidth=1.4, alpha=0.85, color="black")
    ax.set_xlabel("Score Shift (seconds)")
    ax.set_ylabel("Mean Abs Error")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if y_max_axis is not None:
        ax.set_ylim(0.0, y_max_axis)

    ax.legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", marker="o", linewidth=2.2, label="MAE -- Onset"),
            Line2D([0], [0], color="#d62728", marker="s", linewidth=2.2, label="MAE -- Frame"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Aligned"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    print(f"[done] Wrote figure: {output_png}")
    plt.close(fig)


class UnalignedEvaluator:
    BASE_SUMMARY_FIELDS = [
        "unaligned_profile",
        "unaligned_eval_target",
        "unaligned_shift",
        "frame_max_error_avg",
        "frame_max_std_avg",
        "onset_masked_error_avg",
        "onset_masked_std_avg",
        "n_files",
    ]
    MIREVAL_FIELDS = [
        "mir_eval_precision_avg",
        "mir_eval_recall_avg",
        "mir_eval_f1_avg",
    ]

    def __init__(
        self,
        cfg,
        shift_profile: str,
        shift_mode: str,
        enable_mireval: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        self.cfg = cfg
        self.fps = int(cfg.feature.frames_per_second)
        self.score_method = str(getattr(cfg.score_informed, "method", "direct"))
        self.enable_mireval = enable_mireval
        self.shift_profile = shift_profile
        self.eval_target_mode = shift_mode
        results_subdir = "kim_eval_unaligned_random" if self.eval_target_mode == "random_shifted" else "kim_eval_unaligned"
        self.summary_fields = list(self.BASE_SUMMARY_FIELDS)
        if self.enable_mireval:
            self.summary_fields[7:7] = self.MIREVAL_FIELDS

        model_name = get_model_name(cfg)
        if self.score_method != "direct":
            model_name = f"{model_name}+score_{self.score_method}"

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else resolve_checkpoint(cfg, explicit_path=None)
        self.ckpt_iteration = self.checkpoint_path.stem.replace("_iterations", "")
        self.model_name = self.checkpoint_path.parent.name or model_name

        self.transcriptor = VeloTranscription(str(self.checkpoint_path), cfg)
        self.params_count = int(sum(p.numel() for p in self.transcriptor.model.parameters()))

        hdf5_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
        _, self.hdf5_paths = traverse_folder(hdf5_dir)

        self.results_dir = (
            Path(cfg.exp.workspace)
            / results_subdir
            / cfg.dataset.test_set
            / self.model_name
            / f"{self.ckpt_iteration}_iterations"
        )
        create_folder(str(self.results_dir))

        self.shifts_sec = _shifts(self.shift_profile, self.eval_target_mode)
        self.rng = np.random.default_rng(int(getattr(cfg.exp, "random_seed", 0)))
        self.output_stem = (
            f"{self.model_name}_{self.cfg.dataset.test_set}_"
            f"{self.shift_profile}_{self.eval_target_mode}_unaligned"
        )

    def _prepare_inputs(self, target: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        model_type = str(self.cfg.model.type)
        if model_type in _FILM_TYPES:
            return (target["frame_roll"] if self.cfg.model.kim_condition == "frame" else None), None
        if model_type in _TRANSKUN_TYPES:
            return None, None
        i2 = target.get(f"{self.cfg.model.input2}_roll") if self.cfg.model.input2 else None
        i3 = target.get(f"{self.cfg.model.input3}_roll") if self.cfg.model.input3 else None
        return i2, i3

    def _load_file(
        self,
        hdf5_path: str,
    ) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], List[Dict[str, float]]]]:
        with h5py.File(hdf5_path, "r") as hf:
            if hf.attrs["split"].decode() != "test":
                return None
            audio = int16_to_float32(hf["waveform"][:])
            midi_events = [e.decode() for e in hf["midi_event"][:]]
            midi_events_time = hf["midi_event_time"][:]

        duration = len(audio) / self.cfg.feature.sample_rate
        tp = TargetProcessor(segment_seconds=duration, cfg=self.cfg)
        target, note_events, _ = tp.process(
            start_time=0.0,
            midi_events_time=midi_events_time,
            midi_events=midi_events,
            extend_pedal=True,
        )
        if "exframe_roll" not in target:
            target["exframe_roll"] = target["frame_roll"] * (1 - target["onset_roll"])
        return audio, target, note_events

    def _eval_shift(
        self,
        audio: np.ndarray,
        target_ref: Dict[str, np.ndarray],
        note_events_ref: Sequence[Dict[str, float]],
        shift_sec: float,
    ) -> Dict[str, float]:
        applied_shift = (
            _random_shift(shift_sec, self.rng)
            if self.eval_target_mode == "random_shifted"
            else float(shift_sec)
        )
        shifted = _shift_target(target_ref, _shift_to_frames(applied_shift, self.fps))
        i2, i3 = self._prepare_inputs(shifted)
        out = self.transcriptor.transcribe(audio, i2, i3, midi_path=None)["output_dict"]["velocity_output"]
        metric_target = shifted

        if str(self.cfg.model.type) in _TRANSKUN_TYPES:
            out = align_prediction_to_gt_intervals(out, metric_target["velocity_roll"])

        t = min(out.shape[0], metric_target["velocity_roll"].shape[0])
        outputs = [{"velocity_output": out[:t]}]
        targets = [{
            "velocity_roll": metric_target["velocity_roll"][:t],
            "frame_roll": metric_target["frame_roll"][:t],
            "onset_roll": metric_target["onset_roll"][:t],
        }]
        mireval_output = {
            "velocity_output": out[:t],
            "frame_output": shifted["frame_roll"][:t],
            "onset_output": shifted["onset_roll"][:t],
            "offset_output": shifted["offset_roll"][:t],
        }
        frame_err, frame_std = frame_max_metrics_from_list(outputs, targets)
        onset_err, onset_std = onset_pick_metrics_from_list(outputs, targets)
        metrics = {
            "frame_max_error": frame_err,
            "frame_max_std": frame_std,
            "onset_masked_error": onset_err,
            "onset_masked_std": onset_std,
        }
        if self.enable_mireval:
            shifted_note_events = _shift_notes(
                note_events_ref,
                shift_sec=applied_shift,
                duration_sec=float(t / self.fps),
            )
            precision, recall, f1 = mir_eval_metrics(
                self.cfg,
                mireval_output,
                shifted_note_events,
            )
            metrics.update(
                {
                    "mir_eval_precision": precision,
                    "mir_eval_recall": recall,
                    "mir_eval_f1": f1,
                }
            )
        return metrics

    def run(self) -> List[Dict[str, object]]:
        for pattern in [
            f"{self.output_stem}*.csv",
            f"{self.output_stem}*.png",
        ]:
            for path in self.results_dir.glob(pattern):
                path.unlink(missing_ok=True)

        agg: Dict[float, Dict[str, List[float]]] = {
            s: {
                "frame_max_error": [],
                "frame_max_std": [],
                "onset_masked_error": [],
                "onset_masked_std": [],
                "mir_eval_precision": [],
                "mir_eval_recall": [],
                "mir_eval_f1": [],
            }
            for s in self.shifts_sec
        }

        progress = tqdm(sorted(self.hdf5_paths), desc="Kim Unaligned Eval", unit="file", ncols=96)
        for h5 in progress:
            loaded = self._load_file(h5)
            if loaded is None:
                continue
            audio, target_ref, note_events_ref = loaded

            for shift in self.shifts_sec:
                m = self._eval_shift(audio, target_ref, note_events_ref, shift)
                for k, v in m.items():
                    agg[shift][k].append(float(v))

            ref_shift = 0.0 if 0.0 in agg else self.shifts_sec[0]
            cur = agg[ref_shift]["frame_max_error"]
            progress.set_postfix({"shift0_frame_err": f"{float(np.mean(cur)) if cur else 0.0:.2f}"}, refresh=False)

        rows: List[Dict[str, object]] = []
        summary_csv = self.results_dir / f"{self.output_stem}_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_fields)
            writer.writeheader()
            for shift in self.shifts_sec:
                mean = _mean_dict(agg[shift])
                row: Dict[str, object] = {
                    "unaligned_profile": self.shift_profile,
                    "unaligned_eval_target": self.eval_target_mode,
                    "unaligned_shift": float(shift),
                    "frame_max_error_avg": mean.get("frame_max_error", 0.0),
                    "frame_max_std_avg": mean.get("frame_max_std", 0.0),
                    "onset_masked_error_avg": mean.get("onset_masked_error", 0.0),
                    "onset_masked_std_avg": mean.get("onset_masked_std", 0.0),
                    "n_files": float(len(agg[shift]["frame_max_error"])),
                }
                if self.enable_mireval:
                    row.update(
                        {
                            "mir_eval_precision_avg": mean.get("mir_eval_precision", 0.0),
                            "mir_eval_recall_avg": mean.get("mir_eval_recall", 0.0),
                            "mir_eval_f1_avg": mean.get("mir_eval_f1", 0.0),
                        }
                    )
                writer.writerow(row)
                rows.append(row)
        print(f"[done] Wrote summary CSV: {summary_csv}")
        return rows


def _print_shift_table(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    profile = str(rows[0].get("unaligned_profile", "unknown"))
    eval_target = str(rows[0].get("unaligned_eval_target", "reference"))
    print("\n===== Unaligned Shift Summary =====")
    print(f"profile : {profile}")
    print(f"eval target : {eval_target}")
    shift_label = "shift range(s)" if eval_target == "random_shifted" else "shift(s)"
    has_mireval = "mir_eval_f1_avg" in rows[0]
    header = f"{shift_label} | frame_max_err | frame_std | onset_masked_err | onset_std |"
    if has_mireval:
        header += " mir_recall_10% |"
    print(header)
    for r in rows:
        line = (
            f"{r['unaligned_shift']:>7.2f} | "
            f"{r['frame_max_error_avg']:>13.4f} | "
            f"{r['frame_max_std_avg']:>9.4f} | "
            f"{r['onset_masked_error_avg']:>16.4f} | "
            f"{r['onset_masked_std_avg']:>9.4f} |"
        )
        if has_mireval:
            line += f" {r['mir_eval_recall_avg']:>5.4f} |"
        print(line)


def _run_single_mode(
    cfg,
    shift_profile: str,
    shift_mode: str,
    enable_mireval: bool = False,
) -> None:
    evaluator = UnalignedEvaluator(
        cfg,
        shift_profile=shift_profile,
        shift_mode=shift_mode,
        enable_mireval=enable_mireval,
    )
    custom_plot = str(getattr(cfg.exp, "unaligned_plot_path", "")).strip()
    y_max_axis_raw = getattr(cfg.exp, "y_max_axis", None)
    y_max_axis = None if y_max_axis_raw is None else float(y_max_axis_raw)

    print("=" * 96)
    print("Evaluation Mode : Kim-style Unaligned (single checkpoint)")
    print(f"Model Name      : {evaluator.model_name}")
    print(f"Test Set        : {cfg.dataset.test_set}")
    print(f"Checkpoint      : {evaluator.checkpoint_path}")
    print(f"Params          : {evaluator.params_count} ({evaluator.params_count / 1e6:.3f} M)")
    shift_label = "Shift Ranges (sec)" if evaluator.eval_target_mode == "random_shifted" else "Shifts (sec)"
    print(f"{shift_label:<16}: {evaluator.shifts_sec}")
    print(f"Shift Profile   : {evaluator.shift_profile}")
    print(f"Eval Target     : {evaluator.eval_target_mode}")
    print(f"Enable MirEval  : {evaluator.enable_mireval}")
    print("=" * 96)

    rows = evaluator.run()
    _print_shift_table(rows)

    base_name = evaluator.output_stem
    together_png = _plot_path(custom_plot, evaluator.results_dir, base_name)
    title_base = f"Unaligned Robustness ({evaluator.model_name}, {cfg.dataset.test_set})"
    _plot_together(
        rows,
        output_png=together_png,
        title=f"{title_base} - Together",
        y_max_axis=y_max_axis,
    )


def main() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    show_help, args = _consume_flag(sys.argv[1:], "--help")
    show_help_short, args = _consume_flag(args, "-h")
    if show_help or show_help_short:
        _help()
        return

    enable_mireval, args = _consume_flag(args, "--enable_mireval")
    shift_profile = (
        "fast" if "--fast" in args else
        "coarse" if "--coarse" in args else
        "kim_full" if "--kim_full" in args else
        None
    )
    if shift_profile is None:
        _help()
        sys.exit(1)

    shift_mode = (
        "shifted" if "--fix_shifted" in args else
        "random_shifted" if "--random_shifted" in args else
        None
    )
    if shift_mode is None:
        _help()
        sys.exit(1)

    overrides = [
        arg for arg in args
        if arg not in {"--fast", "--coarse", "--kim_full", "--fix_shifted", "--random_shifted"}
    ]
    initialize(config_path="./config", job_name="kim_eval_unaligned", version_base=None)
    cfg = compose(config_name="config", overrides=overrides)
    _run_single_mode(
        cfg,
        shift_profile=shift_profile,
        shift_mode=shift_mode,
        enable_mireval=enable_mireval,
    )


if __name__ == "__main__":
    main()