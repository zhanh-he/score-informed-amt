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
from omegaconf import OmegaConf
from tqdm import tqdm

from calculate_scores import (
    _FILM_TYPES,
    _TRANSKUN_TYPES,
    align_prediction_to_gt_intervals,
    frame_max_metrics_from_list,
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
    "wide": [
        0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0,
        0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
    ],
}


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float_list(raw) -> List[float]:
    if raw is None:
        return []
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, str):
        text = raw.strip().strip("[]")
        if not text:
            return []
        return [float(t.strip()) for t in text.split(",") if t.strip()]
    return [float(x) for x in raw]


def _resolve_profile_and_shifts(cfg) -> Tuple[str, List[float]]:
    explicit = _to_float_list(getattr(cfg.exp, "unaligned_shifts", None))
    if explicit:
        shifts = explicit
        profile = "custom"
    else:
        profile = str(getattr(cfg.exp, "unaligned_profile", "kim_full")).lower()
        shifts = SHIFT_PROFILES.get(profile, SHIFT_PROFILES["kim_full"])

    seen = set()
    dedup: List[float] = []
    for s in shifts:
        key = round(float(s), 6)
        if key not in seen:
            seen.add(key)
            dedup.append(float(s))
    return profile, dedup


def _shift_frames(shift_sec: float, fps: int) -> int:
    return int(round(float(shift_sec) * float(fps)))


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


def _build_shifted_target(base_target: Dict[str, np.ndarray], shift_frames: int) -> Dict[str, np.ndarray]:
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


def _mean_dict(data: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in data.items() if v}


def _resolve_plot_paths(custom_plot_path: str, output_dir: Path, base_name: str) -> Tuple[Path, Path, Path]:
    if custom_plot_path:
        custom = Path(custom_plot_path)
        if custom.suffix:
            parent, stem = custom.parent, custom.stem
        else:
            parent = custom.parent if str(custom.parent) != "." else output_dir
            stem = custom.name
    else:
        parent, stem = output_dir, base_name
    return (
        parent / f"{stem}_frame_max.png",
        parent / f"{stem}_onset_masked.png",
        parent / f"{stem}_together.png",
    )


def _parse_fig_size(size: Union[str, Sequence[float], Tuple[float, float]]) -> Tuple[float, float]:
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
    if metric == "onset_masked":
        metric_key, std_key = "onset_masked_error_avg", "onset_masked_std_avg"
        default_color = "#1f77b4"
        metric_label = "Onset Masked MAE"
    else:
        metric_key, std_key = "frame_max_error_avg", "frame_max_std_avg"
        default_color = "#d62728"
        metric_label = "Frame Max MAE"

    fig_w, fig_h = _parse_fig_size(size)
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
    s = np.asarray([float(r[std_key]) for r in rows_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(x, y, marker="o", linewidth=2.2, color=line_color, label="MAE")
    ax.fill_between(x, y - s, y + s, alpha=0.18, color=line_color, label="STD")
    ax.axvline(0.0, linestyle="--", linewidth=1.4, alpha=0.85, color="black", label="Aligned")
    ax.set_xlabel("Score Shift (seconds)")
    ax.set_ylabel("Mean Abs Error")
    ax.set_title(title or f"{metric_label} vs Shift")
    ax.grid(alpha=0.25)
    if y_max_axis is not None:
        ax.set_ylim(0.0, y_max_axis)
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


def _plot_one(
    rows: Sequence[Dict[str, object]],
    output_png: Path,
    title: str,
    show_plot: bool,
    metric_key: str,
    std_key: str,
    color: str,
    y_max_axis: Optional[float],
) -> None:
    rows_sorted = sorted(rows, key=lambda x: float(x["unaligned_shift"]))
    x = np.asarray([float(r["unaligned_shift"]) for r in rows_sorted], dtype=float)
    y = np.asarray([float(r[metric_key]) for r in rows_sorted], dtype=float)
    s = np.asarray([float(r[std_key]) for r in rows_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, marker="o", linewidth=2.2, color=color)
    ax.fill_between(x, y - s, y + s, alpha=0.18, color=color)
    ax.axvline(0.0, linestyle="--", linewidth=1.4, alpha=0.85, color="black")
    ax.set_xlabel("Score Shift (seconds)")
    ax.set_ylabel("Mean Abs Error")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if y_max_axis is not None:
        ax.set_ylim(0.0, y_max_axis)

    ax.legend(
        handles=[
            Line2D([0], [0], color=color, marker="o", linewidth=2.2, label="MAE"),
            Patch(facecolor=color, alpha=0.18, edgecolor="none", label="STD"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Aligned"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    print(f"[done] Wrote figure: {output_png}")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_together(
    rows: Sequence[Dict[str, object]],
    output_png: Path,
    title: str,
    show_plot: bool,
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
    if show_plot:
        plt.show()
    plt.close(fig)


class UnalignedEvaluator:
    SUMMARY_FIELDS = [
        "unaligned_profile",
        "unaligned_eval_target",
        "unaligned_shift",
        "frame_max_error_avg",
        "frame_max_std_avg",
        "onset_masked_error_avg",
        "onset_masked_std_avg",
        "n_files",
    ]

    def __init__(self, cfg, checkpoint_path: Optional[str] = None, results_subdir: str = "kim_eval_unaligned"):
        self.cfg = cfg
        self.fps = int(cfg.feature.frames_per_second)
        self.score_method = str(getattr(cfg.score_informed, "method", "direct"))
        self.eval_target_mode = str(getattr(cfg.exp, "unaligned_eval_target", "reference")).strip().lower()
        if self.eval_target_mode not in {"reference", "shifted"}:
            raise ValueError("exp.unaligned_eval_target must be 'reference' or 'shifted'.")

        model_name = get_model_name(cfg)
        if self.score_method != "direct":
            model_name = f"{model_name}+score_{self.score_method}"

        if checkpoint_path:
            ckpt_path = Path(checkpoint_path)
        else:
            ckpt_path = resolve_checkpoint(cfg, explicit_path=None)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.checkpoint_path = ckpt_path
        self.ckpt_iteration = ckpt_path.stem.replace("_iterations", "")
        self.model_name = ckpt_path.parent.name or model_name

        self.transcriptor = VeloTranscription(str(ckpt_path), cfg)
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

        self.shift_profile, self.shifts_sec = _resolve_profile_and_shifts(cfg)

    def _prepare_inputs(self, target: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        model_type = str(self.cfg.model.type)
        if model_type in _FILM_TYPES:
            return (target["frame_roll"] if self.cfg.model.kim_condition == "frame" else None), None
        if model_type in _TRANSKUN_TYPES:
            return None, None
        i2 = target.get(f"{self.cfg.model.input2}_roll") if self.cfg.model.input2 else None
        i3 = target.get(f"{self.cfg.model.input3}_roll") if self.cfg.model.input3 else None
        return i2, i3

    def _load_reference(self, hdf5_path: str) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], str]]:
        with h5py.File(hdf5_path, "r") as hf:
            if hf.attrs["split"].decode() != "test":
                return None
            audio = int16_to_float32(hf["waveform"][:])
            midi_events = [e.decode() for e in hf["midi_event"][:]]
            midi_events_time = hf["midi_event_time"][:]
            name = Path(hdf5_path).name

        duration = len(audio) / self.cfg.feature.sample_rate
        tp = TargetProcessor(segment_seconds=duration, cfg=self.cfg)
        target, _, _ = tp.process(
            start_time=0.0,
            midi_events_time=midi_events_time,
            midi_events=midi_events,
            extend_pedal=True,
        )
        if "exframe_roll" not in target:
            target["exframe_roll"] = target["frame_roll"] * (1 - target["onset_roll"])
        return audio, target, name

    def _predict_metrics(self, audio: np.ndarray, target_ref: Dict[str, np.ndarray], shift_sec: float) -> Dict[str, float]:
        shifted = _build_shifted_target(target_ref, _shift_frames(shift_sec, self.fps))
        i2, i3 = self._prepare_inputs(shifted)
        out = self.transcriptor.transcribe(audio, i2, i3, midi_path=None)["output_dict"]["velocity_output"]
        metric_target = shifted if self.eval_target_mode == "shifted" else target_ref

        if str(self.cfg.model.type) in _TRANSKUN_TYPES:
            out = align_prediction_to_gt_intervals(out, metric_target["velocity_roll"])

        t = min(out.shape[0], metric_target["velocity_roll"].shape[0])
        outputs = [{"velocity_output": out[:t]}]
        targets = [{
            "velocity_roll": metric_target["velocity_roll"][:t],
            "frame_roll": metric_target["frame_roll"][:t],
            "onset_roll": metric_target["onset_roll"][:t],
            "pedal_frame_roll": metric_target["pedal_frame_roll"][:t],
        }]
        frame_err, frame_std = frame_max_metrics_from_list(outputs, targets)
        onset_err, onset_std = onset_pick_metrics_from_list(outputs, targets)
        return {
            "frame_max_error": frame_err,
            "frame_max_std": frame_std,
            "onset_masked_error": onset_err,
            "onset_masked_std": onset_std,
        }

    def _validate_note_editor_shift_input(self, target_ref: Dict[str, np.ndarray]) -> None:
        if self.score_method != "note_editor" or len(self.shifts_sec) <= 1:
            return
        z = 0.0 if 0.0 in self.shifts_sec else self.shifts_sec[0]
        base = _build_shifted_target(target_ref, _shift_frames(z, self.fps))
        b2, b3 = self._prepare_inputs(base)
        for s in self.shifts_sec:
            if abs(float(s) - float(z)) < 1e-9:
                continue
            cur = _build_shifted_target(target_ref, _shift_frames(s, self.fps))
            c2, c3 = self._prepare_inputs(cur)
            d2 = 0.0 if (b2 is None or c2 is None) else float(np.mean(np.abs(b2 - c2)))
            d3 = 0.0 if (b3 is None or c3 is None) else float(np.mean(np.abs(b3 - c3)))
            if d2 > 0.0 or d3 > 0.0:
                return
        raise RuntimeError("Shifted conditioning is identical across shifts for note_editor.")

    def run(self) -> List[Dict[str, object]]:
        for pattern in [
            f"{self.model_name}_{self.cfg.dataset.test_set}_shift_*_kim.csv",
            f"{self.model_name}_{self.cfg.dataset.test_set}_unaligned_curve.png",
            f"{self.model_name}_{self.cfg.dataset.test_set}_unaligned_summary_curve.png",
        ]:
            for path in self.results_dir.glob(pattern):
                path.unlink(missing_ok=True)

        agg: Dict[float, Dict[str, List[float]]] = {
            s: {
                "frame_max_error": [],
                "frame_max_std": [],
                "onset_masked_error": [],
                "onset_masked_std": [],
            }
            for s in self.shifts_sec
        }

        progress = tqdm(sorted(self.hdf5_paths), desc="Kim Unaligned Eval", unit="file", ncols=96)
        for h5 in progress:
            loaded = self._load_reference(h5)
            if loaded is None:
                continue
            audio, target_ref, audio_name = loaded

            self._validate_note_editor_shift_input(target_ref)

            for shift in self.shifts_sec:
                m = self._predict_metrics(audio, target_ref, shift)
                for k, v in m.items():
                    agg[shift][k].append(float(v))

            ref_shift = 0.0 if 0.0 in agg else self.shifts_sec[0]
            cur = agg[ref_shift]["frame_max_error"]
            progress.set_postfix({"shift0_frame_err": f"{float(np.mean(cur)) if cur else 0.0:.2f}"}, refresh=False)

        rows: List[Dict[str, object]] = []
        summary_csv = self.results_dir / f"{self.model_name}_{self.cfg.dataset.test_set}_unaligned_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_FIELDS)
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
                writer.writerow(row)
                rows.append(row)
        print(f"[done] Wrote summary CSV: {summary_csv}")
        return rows


def _print_shift_table(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No summary rows.")
    profile = str(rows[0].get("unaligned_profile", "unknown"))
    eval_target = str(rows[0].get("unaligned_eval_target", "reference"))
    print("\n===== Unaligned Shift Summary =====")
    print(f"profile : {profile}")
    print(f"eval target : {eval_target}")
    print("shift(s) | frame_max_err | frame_std | onset_masked_err | onset_std | n_files")
    for r in rows:
        print(
            f"{r['unaligned_shift']:>7.2f} | "
            f"{r['frame_max_error_avg']:>13.4f} | "
            f"{r['frame_max_std_avg']:>9.4f} | "
            f"{r['onset_masked_error_avg']:>16.4f} | "
            f"{r['onset_masked_std_avg']:>9.4f} | "
            f"{int(r['n_files']):>7d}"
        )


def _run_single_mode(cfg) -> None:
    evaluator = UnalignedEvaluator(cfg)
    save_plot = _as_bool(getattr(cfg.exp, "unaligned_save_plot", True), True)
    show_plot = _as_bool(getattr(cfg.exp, "unaligned_show_plot", False), False)
    custom_plot = str(getattr(cfg.exp, "unaligned_plot_path", "")).strip()
    y_max_axis_raw = getattr(cfg.exp, "y_max_axis", None)
    y_max_axis = None if y_max_axis_raw is None else float(y_max_axis_raw)

    print("=" * 96)
    print("Evaluation Mode : Kim-style Unaligned (single checkpoint)")
    print(f"Model Name      : {evaluator.model_name}")
    print(f"Test Set        : {cfg.dataset.test_set}")
    print(f"Checkpoint      : {evaluator.checkpoint_path}")
    print(f"Params          : {evaluator.params_count} ({evaluator.params_count / 1e6:.3f} M)")
    print(f"Shifts (sec)    : {evaluator.shifts_sec}")
    print(f"Shift Profile   : {evaluator.shift_profile}")
    print(f"Eval Target     : {evaluator.eval_target_mode}")
    print(f"Y Max Axis      : {y_max_axis}")
    print("=" * 96)

    rows = evaluator.run()
    _print_shift_table(rows)

    if save_plot:
        base_name = f"{evaluator.model_name}_{cfg.dataset.test_set}_unaligned"
        frame_png, onset_png, together_png = _resolve_plot_paths(custom_plot, evaluator.results_dir, base_name)
        title_base = f"Unaligned Robustness ({evaluator.model_name}, {cfg.dataset.test_set})"
        _plot_one(
            rows,
            output_png=onset_png,
            title=f"{title_base} - Onset Masked",
            show_plot=show_plot,
            metric_key="onset_masked_error_avg",
            std_key="onset_masked_std_avg",
            color="#1f77b4",
            y_max_axis=y_max_axis,
        )
        _plot_one(
            rows,
            output_png=frame_png,
            title=f"{title_base} - Frame Max",
            show_plot=show_plot,
            metric_key="frame_max_error_avg",
            std_key="frame_max_std_avg",
            color="#d62728",
            y_max_axis=y_max_axis,
        )
        _plot_together(
            rows,
            output_png=together_png,
            title=f"{title_base} - Together",
            show_plot=show_plot,
            y_max_axis=y_max_axis,
        )


def main() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    shortcut_flags = {
        "--fast": "fast",
        "--coarse": "coarse",
        "--wide": "wide",
        "--kim_full": "kim_full",
    }
    overrides: List[str] = []
    selected_profile: Optional[str] = None
    for arg in sys.argv[1:]:
        if arg in shortcut_flags:
            selected_profile = shortcut_flags[arg]
        else:
            overrides.append(arg)
    if selected_profile is not None:
        overrides.append(f"+exp.unaligned_profile={selected_profile}")

    initialize(config_path="./config", job_name="kim_eval_unaligned", version_base=None)
    cfg = compose(config_name="config", overrides=overrides)
    run_mode = str(getattr(cfg.exp, "run_infer", "single")).lower()
    if run_mode != "single":
        raise ValueError(f"calculate_scores_unaligned.py is single-only. Got exp.run_infer={run_mode!r}.")
    _run_single_mode(cfg)


if __name__ == "__main__":
    main()
