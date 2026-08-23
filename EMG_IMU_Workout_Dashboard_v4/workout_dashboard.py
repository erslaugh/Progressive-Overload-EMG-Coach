
import os
import csv
import json
import math
import zipfile
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks, resample_poly


# -----------------------------
# Configuration / calibration
# -----------------------------
DEFAULT_CONFIG = {
    "emg_bandpass_hz": [20.0, 450.0],
    "emg_rms_window_ms": 100.0,
    "rep_min_distance_s": 0.7,
    "rep_prominence_fraction": 0.15,
    "sync_activity_threshold_fraction": 0.20,

    # PROVISIONAL decision thresholds.
    # Replace these after collecting calibration/training data.
    "velocity_drop_stop": 0.30,
    "velocity_drop_more_reps": 0.15,
    "rms_rise_stop": 0.35,
    "rms_rise_more_reps": 0.15,
    "jerk_rise_stop": 0.50,
    "minimum_good_reps_for_increase": 8,
    "maximum_velocity_drop_for_increase": 0.10,
    "maximum_rms_rise_for_increase": 0.15,
}

def load_config(config_path=None):
    cfg = dict(DEFAULT_CONFIG)
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


# -----------------------------
# Helpers
# -----------------------------
def _safe_float_array(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _normalize(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo, hi = np.nanpercentile(x, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _linear_slope(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _fraction_change(first, last, eps=1e-9):
    if not np.isfinite(first) or not np.isfinite(last):
        return 0.0
    return float((last - first) / max(abs(first), eps))


def _first_activity_time(t, activity, threshold_fraction):
    if len(t) == 0 or len(activity) == 0:
        return 0.0
    a = _normalize(activity)
    idx = np.flatnonzero(a >= threshold_fraction)
    return float(t[idx[0]]) if len(idx) else float(t[0])


# -----------------------------
# EMG processing
# -----------------------------
def load_emg_wav(path):
    fs, y = wavfile.read(path)

    if y.ndim > 1:
        y = y[:, 0]

    if np.issubdtype(y.dtype, np.integer):
        maxv = max(abs(np.iinfo(y.dtype).min), np.iinfo(y.dtype).max)
        y = y.astype(np.float64) / maxv
    else:
        y = y.astype(np.float64)

    y = y - np.nanmedian(y)
    t = np.arange(len(y), dtype=float) / float(fs)
    return fs, t, y


def bandpass_emg(y, fs, low=20.0, high=450.0):
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    low = min(low, high * 0.5)
    if low <= 0 or high <= low:
        return y.copy()
    b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, y)


def rolling_rms(y, fs, window_ms=100.0):
    n = max(1, int(round(fs * window_ms / 1000.0)))
    kernel = np.ones(n, dtype=float) / n
    return np.sqrt(np.convolve(y * y, kernel, mode="same"))


def process_emg(path, cfg):
    fs, t, raw = load_emg_wav(path)
    low, high = cfg["emg_bandpass_hz"]
    filtered = bandpass_emg(raw, fs, low, high)
    rms = rolling_rms(filtered, fs, cfg["emg_rms_window_ms"])
    return {
        "fs": fs,
        "t": t,
        "raw": raw,
        "filtered": filtered,
        "rms": rms,
    }


# -----------------------------
# IMU processing
# -----------------------------
def _read_csv_flexible(path):
    # sep=None lets pandas infer comma / semicolon / tab in most exports.
    return pd.read_csv(path, sep=None, engine="python")


def _find_time_col(df):
    for c in df.columns:
        if "time" in str(c).lower():
            return c
    raise ValueError(f"Could not find a time column in {list(df.columns)}")


def _xyz_cols(df):
    cols = []
    for axis in ["x", "y", "z"]:
        matches = [c for c in df.columns if str(c).strip().lower().startswith(axis)]
        if matches:
            cols.append(matches[0])
    if len(cols) < 3:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        time_col = _find_time_col(df)
        numeric = [c for c in numeric if c != time_col]
        cols = numeric[:3]
    if len(cols) < 3:
        raise ValueError(f"Could not identify X/Y/Z columns in {list(df.columns)}")
    return cols


def _classify_csv(name, df):
    s = name.lower()
    cols = " ".join(map(str, df.columns)).lower()
    if "gyro" in s or "angular" in s or "rad/s" in cols:
        return "gyro"
    if "acc" in s or "m/s" in cols:
        return "acc"
    return None


def load_imu_zip(zip_path):
    tempdir = tempfile.TemporaryDirectory()
    root = Path(tempdir.name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    csv_paths = list(root.rglob("*.csv"))
    if not csv_paths:
        tempdir.cleanup()
        raise ValueError("No CSV files were found inside the IMU ZIP.")

    data = {}
    diagnostics = []

    for p in csv_paths:
        try:
            df = _read_csv_flexible(p)
            kind = _classify_csv(p.name, df)
            diagnostics.append((p.name, list(df.columns), kind))
            if kind and kind not in data:
                data[kind] = (p, df)
        except Exception:
            pass

    if "gyro" not in data or "acc" not in data:
        tempdir.cleanup()
        detail = "\n".join(f"{n}: {c} -> {k}" for n, c, k in diagnostics)
        raise ValueError(
            "Could not identify both gyroscope and accelerometer CSV files.\n\n"
            + detail
        )

    result = {}
    for kind, (_, df) in data.items():
        tc = _find_time_col(df)
        xyz = _xyz_cols(df)
        t = pd.to_numeric(df[tc], errors="coerce").to_numpy(float)
        arr = df[xyz].apply(pd.to_numeric, errors="coerce").to_numpy(float)

        good = np.isfinite(t) & np.all(np.isfinite(arr), axis=1)
        t, arr = t[good], arr[good]
        order = np.argsort(t)
        t, arr = t[order], arr[order]

        # Start each IMU stream at t=0; absolute offset is handled in synchronization.
        t = t - t[0]

        result[kind] = {
            "t": t,
            "xyz": arr,
            "mag": np.linalg.norm(arr, axis=1),
            "columns": xyz,
        }

    result["_tempdir"] = tempdir
    return result


def dominant_gyro_axis(gyro_xyz):
    # Dynamic range is more useful here than total magnitude for curling.
    ranges = np.nanpercentile(gyro_xyz, 95, axis=0) - np.nanpercentile(gyro_xyz, 5, axis=0)
    return int(np.nanargmax(ranges))


def detect_reps(imu, cfg):
    t = imu["gyro"]["t"]
    xyz = imu["gyro"]["xyz"]
    axis = dominant_gyro_axis(xyz)
    signal = xyz[:, axis]

    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid gyroscope timestamps.")
    fs = 1.0 / dt

    centered = signal - np.nanmedian(signal)
    amplitude = np.abs(centered)
    p95 = np.nanpercentile(amplitude, 95)
    prominence = max(1e-6, cfg["rep_prominence_fraction"] * p95)
    distance = max(1, int(cfg["rep_min_distance_s"] * fs))

    peaks, props = find_peaks(amplitude, distance=distance, prominence=prominence)

    if len(peaks) < 2:
        raise ValueError(
            f"Only {len(peaks)} movement peak(s) detected. "
            "Try a longer recording or adjust rep detection thresholds."
        )

    return {
        "axis": axis,
        "signal": signal,
        "peaks": peaks,
        "peak_times": t[peaks],
        "peak_values": signal[peaks],
        "fs": fs,
    }


def compute_imu_rep_features(imu, reps):
    gt = imu["gyro"]["t"]
    gs = reps["signal"]
    at = imu["acc"]["t"]
    amag = imu["acc"]["mag"]

    times = reps["peak_times"]
    features = []

    for i, tp in enumerate(times):
        if i == 0:
            left = max(0.0, tp - 0.45)
        else:
            left = (times[i - 1] + tp) / 2.0

        if i == len(times) - 1:
            right = min(float(gt[-1]), tp + 0.45)
        else:
            right = (tp + times[i + 1]) / 2.0

        gmask = (gt >= left) & (gt <= right)
        amask = (at >= left) & (at <= right)

        peak_speed = float(np.nanmax(np.abs(gs[gmask]))) if np.any(gmask) else np.nan

        jerk_rms = np.nan
        if np.count_nonzero(amask) >= 3:
            a_seg = amag[amask]
            t_seg = at[amask]
            jerk = np.gradient(a_seg, t_seg)
            jerk_rms = float(np.sqrt(np.nanmean(jerk ** 2)))

        features.append({
            "rep": i + 1,
            "imu_peak_time_s": float(tp),
            "peak_angular_speed": peak_speed,
            "jerk_rms": jerk_rms,
        })

    return pd.DataFrame(features)


# -----------------------------
# Synchronization
# -----------------------------
def estimate_sync_offset(emg, imu, cfg):
    """
    Estimate EMG_time - IMU_time from first meaningful activity onset.

    This makes no assumption that the recordings started together.
    It is a practical fallback for hackathon/prototype data.
    A deliberate synchronization event is preferred for calibration.
    """
    emg_on = _first_activity_time(
        emg["t"], emg["rms"], cfg["sync_activity_threshold_fraction"]
    )

    imu_activity = np.abs(reps_signal_for_sync(imu))
    imu_on = _first_activity_time(
        imu["gyro"]["t"], imu_activity, cfg["sync_activity_threshold_fraction"]
    )
    return emg_on - imu_on, emg_on, imu_on


def reps_signal_for_sync(imu):
    xyz = imu["gyro"]["xyz"]
    axis = dominant_gyro_axis(xyz)
    return xyz[:, axis] - np.nanmedian(xyz[:, axis])


# -----------------------------
# Combined rep features
# -----------------------------
def add_emg_features_by_rep(rep_df, emg, sync_offset_s):
    """
    Map each IMU rep window into EMG time using:
        EMG time = IMU time + sync_offset_s
    """
    peak_times = rep_df["imu_peak_time_s"].to_numpy(float)
    emg_t = emg["t"]
    emg_rms = emg["rms"]

    emg_means = []
    emg_peaks = []

    for i, tp in enumerate(peak_times):
        if i == 0:
            left = tp - 0.45
        else:
            left = (peak_times[i - 1] + tp) / 2.0

        if i == len(peak_times) - 1:
            right = tp + 0.45
        else:
            right = (tp + peak_times[i + 1]) / 2.0

        left += sync_offset_s
        right += sync_offset_s

        mask = (emg_t >= left) & (emg_t <= right)
        if np.any(mask):
            emg_means.append(float(np.nanmean(emg_rms[mask])))
            emg_peaks.append(float(np.nanmax(emg_rms[mask])))
        else:
            emg_means.append(np.nan)
            emg_peaks.append(np.nan)

    rep_df = rep_df.copy()
    rep_df["emg_rms_mean"] = emg_means
    rep_df["emg_rms_peak"] = emg_peaks
    return rep_df


# -----------------------------
# Recommendation logic
# -----------------------------
def summarize_set(rep_df):
    n = len(rep_df)
    if n == 0:
        raise ValueError("No reps available for summary.")

    k = max(1, min(3, n // 3 if n >= 6 else 1))

    speed = rep_df["peak_angular_speed"].to_numpy(float)
    rms = rep_df["emg_rms_mean"].to_numpy(float)
    jerk = rep_df["jerk_rms"].to_numpy(float)

    speed_first = float(np.nanmean(speed[:k]))
    speed_last = float(np.nanmean(speed[-k:]))
    rms_first = float(np.nanmean(rms[:k]))
    rms_last = float(np.nanmean(rms[-k:]))
    jerk_first = float(np.nanmean(jerk[:k]))
    jerk_last = float(np.nanmean(jerk[-k:]))

    velocity_drop = -_fraction_change(speed_first, speed_last)
    rms_rise = _fraction_change(rms_first, rms_last)
    jerk_rise = _fraction_change(jerk_first, jerk_last)

    return {
        "n_reps": int(n),
        "velocity_drop_fraction": float(velocity_drop),
        "rms_rise_fraction": float(rms_rise),
        "jerk_rise_fraction": float(jerk_rise),
        "speed_first": speed_first,
        "speed_last": speed_last,
        "rms_first": rms_first,
        "rms_last": rms_last,
        "jerk_first": jerk_first,
        "jerk_last": jerk_last,
    }


def recommend(summary, cfg):
    vd = summary["velocity_drop_fraction"]
    rr = summary["rms_rise_fraction"]
    jr = summary["jerk_rise_fraction"]
    n = summary["n_reps"]

    # Safety-oriented prototype logic:
    # severe slowing / rising jerk / strong EMG rise => stop
    if (
        vd >= cfg["velocity_drop_stop"]
        or rr >= cfg["rms_rise_stop"]
        or jr >= cfg["jerk_rise_stop"]
    ):
        return (
            "STOP / END SET",
            "Strong fatigue or movement-quality deterioration was detected."
        )

    # Moderate fatigue => keep same weight, perform more reps only if form remains acceptable.
    if (
        vd >= cfg["velocity_drop_more_reps"]
        or rr >= cfg["rms_rise_more_reps"]
    ):
        return (
            "KEEP WEIGHT",
            "Moderate fatigue was detected. Keep the same load rather than increasing it."
        )

    # Very stable set with enough reps => candidate to increase next set/session.
    if (
        n >= cfg["minimum_good_reps_for_increase"]
        and vd <= cfg["maximum_velocity_drop_for_increase"]
        and rr <= cfg["maximum_rms_rise_for_increase"]
    ):
        return (
            "INCREASE WEIGHT",
            "The set remained relatively stable with enough completed reps."
        )

    return (
        "DO MORE REPS / KEEP WEIGHT",
        "The set looks stable, but there is not enough evidence yet to increase the load."
    )


# -----------------------------
# Pipeline entry point
# -----------------------------
def run_pipeline(subject_id, weight, units, exercise, emg_wav_path, imu_zip_path,
                 output_dir=None, config_path=None):
    cfg = load_config(config_path)

    emg = process_emg(emg_wav_path, cfg)
    imu = load_imu_zip(imu_zip_path)

    try:
        reps = detect_reps(imu, cfg)
        rep_df = compute_imu_rep_features(imu, reps)

        sync_offset_s, emg_on, imu_on = estimate_sync_offset(emg, imu, cfg)
        rep_df = add_emg_features_by_rep(rep_df, emg, sync_offset_s)

        summary = summarize_set(rep_df)
        decision, reason = recommend(summary, cfg)

        metadata = {
            "subject_id": subject_id,
            "weight": float(weight),
            "units": units,
            "exercise": exercise,
            "emg_file": str(Path(emg_wav_path).resolve()),
            "imu_file": str(Path(imu_zip_path).resolve()),
            "sync_offset_s_emg_minus_imu": float(sync_offset_s),
            "emg_activity_onset_s": float(emg_on),
            "imu_activity_onset_s": float(imu_on),
            "dominant_gyro_axis_index": int(reps["axis"]),
            "recommendation": decision,
            "recommendation_reason": reason,
            "calibration_status": "PROVISIONAL_THRESHOLDS",
        }

        result = {**metadata, **summary}

        if output_dir is None:
            output_dir = Path(emg_wav_path).resolve().parent / "analysis_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_subject = "".join(c for c in subject_id if c.isalnum() or c in "-_") or "subject"
        stem = f"{safe_subject}_{exercise.replace(' ', '_')}_{weight:g}{units}"

        rep_csv = output_dir / f"{stem}_rep_features.csv"
        summary_json = output_dir / f"{stem}_summary.json"

        rep_df.to_csv(rep_csv, index=False)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return {
            "result": result,
            "rep_features": rep_df,
            "rep_csv": str(rep_csv),
            "summary_json": str(summary_json),
            "output_dir": str(output_dir),
        }
    finally:
        tmp = imu.get("_tempdir")
        if tmp is not None:
            tmp.cleanup()


# -----------------------------
# Tkinter dashboard
# -----------------------------
class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG + IMU Workout Analysis")
        self.geometry("760x690")
        self.minsize(680, 620)

        self.name_var = tk.StringVar()
        self.weight_var = tk.StringVar()
        self.units_var = tk.StringVar(value="lb")
        self.exercise_var = tk.StringVar(value="Bicep curl")
        self.emg_var = tk.StringVar()
        self.imu_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.data_folder_var = tk.StringVar(value=str(Path.home() / "Downloads"))

        self._build()

    def _build(self):
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Workout Analysis",
            font=("Segoe UI", 20, "bold")
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text="Enter the name and weight. The app automatically uses the newest EMG and IMU recordings from your data folder."
        ).pack(anchor="w", pady=(4, 16))

        # Session
        session = ttk.LabelFrame(outer, text="1. Session information", padding=14)
        session.pack(fill="x", pady=(0, 12))

        for c in range(4):
            session.columnconfigure(c, weight=1)

        ttk.Label(session, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(session, textvariable=self.name_var).grid(row=1, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(session, text="Weight").grid(row=0, column=1, sticky="w")
        ttk.Entry(session, textvariable=self.weight_var).grid(row=1, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(session, text="Units").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            session, textvariable=self.units_var,
            values=["lb", "kg"], state="readonly", width=8
        ).grid(row=1, column=2, sticky="ew", padx=(0, 8))

        ttk.Label(session, text="Exercise").grid(row=0, column=3, sticky="w")
        ttk.Entry(session, textvariable=self.exercise_var).grid(row=1, column=3, sticky="ew")

        # Files
        files = ttk.LabelFrame(outer, text="2. Most recent recordings", padding=14)
        files.pack(fill="x", pady=(0, 12))
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Data folder").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(files, textvariable=self.data_folder_var).grid(
            row=0, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(files, text="Browse…", command=self.pick_data_folder).grid(
            row=0, column=2, pady=5
        )

        ttk.Button(
            files, text="Find Most Recent Data", command=self.find_most_recent_data
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 10))

        self._file_row(files, 2, "Newest EMG WAV", self.emg_var, self.pick_emg, optional=True)
        self._file_row(files, 3, "Newest IMU ZIP", self.imu_var, self.pick_imu, optional=True)
        self._file_row(files, 4, "Output folder", self.output_var, self.pick_output, optional=True)

        ttk.Label(
            files,
            text="Run Analysis automatically refreshes these fields to the newest .wav and .zip in the selected folder. You can still browse to override either file."
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # Actions
        action = ttk.LabelFrame(outer, text="3. Analysis", padding=14)
        action.pack(fill="both", expand=True)

        btns = ttk.Frame(action)
        btns.pack(fill="x")

        self.run_btn = ttk.Button(btns, text="Run Analysis", command=self.on_run)
        self.run_btn.pack(side="left")
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=8)

        self.result_banner = tk.Label(
            action,
            text="",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#2f8f4e",
            padx=18,
            pady=16,
            relief="flat",
            anchor="center"
        )

        self.summary_label = ttk.Label(
            action,
            text="Summary",
            font=("Segoe UI", 11, "bold")
        )
        self.summary_label.pack(anchor="w", pady=(16, 6))

        self.status = tk.Text(action, height=14, wrap="word")
        self.status.pack(fill="both", expand=True)
        self.status.insert("1.0", "Ready.\n")
        self.status.configure(state="disabled")

    def _file_row(self, parent, row, label, var, command, optional=False):
        ttk.Label(parent, text=label + (" (optional)" if optional else "")).grid(
            row=row, column=0, sticky="w", pady=5
        )
        ttk.Entry(parent, textvariable=var).grid(
            row=row, column=1, sticky="ew", padx=8, pady=5
        )
        parent.columnconfigure(1, weight=1)
        ttk.Button(parent, text="Browse…", command=command).grid(
            row=row, column=2, pady=5
        )

    def pick_data_folder(self):
        p = filedialog.askdirectory(title="Select data folder")
        if p:
            self.data_folder_var.set(p)
            self.find_most_recent_data(show_message=False)

    def find_most_recent_data(self, show_message=True):
        folder = Path(self.data_folder_var.get().strip()).expanduser()
        if not folder.is_dir():
            if show_message:
                messagebox.showerror("Invalid data folder", "Select a valid data folder.")
            return False

        wavs = [p for p in folder.rglob("*.wav") if p.is_file()]
        zips = [p for p in folder.rglob("*.zip") if p.is_file()]

        if wavs:
            newest_wav = max(wavs, key=lambda p: p.stat().st_mtime)
            self.emg_var.set(str(newest_wav))
        else:
            self.emg_var.set("")

        if zips:
            newest_zip = max(zips, key=lambda p: p.stat().st_mtime)
            self.imu_var.set(str(newest_zip))
        else:
            self.imu_var.set("")

        ok = bool(wavs and zips)
        if show_message:
            if ok:
                self.log(
                    "Most recent recordings selected.\n\n"
                    f"EMG: {Path(self.emg_var.get()).name}\n"
                    f"IMU: {Path(self.imu_var.get()).name}"
                )
            else:
                missing = []
                if not wavs:
                    missing.append("EMG .wav")
                if not zips:
                    missing.append("IMU .zip")
                messagebox.showerror(
                    "Recent data not found",
                    "Could not find: " + ", ".join(missing) + " in the selected folder."
                )
        return ok

    def pick_emg(self):
        p = filedialog.askopenfilename(
            title="Select EMG WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if p:
            self.emg_var.set(p)

    def pick_imu(self):
        p = filedialog.askopenfilename(
            title="Select IMU ZIP",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        if p:
            self.imu_var.set(p)

    def pick_output(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_var.set(p)

    def log(self, text):
        self.status.configure(state="normal")
        self.status.delete("1.0", "end")
        self.status.insert("1.0", text)
        self.status.configure(state="disabled")

    def show_recommendation(self, decision):
        decision_upper = decision.upper()

        if "STOP" in decision_upper:
            color = "#c43d43"
            label = "STOP"
        elif "INCREASE WEIGHT" in decision_upper:
            color = "#2f8f4e"
            label = "MORE WEIGHT"
        elif "MORE REPS" in decision_upper:
            color = "#2f8f4e"
            label = "MORE REPS"
        elif "KEEP WEIGHT" in decision_upper:
            color = "#c28a2c"
            label = "KEEP WEIGHT"
        else:
            color = "#5b6470"
            label = decision_upper

        self.result_banner.configure(text=label, bg=color)
        if not self.result_banner.winfo_ismapped():
            self.result_banner.pack(fill="x", pady=(14, 0), before=self.summary_label)

    def clear(self):
        self.name_var.set("")
        self.weight_var.set("")
        self.units_var.set("lb")
        self.exercise_var.set("Bicep curl")
        self.emg_var.set("")
        self.imu_var.set("")
        self.output_var.set("")
        self.find_most_recent_data(show_message=False)
        if self.result_banner.winfo_ismapped():
            self.result_banner.pack_forget()
        self.result_banner.configure(text="")
        self.log("Ready.\n")

    def on_run(self):
        name = self.name_var.get().strip()
        exercise = self.exercise_var.get().strip() or "Bicep curl"

        # Always refresh to the newest recordings before a run.
        self.find_most_recent_data(show_message=False)
        emg_path = self.emg_var.get().strip()
        imu_path = self.imu_var.get().strip()

        try:
            weight = float(self.weight_var.get())
        except ValueError:
            messagebox.showerror("Invalid weight", "Enter a numeric weight.")
            return

        if not name:
            messagebox.showerror("Missing name", "Enter a name.")
            return
        if weight <= 0:
            messagebox.showerror("Invalid weight", "Weight must be greater than 0.")
            return
        if not Path(emg_path).is_file():
            messagebox.showerror("Missing EMG file", "Select a valid .wav file.")
            return
        if not Path(imu_path).is_file():
            messagebox.showerror("Missing IMU file", "Select a valid .zip file.")
            return

        self.run_btn.configure(state="disabled")
        self.update_idletasks()

        try:
            result = run_pipeline(
                subject_id=name,
                weight=weight,
                units=self.units_var.get(),
                exercise=exercise,
                emg_wav_path=emg_path,
                imu_zip_path=imu_path,
                output_dir=self.output_var.get().strip() or None,
            )

            r = result["result"]
            self.show_recommendation(r["recommendation"])

            report = (
                f"{r['recommendation_reason']}\n\n"
                f"Subject: {r['subject_id']}\n"
                f"Exercise: {r['exercise']}\n"
                f"Weight: {r['weight']:g} {r['units']}\n"
                f"Detected reps: {r['n_reps']}\n\n"
                f"Velocity drop: {100*r['velocity_drop_fraction']:.1f}%\n"
                f"EMG RMS change: {100*r['rms_rise_fraction']:.1f}%\n"
                f"Jerk change: {100*r['jerk_rise_fraction']:.1f}%\n"
                f"Estimated EMG−IMU offset: {r['sync_offset_s_emg_minus_imu']:.3f} s\n\n"
                f"EMG used: {Path(emg_path).name}\n"
                f"IMU used: {Path(imu_path).name}\n\n"
                f"Saved rep features:\n{result['rep_csv']}\n\n"
                f"Saved summary:\n{result['summary_json']}\n\n"
                "IMPORTANT: recommendation thresholds are provisional and need "
                "subject-specific calibration/validation before being treated as a training prescription."
            )
            self.log(report)

        except Exception as e:
            self.log("ANALYSIS ERROR\n\n" + str(e))
            messagebox.showerror("Analysis failed", str(e))
        finally:
            self.run_btn.configure(state="normal")


if __name__ == "__main__":
    Dashboard().mainloop()
