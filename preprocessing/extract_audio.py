import argparse
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- Config ------------------------------------------------------------------

FILENAME_RE  = re.compile(r"q_(?P<qid>\d+)_main(?:_(?P<label>[01]))?\.mp4$", re.IGNORECASE)
GLOB_PATTERN = "*/q_*.mp4"
SAMPLE_RATE  = 16000

# Spike detector sub-window settings (within each evaluation window)
SPIKE_SUBWIN_S    = 0.064   # 64 ms
SPIKE_HOP_S       = 0.016   # 16 ms
SPIKE_MULTIPLIER  = 3.0
BASELINE_FRACTION = 0.25    # use first 25% of CLIP for baseline (not per-window)

# ---- Audio extraction --------------------------------------------------------

def extract_audio_to_tempfile(mp4_path: str) -> str:
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="vad_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp4_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        os.unlink(wav_path)
        raise RuntimeError(f"ffmpeg failed for {mp4_path}: {e}")
    return wav_path

# ---- Per-clip baseline -------------------------------------------------------

def compute_clip_baseline(audio: np.ndarray) -> float:
    """RMS baseline from first 25% of clip using 64ms windows. Same as v2."""
    total = len(audio)
    win = max(1, int(SPIKE_SUBWIN_S * SAMPLE_RATE))
    hop = max(1, int(SPIKE_HOP_S * SAMPLE_RATE))
    end = int(total * BASELINE_FRACTION)
    rms_vals = []
    i = 0
    while i + win <= end:
        w = audio[i:i+win]
        rms_vals.append(float(np.sqrt(np.mean(w**2) + 1e-12)))
        i += hop
    return float(np.mean(rms_vals)) if rms_vals else 0.0

# ---- Per-window features -----------------------------------------------------

def window_features(audio: np.ndarray, start_samp: int, end_samp: int,
                    baseline: float) -> dict:
    """Compute spike/energy features within one window."""
    win = max(1, int(SPIKE_SUBWIN_S * SAMPLE_RATE))
    hop = max(1, int(SPIKE_HOP_S * SAMPLE_RATE))
    spike_threshold = baseline * SPIKE_MULTIPLIER

    seg = audio[start_samp:end_samp]
    if len(seg) < win:
        # Window shorter than sub-window — single RMS over what we have
        rms = float(np.sqrt(np.mean(seg**2) + 1e-12)) if len(seg) else 0.0
        return {
            "spike_detected":  int(rms > spike_threshold and baseline > 0),
            "spike_max_rms":   round(rms, 6),
            "window_max_rms":  round(rms, 6),
        }

    max_rms = 0.0
    i = 0
    n = len(seg)
    while i + win <= n:
        w = seg[i:i+win]
        rms = float(np.sqrt(np.mean(w**2) + 1e-12))
        if rms > max_rms:
            max_rms = rms
        i += hop
    # Tail
    if i < n:
        w = seg[i:]
        rms = float(np.sqrt(np.mean(w**2) + 1e-12))
        if rms > max_rms:
            max_rms = rms

    return {
        "spike_detected":  int(max_rms > spike_threshold and baseline > 0),
        "spike_max_rms":   round(max_rms, 6),
        "window_max_rms":  round(max_rms, 6),
    }

# ---- Per-clip processing -----------------------------------------------------

_VAD_MODEL = None
def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def process_one_clip(args_tuple):
    (mp4_path, participant_id, video_id, label,
     fps, window_size, slide,
     vad_threshold, min_speech_ms) = args_tuple

    import torch
    import soundfile as sf
    from silero_vad import get_speech_timestamps

    wav_path = None
    try:
        wav_path = extract_audio_to_tempfile(mp4_path)
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Unexpected sample rate: {sr}")

        duration_s = len(audio) / SAMPLE_RATE
        # Number of frames at the chosen fps (used to define window indices).
        # The README/eval expect frame indexing at the extraction fps.
        n_frames = int(duration_s * fps)
        # Number of windows of size `window_size` with hop `slide`
        if n_frames < window_size:
            # Pathologically short clip — produce 1 window covering [0, n_frames)
            n_windows = 1
        else:
            n_windows = (n_frames - window_size) // slide + 1

        half_point_s = duration_s / 2.0
        # A window's "is_2nd_half" — we use START time >= half point
        # (matches the convention from prior analysis)

        # Per-clip baseline for spike detection
        baseline_rms = compute_clip_baseline(audio)

        # Run VAD ONCE on the full audio and record segment time spans
        model = _get_vad_model()
        speech_ts = get_speech_timestamps(
            torch.from_numpy(audio),
            model,
            threshold=vad_threshold,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=100,
            return_seconds=False,
        )
        # Precompute segment intervals in samples and their RMS
        seg_intervals = []
        for seg in speech_ts:
            s, e = int(seg["start"]), int(seg["end"])
            seg_rms = float(np.sqrt(np.mean(audio[s:e]**2) + 1e-12)) if e > s else 0.0
            seg_intervals.append((s, e, seg_rms))

        window_rows = []
        first_2h_window = None  # for sanity reporting
        for w_id in range(n_windows):
            # Frame indices
            f_start = w_id * slide
            f_end = f_start + window_size
            # Convert to time
            t_start = f_start / fps
            t_end = f_end / fps
            # Convert to samples for audio analysis
            s_start = int(t_start * SAMPLE_RATE)
            s_end = min(int(t_end * SAMPLE_RATE), len(audio))

            is_2nd_half = int(t_start >= half_point_s)
            if is_2nd_half and first_2h_window is None:
                first_2h_window = w_id

            # VAD segments overlapping this window
            n_vad = 0
            vad_max_rms = 0.0
            for (s, e, srms) in seg_intervals:
                if e > s_start and s < s_end:
                    n_vad += 1
                    if srms > vad_max_rms:
                        vad_max_rms = srms

            # Window-level spike features
            wf = window_features(audio, s_start, s_end, baseline_rms)

            # ── Sub-window features (NEW) ──────────────────────────────────
            # Split this window into 0.2s sub-windows and report per-sub-window
            # VAD and RMS stats. This catches short vocalizations (gasps, "oh",
            # sharp inhales) that get diluted when averaged over the full 5s window.
            subwin_s       = 0.2
            subwin_samples = int(subwin_s * SAMPLE_RATE)
            subwin_hop     = subwin_samples  # non-overlapping
            win_audio      = audio[s_start:s_end]

            subwin_rms_vals  = []  # raw RMS per sub-window
            subwin_vad_count = 0   # sub-windows with at least one VAD segment
            subwin_max_rms   = 0.0 # loudest sub-window RMS (raw, no VAD filter)
            subwin_vad_max_rms = 0.0  # loudest sub-window RMS where VAD fired

            i = 0
            while i + subwin_samples <= len(win_audio):
                sw_start_abs = s_start + i
                sw_end_abs   = s_start + i + subwin_samples
                sw_audio     = win_audio[i : i + subwin_samples]
                sw_rms = float(np.sqrt(np.mean(sw_audio ** 2) + 1e-12))
                subwin_rms_vals.append(sw_rms)
                if sw_rms > subwin_max_rms:
                    subwin_max_rms = sw_rms

                # Check if any VAD segment overlaps this sub-window
                sw_has_vad = False
                for (s, e, srms) in seg_intervals:
                    if e > sw_start_abs and s < sw_end_abs:
                        sw_has_vad = True
                        if sw_rms > subwin_vad_max_rms:
                            subwin_vad_max_rms = sw_rms
                        break
                if sw_has_vad:
                    subwin_vad_count += 1

                i += subwin_hop

            n_subwins = len(subwin_rms_vals) if subwin_rms_vals else 1
            subwin_vad_ratio   = subwin_vad_count / n_subwins
            subwin_rms_std     = float(np.std(subwin_rms_vals)) if subwin_rms_vals else 0.0
            # ── End sub-window features ────────────────────────────────────

            window_rows.append({
                "participant_id":    participant_id,
                "video_id":          video_id,
                "label":             label,
                "window_id":         w_id,
                "frame_start":       f_start,
                "frame_end":         f_end,
                "time_start_s":      round(t_start, 3),
                "time_end_s":        round(t_end, 3),
                "is_2nd_half":       is_2nd_half,
                # Original features
                "n_vad_segments":    n_vad,
                "vad_max_rms":       round(vad_max_rms, 6),
                "spike_detected":    wf["spike_detected"],
                "spike_max_rms":     wf["spike_max_rms"],
                "window_max_rms":    wf["window_max_rms"],
                # Sub-window features
                "n_subwins":         n_subwins,
                "subwin_vad_count":  subwin_vad_count,
                "subwin_vad_ratio":  round(subwin_vad_ratio, 4),
                "subwin_max_rms":    round(subwin_max_rms, 6),
                "subwin_vad_max_rms":round(subwin_vad_max_rms, 6),
                "subwin_rms_std":    round(subwin_rms_std, 6),
            })

        clip_meta = {
            "participant_id":      participant_id,
            "video_id":            video_id,
            "label":               label,
            "duration_s":          round(duration_s, 3),
            "n_frames_5fps":       n_frames,
            "n_windows":           n_windows,
            "baseline_rms":        round(baseline_rms, 6),
            "half_point_s":        round(half_point_s, 3),
            "first_2nd_half_win":  first_2h_window if first_2h_window is not None else -1,
        }
        return window_rows, clip_meta

    except Exception as e:
        return None, {
            "participant_id": participant_id,
            "video_id":       video_id,
            "label":          label,
            "error":          str(e),
        }
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

# ---- Discovery ---------------------------------------------------------------

def discover_clips(data_dir: Path):
    found = list(data_dir.glob(GLOB_PATTERN))
    skipped = []
    for p in found:
        m = FILENAME_RE.search(p.name)
        if not m:
            skipped.append(str(p))
            continue
        video_id = f"q_{m.group('qid')}"   # e.g. q_10_main_1 → q_10
        label = m.group("label")
        label = int(label) if label is not None else -1
        yield (str(p), p.parent.name, video_id, label)
    if skipped:
        print(f"[warn] {len(skipped)} files skipped (name mismatch). First 5:")
        for s in skipped[:5]: print(f"  {s}")

# ---- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",       required=True, type=Path)
    ap.add_argument("--out_dir",        required=True, type=Path)
    ap.add_argument("--fps",            type=int, default=5,
                    help="Frame rate at which windows are indexed (default 5)")
    ap.add_argument("--window_size",    type=int, default=10,
                    help="Window size in frames (default 10 = 2s at 5fps)")
    ap.add_argument("--slide",          type=int, default=5,
                    help="Window slide in frames (default 5 = 1s at 5fps)")
    ap.add_argument("--vad-threshold",  type=float, default=0.15)
    ap.add_argument("--min-speech-ms",  type=int, default=50)
    ap.add_argument("--max_workers",    type=int, default=4)
    ap.add_argument("--limit",          type=int, default=None)
    args = ap.parse_args()

    # Validate eval constraints
    if args.slide > args.window_size:
        sys.exit(f"[fatal] slide ({args.slide}) > window_size ({args.window_size}) "
                 f"violates eval constraint.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    clips = list(discover_clips(args.data_dir))
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        sys.exit(f"[fatal] No clips found under {args.data_dir}")

    print(f"[info] Found {len(clips)} clips. Window setup: "
          f"fps={args.fps}, window_size={args.window_size}, slide={args.slide}")
    print(f"[info] VAD threshold={args.vad_threshold}, min_speech_ms={args.min_speech_ms}")

    clip_args = [
        (mp4, pid, vid, lbl,
         args.fps, args.window_size, args.slide,
         args.vad_threshold, args.min_speech_ms)
        for mp4, pid, vid, lbl in clips
    ]

    all_windows = []
    all_metas = []
    errors = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(process_one_clip, c) for c in clip_args]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="extract"):
            wins, meta = fut.result()
            if wins is None:
                errors.append(meta)
            else:
                all_windows.extend(wins)
                all_metas.append(meta)

    w_df = pd.DataFrame(all_windows)
    m_df = pd.DataFrame(all_metas)
    e_df = pd.DataFrame(errors)

    w_path = args.out_dir / "per_window_features.csv"
    m_path = args.out_dir / "clip_metadata.csv"
    e_path = args.out_dir / "extract_errors.csv"

    w_df.to_csv(w_path, index=False)
    m_df.to_csv(m_path, index=False)
    if not e_df.empty:
        e_df.to_csv(e_path, index=False)
        print(f"[warn] {len(e_df)} clips failed. See {e_path}")

    print(f"[saved] {w_path} ({len(w_df)} windows from {len(m_df)} clips)")
    print(f"[saved] {m_path}")

    # Quick sanity summary
    if not m_df.empty:
        print(f"\nWindows per clip — min: {m_df['n_windows'].min()}, "
              f"median: {m_df['n_windows'].median():.0f}, "
              f"max: {m_df['n_windows'].max()}")
        print(f"Total windows: {m_df['n_windows'].sum()}")
        n_2h_fired = w_df[(w_df['is_2nd_half']==1) &
                          (w_df['n_vad_segments']>0) &
                          (w_df['spike_detected']==1)].groupby(
                          ['participant_id','video_id']).size().reset_index()
       
if __name__ == "__main__":
    main()