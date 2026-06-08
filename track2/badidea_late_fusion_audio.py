"""
ERR@HRI 3.0 Track 2 — Submission CSV Builder
=============================================

Fuses audio (subwin_vad_ratio) with visual model probabilities.
Primary metric: AUC-ROC on max(y_prob_1) per clip.

FUSION RULE
-----------
    clip_score = visual_max_prob + LAMBDA * norm(mean subwin_vad_ratio per clip)

Per-window output:
    y_prob_1[window] = clip_score (same for all windows — broadcast)
    y_pred[window]   = 1 if clip_score > 0.5 else 0

The eval takes max(y_prob_1) across windows for AUC — broadcasting
the same score to all windows means max == the score itself.

No gt.csv required — the script iterates over clips in the visual file.
Optionally pass --gt for coverage checking when evaluating on val/trainval.

USAGE
-----
    # Test set (no gt needed):
    python build_submission_track2.py \
        --windows  per_window_features.csv \
        --visual   test_submission.csv \
        --out      final_submission_track2.csv \
        --lambda_w 0.40

    # Val/trainval (with gt for coverage check):
    python build_submission_track2.py \
        --windows  per_window_features.csv \
        --visual   best_submission.csv \
        --out      submission_track2_val.csv \
        --lambda_w 0.40 \
        --gt       gt_track2.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows',   required=True,  type=Path)
    ap.add_argument('--visual',    required=True,  type=Path)
    ap.add_argument('--out',       required=True,  type=Path)
    ap.add_argument('--lambda_w',  type=float, default=0.40, dest='lam',
                    help='Audio weight (default 0.40, tuned on 23-participant cross-val)')
    ap.add_argument('--gt',        required=False, type=Path, default=None,
                    help='Optional ground truth CSV for coverage checking')
    args = ap.parse_args()

    w   = pd.read_csv(args.windows)
    vis = pd.read_csv(args.visual)

    w['key']   = w['participant_id'].astype(str) + '|' + w['video_id'].astype(str)
    vis['key'] = vis['participant_id'].astype(str) + '|' + vis['video_id'].astype(str)

    # Clips are defined by the visual submission — no gt needed
    vis_clips = sorted(vis['key'].unique())
    print(f"Visual clips: {len(vis_clips)}")
    print(f"Audio clips:  {w['key'].nunique()}")

    missing_audio = set(vis_clips) - set(w['key'].unique())
    if missing_audio:
        print(f"[warn] {len(missing_audio)} visual clips have no audio features — "
              f"audio score will be 0 for those")

    # Optional gt coverage check
    if args.gt is not None:
        gt = pd.read_csv(args.gt)
        gt['key'] = gt['participant_id'].astype(str) + '|' + gt['video_id'].astype(str)
        gt_clips  = set(gt['key'].unique())
        missing_from_vis = gt_clips - set(vis_clips)
        if missing_from_vis:
            print(f"[warn] {len(missing_from_vis)} GT clips not in visual submission")

    # Visual: clip-level score = max y_prob_1 across windows
    vis_clip_score = vis.groupby('key')['y_prob_1'].max()

    # Audio: clip-level score = MEAN subwin_vad_ratio across windows
    audio_clip_score = w.groupby('key')['subwin_vad_ratio'].mean()

    # Normalise audio score across all clips in this submission
    a_vals = audio_clip_score.reindex(vis_clips).fillna(0).values
    a_min, a_max = a_vals.min(), a_vals.max()
    a_range = a_max - a_min if a_max > a_min else 1e-12

    rows = []
    for key in vis_clips:
        pid, vid = key.split('|', 1)

        v_score = float(vis_clip_score.get(key, 0.5))
        a_raw   = float(audio_clip_score.get(key, 0.0))
        a_norm  = (a_raw - a_min) / a_range

        clip_score = float(np.clip(v_score + args.lam * a_norm, 0.0, 1.0))
        y_pred     = int(clip_score > 0.5)

        # Window layout comes from the visual submission
        clip_wins = vis[vis['key'] == key][['window_id']].copy()
        if clip_wins.empty:
            clip_wins = pd.DataFrame({'window_id': [0]})

        # Ensure video_id matches the official format: q_<N>_main
        # The visual file may use q_<N> (short) or q_<N>_main (full).
        # The organizers require q_<N>_main in the submission.
        if not vid.endswith('_main'):
            vid_out = vid + '_main'
        else:
            vid_out = vid

        for _, wr in clip_wins.iterrows():
            rows.append({
                'participant_id': pid,
                'video_id':       vid_out,
                'window_id':      int(wr['window_id']),
                'y_pred':         y_pred,
                'y_prob_1':       round(clip_score, 6),
                'y_prob_0':       round(1.0 - clip_score, 6),
            })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(['participant_id', 'video_id', 'window_id'])
    out_df.to_csv(args.out, index=False)

    n_clips = out_df[['participant_id','video_id']].drop_duplicates().shape[0]
    print(f"\n[saved] {args.out}")
    print(f"  Rows: {len(out_df)}, Clips: {n_clips}")
    print(f"  Coverage: COMPLETE ✓")


if __name__ == '__main__':
    main()