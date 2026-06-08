#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for Bad Idea Track 2 test set.
Loads best model checkpoint and generates submission CSV.
"""

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from badnet_pytorch import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_base_path", type=str, required=True,
                        help="Path to test_npy folder")
    parser.add_argument("--landmark_dir", type=str, default=None,
                        help="Path to test_npy_landmark folder")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output submission CSV path")

    # Model
    parser.add_argument("--frames_per_token", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--lm_proj_dim", type=int, default=128)
    parser.add_argument("--lm_d_model", type=int, default=128)
    parser.add_argument("--lm_n_heads", type=int, default=2)
    parser.add_argument("--lm_n_layers", type=int, default=1)
    parser.add_argument("--lm_dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze_top_blocks", type=int, default=0)

    # Model type
    parser.add_argument("--use_late_fusion_padding", action="store_true")
    parser.add_argument("--use_late_fusion", action="store_true")

    # Submission
    parser.add_argument("--window_size", type=int, default=150)
    parser.add_argument("--slide", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")

    return parser.parse_args()


# ── Test Dataset ──────────────────────────────────────────────────────────────

class BadIdeaTestDataset(Dataset):
    """
    Test dataset — no labels.
    Returns (frames_tensor, pid, q_id)
    """
    def __init__(self, npy_base_path):
        self.samples = []

        participants = sorted([
            p for p in os.listdir(npy_base_path)
            if os.path.isdir(os.path.join(npy_base_path, p))
            and p != 'label_data.csv'
        ])

        for pid in participants:
            pid_dir = os.path.join(npy_base_path, pid)
            video_frames = {}
            for npy_file in sorted(os.listdir(pid_dir)):
                if not npy_file.endswith(".npy"):
                    continue
                match = re.match(r"(q_\d+)_main_-?\d+_\d+fps_frame\d+", npy_file)
                if not match:
                    # test set has no label: q_10_main_30fps_frame0001.npy
                    match = re.match(r"(q_\d+)_main_-?\d+_\d+fps_frame\d+", npy_file)
                    if not match:
                        continue
                q_id = match.group(1)
                video_frames.setdefault(q_id, []).append(
                    os.path.join(pid_dir, npy_file)
                )

            for q_id, paths in video_frames.items():
                self.samples.append((pid, q_id, sorted(paths)))

        print(f"BadIdeaTestDataset: {len(self.samples)} videos loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, q_id, paths = self.samples[idx]
        frames_np = np.stack([np.load(p) for p in paths]).astype(np.float32)
        return torch.from_numpy(frames_np), pid, q_id


class BadIdeaTestDatasetWithLandmark(Dataset):
    """
    Test dataset with landmarks — no labels.
    """
    def __init__(self, npy_base_path, landmark_dir):
        self.samples = []

        # Build landmark lookup
        lm_lookup = {}
        for fname in os.listdir(landmark_dir):
            if not fname.endswith(".npy"):
                continue
            # format: _media_Datasets_bad_idea_test_data_badidea_2313_q_10_main.mp4.npy
            base = fname.replace(".mp4.npy", "")
            parts = base.split("_")
            for i, p in enumerate(parts):
                if p.isdigit() and len(p) == 4:
                    pid = p
                    # q_id 찾기
                    for j in range(i+1, len(parts)):
                        if parts[j] == 'q' and j+1 < len(parts) and parts[j+1].isdigit():
                            q_id = f"q_{parts[j+1]}"
                            lm_lookup[(pid, q_id)] = os.path.join(landmark_dir, fname)
                            break
                    break
        participants = sorted([
            p for p in os.listdir(npy_base_path)
            if os.path.isdir(os.path.join(npy_base_path, p))
        ])

        for pid in participants:
            pid_dir = os.path.join(npy_base_path, pid)
            video_frames = {}
            for npy_file in sorted(os.listdir(pid_dir)):
                if not npy_file.endswith(".npy"):
                    continue
                match = re.match(r"(q_\d+)_main_-?\d+_\d+fps_frame\d+", npy_file)
                if not match:
                    continue
                q_id = match.group(1)
                video_frames.setdefault(q_id, []).append(
                    os.path.join(pid_dir, npy_file)
                )

            for q_id, paths in video_frames.items():
                lm_key = (pid, q_id)
                if lm_key not in lm_lookup:
                    print(f"[WARN] No landmark for {pid}/{q_id}")
                    continue
                self.samples.append((pid, q_id, sorted(paths), lm_lookup[lm_key]))

        print(f"BadIdeaTestDatasetWithLandmark: {len(self.samples)} videos loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, q_id, paths, lm_path = self.samples[idx]
        frames_np = np.stack([np.load(p) for p in paths]).astype(np.float32)
        landmarks_np = np.load(lm_path).astype(np.float32)
        return torch.from_numpy(frames_np), torch.from_numpy(landmarks_np), pid, q_id


def collate_fn_test(batch):
    frames_list, pids, qids = zip(*batch)
    max_len = max(f.shape[0] for f in frames_list)
    B = len(frames_list)
    padded = torch.zeros(B, max_len, 3, 224, 224)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, f in enumerate(frames_list):
        n = f.shape[0]
        padded[i, :n] = f
        mask[i, :n] = True
    return padded, mask, pids, qids


def collate_fn_test_lm(batch):
    frames_list, lm_list, pids, qids = zip(*batch)
    aligned_frames, aligned_lm = [], []
    for f, lm in zip(frames_list, lm_list):
        n = min(f.shape[0], lm.shape[0])
        aligned_frames.append(f[:n])
        aligned_lm.append(lm[:n])
    max_len = max(f.shape[0] for f in aligned_frames)
    B = len(aligned_frames)
    padded_frames = torch.zeros(B, max_len, 3, 224, 224)
    padded_lm = torch.zeros(B, max_len, 468, 3)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (f, lm) in enumerate(zip(aligned_frames, aligned_lm)):
        n = f.shape[0]
        padded_frames[i, :n] = f
        padded_lm[i, :n] = lm
        mask[i, :n] = True
    return padded_frames, padded_lm, mask, pids, qids


def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.use_late_fusion_padding:
        from badidea_late_fusion_padding import BadIdeaLateFusionTransformer
        model = BadIdeaLateFusionTransformer(
            num_classes=2,
            frames_per_token=args.frames_per_token,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
            lm_proj_dim=args.lm_proj_dim,
            lm_d_model=args.lm_d_model,
            lm_n_heads=args.lm_n_heads,
            lm_n_layers=args.lm_n_layers,
            lm_dropout=args.lm_dropout,
            unfreeze_top_blocks=args.unfreeze_top_blocks,
        ).to(device)
    else:
        from badidea_model import BadIdeaTransformer
        model = BadIdeaTransformer(
            num_classes=2,
            frames_per_token=args.frames_per_token,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
        ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch')}, AUC={ckpt.get('val_auc', 0):.4f}")
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.use_late_fusion_padding and args.landmark_dir:
        dataset = BadIdeaTestDatasetWithLandmark(args.npy_base_path, args.landmark_dir)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn_test_lm)
    else:
        dataset = BadIdeaTestDataset(args.npy_base_path)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn_test)

    # ── Inference ─────────────────────────────────────────────────────────────
    submission_rows = []
    window_size = args.window_size
    slide = args.slide

    with torch.no_grad():
        for batch in loader:
            if args.use_late_fusion_padding and args.landmark_dir:
                frames, landmarks, mask, pids, qids = batch
                frames = frames.to(device)
                landmarks = landmarks.to(device)
                mask = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                logits = model(frames, landmarks, mask)
            else:
                frames, mask, pids, qids = batch
                frames = frames.to(device)
                mask = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                logits = model(frames, mask)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            for i in range(len(pids)):
                pid, qid = pids[i], qids[i]
                n_frames = int(mask[i].sum().item())
                n_windows = max((n_frames - window_size) // slide + 1, 1)
                for w in range(n_windows):
                    submission_rows.append({
                        "participant_id": pid,
                        "video_id": qid,
                        "window_id": w,
                        "y_pred": int(preds[i]),
                        "y_prob_0": float(probs[i][0]),
                        "y_prob_1": float(probs[i][1]),
                    })

    sub_df = pd.DataFrame(submission_rows)
    sub_df = sub_df.sort_values(["participant_id", "video_id", "window_id"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    sub_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved submission: {args.out_csv}")
    print(f"  Rows: {len(sub_df)}")
    print(f"  Videos: {sub_df[['participant_id','video_id']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()