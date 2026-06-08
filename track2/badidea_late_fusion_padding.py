#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Late Fusion model for Bad Idea Track 2.

Each modality processes independently through its own Transformer,
then pooled features are concatenated and classified via MLP.

Pipeline:
  frames    → EfficientNet → Transformer → TemporalBias → pooled_vis (d_model)
  landmarks → LandmarkFeatureExtractor → Transformer → mean pool → pooled_lm (lm_d_model)
                                                    ↓
                                  concat → MLP → label
"""

#unfreeze option added 
#lm only for now > best AUC
#added temporalBias for landmark and changed mean pooling into attention pooling

import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── MediaPipe facial region indices ──────────────────────────────────────────
EYEBROW_LEFT  = [46, 53, 52, 51, 50, 49, 48, 47]
EYEBROW_RIGHT = [285, 295, 282, 283, 276, 300, 293, 334]
EYE_LEFT      = [33, 7, 163, 144, 145, 153, 154, 155, 133]
EYE_RIGHT     = [362, 382, 381, 380, 374, 373, 390, 249, 263]
LIP_OUTER     = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LIP_INNER     = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78]

FACE_INDICES = list(set(
    EYEBROW_LEFT + EYEBROW_RIGHT +
    EYE_LEFT + EYE_RIGHT +
    LIP_OUTER + LIP_INNER
))
LM_FEAT_DIM = len(FACE_INDICES) * 3


# ── Dataset ───────────────────────────────────────────────────────────────────

class BadIdeaLateFusionDataset(Dataset):
    """
    Returns (frames_tensor, landmarks_tensor, label, pid, q_id)
    """

    def __init__(self, participants, npy_base_path, landmark_dir, label_mapping):
        self.samples = []

        # Build landmark lookup: (pid, q_id) -> path
        lm_lookup = {}
        for fname in os.listdir(landmark_dir):
            if not fname.endswith(".npy"):
                continue
            base = fname.replace(".mp4.npy", "")
            idx = base.rfind("trainval_")
            if idx == -1:
                continue
            rest = base[idx + len("trainval_"):]
            parts = rest.split("_")
            if len(parts) < 5:
                continue
            pid = parts[0]
            q_id = f"q_{parts[2]}"
            lm_lookup[(pid, q_id)] = os.path.join(landmark_dir, fname)

        for pid in participants:
            pid_dir = os.path.join(npy_base_path, pid)
            if not os.path.exists(pid_dir):
                print(f"[WARN] Not found: {pid_dir}")
                continue

            video_frames = {}
            for npy_file in sorted(os.listdir(pid_dir)):
                if not npy_file.endswith(".npy"):
                    continue
                match = re.match(r"(q_\d+)_main_(\d+)_\d+fps_frame(\d+)", npy_file)
                if not match:
                    continue
                q_id = match.group(1)
                label_from_file = int(match.group(2))
                key = (q_id, label_from_file)
                video_frames.setdefault(key, []).append(
                    os.path.join(pid_dir, npy_file)
                )

            for (q_id, _), paths in video_frames.items():
                lm_key = (pid, q_id)
                if lm_key not in label_mapping:
                    continue
                if lm_key not in lm_lookup:
                    print(f"[WARN] No landmark for {pid}/{q_id}")
                    continue
                label = label_mapping[lm_key]
                self.samples.append((pid, q_id, label, sorted(paths), lm_lookup[lm_key]))

        print(f"BadIdeaLateFusionDataset: {len(self.samples)} videos loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, q_id, label, paths, lm_path = self.samples[idx]
        frames_np = np.stack([np.load(p) for p in paths]).astype(np.float32)
        landmarks_np = np.load(lm_path).astype(np.float32)
        return (
            torch.from_numpy(frames_np),
            torch.from_numpy(landmarks_np),
            label, pid, q_id
        )


def collate_fn(batch):
    frames_list, lm_list, labels, pids, qids = zip(*batch)

    # align frame counts
    aligned_frames, aligned_lm = [], []
    for f, lm in zip(frames_list, lm_list):
        n = min(f.shape[0], lm.shape[0])
        aligned_frames.append(f[:n])
        aligned_lm.append(lm[:n])

    max_len = max(f.shape[0] for f in aligned_frames)
    B = len(aligned_frames)

    padded_frames = torch.zeros(B, max_len, 3, 224, 224)
    padded_lm     = torch.zeros(B, max_len, 468, 3)
    mask          = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (f, lm) in enumerate(zip(aligned_frames, aligned_lm)):
        n = f.shape[0]
        padded_frames[i, :n] = f
        padded_lm[i, :n]     = lm
        mask[i, :n]           = True

    return padded_frames, padded_lm, torch.tensor(labels), mask, pids, qids


# ── Shared modules ────────────────────────────────────────────────────────────

class TemporalBiasAttention(nn.Module):
    def __init__(self, max_len=512):
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(0, 1, max_len).unsqueeze(0))

    def forward(self, x, mask=None):
        T = x.size(1)
        bias = self.bias[:, :T]
        weights = torch.sigmoid(bias)
        if mask is not None:
            weights = weights * mask.float()
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class LandmarkFeatureExtractor(nn.Module):
    """
    Extracts delta features from facial landmarks (eyebrow, eye, lip regions).
    Input:  (B, T, 468, 3)
    Output: (B, n_tokens, proj_dim), token_mask
    """

    def __init__(self, frames_per_token=30, lm_feat_dim=LM_FEAT_DIM, proj_dim=64):
        super().__init__()
        self.frames_per_token = frames_per_token
        self.face_indices = FACE_INDICES
        self.lm_feat_dim = lm_feat_dim
        #self.lm_temporal_bias = TemporalBiasAttention(max_len=512)
        
        self.projection = nn.Sequential(
            nn.Linear(lm_feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

    def forward(self, x, mask=None):
        B, T, _, _ = x.shape
        k = self.frames_per_token

        x_face = x[:, :, self.face_indices, :]
        x_flat = x_face.reshape(B, T, -1)

        delta = torch.cat([
            torch.zeros_like(x_flat[:, :1, :]),
            x_flat[:, 1:, :] - x_flat[:, :-1, :]
        ], dim=1)

        T2 = delta.shape[1]
        n_tokens = T2 // k
        if n_tokens == 0:
            n_tokens = 1
            k = T2

        delta_trunc = delta[:, :n_tokens * k, :].reshape(B, n_tokens, k, self.lm_feat_dim)
        seq = delta_trunc.mean(dim=2)

        if mask is not None:
            mask_delta = mask
            mask_trunc = mask_delta[:, :n_tokens * k].reshape(B, n_tokens, k)
            token_mask = mask_trunc.any(dim=2)
        else:
            token_mask = torch.ones(B, n_tokens, dtype=torch.bool, device=x.device)

        return self.projection(seq), token_mask


# ── Late Fusion Model ─────────────────────────────────────────────────────────

class BadIdeaLateFusionTransformer(nn.Module):
    """
    Late fusion: each modality has its own Transformer,
    pooled features are concatenated before classification.

    Visual branch:   EfficientNet → Transformer → TemporalBias → pooled_vis (d_model)
    Landmark branch: LandmarkFeatureExtractor → Transformer → mean pool → pooled_lm (lm_d_model)
    Fusion:          concat → MLP → label
    """

    def __init__(
        self,
        num_classes=2,
        frames_per_token=30,
        # Visual branch
        d_model=256,
        n_heads=4,
        n_layers=2,
        dropout=0.5,
        freeze_backbone=True,
        unfreeze_top_blocks=0, #added unfreeze option
        # Landmark branch
        lm_d_model=128,
        lm_proj_dim=128,
        lm_n_heads=2,
        lm_n_layers=1,
        lm_dropout=0.3,
    ):
        super().__init__()
        self.frames_per_token = frames_per_token

        # ── Visual branch ─────────────────────────────────────────────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)
        self.backbone = backbone.features
        self.unfreeze_top_blocks = unfreeze_top_blocks
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_top_blocks > 0:
                for i in range(9 - unfreeze_top_blocks, 9):
                    for p in self.backbone[i].parameters():
                        p.requires_grad = True
                print(f"[BadIdeaLateFusionTransformer] Backbone PARTIALLY FROZEN (top {unfreeze_top_blocks} blocks trainable)")
            else:
                print("[BadIdeaLateFusionTransformer] Backbone FULLY FROZEN")

        self.vis_projection = nn.Sequential(
            nn.Linear(1280, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.vis_pos_enc = nn.Embedding(512, d_model)

        vis_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.vis_transformer = nn.TransformerEncoder(
            vis_encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.vis_temporal_bias = TemporalBiasAttention(max_len=512)

        # ── Landmark branch ───────────────────────────────────────────────────
        self.lm_extractor = LandmarkFeatureExtractor(
            frames_per_token=frames_per_token,
            lm_feat_dim=LM_FEAT_DIM,
            proj_dim=lm_proj_dim,
        )
        self.lm_pos_enc = nn.Embedding(512, lm_d_model)
        self.lm_temporal_bias = TemporalBiasAttention(max_len=512)

        lm_encoder_layer = nn.TransformerEncoderLayer(
            d_model=lm_d_model, nhead=lm_n_heads,
            dim_feedforward=lm_d_model * 4,
            dropout=lm_dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.lm_transformer = nn.TransformerEncoder(
            lm_encoder_layer, num_layers=lm_n_layers,
            norm=nn.LayerNorm(lm_d_model),
        )

        # ── Fusion MLP ────────────────────────────────────────────────────────
        fusion_dim = d_model + lm_d_model
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, frames, landmarks, mask=None):
        """
        frames:    (B, N, 3, 224, 224)
        landmarks: (B, N, 468, 3)
        mask:      (B, N) True where valid
        """
        B, N, C, H, W = frames.shape
        k = self.frames_per_token

        # ── Visual branch ─────────────────────────────────────────────────────
        x_flat = frames.reshape(B * N, C, H, W)
        chunk_size = 60
        feats = []

        if self.unfreeze_top_blocks > 0:
            for i in range(0, B * N, chunk_size):
                chunk = x_flat[i:i + chunk_size]
                f = self.backbone(chunk)
                f = self.pool(f).flatten(1)
                feats.append(f)
        else:
            with torch.no_grad():
                for i in range(0, B * N, chunk_size):
                    chunk = x_flat[i:i + chunk_size]
                    f = self.backbone(chunk)
                    f = self.pool(f).flatten(1)
                    feats.append(f)

        feat = torch.cat(feats, dim=0).reshape(B, N, 1280)

        n_tokens = N // k
        if n_tokens == 0:
            n_tokens = 1
            k = N

        feat_trunc = feat[:, :n_tokens * k].reshape(B, n_tokens, k, 1280)
        vis_tokens = feat_trunc.mean(dim=2)
        

        if mask is not None:
            mask_trunc = mask[:, :n_tokens * k].reshape(B, n_tokens, k)
            token_mask = mask_trunc.any(dim=2)
        else:
            token_mask = torch.ones(B, n_tokens, dtype=torch.bool, device=frames.device)

        vis_seq = self.vis_projection(vis_tokens)
        T = vis_seq.size(1)
        pos = torch.arange(T, device=frames.device).unsqueeze(0)
        vis_seq = vis_seq + self.vis_pos_enc(pos)
        vis_seq = self.vis_transformer(vis_seq, src_key_padding_mask=~token_mask)
        pooled_vis = self.vis_temporal_bias(vis_seq, mask=token_mask)  # (B, d_model)

        # ── Landmark branch ───────────────────────────────────────────────────
        lm_tokens, lm_mask = self.lm_extractor(landmarks, mask)  # (B, n_tokens, lm_proj_dim)

        # align token counts
        min_t = min(lm_tokens.shape[1], n_tokens)
        lm_tokens = lm_tokens[:, :min_t, :]
        lm_mask   = lm_mask[:, :min_t]

        lm_seq = lm_tokens
        T_lm = lm_seq.size(1)
        pos_lm = torch.arange(T_lm, device=frames.device).unsqueeze(0)
        lm_seq = lm_seq + self.lm_pos_enc(pos_lm)
        lm_seq = self.lm_transformer(lm_seq, src_key_padding_mask=~lm_mask)

        # mean pooling for landmark
        #lm_mask_f = lm_mask.float().unsqueeze(-1)
        #pooled_lm = (lm_seq * lm_mask_f).sum(dim=1) / (lm_mask_f.sum(dim=1) + 1e-8)  # (B, lm_d_model)
        #attention pooling for landmark
        pooled_lm = self.lm_temporal_bias(lm_seq, mask=lm_mask)

        # ── Late Fusion ───────────────────────────────────────────────────────
        fused = torch.cat([pooled_vis, pooled_lm], dim=-1)  # (B, d_model + lm_d_model)
        return self.classifier(fused)


if __name__ == "__main__":
    print("=== Sanity check ===")
    model = BadIdeaLateFusionTransformer(freeze_backbone=True)
    frames    = torch.randn(2, 300, 3, 224, 224)
    landmarks = torch.randn(2, 300, 468, 3)
    mask      = torch.ones(2, 300, dtype=torch.bool)
    out = model(frames, landmarks, mask)
    print(f"Output: {out.shape}")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")