#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# metrics updated and Cross-validation added (newesttttttt)
# landmarks late-fusion added
# top layer unfreezing added
# focal_loss

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from badnet_pytorch import set_seed
from badidea_model import BadIdeaDatasetVideo, BadIdeaTransformer, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_base_path", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--train_labels_csv", type=str, required=True)
    parser.add_argument("--val_labels_csv", type=str, required=True)
 
    # Cross-validation
    parser.add_argument("--use_cv", action="store_true")
    parser.add_argument("--n_folds", type=int, default=5)
 
    # Model
    parser.add_argument("--frames_per_token", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--freeze_backbone", action="store_true")
 
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/badidea")
    parser.add_argument("--run_name", type=str, default="badidea_transformer")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="errhri3-badidea")
    parser.add_argument("--load_checkpoint", type=str, default=None)
 
    # Landmark Fusion
    parser.add_argument("--use_fusion", action="store_true")
    parser.add_argument("--landmark_dir", type=str, default=None)
    parser.add_argument("--lm_proj_dim", type=int, default=64)

    # Audio Fusion
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--audio_d_model", type=int, default=32)
    parser.add_argument("--audio_n_heads", type=int, default=2)
    parser.add_argument("--audio_n_layers", type=int, default=1)
    
    # Late Fusion
    parser.add_argument("--use_late_fusion", action="store_true")
    parser.add_argument("--lm_d_model", type=int, default=128)
    parser.add_argument("--lm_n_heads", type=int, default=2)
    parser.add_argument("--lm_n_layers", type=int, default=1)
    parser.add_argument("--lm_dropout", type=float, default=0.3)

    #unfreeze
    parser.add_argument("--unfreeze_top_blocks", type=int, default=0,
                    help="Number of top backbone blocks to unfreeze (0=fully frozen)")
 
    #padding
    parser.add_argument("--use_late_fusion_padding", action="store_true")

    #ViT model
    parser.add_argument("--use_vit", action="store_true")
    return parser.parse_args()
 
 
def build_label_mapping(label_csv):
    df = pd.read_csv(label_csv)
    df["participant_id"] = df["participant_id"].astype(str)
    df["q_id"] = df["q_id"].astype(str)
    mapping = {}
    for _, row in df.drop_duplicates(["participant_id", "q_id"]).iterrows():
        mapping[(row["participant_id"], row["q_id"])] = int(row["label"])
    return mapping
 
 
def evaluate(model, loader, device, frames_per_token, window_size, slide, 
             use_fusion=False, use_late_fusion=False, use_late_fusion_padding=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    submission_rows = []
 
    with torch.no_grad():
        for batch in loader:
            if use_fusion or use_late_fusion:
                frames, landmarks, audio, labels, mask, audio_mask, pids, qids = batch
                audio = audio.to(device)
                audio_mask = audio_mask.to(device)
                frames    = frames.to(device)
                landmarks = landmarks.to(device)
                mask      = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                logits    = model(frames, landmarks, audio, mask, audio_mask)
            
            elif use_late_fusion_padding:
                frames, landmarks, labels, mask, pids, qids = batch
                frames    = frames.to(device)
                landmarks = landmarks.to(device)
                mask      = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                logits    = model(frames, landmarks, mask)

            else:
                frames, labels, mask, pids, qids = batch
                frames = frames.to(device)
                mask   = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                logits = model(frames, mask)
 
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
 
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels_np)
 
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
 
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    except:
        auc = 0.0
    return f1, bal_acc, auc, pd.DataFrame(submission_rows)
 
 
def train_one_fold(args, train_participants, val_participants, label_mapping, device, fold_idx=0):
    set_seed(args.seed)
 
    if args.use_fusion:
        from badidea_fusion_model import BadIdeaFusionDataset, BadIdeaFusionTransformer, collate_fn
        train_dataset = BadIdeaFusionDataset(
            train_participants, args.npy_base_path, args.landmark_dir, label_mapping)
        val_dataset = BadIdeaFusionDataset(
            val_participants, args.npy_base_path, args.landmark_dir, label_mapping)
    elif args.use_late_fusion:
        from badidea_late_fusion_model import BadIdeaLateFusionDataset, BadIdeaLateFusionTransformer, collate_fn
        train_dataset = BadIdeaLateFusionDataset(
            train_participants, args.npy_base_path, args.landmark_dir, args.audio_dir, label_mapping)
        val_dataset = BadIdeaLateFusionDataset(
            val_participants, args.npy_base_path, args.landmark_dir, args.audio_dir, label_mapping)
    elif args.use_late_fusion_padding:
        from badidea_late_fusion_padding import BadIdeaLateFusionDataset, BadIdeaLateFusionTransformer, collate_fn
        train_dataset = BadIdeaLateFusionDataset(
            train_participants, args.npy_base_path, args.landmark_dir, label_mapping)
        val_dataset = BadIdeaLateFusionDataset(
            val_participants, args.npy_base_path, args.landmark_dir, label_mapping)
    elif args.use_vit:
        from badidea_vit_model import BadIdeaDatasetVideo, BadIdeaTransformer, collate_fn
        train_dataset = BadIdeaDatasetVideo(train_participants, args.npy_base_path, label_mapping)
        val_dataset   = BadIdeaDatasetVideo(val_participants,   args.npy_base_path, label_mapping)
    else:
        from badidea_model import BadIdeaDatasetVideo, BadIdeaTransformer, collate_fn
        train_dataset = BadIdeaDatasetVideo(train_participants, args.npy_base_path, label_mapping)
        val_dataset   = BadIdeaDatasetVideo(val_participants,   args.npy_base_path, label_mapping)
 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)
 
    print(f"Train videos: {len(train_dataset)}, Val videos: {len(val_dataset)}")
 
    if args.use_fusion:
        from badidea_fusion_model import BadIdeaFusionTransformer
        model = BadIdeaFusionTransformer(
            num_classes=2,
            frames_per_token=args.frames_per_token,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
            lm_proj_dim=args.lm_proj_dim,
        ).to(device)
    
    elif args.use_late_fusion_padding:
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
 
    elif args.use_late_fusion:
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
            audio_d_model=args.audio_d_model,
            audio_n_heads=args.audio_n_heads,
            audio_n_layers=args.audio_n_layers,
        ).to(device)

    elif args.use_vit:
        from badidea_vit_model import BadIdeaTransformer
        model = BadIdeaTransformer(
            num_classes=2,
            frames_per_token=args.frames_per_token,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
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
 
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
 
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: epoch {ckpt.get('epoch')}, AUC={ckpt.get('val_auc', 0):.4f}")
 
    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=1e-3,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
 
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name, f"fold_{fold_idx}")
    os.makedirs(checkpoint_dir, exist_ok=True)
 
    best_auc = 0.0
    patience_counter = 0
    window_size = args.frames_per_token * 5
    slide = args.frames_per_token
 
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
 
        for batch in train_loader:
            if args.use_late_fusion:
                frames, landmarks, audio, labels, mask, audio_mask, pids, qids = batch
                frames    = frames.to(device)       
                landmarks = landmarks.to(device)    
                labels    = labels.to(device)
                audio = audio.to(device)
                audio_mask = audio_mask.to(device)
                mask      = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                optimizer.zero_grad() 
                logits = model(frames, landmarks, audio, mask, audio_mask)
            elif args.use_late_fusion_padding:
                frames, landmarks, labels, mask, pids, qids = batch
                frames    = frames.to(device)
                landmarks = landmarks.to(device)
                labels    = labels.to(device)
                mask      = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                optimizer.zero_grad()
                logits = model(frames, landmarks, mask)
            elif args.use_fusion:
                frames, landmarks, labels, mask, pids, qids = batch
                frames    = frames.to(device)       
                landmarks = landmarks.to(device)   
                labels    = labels.to(device)     
                mask      = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                optimizer.zero_grad()             
                logits = model(frames, landmarks, mask)
            else:
                frames, labels, mask, pids, qids = batch
                frames = frames.to(device)
                labels = labels.to(device)
                mask   = (frames.abs().sum(dim=(2, 3, 4)) > 0)
                optimizer.zero_grad()
                logits = model(frames, mask)
 
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
 
        train_loss /= len(train_loader)
        val_f1, val_bal_acc, val_auc, sub_df = evaluate(
            model, val_loader, device, args.frames_per_token,
            window_size, slide, use_fusion=args.use_fusion,
            use_late_fusion=args.use_late_fusion, use_late_fusion_padding=args.use_late_fusion_padding
        )
        scheduler.step(val_auc)
 
        print(f"[Fold {fold_idx}] Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val BalAcc: {val_bal_acc:.4f}")
 
        if not args.no_wandb:
            import wandb
            wandb.log({
                f"fold_{fold_idx}_train_loss": train_loss,
                f"fold_{fold_idx}_val_auc": val_auc,
                f"fold_{fold_idx}_val_f1": val_f1,
                f"fold_{fold_idx}_val_bal_acc": val_bal_acc,
                "epoch": epoch,
            })
 
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc},
                os.path.join(checkpoint_dir, "best_model.pth")
            )
            sub_df.to_csv(os.path.join(checkpoint_dir, "best_submission.csv"), index=False)
            print(f"  ✓ Best model saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
 
    return best_auc
 
 
def main():
    args = parse_args()
    set_seed(args.seed)
 
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    print(f"Run:    {args.run_name}")
    print(f"Fusion: {args.use_fusion}")
 
    if (args.use_fusion or args.use_late_fusion or args.use_late_fusion_padding) and args.landmark_dir is None:
        raise ValueError("--landmark_dir required when --use_fusion or --use_late_fusion is set")
 
    if args.use_late_fusion and args.audio_dir is None:
        raise ValueError("--audio_dir required when --use_late_fusion is set")
    
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
 
    label_mapping = build_label_mapping(args.label_csv)
 
    if args.use_cv:
        train_df = pd.read_csv(args.train_labels_csv)
        val_df   = pd.read_csv(args.val_labels_csv)
        all_df   = pd.concat([train_df, val_df], ignore_index=True)
        all_participants = sorted(all_df["participant_id"].astype(str).unique())
 
        print(f"\nCross-validation: {args.n_folds} folds over {len(all_participants)} participants")
 
        kf = GroupKFold(n_splits=args.n_folds)
        fold_aucs = []
 
        for fold_idx, (train_idx, val_idx) in enumerate(
            kf.split(all_participants, groups=all_participants)
        ):
            train_participants = [all_participants[i] for i in train_idx]
            val_participants   = [all_participants[i] for i in val_idx]
 
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx + 1}/{args.n_folds}")
            print(f"Train: {train_participants}")
            print(f"Val:   {val_participants}")
            print(f"{'='*60}")
 
            best_auc = train_one_fold(args, train_participants, val_participants,
                                      label_mapping, device, fold_idx=fold_idx)
            fold_aucs.append(best_auc)
            print(f"Fold {fold_idx + 1} Best AUC: {best_auc:.4f}")
 
        print(f"\n{'='*60}")
        print(f"CV Results: {[f'{a:.4f}' for a in fold_aucs]}")
        print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"Baseline to beat: 0.561")
 
    else:
        train_df = pd.read_csv(args.train_labels_csv)
        val_df   = pd.read_csv(args.val_labels_csv)
        train_participants = sorted(train_df["participant_id"].astype(str).unique())
        val_participants   = sorted(val_df["participant_id"].astype(str).unique())
 
        print(f"Train participants ({len(train_participants)}): {train_participants}")
        print(f"Val participants   ({len(val_participants)}):   {val_participants}")
 
        best_auc = train_one_fold(args, train_participants, val_participants,
                                  label_mapping, device, fold_idx=0)
 
        print(f"\nDone! Best Val AUC: {best_auc:.4f}")
        print(f"Baseline to beat: 0.561")
 
    if not args.no_wandb:
        import wandb
        wandb.finish()
 
 
if __name__ == "__main__":
    main()