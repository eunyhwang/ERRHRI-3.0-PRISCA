# ICMI ERR@HRI 3.0 Challenge Submission

Multimodal Detection of Errors and Anticipation in Human-Robot Interactions
- [Challenge Website](https://sites.google.com/view/errhri30/)
- [Challenge Repository](https://github.com/IRL-CT/errhri-3-0/tree/main)


---

## Repository Structure

```
├── track1/
├── track2/
│   ├── badidea_late_fusion_padding.py        # Track 2 v1: Visual + Landmark
│   ├── badidea_late_fusion_audio.py (tentative)          # Track 2 v2: Visual + Landmark + Audio
│   ├── train_badidea.py                      # Training script 
│   ├── inference_badidea.py                  # Inference script 
│   └── badnet_pytorch.py                     # Utility functions
├── preprocessing/
│   ├── extract_badidea_npy.py                # Frame extraction
│   ├── extract_landmark_badidea.py           # Landmark extraction (trainval)
│   ├── extract_landmark_badidea_testdata.py  # Landmark extraction (test)
│   └── extract_aduio.py(#vad_spike_gate_track2_lesstime.py)     # Audio feature extraction
└── data/badidea/
    ├── train_labels.csv
    └── val_labels.csv
```

---

## Environment Setup

```bash
# System dependency
sudo apt install ffmpeg

# Python dependencies
pip install -r requirements.txt
```

---

## Track 1. Bad (Bystander Affect Detection) Dataset
*[to be filled]*

---

## Track 2. Bad Idea Dataset

### Window Parameters: `fps=30, window_size=150, slide=30`

### Step 1: Frame Extraction
Edit `TRAINVAL_DIR` and `NPY_ROOT` in `preprocessing/extract_badidea_npy.py`, then:
```bash
python preprocessing/extract_badidea_npy.py
```

### Step 2: Landmark Extraction
Edit `TEST_DIR` and `CACHE_DIR` in `preprocessing/extract_landmark_badidea_testdata.py`, then:
```bash
python preprocessing/extract_landmark_badidea_testdata.py
```

### Step 3: Inference (v1: Visual + Landmark)
```bash
python badnet/inference_badidea.py \
  --npy_base_path /path/to/test_npy \
  --landmark_dir /path/to/test_npy_landmark \
  --checkpoint /path/to/best_model.pth \
  --out_csv /path/to/test_submission.csv \
  --use_late_fusion_padding --freeze_backbone \
  --lm_proj_dim 128 --seed 42
```

### Step 3: Inference (v2: Visual + Landmark + Audio)
*[to be filled]*
