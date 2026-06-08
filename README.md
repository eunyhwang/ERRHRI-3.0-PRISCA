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
│   ├── badidea_late_fusion_audio.py          # Track 2 v2: Visual + Landmark + Audio
│   ├── train_badidea.py                      # Training script 
│   ├── inference_badidea.py                  # Inference script 
│   └── badnet_pytorch.py                     # Utility functions
├── preprocessing/
│   ├── extract_badidea_npy.py                # Frame extraction
│   ├── extract_landmark_badidea.py           # Landmark extraction (trainval)
│   ├── extract_landmark_badidea_testdata.py  # Landmark extraction (test)
│   └── extract_aduio.py                      # Audio feature extraction
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
It should be known that we are making 2 submissions for track 2. One is a visual model only and the other one is the visual and audio models fused. The files related to the audio model are called "extract_audio.py" and "badidea_late_fusion_audio.py" which can be found under the folders "preprocessing" and "track2" respectively.

To replicate the results for Submission 1 (Video Only): Follow the steps up until Step 3 to get the CSV file. To replicate the results for Submission 2 (Video + Audio_: Follow all the steps (including the first 3 steps).

#### Window Parameters: `fps=30, window_size=150, slide=30`

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

### Step 4: Audio Extraction (v2: Visual + Landmark + Audio)
```bash
python preprocessing/extract_audio.py
--data_dir "/path/to/test/data"
--out_dir  ./Output --fps 30
--window_size 150 --slide 30
--vad-threshold 0.15
--min-speech-ms 50
--max_workers 4
```
This outputs a csv file called "per_window_features.csv" which will be used in the next step.

### Step 5: Inference (v2: Visual + Landmark + Audio)
```bash
 python track2/badidea_late_fusion_audio.py
--windows ./Output/per_window_features.csv
--visual /path/to/test_submission.csv (From  Step 3: Visual Model CSV output file)
--out ./Output/test_submission_audio_and_video.csv
--lambda_w 0.40
 ```


 This concludes all steps and gives the final CSV files for both submissions.  //
 Submission 1: test_submission.csv
 Submission 2: test_submission_audio_and_video.csv

