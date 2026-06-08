#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Extract Bad Idea trainval videos into 30 fps NPY tensors, modified from Mahmoud's code.
Reads all videos under data/badidea/trainval/<participant_id>/q_<stimulus_id>_main_<label>.mp4 and then it extracts
frames at 30 fps, crops the face region using MediaPipe, uses only the second half of each video (temporal crop),
resizes and normalizes as used in the original baseline code and saves as .npy too and
writes data/badidea/trainval_npy/label_data.csv with (participant_id, q_id, label) per frame.
The output from this script is used in train_badnet_valonly_baddataset.py to be in line with pradip's split.
"""

import os
import csv
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
import mediapipe as mp

#removed face-cropping & temporal cropping
#modified the dir for test data
#updated qid_from_filename function for both test & trainval data

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/Datasets/bad_idea")
TRAINVAL_DIR = Path("/media/Datasets/bad_idea/test_data_badidea")
NPY_ROOT = Path("/media/Datasets/bad_idea/test_npy")
LABEL_CSV = NPY_ROOT / "label_data.csv"

TARGET_FPS = 30
IMG_SIZE = 224
FACE_PADDING = 0.2  # 20% padding around face bbox

NORMALIZE = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

mp_face_detection = mp.solutions.face_detection

def qid_from_filename(file_name: str):
    base = os.path.splitext(file_name)[0]  # q_10_main
    parts = base.split("_")
    # test set: q_10_main (len=3)
    # trainval: q_10_main_0 (len=4)
    stim_str = parts[1]
    q_id = f"q_{int(stim_str)}"
    label = int(parts[3]) if len(parts) == 4 else -1
    return q_id, label


def crop_face(pil_img: Image.Image, detector, last_bbox: list) -> Image.Image:
    """
    Detect face and crop with padding.
    Falls back to previous bbox if detection fails.
    Falls back to full image if no bbox available.
    """
    img_np = np.array(pil_img)
    h, w = img_np.shape[:2]

    results = detector.process(img_np)

    if results.detections:
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        x1 = int((bbox.xmin - FACE_PADDING) * w)
        y1 = int((bbox.ymin - FACE_PADDING) * h)
        x2 = int((bbox.xmin + bbox.width + FACE_PADDING) * w)
        y2 = int((bbox.ymin + bbox.height + FACE_PADDING) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        last_bbox[:] = [x1, y1, x2, y2]

    if last_bbox:
        x1, y1, x2, y2 = last_bbox
        return pil_img.crop((x1, y1, x2, y2))

    return pil_img  # fallback: full image


def extract_frames_ffmpeg(video_path: Path):
    """
    Stream raw RGB frames from ffmpeg at TARGET_FPS.
    Only yields the second half of the video (temporal crop).
    """
    # Probe width/height
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0", str(video_path)
        ],
        capture_output=True, text=True
    )
    w_h = probe.stdout.strip().split(",")
    if len(w_h) != 2:
        raise RuntimeError(f"ffprobe failed on {video_path}: '{probe.stdout}' '{probe.stderr}'")
    w, h = int(w_h[0]), int(w_h[1])

    # Probe duration for temporal crop (second half only)
    dur_probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video_path)
        ],
        capture_output=True, text=True
    )
    try:
        duration = float(dur_probe.stdout.strip())
        #start_time = duration / 2.0  # second half only
        start_time = 0.0
    except ValueError:
        start_time = 0.0  # fallback: full video

    cmd = [
        "ffmpeg", "-ss", str(start_time),
        "-i", str(video_path),
        "-vf", f"fps={TARGET_FPS}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 3
    idx = 1
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            yield idx, Image.fromarray(arr)
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


def main():
    NPY_ROOT.mkdir(parents=True, exist_ok=True)

    label_rows = []

    participants = sorted([p for p in os.listdir(TRAINVAL_DIR)
                           if (TRAINVAL_DIR / p).is_dir()])

    print(f"Found {len(participants)} participants under {TRAINVAL_DIR}")

    total_videos = 0
    for pid in participants:
        pid_dir = TRAINVAL_DIR / pid
        vids = sorted([f for f in os.listdir(pid_dir) if f.endswith(".mp4")])
        total_videos += len(vids)

    done = 0
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as detector:

        for pid in participants:
            pid_dir = TRAINVAL_DIR / pid
            out_dir = NPY_ROOT / pid
            out_dir.mkdir(parents=True, exist_ok=True)

            video_files = sorted([f for f in os.listdir(pid_dir) if f.endswith(".mp4")])

            for vid_file in video_files:
                video_path = pid_dir / vid_file
                q_id, label = qid_from_filename(vid_file)
                if q_id is None:
                    print(f"[WARN] Skipping file with unexpected name: {video_path}")
                    continue

                done += 1
                print(f"[{done}/{total_videos}] {pid}/{vid_file}  q_id={q_id}, label={label}")

                # Check if already processed
                existing = [f for f in os.listdir(out_dir)
                            if f.startswith(f"{q_id}_main_{label}_30fps_") and f.endswith(".npy")]
                if existing:
                    for _ in existing:
                        label_rows.append((pid, q_id, label))
                    continue

                # Extract second half frames + face crop
                last_bbox = []
                for frame_idx, pil_img in extract_frames_ffmpeg(video_path):
                    #cropped = crop_face(pil_img, detector, last_bbox)

                    stem = f"{q_id}_main_{label}_30fps_frame{frame_idx:04d}"
                    npy_path = out_dir / (stem + ".npy")

                    if not npy_path.exists():
                        arr = NORMALIZE(pil_img).numpy()  # (3, 224, 224) float32
                        np.save(npy_path, arr)

                    label_rows.append((pid, q_id, label))

    # Write label_data.csv
    with LABEL_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "q_id", "label"])
        w.writerows(label_rows)

    print(f"\nWrote {len(label_rows)} frame-label rows to {LABEL_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()