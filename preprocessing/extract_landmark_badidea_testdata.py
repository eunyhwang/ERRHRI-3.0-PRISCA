import cv2
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import mediapipe as mp

DATASET = "bad"
DATASET = "badidea"

def precompute_all(samples, cache_dir):
    """
    Scorre tutto il dataset ed estrae i landmark grezzi a step=1.
    Se il file .npy esiste già, lo salta automaticamente.
    """
    print(f"Avvio pre-estrazione su {len(samples)} video nella cartella '{cache_dir}'...")
    os.makedirs(cache_dir, exist_ok=True)
    skipped = []
    
    for video_path, _ in tqdm(samples, desc="Estrazione landmark"):
        cp = cache_path_for(video_path, cache_dir)
        
        if os.path.exists(cp):
            continue  
            
        seq = extract_landmarks_from_video(video_path, max_frames=None, frame_step=1)
        
        if seq is None:
            skipped.append(video_path)
            continue
            
        np.save(cp, seq)
        
    print("\nEstrazione completata")
    if skipped:
        print(f"Skippati {len(skipped)} video.")

def extract_landmarks_from_video(video_path: str, max_frames=None, frame_step=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is None:
        start_frame, limit_frames = 0, float('inf')
    else:
        frames_needed = max_frames * frame_step
        start_frame   = max(0, total_frames - frames_needed)
        limit_frames  = max_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    all_landmarks, frame_idx = [], 0

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened() and len(all_landmarks) < limit_frames:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_step != 0:
                frame_idx += 1; continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                coords = normalize_and_align_landmarks(np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32))
                all_landmarks.append(coords)
            elif all_landmarks: 
                all_landmarks.append(all_landmarks[-1].copy())
            frame_idx += 1
            
    cap.release()
    return np.stack(all_landmarks, axis=0) if len(all_landmarks) >= 5 else None

def normalize_and_align_landmarks(lm: np.ndarray) -> np.ndarray:
    """Trasla su origine, allinea rotazione 3D e scala rispetto agli occhi."""
    centroid = (lm[33] + lm[263] + lm[1]) / 3.0
    lm_centered = lm - centroid

    x_axis = lm_centered[263] - lm_centered[33]
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
    mid_eyes = (lm_centered[33] + lm_centered[263]) / 2.0
    y_axis = mid_eyes - lm_centered[1]
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    lm_rotated = np.dot(lm_centered, R.T)

    inter_eye = np.linalg.norm(lm_rotated[33] - lm_rotated[263]) + 1e-8
    return lm_rotated / inter_eye

def cache_path_for(video_path: str, cache_dir: str) -> str:
    safe_name = video_path.replace(os.sep, "_").replace(":", "_")
    return os.path.join(cache_dir, safe_name + ".npy")

if __name__ == "__main__":
    import glob
    
    CACHE_DIR = "/media/Datasets/bad_idea/test_npy_landmark"
    TEST_DIR = "/media/Datasets/bad_idea/test_data_badidea"
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # scanning without csv
    all_samples = []
    for pid in os.listdir(TEST_DIR):
        pid_dir = os.path.join(TEST_DIR, pid)
        if not os.path.isdir(pid_dir):
            continue
        for fname in os.listdir(pid_dir):
            if fname.endswith(".mp4"):
                full_path = os.path.join(pid_dir, fname)
                #label = int(fname.split("_")[-1].replace(".mp4", ""))
                all_samples.append((full_path, -1))
    
    print(f"Found {len(all_samples)} test videos")
    precompute_all(all_samples, cache_dir=CACHE_DIR)