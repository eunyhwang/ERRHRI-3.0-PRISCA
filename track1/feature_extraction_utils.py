import cv2
from tqdm import tqdm
import os
import numpy as np
import mediapipe as mp
from torch_geometric.utils import subgraph
import torch
from torch_geometric.data import Data

mp_face_mesh = mp.solutions.face_mesh

def get_mediapipe_edges() -> torch.Tensor:
    connections = mp_face_mesh.FACEMESH_TESSELATION
    edges = set()
    for src, dst in connections:
        edges.add((src, dst))
        edges.add((dst, src))
    src_nodes, dst_nodes = zip(*edges)
    return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

EDGE_INDEX_RAW = get_mediapipe_edges()

def get_expressive_nodes():
    expressive_edges = (
        list(mp_face_mesh.FACEMESH_LIPS) +
        list(mp_face_mesh.FACEMESH_LEFT_EYE) +
        list(mp_face_mesh.FACEMESH_RIGHT_EYE) +
        list(mp_face_mesh.FACEMESH_LEFT_EYEBROW) +
        list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
    )
    nodes = set()
    for src, dst in expressive_edges:
        nodes.add(src)
        nodes.add(dst)
    return sorted(list(nodes))

def get_optimized_expressive_nodes(include_nose: bool = True, 
                                   include_irises: bool = True, 
                                   include_chin: bool = True,
                                   include_glabella: bool = True) -> list:
    """
    Extracts a robust subset of MediaPipe FACEMESH nodes.
    Includes explicit support for AU4 (Corrugator) via Glabella nodes.
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    # 1. Base Arches
    expressive_edges = (
        list(mp_face_mesh.FACEMESH_LIPS) +
        list(mp_face_mesh.FACEMESH_LEFT_EYE) +
        list(mp_face_mesh.FACEMESH_RIGHT_EYE) +
        list(mp_face_mesh.FACEMESH_LEFT_EYEBROW) +
        list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
    )
    
    if include_nose:
        expressive_edges += list(mp_face_mesh.FACEMESH_NOSE)
    if include_irises:
        expressive_edges += list(mp_face_mesh.FACEMESH_IRISES)
        
    nodes = set()
    for src, dst in expressive_edges:
        nodes.add(src)
        nodes.add(dst)
        
    if include_chin:
        nodes.add(152) 
        
    # 2. AU4 (Brow Lowerer / Corrugator) Precision Nodes
    if include_glabella:
        nodes.add(9)   # Central Glabella Epicenter
        nodes.add(107) # Left Inner Brow Tensor
        nodes.add(336) # Right Inner Brow Tensor
        nodes.add(55)  # Left Upper Glabella
        nodes.add(285) # Right Upper Glabella

    final_nodes = sorted(list(nodes))
    
    print(f"Extraction complete. Selected {len(final_nodes)} highly expressive nodes.")
    return final_nodes

EXPRESSIVE_NODES = get_expressive_nodes()
#EXPRESSIVE_NODES = get_optimized_expressive_nodes(include_irises=False)
EXPRESSIVE_TENSOR = torch.tensor(EXPRESSIVE_NODES, dtype=torch.long)

EDGE_INDEX, _ = subgraph(EXPRESSIVE_TENSOR, EDGE_INDEX_RAW, relabel_nodes=True)

def compute_corrected_au_proxies(lm: np.ndarray) -> np.ndarray:
    def dist(a, b): 
        return np.linalg.norm(lm[:, a, :] - lm[:, b, :], axis=-1)

    face_width = dist(234, 454) + 1e-6

    au_features = np.stack([
        dist(70, 63) / face_width,        # AU1: inner brow raise L
        dist(300, 293) / face_width,      # AU2: inner brow raise R
        dist(55, 285) / face_width,       # AU4: brow lowerer
        dist(159, 145) / (dist(33, 133) + 1e-6),   # AU5/7: EAR L (Verticale / Orizzontale occhio L)
        dist(386, 374) / (dist(362, 263) + 1e-6),  # AU5/7: EAR R (Verticale / Orizzontale occhio R)
        dist(48, 220) / face_width,       # AU9: nasolabial depth L
        dist(278, 440) / face_width,      # AU9: nasolabial depth R
        dist(61, 291) / face_width,       # AU12/20: lip corners width
        dist(0, 17) / face_width,         # AU25: vertical mouth opening
        dist(13, 14) / face_width,        # AU26: jaw drop (lips gap)
        dist(61, 76) / face_width,        # AU15: lip corner down L
        dist(291, 306) / face_width,      # AU15: lip corner down R
        dist(159, 386) / face_width,      # AU45: blink ratio
    ], axis=-1)

    return au_features

def extract_enhanced_features(seq: np.ndarray, 
                             expressive_nodes: list = EXPRESSIVE_NODES, fps=10) -> np.ndarray:
    T, N_tot, _ = seq.shape
    dt = 1.0 / fps
    
    au_raw = compute_corrected_au_proxies(seq) 
    au_expanded = np.repeat(au_raw[:, np.newaxis, :], len(expressive_nodes), axis=1)

    seq_sub = seq[:, expressive_nodes, :]
    radial_dist = np.linalg.norm(seq_sub, axis=-1, keepdims=True)
    
    velocity = np.gradient(seq_sub, dt, axis=0) 
    acceleration = np.gradient(velocity, dt, axis=0)
    
    features_final = np.concatenate([seq_sub, velocity, acceleration, radial_dist, au_expanded], axis=-1)
    
    return features_final.astype(np.float32)

def sequence_to_graph_list(seq: np.ndarray, edge_index: torch.Tensor) -> list[Data]:
    feats = extract_enhanced_features(seq)
    graphs = [Data(x=torch.tensor(f, dtype=torch.float), edge_index=edge_index) for f in feats]
    return graphs

def precompute_all(samples, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    skipped = []
    
    for video_path in tqdm(samples, desc="Landmark Extraction"):
        cp = cache_path_for(video_path, cache_dir)
        
        if os.path.exists(cp):
            continue  
            
        seq = extract_landmarks_from_video(video_path, max_frames=None, frame_step=1)
        
        if seq is None:
            skipped.append(video_path)
            continue
            
        np.save(cp, seq)

    print("Extraction Completed")

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

def scan_dataset_test_bad(root):
    samples = []
    test_subjects = sorted([s for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))])
    for subj_id in test_subjects:
        subj_folder = os.path.join(root, subj_id)
        for fname in sorted(os.listdir(subj_folder)):
            samples.append(os.path.join(subj_folder, fname))
    return samples