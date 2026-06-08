import torch
from tqdm import tqdm
import os
import glob
import numpy as np
import pandas as pd
from feature_extraction_utils import sequence_to_graph_list

@torch.no_grad()
def generate_official_submission(model, samples_list, cache_dir, edge_index, device,
                                        fps=5, window_size=10, slide=5, original_fps=30,
                                        threshold=0.5, min_consecutive=2, output_file="sub.csv"):
    model.eval()
    submission_data = []
    
    frame_step = original_fps // fps
    orig_window_frames = window_size * frame_step
    orig_slide_frames = slide * frame_step

    for video_path in tqdm(samples_list):
        parts = video_path.replace("\\", "/").split("/")
        pid = str(parts[-2]).strip()
        vid = str(parts[-1].split(".")[0]).strip()
        
        search_pattern = os.path.join(cache_dir, f"*{pid}*{vid}*.npy")
        matches = glob.glob(search_pattern)
        if not matches: continue
            
        seq = np.load(matches[0])
        total_frames = seq.shape[0]
        
        video_probabilities = []
        
        for start in range(0, total_frames, orig_slide_frames):
            end = start + orig_window_frames
            window_seq = seq[start:end][::frame_step]
            
            T = window_seq.shape[0]
            if T < window_size:
                if T > 0:
                    pad = np.tile(window_seq[-1:], (window_size - T, 1, 1))
                    window_seq = np.concatenate([window_seq, pad], axis=0)
                else: break
                    
            graphs = sequence_to_graph_list(window_seq, edge_index)
            x_tensor = torch.stack([g.x for g in graphs], dim=0).unsqueeze(0).to(device)
            
            logit = model(x_tensor, edge_index.to(device))
            video_probabilities.append(torch.sigmoid(logit).item())
            
            if end >= total_frames: break
                
        if not video_probabilities: continue
        
        consecutive_count = 0
        video_has_reaction = False
        
        for p in video_probabilities:
            if p >= threshold:
                consecutive_count += 1
                if consecutive_count >= min_consecutive:
                    video_has_reaction = True
                    break 
            else:
                consecutive_count = 0 
                
        global_video_label = 1 if video_has_reaction else 0
        
        for w_id, p1 in enumerate(video_probabilities):
            if global_video_label==0:
                submission_data.append({
                    "participant_id": pid,
                    "video_id": vid,
                    "window_id": w_id,
                    "y_pred": global_video_label,
                    "y_prob_0": 1.0 - p1,
                    "y_prob_1": p1
                })
            else:
                submission_data.append({
                    "participant_id": pid,
                    "video_id": vid,
                    "window_id": w_id,
                    "y_pred": global_video_label,
                    "y_prob_0": 1.0 - p1,
                    "y_prob_1": p1
                })

    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(output_file, index=False)
    return df_sub

