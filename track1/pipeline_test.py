from feature_extraction_utils import *
from model import *
from test_utils import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CACHE_DIR = "landmarks_bad_test"
ROOT_DIR = "test_data_baddataset" 
SUB_FILE_NAME = "sub_test.csv"

if __name__ == "__main__":
    print(f"Execution on {DEVICE}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    VAL_THRESHOLD = np.float64(0.6499999999999999)
    VAL_MIN_CONSECUTIVE = 1

    model = FaceReactionGNNSimple(
        node_feat_dim=23, 
        gat_hidden=32, 
        gru_hidden=16, 
        dropout=0.3 
    ).to(DEVICE)
    model.eval()
    print("#####LOADING WEIGHTS#####")
    chkp = torch.load("face_reaction_gnn_windows_fps5_ws10_s5.pth", weights_only=True)
    model.load_state_dict(chkp)
    print("Done\n")
    print("#####SCAN AND FEATURE EXTRACTION#####")
    test_samples = scan_dataset_test_bad(ROOT_DIR)
    precompute_all(test_samples, cache_dir=CACHE_DIR)
    print("Done\n")
    print("#####SUBMISSION GENERATION#####")
    df_sub = generate_official_submission(
                                        model=model,                  
                                        samples_list=test_samples, 
                                        cache_dir=CACHE_DIR, 
                                        edge_index=EDGE_INDEX, 
                                        device=DEVICE,
                                        fps=5, 
                                        window_size=10, 
                                        slide=5,
                                        original_fps=30,
                                        threshold=VAL_THRESHOLD, 
                                        min_consecutive=VAL_MIN_CONSECUTIVE,              
                                        output_file=SUB_FILE_NAME
                                        )
    print(f"Done, saved as {SUB_FILE_NAME}")