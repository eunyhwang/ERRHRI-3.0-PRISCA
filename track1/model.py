import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import Data, Batch

class FaceReactionGNNSimple(nn.Module):
    def __init__(self, node_feat_dim=23, gat_hidden=32, gru_hidden=16, dropout=0.3):
        super().__init__()
        
        self.gat1 = GATConv(node_feat_dim, gat_hidden, heads=2, concat=True, dropout=dropout)
        self.gat2 = GATConv(gat_hidden * 2, gat_hidden, heads=1, concat=False, dropout=dropout)
        self.bn1  = nn.BatchNorm1d(gat_hidden * 2)
        self.bn2  = nn.BatchNorm1d(gat_hidden)
        self.gru = nn.GRU(gat_hidden, gru_hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.attn_net = nn.Sequential(
            nn.Linear(gat_hidden, gat_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(gat_hidden // 2, 1)
        )
        self.global_attn_pool = AttentionalAggregation(gate_nn=self.attn_net)

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_batch, edge_index):
        B, T, N, Fe = x_batch.shape
        device = x_batch.device

        x_flat = x_batch.view(B * T, N, Fe)
        data_list = [Data(x=x_flat[bt], edge_index=edge_index) for bt in range(B * T)]
        big_batch = Batch.from_data_list(data_list).to(device)

        h = F.relu(self.bn1(self.gat1(big_batch.x, big_batch.edge_index)))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.bn2(self.gat2(h, big_batch.edge_index)))


        h_frames = self.global_attn_pool(h, big_batch.batch)
        h_frames = h_frames.view(B, T, -1)
        gru_out, _ = self.gru(h_frames)
        h_video = torch.mean(gru_out, dim=1)

        logit = self.classifier(h_video).squeeze(-1)
        return logit
    
