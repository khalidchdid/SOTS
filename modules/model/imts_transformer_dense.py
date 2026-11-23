import math
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Positional encoding
def positional_encoding(L: int, d_model: int, device="cpu") -> torch.Tensor:
    """Return (L, d_model) sinusoidal matrix."""
    pos = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)  # (L,1)
    dim = torch.arange(d_model, dtype=torch.float32, device=device).unsqueeze(0)  # (1,d)
    denom = torch.pow(500.0, (2 * (dim // 2)) / d_model)
    phase = pos / denom
    pe = torch.zeros_like(phase)
    pe[:, 0::2] = torch.sin(phase[:, 0::2])
    pe[:, 1::2] = torch.cos(phase[:, 1::2])
    return pe  # (L,d_model)

class DenseIMTSTransformer(nn.Module):
    """
    Transformer seft : measurement indicators
    input per unique timestamp :  [v1, … vM; m1, … mM]  (size = 2*M per mesurement)
    m_j = 1 if variable j observed, else 0
    """
    def __init__(self,
                 n_modalities: int,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dim_ff: int | None = None,
                 dropout: float = 0.3,
                 n_classes: int = 2):
        super().__init__()
        self.n_mods = n_modalities
        self.d_model = d_model

        dim_ff = 2 * d_model

        # Linear projection from raw feature vector (2*M) -> d_model
        self.input_lin = nn.Linear(2 * n_modalities, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
            # norm_first=True
            )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # self.cls_head = nn.Sequential(
        #     nn.LayerNorm(d_model),
        #     nn.Linear(d_model, n_classes)
        # )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model+2, d_model+2),
            nn.ReLU(),
            nn.Linear(d_model+2, n_classes),
        )
    
    def forward(self,
                value_mask_tensor,   # (B,L, 2*M)
                pad_mask, static_5, static_6):           # (B,L) True = pad
        B, L, _ = value_mask_tensor.shape
        device  = value_mask_tensor.device

        pe = positional_encoding(L, self.d_model, device)  # (L,d)
        x  = self.input_lin(value_mask_tensor) + pe  # (B,L,d)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x_masked = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled = x_masked.sum(dim=1) / lengths         # mean aggregation

        # static_5, static_6 : error_start_time, error_duration

        pooled_cat = torch.cat([pooled, static_5, static_6], dim=1)         # mean aggregation
        return self.cls_head(pooled_cat)                    # (B,n_classes)

class DenseTxtDataset(Dataset):
    """
    extract data from txt files (dataset)
    Each txt file: rows "t, param,  val"
    """
    def __init__(self,
                 root_dir: Path,
                 indices: List[int],
                 label_map: Dict[int, int],
                 n_modalities: int):
        self.root_dir = Path(root_dir)
        self.idx_list = indices
        self.label_map = label_map
        self.M = n_modalities
    
    def _load_one(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = []
        for line in (self.root_dir / f"{idx}.txt").open():
            if not line.startswith('Time'):
                t, m, v = line.strip().split(',')
                rows.append((np.float16(t), int(m), float(v)))
                if int(m) == 5:
                    static_5 = torch.tensor(float(v), dtype=torch.float32)
                if int(m) == 6:
                    static_6 = torch.tensor(float(v), dtype=torch.float32)

        rows.sort(key=lambda x: x[0])
        times = torch.tensor([r[0] for r in rows], dtype=torch.float32)
        mods  = torch.tensor([r[1] for r in rows], dtype=torch.long)
        vals  = torch.tensor([r[2] for r in rows], dtype=torch.float32)

        uniq_times, inv = torch.unique(times, return_inverse=True)
        L = len(uniq_times)
        V = torch.zeros((L, self.M))
        M = torch.zeros_like(V)
        V[inv, mods] = vals
        M[inv, mods] = 1.0

        feat = torch.cat([V, M], dim=1)  # (L,2M)
        return feat, static_5, static_6, torch.tensor(self.label_map[idx], dtype=torch.long)

    def __len__(self): return len(self.idx_list)
    def __getitem__(self, i): return self._load_one(self.idx_list[i])

def collate_dense(batch):
    # Pre process data for batch
    feats, statics_5, statics_6, labels = zip(*batch)
    lengths = [f.shape[0] for f in feats]
    Lmax = max(lengths)
    M2 = feats[0].shape[1]          # 2*M

    pad_feat = torch.zeros(len(batch), Lmax, M2)
    pad_mask = torch.ones(len(batch), Lmax, dtype=torch.bool)

    for i, f in enumerate(feats):
        l = f.shape[0]
        pad_feat[i, :l, :] = f
        pad_mask[i, :l] = False
    return pad_feat, pad_mask, torch.tensor(statics_5).unsqueeze(1), torch.tensor(statics_6).unsqueeze(1), torch.tensor(labels)