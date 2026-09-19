"""PyTorch Dataset for the EPS fault-diagnosis windows."""
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def load_metadata(data_dir: str = DATA_DIR) -> dict:
    with open(os.path.join(data_dir, "metadata.json")) as f:
        return json.load(f)


class EPSWindowDataset(Dataset):
    """Each item: (X, y) where X is (T, nodes, feat) z-score normalized
    using TRAIN-split statistics, y is a scalar class index."""

    def __init__(self, split: str, data_dir: str = DATA_DIR, normalize: bool = True):
        assert split in ("train", "val", "test")
        d = np.load(os.path.join(data_dir, f"{split}.npz"))
        self.X = d["X"].astype(np.float32)          # (N, T, nodes, feat)
        self.y = d["y"].astype(np.int64)
        self.fault_mask = d["fault_mask"]
        self.eclipse_mask = d["eclipse_mask"]
        self.meta = load_metadata(data_dir)

        if normalize:
            mean = np.array(self.meta["normalization"]["mean"], dtype=np.float32)
            std = np.array(self.meta["normalization"]["std"], dtype=np.float32)
            self.X = (self.X - mean[None, None]) / std[None, None]

        self.n_nodes = self.X.shape[2]
        self.n_features = self.X.shape[3]
        self.seq_len = self.X.shape[1]
        self.n_classes = len(self.meta["fault_classes"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])   # (T, nodes, feat)
        y = torch.tensor(self.y[idx])
        return x, y
