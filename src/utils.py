import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


def preprocess_unsw(path, seq_len=50):
    df = pd.read_csv(path)

    # Extract labels
    if "attack_cat" in df.columns:
        y = df["label"].astype(int).values  # 0 = normal, 1 = attack
        df = df.drop(["attack_cat", "label", 'id'], axis=1)
    else:
        y = np.zeros(len(df))

    for col in ["proto", "service", "state"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Keep only numeric features
    X = df.select_dtypes(include=[np.number]).fillna(0).values
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Slice into fixed-length sequences
    sequences, labels = [], []
    for i in range(0, len(X) - seq_len, seq_len):
        sequences.append(X[i:i + seq_len])
        labels.append(int(y[i:i + seq_len].max()))  # label = 1 if any anomaly inside window

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def preprocess_single_sequence(df_row, scaler=None, feature_columns=None):
    """Preprocess a single row for prediction"""
    if feature_columns is None:
        # Select only numeric features
        feature_columns = df_row.select_dtypes(include=[np.number]).columns

    # Extract features
    features = df_row[feature_columns].fillna(0).values.astype(np.float32)

    # Normalize (you might want to save the scaler during training)
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1)).flatten()

    return features