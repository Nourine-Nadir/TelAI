import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


def preprocess_unsw(path: str, seq_len: int = 50, normalize: bool = True, multiclass: bool = False):
    df = pd.read_csv(path)

    # Extract labels based on multiclass flag
    if multiclass:
        # Multi-class: use attack_cat column
        if "attack_cat" in df.columns:
            # Map attack categories to numerical labels
            attack_categories = ['Normal', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS',
                                 'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
            category_to_label = {cat: idx for idx, cat in enumerate(attack_categories)}

            # Convert attack_cat to numerical labels
            y = df["attack_cat"].map(category_to_label).fillna(0).astype(int).values
        else:
            y = np.zeros(len(df))
    else:
        # Binary: use label column
        if "attack_cat" in df.columns:
            y = df["label"].astype(int).values  # 0 = normal, 1 = attack
        else:
            y = np.zeros(len(df))

    df = df.drop(["attack_cat", "label", 'id'], axis=1, errors='ignore')

    for col in ["proto", "service", "state"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Keep only numeric features
    X = df.select_dtypes(include=[np.number]).fillna(0).values
    if normalize:
        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Slice into fixed-length sequences
    sequences, labels = [], []
    for i in range(0, len(X) - seq_len, seq_len):
        sequences.append(X[i:i + seq_len])
        # For multi-class, use the most frequent class in the window
        if multiclass:
            window_labels = y[i:i + seq_len]
            if len(window_labels) > 0:
                labels.append(np.bincount(window_labels).argmax())
            else:
                labels.append(0)
        else:
            # Binary: label = 1 if any anomaly inside window
            labels.append(int(y[i:i + seq_len].max()))

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