from utils import preprocess_unsw, FlowDataset
from test import evaluate_model
from train import train_model
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    X, y = preprocess_unsw("../Data/UNSW-NB15/UNSW_NB15_training-set.csv", seq_len=10)
    X_test, y_test = preprocess_unsw("../Data/UNSW-NB15/UNSW_NB15_testing-set.csv", seq_len=10)
    dataset_train = FlowDataset(X, y)
    print('train dataset ', len(dataset_train))

    # Split train/val

    n_train = int(0.8 * len(dataset_train))
    n_val = len(dataset_train) - n_train
    train_ds, val_ds = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(train_dl, val_dl, input_dim=X.shape[2], epochs=10, device=device)

    # Evaluate
    dataset_test = FlowDataset(X_test, y_test)
    test_dl = DataLoader(dataset_test, batch_size=32, shuffle=False)

    print('test dataset ', len(dataset_test))
    evaluate_model(model, test_dl, device=device)
