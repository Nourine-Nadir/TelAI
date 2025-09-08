import os

from utils import preprocess_unsw, FlowDataset
from test import evaluate_model
from train import train_model
from federated import FederatedLearningServer, create_client_dataloaders
from models import AnomalyTransformer, AnomalyAwareTransformer
from torch.utils.data import random_split, DataLoader
import torch
from tqdm import tqdm

def main_centralized():
    """Centralized training"""
    X, y = preprocess_unsw("../Data/UNSW-NB15/UNSW_NB15_training-set.csv", seq_len=15)
    X_test, y_test = preprocess_unsw("../Data/UNSW-NB15/UNSW_NB15_testing-set.csv", seq_len=15)

    dataset_train = FlowDataset(X, y)
    n_train = int(0.8 * len(dataset_train))
    n_val = len(dataset_train) - n_train
    train_ds, val_ds = random_split(dataset_train, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(train_dl, val_dl, input_dim=X.shape[2], epochs=200, device=device, lr=1e-4)

    # Evaluate
    dataset_test = FlowDataset(X_test, y_test)
    test_dl = DataLoader(dataset_test, batch_size=512, shuffle=False)
    evaluate_model(model, test_dl, device=device)


def main_federated():
    """Federated learning approach"""
    X, y = preprocess_unsw("../Data/UNSW-NB15/UNSW_NB15_training-set.csv", seq_len=1)
    X_test, y_test = preprocess_unsw("../Data/UNSW-NB15/UNSW_NB15_testing-set.csv", seq_len=1)

    # Create full training dataset
    dataset_train = FlowDataset(X, y)

    # Split into validation set (20%)
    n_train = int(0.8 * len(dataset_train))
    n_val = len(dataset_train) - n_train
    train_ds, val_ds = random_split(dataset_train, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Create validation dataloader
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    # Federated learning setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_clients = 10  # Number of clients/silos

    # Create client dataloaders
    client_dataloaders = create_client_dataloaders(train_ds, num_clients, batch_size=512)

    # Initialize global model
    global_model = AnomalyAwareTransformer(input_dim=X.shape[2]).to(device)

    # Initialize federated learning server
    server = FederatedLearningServer(global_model, num_clients, device)

    # Federated training
    num_rounds = 70
    epochs_per_client = 10

    for round_idx in tqdm(range(num_rounds), desc="Federated Rounds"):
        server.train_round(client_dataloaders, epochs_per_client, lr=1e-4)

        # Validate every 5 rounds
        if round_idx % 5 == 0:
            server.global_model.eval()
            val_correct, val_samples = 0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    preds, _ = server.global_model(xb)
                    val_correct += (preds.argmax(1) == yb).sum().item()
                    val_samples += xb.size(0)

            val_acc = val_correct / val_samples
            print(f"Round {round_idx + 1}/{num_rounds} | Val Acc: {val_acc:.4f}")

    # Final evaluation
    dataset_test = FlowDataset(X_test, y_test)
    test_dl = DataLoader(dataset_test, batch_size=512, shuffle=False)
    print("Final Evaluation:")
    try:
        os.makedirs('saved_models', exist_ok=True)
        # Save global model weights after training
        torch.save(server.global_model.state_dict(), "saved_models/federated_model_seq1_.pth")
        print('Model Saved !')
    except:
        print('Couldn\'t save the model !' )

    evaluate_model(server.global_model, test_dl, device=device)


if __name__ == "__main__":
    print("Choose training mode:")
    print("1. Centralized Training")
    print("2. Federated Learning")

    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        print("Running centralized training...")
        main_centralized()
    else:
        print("Running federated learning...")
        main_federated()