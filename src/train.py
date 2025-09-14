from models import AnomalyTransformer
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torch


def train_model(train_dl, val_dl, input_dim, epochs=10, lr=1e-3, device="cuda", num_classes=10):
    model = AnomalyTransformer(input_dim=input_dim, num_classes=10).to(device)
    opt = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc='Training'):
        # ---------- Training ----------
        model.train()
        total_loss, total_samples = 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

        avg_loss = total_loss / total_samples

        if epoch % 10 == 0:
            # ---------- Validation ----------
            model.eval()
            val_correct, val_samples = 0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    val_correct += (preds.argmax(1) == yb).sum().item()
                    val_samples += xb.size(0)

            val_acc = val_correct / val_samples
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model


def train_client_model(model, dataloader, epochs=1, lr=1e-4, device="cuda"):
    """Train a client model for federated learning"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    total_loss = 0
    num_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            num_samples += xb.size(0)

        total_loss += epoch_loss

    return total_loss / num_samples if num_samples > 0 else 0, num_samples