from models import AnomalyTransformer
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torch

def train_model(train_dl, val_dl, input_dim, epochs=10, lr=1e-3, device="cuda"):
    model = AnomalyTransformer(input_dim=input_dim).to(device)
    opt = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc='Training'):
        # ---------- Training ----------
        model.train()
        total_loss, total_samples = 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

        avg_loss = total_loss / total_samples

        if epoch % 11 == 0:
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

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model
