import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import copy


class FederatedLearningServer:
    def __init__(self, global_model, num_clients, device="cuda"):
        self.global_model = global_model
        self.device = device
        self.num_clients = num_clients
        self.client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    def federated_averaging(self, client_weights):
        """Perform federated averaging of client model weights"""
        global_dict = self.global_model.state_dict()

        # Initialize averaged weights
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])

        # Sum weights from all clients
        total_samples = sum([samples for _, samples in client_weights])
        for client_dict, num_samples in client_weights:
            for key in global_dict.keys():
                global_dict[key] += client_dict[key] * num_samples

        # Average the weights
        for key in global_dict.keys():
            global_dict[key] /= total_samples

        # Update global model
        self.global_model.load_state_dict(global_dict)

        # Update all client models with new global model
        for client_model in self.client_models:
            client_model.load_state_dict(copy.deepcopy(global_dict))

    def train_round(self, client_dataloaders, epochs_per_client=1, lr=1e-4):
        """Perform one round of federated training"""
        client_weights = []

        for client_idx in range(self.num_clients):
            if client_idx < len(client_dataloaders):
                # Train client model
                client_loss, num_samples = self.train_client(
                    client_idx, client_dataloaders[client_idx], epochs_per_client, lr
                )

                # Get client model weights
                client_dict = self.client_models[client_idx].state_dict()
                client_weights.append((client_dict, num_samples))

                print(f"Client {client_idx}: Loss={client_loss:.4f}, Samples={num_samples}")

        # Perform federated averaging
        if client_weights:
            self.federated_averaging(client_weights)

    def train_client(self, client_idx, dataloader, epochs=1, lr=1e-4):
        """Train a single client model"""
        model = self.client_models[client_idx]
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        total_loss = 0
        num_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)
                num_samples += xb.size(0)

            total_loss += epoch_loss

        return total_loss / num_samples if num_samples > 0 else 0, num_samples


def create_client_dataloaders(dataset, num_clients, batch_size=512):
    """Split dataset into multiple clients"""

    # Split dataset among clients
    client_sizes = [len(dataset) // num_clients] * num_clients
    client_sizes[-1] += len(dataset) - sum(client_sizes)  # Handle remainder

    client_datasets = random_split(
        dataset, client_sizes,
        generator=torch.Generator().manual_seed(42)
    )

    client_dataloaders = []
    for client_dataset in client_datasets:
        dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(dataloader)

    return client_dataloaders