import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.notebook import trange
import torchio as tio


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric_fn: callable) -> list:
    network.eval()
    device = next(network.parameters()).device

    results = []
    for batch in data:
        inputs = torch.cat([
            batch['t1n'][tio.DATA],
            batch['t1c'][tio.DATA],
            batch['t2w'][tio.DATA],
            batch['t2f'][tio.DATA],
        ], dim=1).to(device)

        targets = batch['seg'][tio.DATA].squeeze(1).to(device)

        outputs = network(inputs)
        results.append(metric_fn(outputs, targets))
    return results


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss_fn: nn.Module,
           opt: optim.Optimizer) -> list:
    network.train()
    device = next(network.parameters()).device

    losses = []
    for batch in data:
        inputs = torch.cat([
            batch['t1n'][tio.DATA],
            batch['t1c'][tio.DATA],
            batch['t2w'][tio.DATA],
            batch['t2f'][tio.DATA],
        ], dim=1).to(device)

        targets = batch['seg'][tio.DATA].squeeze(1).to(device)

        opt.zero_grad()
        outputs = network(inputs)
        loss_value = loss_fn(outputs, targets)
        loss_value.backward()
        opt.step()

        losses.append(loss_value)
    return losses

class Trainer:
    def __init__(self, model, optimizer, loss_fn, metric_fn, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.device = device
        self.train_loss_history = []
        self.val_metric_history = []

    def train(self, train_loader, val_loader, epochs):
        for epoch in trange(epochs, desc="Epochs"):
            train_losses = update(self.model, train_loader, self.loss_fn, self.optimizer)
            avg_train_loss = torch.stack(train_losses).mean().item()
            self.train_loss_history.append(avg_train_loss)

            val_metrics = evaluate(self.model, val_loader, self.metric_fn)
            avg_val_metric = torch.stack(val_metrics).mean().item()
            self.val_metric_history.append(avg_val_metric)

            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Metric: {avg_val_metric:.4f}")

        self.plot_curves()

    def plot_curves(self):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='Train Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.val_metric_history, label='Val Metric', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Validation Metric')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
