import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.notebook import trange
import torchio as tio
from utils.logging_utils import Tracker
from pathlib import Path

from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, loss_fn, metric_fn, device='cuda', use_torchio=False, tracker: Tracker = None,):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.device = device
        self.train_loss_history = []
        self.val_metric_history = []
        self.use_torchio = use_torchio

        if tracker is None:
            tracker = Tracker()
        self.tracker = tracker

    def state_dict(self):
        """ Current state of learning. """
        return {
            "model": self.model.state_dict(),
            "objective": self.loss_fn.state_dict(),
            "optimiser": self.optimizer.state_dict(),
            "num_epochs": self.tracker.epoch,
            "num_updates": self.tracker.update,
        }

    @torch.no_grad()
    def evaluate(self, data: DataLoader, use_torchio=False, tag: str = None) -> list:
        self.model.eval()
        device = next(self.model.parameters()).device

        self.tracker.start(tag, num_batches=len(data))

        losses = []
        dice_scores = []
        #for batch in tqdm(data, 'Evaluating'):
        for batch in data:
            if use_torchio:
                inputs = torch.cat([
                    batch['t1n'][tio.DATA],
                    batch['t1c'][tio.DATA],
                    batch['t2w'][tio.DATA],
                    batch['t2f'][tio.DATA],
                ], dim=1).float().to(device)

                targets = batch['seg'][tio.DATA].float().squeeze(1).to(device)
            else:
                inputs, targets = batch
                #inputs, targets = inputs.to(device), targets.squeeze().to(device)
                inputs, targets = inputs.to(device), targets.to(device)

            outputs = self.model(inputs)
            #print("inputs.shape: ", inputs.shape)
            #print("outputs.shape: ", outputs.shape)
            #print("targets.shape: ", targets.shape)
            loss_value = self.loss_fn(outputs, targets)

            losses.append(loss_value.item())
            self.tracker.step(loss_value.item())

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            res = self.metric_fn(preds, targets)

            dice_scores.append(res)

        res = torch.stack(dice_scores).mean().item()
        self.tracker._summary["metric"] = res
        avg_loss = self.tracker.summary()
        return avg_loss

    @torch.enable_grad()
    def update(self, data: DataLoader, use_torchio=False, tag: str = None) -> list:
        self.model.train()
        device = next(self.model.parameters()).device

        self.tracker.start(tag, num_batches=len(data))

        losses = []
        #for batch in tqdm(data, 'Updating'):
        for batch in data:
            if use_torchio:
                inputs = torch.cat([
                    batch['t1n'][tio.DATA],
                    batch['t1c'][tio.DATA],
                    batch['t2w'][tio.DATA],
                    batch['t2f'][tio.DATA],
                ], dim=1).float().to(device)

                targets = batch['seg'][tio.DATA].float().squeeze(1).to(device)
            else:
                inputs, targets = batch
                #inputs, targets = inputs.to(device), targets.squeeze().to(device)
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            loss_value = self.loss_fn(outputs, targets)
            losses.append(loss_value.item())

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            self.tracker.step(loss_value.item())
            self.tracker.count_update()

        avg_loss = self.tracker.summary()
        return avg_loss

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            self.tracker.start_epoch()

            avg_train_loss = self.update(train_loader, use_torchio=self.use_torchio, tag="train")
            #avg_train_loss = torch.stack(train_losses).mean().item()
            self.train_loss_history.append(avg_train_loss)

            avg_val_metric = self.evaluate(val_loader, use_torchio=self.use_torchio, tag="valid")

            #val_metrics = evaluate(self.model, val_loader, self.metric_fn, use_torchio=self.use_torchio)
            #avg_val_metric = torch.stack(val_metrics).mean().item()
            self.val_metric_history.append(avg_val_metric)

            #print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Metric: {avg_val_metric:.4f}")
            self.tracker.end_epoch()

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

def load_trained_model(run_dir, model_class, device='cuda', **kwargs):
    """
    Load a trained model from a checkpoint in the specified run directory.

    Parameters
    ----------
    run_dir : str or Path
        Path to the run folder (containing epoch_XYZ.pth files).
    model_class : nn.Module
        The class of the model (e.g.).
    device : str
        Device to map the model to ('cuda' or 'cpu').

    Returns
    -------
    model : nn.Module
        Model with loaded weights.
    """
    run_dir = Path(run_dir)
    checkpoint_files = sorted(run_dir.glob("epoch_*.pth"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {run_dir}")

    checkpoint_path = checkpoint_files[-1]  # load latest
    print(f"Loading checkpoint: {checkpoint_path.name}")

    # Instantiate model
    model = model_class(**kwargs).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model
