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
def compute_braTS_dice(pred, target, num_classes=5):
    """
    Compute Dice scores for Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).

    Args:
        pred: Tensor of shape (B, D, H, W), predicted class labels (ints)
        target: Tensor of shape (B, D, H, W), ground truth class labels (ints)
        num_classes: total number of classes including background

    Returns:
        dict with Dice scores: {'WT': ..., 'TC': ..., 'ET': ...}
    """
    eps = 1e-5
    dice_scores = {}

    # Binarized masks
    pred_wt = (pred > 0)  # WT: whole tumor
    target_wt = (target > 0)

    pred_tc = (pred == 1) | (pred == 3) | (pred == 4)  # Tumor Core: NCR, ET, RC => labels 1, 3, 4
    target_tc = (pred == 1) | (pred == 3) | (pred == 4)

    pred_et = (pred == 3)  # ET: enhancing only
    target_et = (target == 3)

    def dice(x, y):
        return (2. * (x & y).sum().float()) / (x.sum() + y.sum() + eps)

    dice_scores['WT'] = dice(pred_wt, target_wt)
    dice_scores['TC'] = dice(pred_tc, target_tc)
    dice_scores['ET'] = dice(pred_et, target_et)

    return dice_scores


class Trainer:
    def __init__(self, model, optimizer, loss_fn, metric_fn, device='cuda', use_torchio=False, tracker: Tracker = None,):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.device = device
        self.train_loss_history = []
        self.val_metric_history = []
        self.val_loss_history = []
        self.use_torchio = use_torchio


        self.wt_scores = []
        self.tc_scores = []
        self.et_scores = []

        if tracker is None:
            tracker = Tracker()
        self.tracker = tracker

    def state_dict(self):
        """ Current state of learning. """
        return {
            "model": self.model.state_dict(),
            #"objective": self.loss_fn.state_dict(),
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
        wt = []
        tc = []
        et = []
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

            dice = compute_braTS_dice(preds, targets)
            wt.append(dice["WT"].item())
            tc.append(dice["TC"].item())
            et.append(dice["ET"].item())

            dice_scores.append(res)

        res = torch.stack(dice_scores).mean().item()
        self.tracker._summary["mean-dice-score"] = res
        self.val_metric_history.append(res)

        self.wt_scores.append(torch.tensor(wt).mean().item())
        self.tc_scores.append(torch.tensor(tc).mean().item())
        self.et_scores.append(torch.tensor(et).mean().item())

        self.tracker._summary["dice-score-wt"] = torch.tensor(wt).mean().item()
        self.tracker._summary["dice-score-tc"] = torch.tensor(tc).mean().item()
        self.tracker._summary["dice-score-et"] = torch.tensor(et).mean().item()

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

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            losses.append(loss_value.item())
            self.tracker.step(loss_value.item())
            self.tracker.count_update()

        avg_loss = self.tracker.summary()
        return avg_loss

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            self.tracker.start_epoch()

            avg_train_loss = self.update(train_loader, use_torchio=self.use_torchio, tag="train-loss")
            #avg_train_loss = torch.stack(train_losses).mean().item()
            self.train_loss_history.append(avg_train_loss)

            avg_val_loss = self.evaluate(val_loader, use_torchio=self.use_torchio, tag="valid-loss")

            #val_metrics = evaluate(self.model, val_loader, self.metric_fn, use_torchio=self.use_torchio)
            #avg_val_metric = torch.stack(val_metrics).mean().item()
            self.val_loss_history.append(avg_val_loss)

            #print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Metric: {avg_val_metric:.4f}")
            self.tracker.end_epoch()

        self.plot_curves()

    def plot_curves(self):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.plot(self.train_loss_history, label='Train Loss')#, marker='o')
        plt.plot(self.val_loss_history, label='Valid Loss')#, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.val_metric_history, label='Val Metric', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Mean Dice Score per Class')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.wt_scores, label='WT Dice')
        plt.plot(self.tc_scores, label='TC Dice')
        plt.plot(self.et_scores, label='ET Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Per-epoch Dice Scores')
        plt.ylim(0, 1)
        plt.legend()
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

class Att3DUNET_Trainer(Trainer):
    @torch.no_grad()
    def evaluate(self, data: DataLoader, use_torchio=False, tag: str = None) -> list:
        self.model.eval()
        device = next(self.model.parameters()).device

        self.tracker.start(tag, num_batches=len(data))

        losses = []
        dice_scores = []
        wt = []
        tc = []
        et = []
        #for batch in tqdm(data, 'Evaluating'):
        for batch in data:

            inputs, targets = batch
            #inputs, targets = inputs.to(device), targets.squeeze().to(device)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = self.model(inputs)
            #print("inputs.shape: ", inputs.shape)
            #print("outputs.shape: ", outputs.shape)
            #print("targets.shape: ", targets.shape)
            loss_value = self.loss_fn(outputs, targets)

            losses.append(loss_value.item())
            self.tracker.step(loss_value.item())

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            res = self.metric_fn(preds, targets)

            dice = compute_braTS_dice(preds, targets)
            wt.append(dice["WT"].item())
            tc.append(dice["TC"].item())
            et.append(dice["ET"].item())

            dice_scores.append(res)

        res = torch.stack(dice_scores).mean().item()
        self.tracker._summary["mean-dice-score"] = res

        self.wt_scores.append(torch.tensor(wt).mean().item())
        self.tc_scores.append(torch.tensor(tc).mean().item())
        self.et_scores.append(torch.tensor(et).mean().item())

        self.tracker._summary["dice-score-wt"] = torch.tensor(wt).mean().item()
        self.tracker._summary["dice-score-tc"] = torch.tensor(tc).mean().item()
        self.tracker._summary["dice-score-et"] = torch.tensor(et).mean().item()

        self.val_metric_history.append(res)
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
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs,_ = self.model(inputs)
            loss_value = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            losses.append(loss_value.item())
            self.tracker.step(loss_value.item())
            self.tracker.count_update()

        avg_loss = self.tracker.summary()
        return avg_loss

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            self.tracker.start_epoch()

            avg_train_loss = self.update(train_loader, use_torchio=self.use_torchio, tag="train-loss")
            #avg_train_loss = torch.stack(train_losses).mean().item()
            self.train_loss_history.append(avg_train_loss)

            avg_val_loss = self.evaluate(val_loader, use_torchio=self.use_torchio, tag="valid-loss")

            #val_metrics = evaluate(self.model, val_loader, self.metric_fn, use_torchio=self.use_torchio)
            #avg_val_metric = torch.stack(val_metrics).mean().item()
            self.val_loss_history.append(avg_val_loss)

            #print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Metric: {avg_val_metric:.4f}")
            self.tracker.end_epoch()

        self.plot_curves()