import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.notebook import trange
import torchio as tio
from utils.logging_utils import Tracker
from pathlib import Path
from surface_distance import compute_surface_distances, compute_average_surface_distance
import numpy as np

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
    target_tc = (target == 1) | (target == 3) | (target == 4)

    pred_et = (pred == 3)  # ET: enhancing only
    target_et = (target == 3)

    def dice(x, y):
        x = x.float()
        y = y.float()
        intersection = (x * y).sum()
        return (2. * intersection + eps) / (x.sum() + y.sum() + eps)

    dice_scores['WT'] = dice(pred_wt, target_wt)
    dice_scores['TC'] = dice(pred_tc, target_tc)
    dice_scores['ET'] = dice(pred_et, target_et)

    return dice_scores

import torch
import numpy as np
from surface_distance import compute_surface_distances

from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

def compute_all_metrics(pred, target, spacing=(1.0, 1.0, 1.0), tolerance_mm=1.0):
    """
    Computes all required lesion-wise metrics for WT, TC, ET regions:
    Dice, NSD, Sensitivity, Specificity, Precision.

    Args:
        pred: Tensor of shape (B, H, W, D), predicted labels
        target: Tensor of shape (B, H, W, D), ground truth labels
        spacing: Tuple of physical voxel spacing (e.g., (1.0, 1.0, 1.0))
        tolerance_mm: Tolerance threshold in mm for NSD

    Returns:
        dict[region][metric] = value
    """
    eps = 1e-5
    def sensitivity(p, t):
        tp = (p & t).sum(dim=(1,2,3))
        fn = (~p & t).sum(dim=(1,2,3))
        return (tp / (tp + fn + eps)).mean().item()

    def specificity(p, t):
        tn = (~p & ~t).sum(dim=(1,2,3))
        fp = (p & ~t).sum(dim=(1,2,3))
        return (tn / (tn + fp + eps)).mean().item()

    def precision(p, t):
        tp = (p & t).sum(dim=(1,2,3))
        fp = (p & ~t).sum(dim=(1,2,3))
        return (tp / (tp + fp + eps)).mean().item()

    def nsd_batch(pred_bin, target_bin, spacing, tolerance):
        """NSD averaged over batch."""
        batch_nsd = []
        for i in range(pred_bin.shape[0]):
            pred_np = pred_bin[i].cpu().numpy().astype(np.bool_)
            target_np = target_bin[i].cpu().numpy().astype(np.bool_)

            if pred_np.sum() == 0 and target_np.sum() == 0:
                batch_nsd.append(1.0)
                continue
            if pred_np.sum() == 0 or target_np.sum() == 0:
                batch_nsd.append(0.0)
                continue

            sd = compute_surface_distances(target_np, pred_np, spacing)
            score = compute_surface_dice_at_tolerance(sd, tolerance)
            batch_nsd.append(score)
        return sum(batch_nsd) / len(batch_nsd)

    def get_mask(x, region):
        if region == "WT":
            return (x > 0)
        elif region == "TC":
            return (x == 1) | (x == 3) | (x == 4)
        elif region == "ET":
            return (x == 3)
        else:
            raise ValueError(f"Unknown region: {region}")

    results = {}
    for region in ["WT", "TC", "ET"]:
        pred_mask = get_mask(pred, region)
        target_mask = get_mask(target, region)

        results[region] = {
            "NSD": nsd_batch(pred_mask, target_mask, spacing, tolerance_mm),
            "Sensitivity": sensitivity(pred_mask, target_mask),
            "Specificity": specificity(pred_mask, target_mask),
            "Precision": precision(pred_mask, target_mask),
        }

    return results



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

        wt_dice, tc_dice, et_dice = [], [], []

        wt_nsd, wt_sens, wt_spec, wt_prec = [], [], [], []
        tc_nsd, tc_sens, tc_spec, tc_prec = [], [], [], []
        et_nsd, et_sens, et_spec, et_prec = [], [], [], []

        for batch in data:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = self.model(inputs)

            loss_value = self.loss_fn(outputs, targets)
            losses.append(loss_value.item())
            self.tracker.step(loss_value.item())

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            res = self.metric_fn(preds, targets)
            dice_scores.append(res)

            dice = compute_braTS_dice(preds, targets)
            metrics = compute_all_metrics(preds, targets)

            wt_dice.append(dice["WT"].item())
            tc_dice.append(dice["TC"].item())
            et_dice.append(dice["ET"].item())

            # NSD, Sensitivity, Specificity, Precision
            for region, accs in [("WT", (wt_nsd, wt_sens, wt_spec, wt_prec)),
                                 ("TC", (tc_nsd, tc_sens, tc_spec, tc_prec)),
                                 ("ET", (et_nsd, et_sens, et_spec, et_prec))]:
                accs[0].append(metrics[region]["NSD"])
                accs[1].append(metrics[region]["Sensitivity"])
                accs[2].append(metrics[region]["Specificity"])
                accs[3].append(metrics[region]["Precision"])

        res = torch.stack(dice_scores).mean().item()
        self.tracker._summary["mean-dice-score"] = res
        self.val_metric_history.append(res)

        self.wt_scores.append(torch.tensor(wt_dice).mean().item())
        self.tc_scores.append(torch.tensor(tc_dice).mean().item())
        self.et_scores.append(torch.tensor(et_dice).mean().item())

        # Track average dice
        self.tracker._summary["WT-Dice"] = np.mean(wt_dice)
        self.tracker._summary["TC-Dice"] = np.mean(tc_dice)
        self.tracker._summary["ET-Dice"] = np.mean(et_dice)

        # Track other metrics
        self.tracker._summary["WT-NSD"] = np.mean(wt_nsd)
        self.tracker._summary["WT-Sensitivity"] = np.mean(wt_sens)
        self.tracker._summary["WT-Specificity"] = np.mean(wt_spec)
        self.tracker._summary["WT-Precision"] = np.mean(wt_prec)

        self.tracker._summary["TC-NSD"] = np.mean(tc_nsd)
        self.tracker._summary["TC-Sensitivity"] = np.mean(tc_sens)
        self.tracker._summary["TC-Specificity"] = np.mean(tc_spec)
        self.tracker._summary["TC-Precision"] = np.mean(tc_prec)

        self.tracker._summary["ET-NSD"] = np.mean(et_nsd)
        self.tracker._summary["ET-Sensitivity"] = np.mean(et_sens)
        self.tracker._summary["ET-Specificity"] = np.mean(et_spec)
        self.tracker._summary["ET-Precision"] = np.mean(et_prec)

        avg_loss = self.tracker.summary()
        return avg_loss

    @torch.enable_grad()
    def update(self, data: DataLoader, use_torchio=False, tag: str = None) -> list:
        self.model.train()
        device = next(self.model.parameters()).device

        self.tracker.start(tag, num_batches=len(data))

        losses = []
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
            self.train_loss_history.append(avg_train_loss)

            avg_val_loss = self.evaluate(val_loader, use_torchio=self.use_torchio, tag="valid-loss")
            self.val_loss_history.append(avg_val_loss)

            self.tracker.end_epoch()

        self.plot_curves()