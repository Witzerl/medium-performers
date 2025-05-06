import argparse
import os
import random
import sys
from subprocess import call

import numpy as np
import torchio as tio
from torchio.data import SubjectsLoader

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet, AttentionUnet

from utils.loading_utils import *
from visualization.visualization import *
from ml.dataset_ import BrainMetDatasetPreloaded, BrainMetDataset, GridSamplerWrapper
from ml.trainer import Trainer

SEED = 42

# NOTE: './training_helper/' is a dummy folder that should hold 5 samples and is used for quick testing to avoid loading the whole dataset
TRAIN_ROOT_DIR = './training_helper/'  # './MICCAI-LH-BraTS2025-MET-Challenge-Training/'
VAL_ROOT_DIR   = './training_helper/'  # './MICCAI-LH-BraTS2025-MET-Challenge-Validation/'


def fix_seed(seed=SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def print_information() -> None:
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')

    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())


# TODO: Also load the validation dataset
def load_data(device: torch.cuda.device, batch_size: int, num_workers: int = 0):
    case_contents = inspect_case_contents(TRAIN_ROOT_DIR)
    filtered_contents = filter_suspicious_cases(TRAIN_ROOT_DIR, case_contents)  # filter sus stuff

    train_transform_patch = tio.Compose([
        tio.ZNormalization(include=('t1n', 't1c', 't2w', 't2f')),
        tio.Resample((1, 1, 1)),
        tio.EnsureShapeMultiple(16),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        tio.RandomElasticDeformation(),
        # tio.RandomNoise(),
        # tio.RandomFlip(axes=('LR',)),
    ])

    # TODO: Speed up this pre-loading because it takes a lot of time for the whole dataset
    train_dataset = BrainMetDatasetPreloaded(
        root_dir=TRAIN_ROOT_DIR,
        case_contents=filtered_contents,
        with_segmentation=True,
        transforms=train_transform_patch,
        to_device=device
    )

    subjects_dataset = tio.SubjectsDataset(train_dataset)
    sampler = GridSamplerWrapper(patch_size=(128, 128, 128), patch_overlap=(32, 32, 32))
    patches_queue = tio.Queue(
        subjects_dataset=subjects_dataset,
        max_length=500,
        samples_per_volume=100,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    train_loader = SubjectsLoader(patches_queue, batch_size=batch_size, num_workers=num_workers)
    return train_loader


def define_model(model: str, device: torch.cuda.device) -> nn.Module:
    match model:
        case 'unet':
            return UNet(
                spatial_dims=3,
                in_channels=4,
                out_channels=5,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2
            ).to(device)

        case 'attention-unet':  # Note: This model configuration is very VRAM-hungry
            return AttentionUnet(
                spatial_dims=3,
                in_channels=4,
                out_channels=5,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
            ).to(device)

        case _:
            raise RuntimeError(f'Model "{model}" is not supported')



def dice_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float =1e-5) -> torch.Tensor:
    # Assumes pred is softmax, target is one-hot
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = 2. * intersection / (union + epsilon)
    return 1 - dice.mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, target.to(torch.long))
    probs = torch.softmax(logits, dim=1)
    one_hot_target = torch.nn.functional.one_hot(target.to(torch.long), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    dice = dice_loss(probs, one_hot_target)
    return ce + dice


def dice_score(pred: torch.Tensor, target: torch.Tensor, epsilon: float =1e-5) -> torch.Tensor:
    """Computes average Dice score per batch"""

    # this is a voxel-wise implementation of dice score... however challenge ranks lesion wise
    probs = torch.softmax(pred, dim=1)  # (B, C, H, W, D)
    preds = torch.argmax(probs, dim=1)  # (B, H, W, D)
    targets = target.long()  # (B, H, W, D)

    dice_scores = []
    num_classes = pred.shape[1]

    for c in range(1, num_classes):  # skip background class 0
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2, 3))  # sum over HWD
        union = pred_c.sum(dim=(1, 2, 3)) + target_c.sum(dim=(1, 2, 3))

        dice = (2 * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice)

    mean_dice = torch.stack(dice_scores).mean()  # average over all foreground classes
    return mean_dice


def main(args) -> None:
    fix_seed(SEED)
    print_information()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    train_dataloader = load_data(device, batch_size=args.batch_size, num_workers=args.num_workers)
    model = define_model(args.model, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(model, optimizer, loss_fn, dice_score, device=device)
    trainer.train(train_loader=train_dataloader, val_loader=train_dataloader, epochs=args.epochs)

    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain Metastases Segmentation - Model Training')

    # Data loading params
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')

    # Model params
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attention-unet'], help='Which model to use')

    # Training params
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')

    command_line_args = parser.parse_args()
    main(command_line_args)