from tqdm import tqdm
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from utils.logging_utils import Logger
from ml.trainer import Trainer
from tqdm import tqdm


class ProgressBar(Logger):
    def __init__(self, path: str = None):
        super().__init__(path)
        self.bar = None
        self.losses = {}

    def on_epoch_start(self, epoch: int, **kwargs):
        self.losses.clear()

    def on_iter_start(self, epoch: int, update: int, tag: str, **kwargs):
        num_steps = kwargs['num_steps_expected']
        self.bar = tqdm(total=num_steps, dynamic_ncols=True, desc=f"Epoch: {epoch} [{tag}]")
        self.losses[tag] = []

    def on_iter_update(self, epoch: int, update: int, tag: str, **kwargs):
        loss = kwargs.get("loss")
        self.losses[tag].append(loss)
        self.bar.set_postfix({f"{tag}_loss": f"{sum(self.losses[tag]) / len(self.losses[tag]):.4f}"})
        self.bar.update(1)

    def on_iter_end(self, epoch: int, update: int, tag: str, avg_loss: float, **kwargs):
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def on_epoch_end(self, epoch: int, **kwargs):
        print("\nEpoch Summary:")
        for tag, avg_loss in kwargs.items():
            print(f"  {tag}-loss: {avg_loss:.4f}")


class TensorBoard(Logger):
    """Log loss values to tensorboard."""

    def __init__(self, path: Path = None, every: int = 1):
        super().__init__(path)
        self.every = every

        self.writer = SummaryWriter(log_dir=str(self.path.resolve()))

    def on_iter_update(self, epoch: int, update: int, tag: str, **kwargs):
        loss = kwargs.get("loss")

        if loss is not None and (update % self.every == 0):
            self.writer.add_scalar(f"{tag}/loss", loss, update)

    def on_iter_end(self, epoch: int, update: int, tag: str, **kwargs):
        avg_loss = kwargs.get("avg_loss")
        if avg_loss is not None:
            self.writer.add_scalar(f"{tag}/avg_loss", avg_loss, epoch)

    def on_epoch_end(self, epoch: int, **kwargs):
        # Not 100% sure if this is necessary but it is in the tensorboard documentation.
        # Call this method to make sure that all pending events have been written to disk.
        self.writer.flush()


class Backup(Logger):
    DEFAULT_FILE = "backup.pth"

    def __init__(self, path: Path = None, every: int = 1):
        super().__init__(path)
        self.trainer = None
        self.every = every

        if self.path.is_dir() or not self.path.suffix:
            self.path = self.path / self.DEFAULT_FILE

        self.path.parent.mkdir(exist_ok=True, parents=True)

    def attach_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_end(self, epoch: int, **kwargs):
        if self.trainer is None:
            warnings.warn("No trainer attached to Backup logger")
            return

        if epoch % self.every == 0:
            # Ensure path is treated as directory
            save_dir = self.path.parent if self.path.suffix else self.path
            save_dir.mkdir(exist_ok=True, parents=True)

            save_path = save_dir / f"epoch_{epoch:03d}.pth"
            torch.save(self.trainer.state_dict(), save_path)
