"""
Training utilities for MBG model.

Includes:
- Google Drive mounting and directory setup
- CheckpointManager for saving/loading training state
- EarlyStopping callback
- MetricLogger for CSV logging to Drive
"""

import os
import json
import csv
import time
import glob
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn


# Default Drive paths
DEFAULT_DRIVE_ROOT = '/content/drive/MyDrive/MBG_Project'
DEFAULT_CHECKPOINT_DIR = os.path.join(DEFAULT_DRIVE_ROOT, 'checkpoints')
DEFAULT_RESULTS_DIR = os.path.join(DEFAULT_DRIVE_ROOT, 'results')
DEFAULT_LOG_DIR = os.path.join(DEFAULT_DRIVE_ROOT, 'logs')


def mount_google_drive(
    mount_point: str = '/content/drive',
    project_root: str = DEFAULT_DRIVE_ROOT,
) -> Dict[str, str]:
    """Mount Google Drive and create project directories.

    Args:
        mount_point: Where to mount Google Drive.
        project_root: Root directory for the MBG project on Drive.

    Returns:
        Dictionary with paths to checkpoint, results, and log directories.
    """
    try:
        from google.colab import drive
        drive.mount(mount_point)
    except ImportError:
        print("[INFO] Not running in Google Colab. Skipping Drive mount.")
        print(f"[INFO] Using local directory: {project_root}")

    dirs = {
        'root': project_root,
        'checkpoints': os.path.join(project_root, 'checkpoints'),
        'results': os.path.join(project_root, 'results'),
        'logs': os.path.join(project_root, 'logs'),
    }

    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)

    print(f"Project root: {dirs['root']}")
    print(f"Checkpoints:  {dirs['checkpoints']}")
    print(f"Results:      {dirs['results']}")
    print(f"Logs:         {dirs['logs']}")

    return dirs


class CheckpointManager:
    """Manages model checkpoints with auto-save and top-K retention.

    Features:
    - Save full training state (model, optimizer, scheduler, scaler, RNG)
    - Auto-save every N epochs
    - Keep only top-K checkpoints per fold to save Drive space
    - Resume from latest checkpoint
    - Save best model based on a metric
    """

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        save_every_n_epochs: int = 10,
        keep_top_k: int = 3,
        metric_name: str = 'balanced_accuracy',
        metric_mode: str = 'max',  # 'max' or 'min'
    ):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Base directory for checkpoints.
            save_every_n_epochs: Auto-save interval.
            keep_top_k: Number of top checkpoints to keep per fold.
            metric_name: Name of the metric to track for best model.
            metric_mode: 'max' if higher is better, 'min' if lower is better.
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_top_k = keep_top_k
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _get_fold_dir(self, fold: int) -> str:
        """Get or create directory for a specific fold."""
        fold_dir = os.path.join(self.checkpoint_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        return fold_dir

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: int = 0,
        fold: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metrics_history: Optional[List[Dict]] = None,
        config: Optional[Dict] = None,
        path: Optional[str] = None,
    ) -> str:
        """Save a full training checkpoint.

        Args:
            model: The model to save.
            optimizer: Optimizer state.
            scheduler: LR scheduler (optional).
            scaler: GradScaler for AMP (optional).
            epoch: Current epoch number.
            fold: Current fold number.
            metrics: Current epoch metrics.
            metrics_history: Full metrics history.
            config: Training configuration dict.
            path: Custom save path (overrides default).

        Returns:
            Path to saved checkpoint.
        """
        if path is None:
            fold_dir = self._get_fold_dir(fold)
            path = os.path.join(fold_dir, f'checkpoint_epoch_{epoch}.pt')

        checkpoint = {
            'epoch': epoch,
            'fold': fold,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_metric': metrics.get(self.metric_name) if metrics else None,
            'metrics': metrics,
            'metrics_history': metrics_history or [],
            'config': config or {},
            'random_state': {
                'torch': torch.random.get_rng_state(),
                'numpy': np.random.get_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            'timestamp': time.time(),
        }

        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved: {path}")

        # Clean up old checkpoints (keep top-K)
        self._cleanup_old_checkpoints(fold)

        return path

    def load_checkpoint(
        self,
        path: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: str = 'cpu',
    ) -> Dict[str, Any]:
        """Load a checkpoint and optionally restore states.

        Args:
            path: Path to the checkpoint file.
            model: Model to load state into (optional).
            optimizer: Optimizer to load state into (optional).
            scheduler: Scheduler to load state into (optional).
            scaler: GradScaler to load state into (optional).
            device: Device to map tensors to.

        Returns:
            Checkpoint dictionary with all saved state.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore RNG states
        rng_state = checkpoint.get('random_state', {})
        if rng_state.get('torch') is not None:
            torch.random.set_rng_state(rng_state['torch'])
        if rng_state.get('numpy') is not None:
            np.random.set_state(rng_state['numpy'])
        if rng_state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['cuda'])

        print(f"[Checkpoint] Loaded: {path} (epoch {checkpoint['epoch']}, fold {checkpoint['fold']})")
        return checkpoint

    def save_best_model(
        self,
        model: nn.Module,
        metrics: Dict[str, float],
        fold: int = 0,
        path: Optional[str] = None,
    ) -> str:
        """Save the best model based on tracked metric.

        Args:
            model: Model to save.
            metrics: Current metrics dict.
            fold: Current fold.
            path: Custom save path.

        Returns:
            Path to the saved best model.
        """
        if path is None:
            fold_dir = self._get_fold_dir(fold)
            path = os.path.join(fold_dir, 'best_model.pt')

        best_state = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'fold': fold,
            'metric_name': self.metric_name,
            'metric_value': metrics.get(self.metric_name),
            'timestamp': time.time(),
        }

        torch.save(best_state, path)
        print(f"[Checkpoint] Best model saved: {path} "
              f"({self.metric_name}={metrics.get(self.metric_name, 'N/A'):.4f})")
        return path

    def get_latest_checkpoint(self, fold: Optional[int] = None) -> Optional[str]:
        """Find the latest checkpoint for a fold or across all folds.

        Args:
            fold: Specific fold to search. If None, searches all folds.

        Returns:
            Path to the latest checkpoint, or None if not found.
        """
        if fold is not None:
            fold_dir = self._get_fold_dir(fold)
            pattern = os.path.join(fold_dir, 'checkpoint_epoch_*.pt')
        else:
            pattern = os.path.join(self.checkpoint_dir, 'fold_*', 'checkpoint_epoch_*.pt')

        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None

        # Sort by modification time
        checkpoints.sort(key=os.path.getmtime)
        return checkpoints[-1]

    def should_save(self, epoch: int) -> bool:
        """Check if checkpoint should be saved at this epoch."""
        return (epoch + 1) % self.save_every_n_epochs == 0

    def _cleanup_old_checkpoints(self, fold: int):
        """Keep only top-K checkpoints based on epoch (most recent)."""
        fold_dir = self._get_fold_dir(fold)
        pattern = os.path.join(fold_dir, 'checkpoint_epoch_*.pt')
        checkpoints = glob.glob(pattern)

        if len(checkpoints) <= self.keep_top_k:
            return

        # Sort by modification time, remove oldest
        checkpoints.sort(key=os.path.getmtime)
        to_remove = checkpoints[:-self.keep_top_k]

        for ckpt_path in to_remove:
            os.remove(ckpt_path)
            print(f"[Checkpoint] Removed old: {os.path.basename(ckpt_path)}")


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs (patience).
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'max',
        verbose: bool = True,
    ):
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'max' if higher metric is better, 'min' if lower is better.
            verbose: Whether to print messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value: float) -> bool:
        """Check if training should stop.

        Args:
            metric_value: Current epoch's metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric_value
            return False

        if self.mode == 'max':
            improved = metric_value > (self.best_score + self.min_delta)
        else:
            improved = metric_value < (self.best_score - self.min_delta)

        if improved:
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")

        self.early_stop = self.counter >= self.patience
        return self.early_stop

    def reset(self):
        """Reset early stopping state (e.g., for a new fold)."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class MetricLogger:
    """Logs training metrics to CSV files on Drive.

    Creates per-fold CSV files with epoch-level metrics.
    """

    def __init__(
        self,
        log_dir: str = DEFAULT_LOG_DIR,
        fold: int = 0,
    ):
        """Initialize MetricLogger.

        Args:
            log_dir: Directory to save log files.
            fold: Current fold number.
        """
        self.log_dir = log_dir
        self.fold = fold
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f'training_log_fold_{fold}.csv')
        self.headers_written = os.path.exists(self.log_file)
        self.history: List[Dict[str, Any]] = []

    def log(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch.
            metrics: Dictionary of metric names to values.
        """
        row = {'epoch': epoch, 'timestamp': time.time(), **metrics}
        self.history.append(row)

        # Write to CSV
        fieldnames = list(row.keys())

        if not self.headers_written:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self.headers_written = True
        else:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all logged metrics."""
        return self.history

    def get_best_epoch(self, metric_name: str, mode: str = 'max') -> Dict[str, Any]:
        """Find the epoch with the best metric value.

        Args:
            metric_name: Name of metric to optimize.
            mode: 'max' or 'min'.

        Returns:
            Dictionary with the best epoch's metrics.
        """
        if not self.history:
            return {}

        if mode == 'max':
            best = max(self.history, key=lambda x: x.get(metric_name, float('-inf')))
        else:
            best = min(self.history, key=lambda x: x.get(metric_name, float('inf')))

        return best


if __name__ == '__main__':
    # Quick test of utilities
    print("Testing CheckpointManager...")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            checkpoint_dir=tmpdir,
            save_every_n_epochs=2,
            keep_top_k=2,
        )

        # Create a simple model
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Save checkpoints
        for epoch in range(6):
            metrics = {'balanced_accuracy': 0.5 + epoch * 0.05}
            if manager.should_save(epoch):
                manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    fold=0,
                    metrics=metrics,
                )

        # Find latest
        latest = manager.get_latest_checkpoint(fold=0)
        print(f"Latest checkpoint: {latest}")
        assert latest is not None

        # Load checkpoint
        ckpt = manager.load_checkpoint(latest, model=model, optimizer=optimizer)
        print(f"Loaded epoch: {ckpt['epoch']}")

    print("\nTesting EarlyStopping...")
    es = EarlyStopping(patience=3, mode='max', verbose=True)
    scores = [0.5, 0.6, 0.65, 0.65, 0.64, 0.64, 0.63]
    for i, score in enumerate(scores):
        stop = es(score)
        if stop:
            print(f"Early stopping triggered at step {i}")
            break

    print("\nTesting MetricLogger...")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricLogger(log_dir=tmpdir, fold=0)
        for epoch in range(5):
            logger.log(epoch, {
                'train_loss': 1.0 - epoch * 0.1,
                'val_accuracy': 0.5 + epoch * 0.05,
            })
        best = logger.get_best_epoch('val_accuracy', mode='max')
        print(f"Best epoch: {best}")

    print("\nAll training_utils tests passed!")
