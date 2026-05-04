"""
Training Pipeline — Error Compensation (train_val_loop_comp.py)

Data schema  (6 columns, z-score normalized):
    Col 0  Δe_scale_x   (a2 - a1)   ← input
    Col 1  Δe_scale_y                ← input
    Col 2  Δe_scale_z                ← input
    Col 3  Δvff_x                    ← target
    Col 4  Δvff_y                    ← target
    Col 5  Δvff_z                    ← target

axis_selection='x' → input col 0, target col 3
axis_selection='y' → input col 1, target col 4
axis_selection='z' → input col 2, target col 5

reverse_time=True  → flip all signals along the time axis at dataset level
                     so the GRU sees the trajectory backwards.
                     Plots always show forward time (unflipped) for readability.

No motion / position filtering — the delta-e signal does not carry velocity
information and filtering is not needed for this dataset.

Everything else (model architecture, optimizer, early stopping, MLflow
logging, Ray/Optuna plumbing) is identical to train_val_loop_v10.py so
run_experiment.py and hyperparam_opt.py work unchanged except for the
train_fn swap.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
import time
import os
import datetime
import mlflow
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# COLUMN LAYOUT
# ============================================================================

COMP_INPUT_COLS  = [0, 1, 2]   # Δe_scale_x/y/z
COMP_TARGET_COLS = [3, 4, 5]   # Δvff_x/y/z
COMP_AXIS_INPUT  = {'x': 0, 'y': 1, 'z': 2}
COMP_AXIS_TARGET = {'x': 3, 'y': 4, 'z': 5}
COMP_N_COLS      = 6


# ============================================================================
# DATASET
# ============================================================================

class CompDataset(Dataset):
    """
    Dataset for the compensation pipeline.

    Loads a (N, T, 6) float32 array from disk using mmap_mode='r' so the
    11 GB file is paged by the OS rather than copied into RAM up-front.

    No filtering is applied — every window is retained.

    Parameters
    ----------
    data_file_path : str
    target_timesteps : int
        Window length the GRU will see. Must divide T exactly.
    axis_selection : 'x' | 'y' | 'z'
        Which axis pair (input col, target col) to extract.
    downsample_factor : int
        Temporal downsampling applied before windowing.
    reverse_time : bool
        If True, flip X and y along the time axis so the GRU processes
        the signal backwards. Plots always show forward time.
    max_samples : int | None
        Cap on the number of windows kept (after reshape). Useful for
        quick debugging runs. None = keep all.
    random_seed : int
        Seed used for max_samples subsampling. Ignored when max_samples
        is None.
    dtype : torch.dtype
    validate_divisibility : bool
    """

    def __init__(
        self,
        data_file_path: str,
        target_timesteps: int,
        axis_selection: str = 'x',
        downsample_factor: int = 1,
        reverse_time: bool = True,
        max_samples: Optional[int] = None,
        random_seed: int = 0,
        dtype: torch.dtype = torch.float32,
        validate_divisibility: bool = True,
    ):
        self.target_timesteps = target_timesteps
        self.axis_selection   = axis_selection.lower()
        self.reverse_time     = reverse_time
        self.dtype            = dtype

        if self.axis_selection not in ('x', 'y', 'z'):
            raise ValueError(
                f"axis_selection must be 'x', 'y', or 'z', got '{axis_selection}'"
            )

        # ── Load (memory-mapped so the OS pages on demand) ──────────────────
        print(f"Loading compensation data from {data_file_path}...")
        data = np.load(data_file_path, mmap_mode='r')
        print(f"  Shape: {data.shape}  dtype: {data.dtype}")

        if data.shape[2] != COMP_N_COLS:
            raise ValueError(
                f"Expected {COMP_N_COLS} columns, got {data.shape[2]}. "
                f"Check that this is the compensation dataset."
            )

        original_timesteps = data.shape[1]

        # ── Downsample ───────────────────────────────────────────────────────
        if downsample_factor > 1:
            print(f"  Downsampling by {downsample_factor}x ...")
            data = data[:, ::downsample_factor, :]
            print(f"  Timesteps: {original_timesteps} → {data.shape[1]}")

        downsampled_timesteps = data.shape[1]

        if validate_divisibility and downsampled_timesteps % target_timesteps != 0:
            raise ValueError(
                f"target_timesteps ({target_timesteps}) must divide "
                f"{downsampled_timesteps} exactly. "
                f"Remainder: {downsampled_timesteps % target_timesteps}"
            )

        # ── Reshape into windows ─────────────────────────────────────────────
        n_windows = downsampled_timesteps // target_timesteps
        m         = data.shape[0]
        print(f"  Windowing: {m} signals × {n_windows} windows = {m * n_windows} total")

        # Fancy-indexing on a mmap returns a copy — safe to reshape.
        # We copy explicitly to detach from the mmap handle before slicing cols.
        data_reshaped = np.array(
            data.reshape(m * n_windows, target_timesteps, COMP_N_COLS),
            dtype=np.float32,
        )

        # ── Optional cap ─────────────────────────────────────────────────────
        if max_samples is not None and len(data_reshaped) > max_samples:
            rng  = np.random.default_rng(random_seed if random_seed != 0 else 42)
            idxs = rng.choice(len(data_reshaped), size=max_samples, replace=False)
            data_reshaped = data_reshaped[idxs]
            print(f"  Capped to {max_samples} samples (max_samples)")

        # ── Column extraction ────────────────────────────────────────────────
        in_col  = COMP_AXIS_INPUT[self.axis_selection]
        tgt_col = COMP_AXIS_TARGET[self.axis_selection]

        # Shape: (N, T, 1) — keep trailing dim so model sees (batch, T, 1)
        X = data_reshaped[:, :, in_col:in_col + 1]    # Δe_scale for this axis
        y = data_reshaped[:, :, tgt_col:tgt_col + 1]  # Δvff for this axis

        # ── Reverse time (GRU sees backward; plots show forward) ─────────────
        if reverse_time:
            # [:, ::-1, :] is a view on a C-contiguous array → copy required
            X = X[:, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
            print("  reverse_time=True: signals flipped along time axis")

        self.X_forward = data_reshaped[:, :, in_col:in_col + 1].copy()   # always forward, for plotting
        self.X = X
        self.y = y

        self.num_input_features  = 1
        self.num_output_features = 1
        self.feature_names       = [f'delta_e_scale_{self.axis_selection}']
        self.target_names        = [f'delta_vff_{self.axis_selection}']

        print(f"\nCompDataset ready:")
        print(f"  Samples          : {len(self.X)}")
        print(f"  Axis             : {self.axis_selection.upper()}")
        print(f"  Input  col {in_col}     : {self.feature_names[0]}")
        print(f"  Target col {tgt_col}     : {self.target_names[0]}")
        print(f"  reverse_time     : {reverse_time}")
        print(f"  X shape          : {self.X.shape}")
        print(f"  y shape          : {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx],         dtype=self.dtype)
        y = torch.tensor(self.y[idx],         dtype=self.dtype)
        X_fwd = torch.tensor(self.X_forward[idx], dtype=self.dtype)
        return X, y, X_fwd   # X_fwd passed through for plotting only


# ============================================================================
# DATASET SPLIT  (same as v10)
# ============================================================================

def split_dataset(
    dataset: Dataset,
    train_fraction: float = 8 / 9,
    random_seed: int = 0,
    verbose: bool = True,
):
    total_size = len(dataset)

    seed_used = (
        np.random.randint(0, 100_000_000) if random_seed == 0 else random_seed
    )
    if verbose:
        print(f"{'Generated' if random_seed == 0 else 'Using'} random seed: {seed_used}")

    torch.manual_seed(seed_used)
    np.random.seed(seed_used)

    train_size = int(train_fraction * total_size)
    val_size   = total_size - train_size
    indices    = torch.randperm(total_size).tolist()

    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset   = Subset(dataset, indices[train_size:])

    if verbose:
        print(f"\nDataset split:")
        print(f"  Total  : {total_size:,}")
        print(f"  Train  : {train_size:,} ({train_size / total_size * 100:.1f}%)")
        print(f"  Val    : {val_size:,}   ({val_size   / total_size * 100:.1f}%)")

    return train_dataset, val_dataset, seed_used


# ============================================================================
# MODEL  (identical to v10 — imported to avoid duplication)
# ============================================================================

from train_val_loop_v10 import (
    SequenceModel,
    initialize_model,
    EarlyStopping,
    train_epoch,
    validate_epoch,
)


# ============================================================================
# JACOBIAN PENALTY  (simple i.i.d. Gaussian — no kinematic chain)
# ============================================================================

def compute_jacobian_frobenius_penalty(
    model:           nn.Module,
    inputs:          torch.Tensor,   # (B, T, F)
    noise_std:       float = 1e-4,
    eps:             float = 1e-8,
) -> torch.Tensor:
    """
    Lipschitz / sensitivity penalty using a pure i.i.d. Gaussian perturbation.

    Unlike the prediction pipeline, the compensation inputs (Δe_scale) are not
    kinematically structured, so there is no reason to build a pos/vel/accel
    perturbation chain. A single Gaussian delta is simpler and architecture-
    agnostic.

    Fully normalised: result is independent of batch size, sequence length,
    input dim, and output dim.
    """
    batch_size, seq_len, num_features = inputs.shape
    device     = inputs.device
    output_dim = model(inputs).detach().shape[-1]

    delta  = torch.randn_like(inputs) * noise_std
    y_base = model(inputs).detach()
    y_pert = model(inputs + delta)

    numerator   = (y_pert - y_base).pow(2).sum(dim=-1)
    with torch.no_grad():
        denominator = delta.pow(2).sum(dim=-1) + eps

    penalty = (numerator / denominator).sum() / (
        batch_size * seq_len * num_features * output_dim
    )
    return penalty


# ============================================================================
# LOSS FUNCTION  (re-used structure from v10, comp-specific Jacobian)
# ============================================================================

class CompositeLossComp(nn.Module):
    """
    Composite loss for the compensation pipeline.

    Components:
        1. Base loss (MSE or MAE) with warmup masking
        2. Jacobian Frobenius penalty (i.i.d. Gaussian, optional)
        3. Hidden-state derivative penalty (optional)
    """

    def __init__(
        self,
        base_loss:                        str   = 'mse',
        jacobian_weight:                  float = 0.0,
        hidden_derivative_weight:         float = 0.0,
        warmup_timesteps:                 int   = 0,
        jacobian_every_n_batches:         int   = 1,
        hidden_derivative_every_n_batches:int   = 1,
    ):
        super().__init__()

        self.base_loss_fn = nn.MSELoss() if base_loss == 'mse' else nn.L1Loss()
        self.jacobian_weight              = jacobian_weight
        self.hidden_derivative_weight     = hidden_derivative_weight
        self.warmup_timesteps             = warmup_timesteps
        self.jacobian_every_n_batches     = jacobian_every_n_batches
        self.hidden_derivative_every_n_batches = hidden_derivative_every_n_batches

        self._cached_jacobian  = 0.0
        self._cached_hidden_d  = 0.0

    def forward(
        self,
        predictions:   torch.Tensor,
        targets:       torch.Tensor,
        model:         nn.Module,
        inputs:        Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        batch_idx:     int = 0,
    ) -> torch.Tensor:

        w = self.warmup_timesteps
        if w > 0:
            predictions   = predictions[:, w:, :]
            targets       = targets[:,     w:, :]
            if inputs        is not None: inputs        = inputs[:,        w:, :]
            if hidden_states is not None: hidden_states = hidden_states[:, w:, :]

        loss = self.base_loss_fn(predictions, targets)

        if self.jacobian_weight > 0 and inputs is not None:
            if batch_idx % self.jacobian_every_n_batches == 0:
                jac = compute_jacobian_frobenius_penalty(model, inputs)
                self._cached_jacobian = jac.item()
            else:
                jac = self._cached_jacobian
            loss = loss + self.jacobian_weight * jac

        if self.hidden_derivative_weight > 0 and hidden_states is not None:
            if batch_idx % self.hidden_derivative_every_n_batches == 0:
                dh   = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
                B, T, H = dh.shape
                hd   = torch.sum(dh ** 2) / (B * T * H)
                hd   = torch.clamp(hd, 0.0, 100.0)
                self._cached_hidden_d = hd
            else:
                hd = self._cached_hidden_d
            loss = loss + self.hidden_derivative_weight * hd

        return loss

    def needs_input_gradients(self) -> bool:
        return self.jacobian_weight > 0


def create_loss_function(
    base_loss:                        str   = 'mse',
    jacobian_weight:                  float = 0.0,
    jacobian_every_n_batches:         int   = 1,
    hidden_derivative_weight:         float = 0.0,
    hidden_derivative_every_n_batches:int   = 1,
    warmup_timesteps:                 int   = 0,
    verbose:                          bool  = True,
) -> CompositeLossComp:
    loss_fn = CompositeLossComp(
        base_loss                        = base_loss,
        jacobian_weight                  = jacobian_weight,
        hidden_derivative_weight         = hidden_derivative_weight,
        warmup_timesteps                 = warmup_timesteps,
        jacobian_every_n_batches         = jacobian_every_n_batches,
        hidden_derivative_every_n_batches= hidden_derivative_every_n_batches,
    )
    if verbose:
        print("\nLoss configuration (comp):")
        print(f"  Base loss          : {base_loss.upper()}")
        print(f"  Jacobian weight    : {jacobian_weight}")
        print(f"  Hidden-deriv weight: {hidden_derivative_weight}")
        print(f"  Warmup timesteps   : {warmup_timesteps}")
    return loss_fn


# ============================================================================
# PLOTTING  (forward-time, comp-specific labels)
# ============================================================================

def plot_predictions(
    model:        nn.Module,
    dataset:      Dataset,
    indices:      list,
    device:       str,
    save_dir:     str  = 'results_comp',
    dataset_name: str  = 'train',
    reverse_time: bool = False,
):
    """
    Plot Δe_scale input (forward time), Δvff target vs prediction, and
    residual error. When reverse_time=True the model saw the signal
    backwards, but we plot forward time throughout.
    """
    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(indices):
            # CompDataset returns (X, y, X_fwd); Subset wraps __getitem__
            item    = dataset[sample_idx]
            X_in    = item[0]          # (T, 1) — possibly reversed
            y_true  = item[1]          # (T, 1) — possibly reversed
            X_fwd   = item[2]          # (T, 1) — always forward

            X_batch = X_in.unsqueeze(0).to(device)
            y_pred  = model(X_batch).squeeze(0).cpu().numpy()   # (T, 1)

            y_true_np = y_true.numpy()
            X_fwd_np  = X_fwd.numpy()[:, 0]

            # If reversed, flip pred and true back to forward time for plotting
            if reverse_time:
                y_pred    = y_pred[::-1,    :]
                y_true_np = y_true_np[::-1, :]

            T        = len(X_fwd_np)
            timesteps = np.arange(T)
            residual  = y_pred[:, 0] - y_true_np[:, 0]

            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            fig.suptitle(
                f'{dataset_name.capitalize()} Sample {sample_idx}'
                f'{"  [reverse_time]" if reverse_time else ""}',
                fontsize=14, fontweight='bold',
            )

            # ── Input ────────────────────────────────────────────────────────
            axes[0].plot(timesteps, X_fwd_np, 'b-', lw=1.2, label='Δe_scale (input)')
            axes[0].set_ylabel('Δe_scale (norm)', fontweight='bold')
            axes[0].set_title('Input signal (forward time)')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc='upper right')

            # ── Target vs Prediction ─────────────────────────────────────────
            axes[1].plot(timesteps, y_true_np[:, 0], 'g-',  lw=1.2, label='Actual Δvff',    alpha=0.8)
            axes[1].plot(timesteps, y_pred[:,    0],  'r--', lw=1.2, label='Predicted Δvff', alpha=0.8)
            axes[1].set_ylabel('Δvff (norm)', fontweight='bold')
            axes[1].set_title('Target vs Prediction (forward time)')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc='upper right')

            # ── Residual ─────────────────────────────────────────────────────
            axes[2].plot(timesteps, residual, color='purple', lw=1.2, label='Residual (pred − actual)', alpha=0.8)
            axes[2].axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
            mae  = np.mean(np.abs(residual))
            rmse = np.sqrt(np.mean(residual ** 2))
            axes[2].set_xlabel(
                f'Timestep   MAE={mae:.6f}  RMSE={rmse:.6f}',
                fontsize=9, fontweight='bold',
            )
            axes[2].set_ylabel('Residual (norm)', fontweight='bold')
            axes[2].set_title('Residual Error')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc='upper right')

            plt.tight_layout()
            fname    = save_path / f'{dataset_name}_sample_{sample_idx:04d}.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            mlflow.log_figure(fig, str(fname))
            plt.close()

            if (plot_idx + 1) % 5 == 0:
                print(f"  Saved {plot_idx + 1}/{len(indices)} plots...")

    print(f"All {len(indices)} plots saved to '{save_dir}/'")


def visualize_predictions(
    model:        nn.Module,
    train_dataset: Dataset,
    val_dataset:   Dataset,
    device:       str,
    num_samples:  int  = 10,
    save_dir:     str  = 'results_comp',
    random_seed:  int  = 42,
    reverse_time: bool = True,
):
    np.random.seed(random_seed)

    train_indices = np.random.choice(
        len(train_dataset), size=min(num_samples, len(train_dataset)), replace=False
    ).tolist()
    val_indices = np.random.choice(
        len(val_dataset), size=min(num_samples, len(val_dataset)), replace=False
    ).tolist()

    for split_name, indices, ds in [
        ('train', train_indices, train_dataset),
        ('val',   val_indices,   val_dataset),
    ]:
        plot_predictions(
            model=model, dataset=ds, indices=indices,
            device=device, save_dir=save_dir,
            dataset_name=split_name, reverse_time=reverse_time,
        )


# ============================================================================
# TRAIN / VAL EPOCHS  (thin wrappers — reuse v10 internals)
# ============================================================================

def _collate_drop_fwd(batch):
    """DataLoader collate that drops the X_fwd tensor (only needed for plots)."""
    X    = torch.stack([b[0] for b in batch])
    y    = torch.stack([b[1] for b in batch])
    X_fwd = torch.stack([b[2] for b in batch])
    return X, y, X_fwd


def _train_epoch_comp(
    model:               nn.Module,
    train_loader:        DataLoader,
    loss_function:       CompositeLossComp,
    optimizer:           torch.optim.Optimizer,
    device:              str,
    gradient_clip_value: Optional[float] = None,
    verbose:             int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, (X_batch, y_batch, _) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)

        hidden_states = None
        if loss_function.hidden_derivative_weight > 0:
            if hasattr(model, 'rnn'):
                hidden_states, _ = model.rnn(X_batch) if isinstance(model.rnn, nn.GRU) \
                    else model.rnn(X_batch)

        loss = loss_function(
            predictions, y_batch, model, X_batch, hidden_states, batch_idx
        )
        loss.backward()

        if gradient_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    avg_loss = total_loss / n_batches
    if verbose >= 1:
        print(f"  Train Loss: {avg_loss:.6f}")
    return avg_loss


def _validate_epoch_comp(
    model:            nn.Module,
    val_loader:       DataLoader,
    warmup_timesteps: int,
    device:           str,
    verbose:          int = 1,
) -> Tuple[float, float]:
    """Validation always uses plain MSE (no regularization terms)."""
    model.eval()
    mse_loss    = nn.MSELoss()
    batch_losses = []

    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(X_batch)
            if warmup_timesteps > 0:
                preds   = preds[:,   warmup_timesteps:, :]
                y_batch = y_batch[:, warmup_timesteps:, :]
            batch_losses.append(mse_loss(preds, y_batch).item())

    avg_loss = float(np.mean(batch_losses))
    std_loss = float(np.std(batch_losses, ddof=1)) if len(batch_losses) > 1 else 0.0

    if verbose >= 1:
        print(f"  Val Loss (MSE): {avg_loss:.6f} ± {std_loss:.6f}")
    return avg_loss, std_loss


# ============================================================================
# MASTER TRAINING FUNCTION
# ============================================================================

def train_model(
    # ── Data ────────────────────────────────────────────────────────────────
    data_file_path:   str,
    target_timesteps: int,
    axis_selection:   str   = 'x',
    downsample_factor:int   = 1,
    reverse_time:     bool  = False,
    max_samples:      Optional[int] = None,
    train_val_split:  float = 8 / 9,
    random_seed:      int   = 0,

    # ── Model ────────────────────────────────────────────────────────────────
    model_type:        str   = 'gru',
    hidden_size:       int   = 64,
    num_layers:        int   = 2,
    dropout:           float = 0.0,
    bidirectional:     bool  = False,
    load_model_path:   Optional[str] = None,
    load_model_strict: bool  = True,

    # ── Loss ─────────────────────────────────────────────────────────────────
    base_loss:                        str   = 'mse',
    l2_weight:                        float = 0.0,
    jacobian_weight:                  float = 0.0,
    jacobian_every_n_batches:         int   = 1,
    hidden_derivative_weight:         float = 0.0,
    hidden_derivative_every_n_batches:int   = 1,

    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs:          int   = 1000,
    batch_size:          int   = 32,
    learning_rate:       float = 0.001,
    optimizer_type:      str   = 'adam',
    optimizer_kwargs:    Optional[Dict[str, Any]] = None,
    gradient_clip_value: Optional[float] = None,
    device:              str   = 'cuda',

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_experiment_id: int = 0,
    tracking_uri:         str = "http://hurco-lambda.hurco.com:8090",

    # ── Early stopping ───────────────────────────────────────────────────────
    early_stopping_patience:  int   = 15,
    early_stopping_min_delta: float = 1e-6,

    # ── Validation ───────────────────────────────────────────────────────────
    validation_frequency: int = 1,

    # ── DataLoader ───────────────────────────────────────────────────────────
    num_workers: int = 0,

    # ── Output ───────────────────────────────────────────────────────────────
    verbose:        int  = 1,
    return_history: bool = True,
    save_model_path: Optional[str] = None,
    save_best:  bool = True,

    # ── Visualization ────────────────────────────────────────────────────────
    generate_plots:  bool = True,
    num_plot_samples:int  = 10,
    plot_save_dir:   str  = 'results_comp',

    # ── Unused v10 kwargs accepted silently so run_experiment stays clean ────
    **_ignored,
) -> Tuple[nn.Module, Optional[Dict[str, list]], int]:
    """
    Master training function for the compensation pipeline.
    Returns (model, history, seed_used) — same contract as v10.
    """

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.start_run(experiment_id=str(mlflow_experiment_id), log_system_metrics=True)

    if verbose >= 1:
        print("=" * 80)
        print("COMPENSATION TRAINING PIPELINE")
        print("=" * 80)

    start_time = time.time()

    # ── STEP 1: Dataset ──────────────────────────────────────────────────────
    if verbose >= 1:
        print("\n" + "-" * 80)
        print("STEP 1: Loading dataset")
        print("-" * 80)

    dataset = CompDataset(
        data_file_path    = data_file_path,
        target_timesteps  = target_timesteps,
        axis_selection    = axis_selection,
        downsample_factor = downsample_factor,
        reverse_time      = reverse_time,
        max_samples       = max_samples,
        random_seed       = random_seed,
        dtype             = torch.float32,
        validate_divisibility = True,
    )

    input_size  = dataset.num_input_features   # always 1
    output_size = dataset.num_output_features  # always 1

    if verbose >= 1:
        print(f"Input  : {dataset.feature_names}")
        print(f"Output : {dataset.target_names}")

    # ── STEP 2: Split ────────────────────────────────────────────────────────
    if verbose >= 1:
        print("\n" + "-" * 80)
        print("STEP 2: Train / val split")
        print("-" * 80)

    train_dataset, val_dataset, seed_used = split_dataset(
        dataset, train_val_split, random_seed, verbose=(verbose >= 1)
    )

    # ── STEP 3: DataLoaders ──────────────────────────────────────────────────
    if verbose >= 1:
        print("\n" + "-" * 80)
        print("STEP 3: DataLoaders")
        print("-" * 80)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device != 'cpu'),
        collate_fn=_collate_drop_fwd,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != 'cpu'),
        collate_fn=_collate_drop_fwd,
    )

    if verbose >= 1:
        print(f"  Batch size : {batch_size}")
        print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── STEP 4: Model ────────────────────────────────────────────────────────
    if verbose >= 1:
        print("\n" + "-" * 80)
        print("STEP 4: Model")
        print("-" * 80)

    model = initialize_model(
        model_type=model_type, input_size=input_size,
        hidden_size=hidden_size, num_layers=num_layers,
        output_size=output_size, dropout=dropout,
        bidirectional=bidirectional, device=device,
        verbose=(verbose >= 1),
    )

    if load_model_path is not None:
        if verbose >= 1:
            print(f"  Loading weights from: {load_model_path}")
        loaded = torch.load(load_model_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict):
            model.load_state_dict(loaded)
        else:
            model.load_state_dict(loaded.state_dict())
            model.to(device)
        if verbose >= 1:
            print("  Weights loaded.")

    # ── STEP 5: Loss ─────────────────────────────────────────────────────────
    if verbose >= 1:
        print("\n" + "-" * 80)
        print("STEP 5: Loss function")
        print("-" * 80)

    resolved_warmup = int(0.05 * target_timesteps)

    loss_fn = create_loss_function(
        base_loss                        = base_loss,
        jacobian_weight                  = jacobian_weight,
        jacobian_every_n_batches         = jacobian_every_n_batches,
        hidden_derivative_weight         = hidden_derivative_weight,
        hidden_derivative_every_n_batches= hidden_derivative_every_n_batches,
        warmup_timesteps                 = resolved_warmup,
        verbose                          = (verbose >= 1),
    )

    # ── STEP 6: Optimizer ────────────────────────────────────────────────────
    if verbose >= 1:
        print("\n" + "-" * 80)
        print("STEP 6: Optimizer")
        print("-" * 80)

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer_map    = {
        'adam':   torch.optim.Adam,
        'adamw':  torch.optim.AdamW,
        'sgd':    torch.optim.SGD,
        'rmsprop':torch.optim.RMSprop,
    }
    if optimizer_type.lower() not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    optimizer = optimizer_map[optimizer_type.lower()](
        model.parameters(), lr=learning_rate,
        weight_decay=l2_weight, **optimizer_kwargs,
    )
    if verbose >= 1:
        print(f"  {optimizer_type.upper()}  lr={learning_rate}  weight_decay={l2_weight}")

    # ── STEP 7: Early stopping ───────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience  = early_stopping_patience,
        min_delta = early_stopping_min_delta,
        verbose   = (verbose >= 1),
    )

    # ── MLflow: log all config ───────────────────────────────────────────────
    all_config = {
        'pipeline':                       'compensation',
        'data_file_path':                 data_file_path,
        'target_timesteps':               target_timesteps,
        'axis_selection':                 axis_selection,
        'downsample_factor':              downsample_factor,
        'reverse_time':                   reverse_time,
        'max_samples':                    max_samples,
        'total_samples':                  len(dataset),
        'train_samples':                  len(train_dataset),
        'val_samples':                    len(val_dataset),
        'train_val_split':                train_val_split,
        'random_seed_used':               seed_used,
        'model_type':                     model_type,
        'hidden_size':                    hidden_size,
        'num_layers':                     num_layers,
        'dropout':                        dropout,
        'bidirectional':                  bidirectional,
        'total_parameters':               model.get_num_parameters(),
        'load_model_path':                load_model_path,
        'base_loss':                      base_loss,
        'l2_weight':                      l2_weight,
        'jacobian_weight':                jacobian_weight,
        'hidden_derivative_weight':       hidden_derivative_weight,
        'warmup_timesteps':               resolved_warmup,
        'num_epochs':                     num_epochs,
        'batch_size':                     batch_size,
        'learning_rate':                  learning_rate,
        'optimizer_type':                 optimizer_type,
        'gradient_clip_value':            gradient_clip_value,
        'early_stopping_patience':        early_stopping_patience,
        'early_stopping_min_delta':       early_stopping_min_delta,
        'validation_frequency':           validation_frequency,
        'device':                         device,
    }
    mlflow.log_params(all_config, synchronous=False)

    # ── STEP 8: Training loop ────────────────────────────────────────────────
    if verbose >= 1:
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)

    history         = {'train_loss': [], 'val_loss': []} if return_history else None
    best_val_loss   = float('inf')
    best_val_std    = 0.0
    best_model_state= None

    training_start_time = datetime.datetime.now().isoformat()
    mlflow.log_param("Training Start Time", training_start_time)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        if verbose >= 1:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

        train_loss = _train_epoch_comp(
            model=model, train_loader=train_loader,
            loss_function=loss_fn, optimizer=optimizer,
            device=device, gradient_clip_value=gradient_clip_value,
            verbose=verbose,
        )

        if return_history:
            history['train_loss'].append(train_loss)
        mlflow.log_metric("Train loss", train_loss, step=epoch, synchronous=False)

        if (epoch + 1) % validation_frequency == 0:
            val_loss, val_loss_std = _validate_epoch_comp(
                model=model, val_loader=val_loader,
                warmup_timesteps=resolved_warmup,
                device=device, verbose=verbose,
            )

            if return_history:
                history['val_loss'].append(val_loss)

            mlflow.log_metric("Val loss",      val_loss,     step=epoch, synchronous=False)
            mlflow.log_metric("val_loss_std",  val_loss_std, step=epoch, synchronous=False)

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_val_std   = val_loss_std
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

                if save_model_path is not None and save_best:
                    Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_model_state, save_model_path)
                    if verbose >= 1:
                        print(f"  Saved best model → {save_model_path}")

            mlflow.log_metric("best val loss", best_val_loss, step=epoch, synchronous=False)
            mlflow.log_metric("best val std",  best_val_std,  step=epoch, synchronous=False)

            status = early_stopper(val_loss)
            if status == 0:
                if verbose >= 1:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        epoch_time = time.time() - epoch_start
        mlflow.log_metric("Epoch Time",            epoch_time,          step=epoch, synchronous=False)
        mlflow.log_metric("Early Stopping Counter",early_stopper.counter,step=epoch, synchronous=False)
        if verbose >= 1:
            print(f"  Epoch time: {epoch_time:.2f}s")

    # Restore best weights only when save_best_only=True; otherwise keep last epoch state
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
    mlflow.pytorch.log_model(model, name="model_best" if save_best else "model_last")

    # ── STEP 9: Summary ──────────────────────────────────────────────────────
    total_time          = time.time() - start_time
    training_end_time   = datetime.datetime.now().isoformat()
    mlflow.log_param("Training End Time", training_end_time)
    mlflow.log_metric("total_training_minutes", total_time / 60)

    if verbose >= 1:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"  Total time   : {total_time / 60:.2f} min")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Seed used    : {seed_used}")
        print("=" * 80)

    # ── STEP 10: Save final model if needed ──────────────────────────────────
    if save_model_path is not None and not save_best:
        final_path = Path(save_model_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, str(final_path.parent / f"{final_path.stem}_final{final_path.suffix}"))
        mlflow.pytorch.log_model(model, name="model_final")

    # ── STEP 11: Plots ───────────────────────────────────────────────────────
    if generate_plots:
        visualize_predictions(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            num_samples=num_plot_samples,
            save_dir=plot_save_dir,
            random_seed=seed_used,
            reverse_time=reverse_time,
        )

    mlflow.end_run()
    return model, history, seed_used


if __name__ == "__main__":
    print("Compensation Training Pipeline")
    print("Schema: (N, T, 6)  cols 0-2=Δe_scale, cols 3-5=Δvff")
    print("Call train_model(...) to train.")