"""
Training Pipeline — Error Compensation v3 (Approach C: Error Prediction Hidden State)
=======================================================================================

Architecture (two-phase):

  Phase 1  — Error Prediction GRU (train_val_loop_v10.py, already done)
             Input : cmd_pos_{x,y,z}
             Output: e_scale_{x,y,z}  (following error)
             Result: frozen GRU weights + hidden states h_t

  Phase 2  — MLP + Error Compensation GRU  (this file)
             Input : [MLP(h_t), ΔE]  (h_t from frozen error-pred GRU)
             Output: ΔVFF
             Loss  : RMSE + λ_mlp · ||W_mlp||² + λ_gru · ||W_comp||²

Full data tensor layout  (loaded once into RAM):
    X: shape (M, N, L, F)
      M = number of unique trajectories  (~1000)
      N = number of VFF runs per traj    (11, including zero-VFF baseline)
      L = sequence length                (8000 at 4 kHz)
      F = number of feature columns      (see DATASET_COL_* below)

Per-sample construction (vectorised, at batch time):
    ΔE   = error_i   - error_j
    ΔVFF = vff_i     - vff_j
    cmd_pos stays as trajectory context (identical for i and j)

GPU allocation:
    GPU 0 — frozen error-pred GRU (inference only → hidden states h_t)
    GPU 1 — MLP + error-comp GRU  (all gradient computation)

Run from run_experiment.py with pipeline='compensation_v3'.
"""

import math
import time
import os
import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from train_val_loop_v10 import SequenceModel, EarlyStopping


# ============================================================================
# SECTION 1: COLUMN LAYOUT
# ============================================================================
# Columns in the raw 4D tensor X[M, N, L, F].
# Adjust these indices to match your actual stored column order.

# kinematics.npy  (M, L, 9) — trajectory-level
# Group offsets: pos=0, vel=3, accel=6  (3 axes each)
KIN_POS_OFFSET   = 0   # pos_x at col 0, pos_y at col 1, pos_z at col 2
KIN_VEL_OFFSET   = 3
KIN_ACCEL_OFFSET = 6

# ev.npy  (M, N, L, 6) — instance-level
# Group offsets: e_enc=0, vff=3  (3 axes each)
EV_ERR_OFFSET    = 0   # e_enc_x at col 0, e_enc_y at col 1, e_enc_z at col 2
EV_VFF_OFFSET    = 3

AXIS_COL = {
    'x': 0,
    'y': 1,
    'z': 2,
}


# ============================================================================
# SECTION 2: DATASET
# ============================================================================

# Standard filenames inside data_dir — never need to change these
DATA_KINEMATICS = 'kinematics.npy'
DATA_EV         = 'ev.npy'
DATA_SPLITS     = 'splits.json'


def load_two_files(
    data_dir:          str,
    axis:              str,
    downsample_factor: int = 1,
) -> tuple:
    """
    Load the two-file dataset and return per-axis subsets.

    Returns
    -------
    pos_shared : np.ndarray, shape (M, L, 1), float32, C-contiguous
        cmd_pos for the selected axis. Trajectory-level — no N dimension.
        Both train and val datasets share the same array reference.

    ev_shared : np.ndarray, shape (M, N, L, 2), float32, C-contiguous
        Columns: [e_enc, vff] for the selected axis. Instance-level.
        Both train and val datasets share the same array reference.
    """
    a = AXIS_COL[axis]

    # ── kinematics.npy ────────────────────────────────────────────────────────
    kinematics_path = os.path.join(data_dir, DATA_KINEMATICS)
    ev_path         = os.path.join(data_dir, DATA_EV)

    print(f"[load_two_files] Loading {kinematics_path} ...")
    kin = np.load(kinematics_path)               # (M, L, 9)
    print(f"  kinematics shape : {kin.shape}  dtype: {kin.dtype}")

    M, L, _ = kin.shape

    if downsample_factor > 1:
        kin = kin[:, ::downsample_factor, :]
        L   = kin.shape[1]
        print(f"  After downsample x{downsample_factor}: L={L}")

    pos_col    = KIN_POS_OFFSET + a
    pos_shared = np.ascontiguousarray(
        kin[:, :, pos_col:pos_col+1], dtype=np.float32   # (M, L, 1)
    )
    del kin

    # ── ev.npy ────────────────────────────────────────────────────────────────
    print(f"[load_two_files] Loading {ev_path} ...")
    ev = np.load(ev_path)                        # (M, N, L, 6)
    print(f"  ev shape         : {ev.shape}  dtype: {ev.dtype}")

    M_ev, N, L_ev, _ = ev.shape
    assert M_ev == M,  f"Trajectory count mismatch: kinematics M={M}, ev M={M_ev}"
    assert L_ev == L, \
    f"Sequence length mismatch: kinematics L={L}, ev L={L_ev} (before downsample)"

    if downsample_factor > 1:
        ev = ev[:, :, ::downsample_factor, :]
        
    err_col   = EV_ERR_OFFSET + a
    vff_col   = EV_VFF_OFFSET + a
    ev_shared = np.ascontiguousarray(
        ev[:, :, :, [err_col, vff_col]], dtype=np.float32   # (M, N, L, 2)
    )
    del ev

    print(f"  pos_shared : {pos_shared.shape}  ({pos_shared.nbytes / 1e9:.3f} GB)")
    print(f"  ev_shared  : {ev_shared.shape}  ({ev_shared.nbytes / 1e9:.3f} GB)")
    return pos_shared, ev_shared


def load_splits(data_dir: str) -> tuple:
    """
    Load predetermined train/val/test triplet lists from splits.json
    produced by Tool 3 Stage 3.

    splits.json schema:
        {
          "train": [[traj_idx, inst_i, inst_j], ...],
          "val":   [[traj_idx, inst_i, inst_j], ...],
          "test":  [[traj_idx, inst_i, inst_j], ...],
          "n_trajectories": int,
          "n_instances":    int,
          "instance_ids":   [int, ...]
        }

    Returns
    -------
    train_indices, val_indices, test_indices : list of (m, i, j) tuples.
        i and j are 0-based positions into ev_shared's N axis, which must
        be assembled in instance_ids order by Tool 3 Stage 3.
    """
    splits_path = os.path.join(data_dir, DATA_SPLITS)
    with open(splits_path) as f:
        splits = json.load(f)

    train = [tuple(t) for t in splits['train']]
    val   = [tuple(t) for t in splits['val']]
    test  = [tuple(t) for t in splits['test']]

    print(f"[load_splits] {splits_path}")  # splits_path resolved above
    print(f"  Train {len(train):,}  Val {len(val):,}  Test {len(test):,}")
    print(f"  Trajectories: {splits['n_trajectories']}  "
          f"Instances: {splits['n_instances']}  "
          f"Instance IDs: {splits['instance_ids']}")
    return train, val, test


def build_index_list(M: int, N: int, random_seed: int) -> list:
    """
    Utility — build all (traj, i, j) permutation indices.
    Not called during training (splits come from splits.json).
    Retained for offline inspection or regenerating splits.
    """
    index_list = [
        (m, i, j)
        for m in range(M)
        for i in range(N)
        for j in range(N)
        if i != j
    ]
    rng = np.random.default_rng(seed=random_seed)
    rng.shuffle(index_list)
    print(f"  Index list: {len(index_list):,} permutations  "
          f"(M={M} x N={N} x {N-1})")
    return index_list


class CompPermutationDataset(Dataset):
    """
    Thin wrapper around two pre-loaded shared arrays and a pre-split index list.

    Both train and val datasets hold references to the same pos_shared and
    ev_shared arrays — no copies are made. OS copy-on-write handles DataLoader
    worker forking transparently.

    Parameters
    ----------
    pos_shared : np.ndarray, shape (M, L, 1)
        cmd_pos for the selected axis. Trajectory-level (no N dimension).
    ev_shared : np.ndarray, shape (M, N, L, 2)
        [e_enc, vff] for the selected axis. Instance-level.
    indices : list of (traj_idx, i, j) tuples
        Pre-split subset loaded from splits.json.
    dtype : torch.dtype
    """

    def __init__(
        self,
        pos_shared: np.ndarray,
        ev_shared:  np.ndarray,
        indices:    list,
        dtype:      torch.dtype = torch.float32,
    ):
        self.pos     = pos_shared   # (M, L, 1)    — shared reference
        self.ev      = ev_shared    # (M, N, L, 2) — shared reference
        self.indices = indices
        self.dtype   = dtype

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        m, i, j = self.indices[idx]

        cmd_pos   = self.pos[m]                          # (L, 1)
        delta_e   = (self.ev[m, i, :, 0:1]
                   - self.ev[m, j, :, 0:1])              # e_enc_i - e_enc_j  (L, 1)
        delta_vff = (self.ev[m, i, :, 1:2]
                   - self.ev[m, j, :, 1:2])              # vff_i   - vff_j    (L, 1)

        return (
            torch.tensor(cmd_pos,   dtype=self.dtype),   # (L, 1)
            torch.tensor(delta_e,   dtype=self.dtype),   # (L, 1)
            torch.tensor(delta_vff, dtype=self.dtype),   # (L, 1)
        )


# ============================================================================
# SECTION 3: MODEL (Approach C)
# ============================================================================

class SparseMLP(nn.Module):
    """
    Single linear layer: hidden_size → hidden_size.
    L2 regularisation on W and b together (caller applies λ_mlp externally).
    Post-training, column L2 norms of W reveal which hidden-state dimensions
    the compensation model actually uses.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, hidden_size)
        return self.linear(h)

    def l2_penalty(self) -> torch.Tensor:
        """||W||² + ||b||²  — caller multiplies by λ_mlp."""
        return (
            self.linear.weight.pow(2).sum()
            + self.linear.bias.pow(2).sum()
        )


class ErrorCompensationModel(nn.Module):
    """
    Approach C two-stage model.

    Stage 1 (frozen, GPU 0):
        error_pred_gru : SequenceModel   (input_size=1, output_size=1)
        Produces hidden states h_t at every timestep.

    Stage 2 (trainable, GPU 1):
        mlp            : SparseMLP       (hidden_size → hidden_size)
        comp_gru       : SequenceModel   (input_size=hidden_size+1, output_size=1)

    Only Stage 2 parameters receive gradients.

    Parameters
    ----------
    error_pred_checkpoint : str
        Path to a saved SequenceModel checkpoint (state_dict or full model).
    pred_hidden_size : int
        Hidden size of the loaded error prediction GRU.
    pred_num_layers : int
    comp_hidden_size : int
        Hidden size of the compensation GRU.
    comp_num_layers : int
    comp_dropout : float
    gpu_pred : int | str
        Device for the frozen error-pred GRU.
    gpu_comp : int | str
        Device for the MLP + compensation GRU.
    """

    def __init__(
        self,
        error_pred_checkpoint: str,
        pred_hidden_size:      int,
        pred_num_layers:       int,
        comp_hidden_size:      int,
        comp_num_layers:       int   = 1,
        comp_dropout:          float = 0.0,
        gpu_pred:              str   = 'cuda:0',
        gpu_comp:              str   = 'cuda:1',
    ):
        super().__init__()

        self.gpu_pred = gpu_pred
        self.gpu_comp = gpu_comp

        # ── Stage 1: frozen error-pred GRU ───────────────────────────────────
        self.error_pred_gru = SequenceModel(
            model_type  = 'gru',
            input_size  = 1,
            hidden_size = pred_hidden_size,
            num_layers  = pred_num_layers,
            output_size = 1,
        )
        ckpt = torch.load(error_pred_checkpoint, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if isinstance(ckpt, dict):
            self.error_pred_gru.load_state_dict(ckpt, strict=True)
        else:
            # Full model saved; copy weights
            self.error_pred_gru.load_state_dict(ckpt.state_dict(), strict=True)

        self.error_pred_gru.to(gpu_pred)
        for p in self.error_pred_gru.parameters():
            p.requires_grad_(False)
        self.error_pred_gru.eval()

        self.pred_hidden_size = pred_hidden_size

        # ── Stage 2: MLP + comp GRU ───────────────────────────────────────────
        self.mlp = SparseMLP(pred_hidden_size).to(gpu_comp)

        # comp GRU input: [MLP(h_t), ΔE]  → size = pred_hidden_size + 1
        self.comp_gru = SequenceModel(
            model_type  = 'gru',
            input_size  = pred_hidden_size + 1,
            hidden_size = comp_hidden_size,
            num_layers  = comp_num_layers,
            output_size = 1,
            dropout     = comp_dropout,
        ).to(gpu_comp)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        cmd_pos: torch.Tensor,   # (B, T, 1)  on CPU or gpu_pred
        delta_e: torch.Tensor,   # (B, T, 1)  on CPU or gpu_comp
    ) -> torch.Tensor:
        """
        Returns ΔVFF prediction: (B, T, 1) on gpu_comp.

        Training:  caller passes measured ΔE directly.
        Inference: caller passes −e_pred (or use self.infer()).
        """
        # ── Stage 1: hidden states ────────────────────────────────────────────
        cmd_pos = cmd_pos.to(self.gpu_pred)
        # GRU requires train() for cuDNN even when frozen (no-grad)
        self.error_pred_gru.train()
        with torch.no_grad():
            h_t = self._extract_hidden_states(cmd_pos)  # (B, T, H)

        # ── Cross-GPU transfer ────────────────────────────────────────────────
        h_t     = h_t.to(self.gpu_comp)
        delta_e = delta_e.to(self.gpu_comp)

        # ── Stage 2: MLP + concat + comp GRU ─────────────────────────────────
        mlp_out = self.mlp(h_t)                        # (B, T, H)
        inp     = torch.cat([mlp_out, delta_e], dim=-1) # (B, T, H+1)
        return self.comp_gru(inp)                       # (B, T, 1)

    def _extract_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the error-pred GRU and return all hidden states per timestep.
        Uses the GRU's internal rnn module directly to get h_t at every step.
        """
        rnn_out, _ = self.error_pred_gru.rnn(x)       # (B, T, H)
        return rnn_out

    def infer(self, cmd_pos: torch.Tensor) -> torch.Tensor:
        """
        Inference-mode forward. Supplies −e_pred as ΔE proxy.
        cmd_pos: (B, T, 1) normalised commanded position.
        Returns: ΔVFF  (B, T, 1) on gpu_comp.
        """
        cmd_pos = cmd_pos.to(self.gpu_pred)
        self.error_pred_gru.train()
        with torch.no_grad():
            h_t    = self._extract_hidden_states(cmd_pos)     # (B, T, H)
            e_pred = self.error_pred_gru.fc(h_t)              # (B, T, 1)
        neg_e_pred = (-e_pred).to(self.gpu_comp)
        h_t        = h_t.to(self.gpu_comp)
        mlp_out    = self.mlp(h_t)
        inp        = torch.cat([mlp_out, neg_e_pred], dim=-1)
        return self.comp_gru(inp)

    def get_num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.error_pred_gru.parameters())


# ============================================================================
# SECTION 4: LOSS
# ============================================================================

class CompensationLoss(nn.Module):
    """
    L = RMSE(ΔVFF_pred, ΔVFF_actual)
      + λ_mlp · (||W_mlp||² + ||b_mlp||²)
      + λ_gru · ||W_comp||²

    λ_mlp  — aggressive; drives sparsification of MLP channel selection.
    λ_gru  — conservative; prevents overfitting on small compensation dataset.
    """

    def __init__(
        self,
        mlp:        SparseMLP,
        comp_gru:   SequenceModel,
        lambda_mlp: float = 1e-4,
        lambda_gru: float = 1e-5,
        warmup_timesteps: int = 0,
    ):
        super().__init__()
        self.mlp              = mlp
        self.comp_gru         = comp_gru
        self.lambda_mlp       = lambda_mlp
        self.lambda_gru       = lambda_gru
        self.warmup_timesteps = warmup_timesteps

    def forward(
        self,
        pred:   torch.Tensor,   # (B, T, 1)
        target: torch.Tensor,   # (B, T, 1)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns (total_loss, component_dict).
        component_dict keys: 'rmse', 'reg_mlp', 'reg_gru', 'total'.
        """
        w = self.warmup_timesteps
        if w > 0:
            pred   = pred[:,   w:, :]
            target = target[:, w:, :]

        rmse    = torch.sqrt(((pred - target) ** 2).mean())
        reg_mlp = self.lambda_mlp * self.mlp.l2_penalty()
        reg_gru = self.lambda_gru * sum(
            p.pow(2).sum()
            for p in self.comp_gru.parameters()
        )
        total = rmse + reg_mlp + reg_gru

        return total, {
            'rmse':    rmse.item(),
            'reg_mlp': reg_mlp.item(),
            'reg_gru': reg_gru.item(),
            'total':   total.item(),
        }


# ============================================================================
# SECTION 5: TRAINING UTILITIES
# ============================================================================

def _train_epoch(
    model:       ErrorCompensationModel,
    loader:      DataLoader,
    loss_fn:     CompensationLoss,
    optimizer:   torch.optim.Optimizer,
    clip_value:  float = 1.0,
    verbose:     bool  = False,
) -> Dict[str, float]:
    """
    One training epoch.  Returns dict of mean loss components.
    """
    model.mlp.train()
    model.comp_gru.train()

    sums = {'rmse': 0.0, 'reg_mlp': 0.0, 'reg_gru': 0.0, 'total': 0.0}
    n_batches = 0

    for cmd_pos, delta_e, delta_vff in loader:
        # delta_vff target goes to gpu_comp
        target = delta_vff.to(model.gpu_comp)

        pred          = model(cmd_pos, delta_e)
        loss, parts   = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        if clip_value > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                clip_value,
            )
        optimizer.step()

        for k, v in parts.items():
            sums[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def _val_epoch(
    model:            ErrorCompensationModel,
    loader:           DataLoader,
    warmup_timesteps: int = 0,
) -> Tuple[float, float]:
    """
    Validation: returns (mean_rmse, std_rmse) over batches.
    Uses pure RMSE (no regularisation) so the metric is comparable across λ configs.
    """
    model.mlp.eval()
    model.comp_gru.eval()

    rmse_vals = []
    for cmd_pos, delta_e, delta_vff in loader:
        target = delta_vff.to(model.gpu_comp)
        pred   = model(cmd_pos, delta_e)

        w = warmup_timesteps
        if w > 0:
            pred   = pred[:,   w:, :]
            target = target[:, w:, :]

        rmse = torch.sqrt(((pred - target) ** 2).mean()).item()
        rmse_vals.append(rmse)

    arr = np.array(rmse_vals)
    return float(arr.mean()), float(arr.std())


# ============================================================================
# SECTION 6: MLP DIAGNOSTIC LOGGING
# ============================================================================

def log_mlp_diagnostics(
    mlp:        SparseMLP,
    epoch:      int,
    n_epochs:   int,
    save_dir:   str,
    col_norms_history: List[np.ndarray],
) -> None:
    """
    Log a composite image for the MLP weight matrix and bias vector.

    Left panel : heatmap of W (hidden_size × hidden_size).
                 Column L2 norms plotted as a bar chart below.
    Right panel: bar chart of b.

    Appends current column norms to col_norms_history for
    across-training evolution plots.
    """
    W   = mlp.linear.weight.detach().cpu().numpy()   # (H_out, H_in)
    b   = mlp.linear.bias.detach().cpu().numpy()     # (H_out,)
    col_norms = np.linalg.norm(W, axis=0)            # (H_in,)
    row_norms = np.linalg.norm(W, axis=1)            # (H_out,)
    col_norms_history.append(col_norms.copy())

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9),
        gridspec_kw={'height_ratios': [4, 1]},
    )
    fig.suptitle(
        f"Epoch {epoch}/{n_epochs} — MLP weight matrix (left): column L2 norm shows "
        f"which hidden-state dims are used. Bias vector (right): large bias + near-zero "
        f"W row norm indicates a degenerate constant output.",
        fontsize=9, wrap=True,
    )

    # Weight heatmap
    im = axes[0, 0].imshow(np.abs(W), aspect='auto', cmap='viridis')
    axes[0, 0].set_title('|W|  (rows=output, cols=input)')
    axes[0, 0].set_xlabel('Input hidden-state dim')
    axes[0, 0].set_ylabel('Output dim')
    plt.colorbar(im, ax=axes[0, 0])

    # Column L2 norms
    axes[1, 0].bar(range(len(col_norms)), col_norms, color='steelblue', alpha=0.8)
    axes[1, 0].set_title('Column L2 norms of W  (primary diagnostic)')
    axes[1, 0].set_xlabel('Input hidden-state dim')
    axes[1, 0].set_ylabel('L2 norm')
    axes[1, 0].grid(True, alpha=0.3)

    # Bias bar chart
    axes[0, 1].bar(range(len(b)), np.abs(b), color='coral', alpha=0.8)
    axes[0, 1].bar(range(len(row_norms)), row_norms, color='steelblue', alpha=0.5,
                   label='W row norm (overlaid)')
    axes[0, 1].set_title('|b|  (coral) vs W row norm (blue overlay)')
    axes[0, 1].set_xlabel('Output dim')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = Path(save_dir) / f"mlp_weights_epoch{epoch:04d}.png"
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    mlflow.log_artifact(str(save_path), artifact_path="mlp_diagnostics")


def log_col_norm_evolution(
    col_norms_history: List[np.ndarray],
    log_epochs:        List[int],
    save_dir:          str,
) -> None:
    """
    After training, plot per-column L2 norm of W over all logged epochs.
    One line per input hidden-state dimension.
    """
    if not col_norms_history:
        return

    history = np.stack(col_norms_history, axis=0)   # (n_checkpoints, H_in)
    H       = history.shape[1]

    fig, ax = plt.subplots(figsize=(12, 5))
    for h in range(H):
        ax.plot(log_epochs[:len(col_norms_history)], history[:, h],
                alpha=0.6, lw=1.0, label=f'dim {h}' if H <= 16 else None)

    ax.set_title('MLP column L2 norm evolution — sparsification dynamics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Column L2 norm of W')
    ax.grid(True, alpha=0.3)
    if H <= 16:
        ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()

    save_path = Path(save_dir) / "mlp_col_norm_evolution.png"
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifact(str(save_path), artifact_path="mlp_diagnostics")


# ============================================================================
# SECTION 7: PREDICTION PLOTS
# ============================================================================

@torch.no_grad()
def visualize_predictions(
    model:       ErrorCompensationModel,
    train_ds:    Dataset,
    val_ds:      Dataset,
    num_samples: int = 3,
    save_dir:    str = 'results_comp_v3',
    random_seed: int = 0,
) -> None:
    """
    5-panel plot per sample:
      0 cmd_pos input
      1 ΔE input
      2 ΔVFF prediction only
      3 ΔVFF target vs prediction overlay
      4 residual
    """
    model.mlp.eval()
    model.comp_gru.eval()

    rng = np.random.default_rng(random_seed if random_seed != 0 else 42)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for ds_name, ds in [('train', train_ds), ('val', val_ds)]:
        n = min(num_samples, len(ds))
        indices = rng.choice(len(ds), size=n, replace=False)

        for plot_idx, sample_idx in enumerate(indices):
            cmd_pos, delta_e, delta_vff = ds[int(sample_idx)]

            pred = model(
                cmd_pos.unsqueeze(0),
                delta_e.unsqueeze(0),
            ).squeeze(0).cpu().numpy()   # (T, 1)

            cmd_np  = cmd_pos.numpy()[:, 0]
            de_np   = delta_e.numpy()[:, 0]
            tgt_np  = delta_vff.numpy()[:, 0]
            pred_np = pred[:, 0]
            resid   = pred_np - tgt_np
            T       = len(cmd_np)
            t       = np.arange(T)

            fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
            fig.suptitle(
                f'{ds_name.capitalize()} Sample {sample_idx}',
                fontsize=13, fontweight='bold',
            )

            axes[0].plot(t, cmd_np,  color='steelblue', lw=1.2)
            axes[0].set_ylabel('cmd_pos (norm)', fontweight='bold')
            axes[0].set_title('Commanded position (input)')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(t, de_np,   color='royalblue', lw=1.2)
            axes[1].set_ylabel('ΔE (norm)', fontweight='bold')
            axes[1].set_title('Error delta (input)')
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(t, pred_np, color='crimson', lw=1.2, alpha=0.9)
            axes[2].set_ylabel('ΔVFF (norm)', fontweight='bold')
            axes[2].set_title('Predicted ΔVFF')
            axes[2].grid(True, alpha=0.3)

            axes[3].plot(t, tgt_np,  color='forestgreen', lw=1.2, alpha=0.8,
                         label='Actual ΔVFF')
            axes[3].plot(t, pred_np, color='crimson',     lw=1.2, alpha=0.8,
                         linestyle='--', label='Predicted ΔVFF')
            axes[3].set_ylabel('ΔVFF (norm)', fontweight='bold')
            axes[3].set_title('Target vs Prediction')
            axes[3].legend(loc='upper right')
            axes[3].grid(True, alpha=0.3)

            axes[4].plot(t, resid,   color='darkorange', lw=1.0, alpha=0.8)
            axes[4].axhline(0, color='k', lw=0.8, linestyle='--')
            axes[4].set_ylabel('Residual (norm)', fontweight='bold')
            axes[4].set_title('Residual (pred − target)')
            axes[4].set_xlabel('Timestep')
            axes[4].grid(True, alpha=0.3)

            plt.tight_layout()
            fname = save_path / f"{ds_name}_sample{plot_idx:02d}.png"
            fig.savefig(fname, dpi=100, bbox_inches='tight')
            plt.close(fig)

            mlflow.log_artifact(str(fname), artifact_path=f"plots/{ds_name}")


# ============================================================================
# SECTION 7b: DYNAMIC BATCH SIZE
# ============================================================================

def compute_dynamic_batch_size(
    sequence_length:  int,
    n_input_features: int   = 1,   # cmd_pos only (1 channel per axis)
    vram_fraction:    float = 0.80,
    cap:              int   = 1024,
    gpu:              str   = 'cuda:0',
) -> int:
    """
    Estimate a safe batch size for the compensation pipeline based on
    available VRAM on the given GPU.

    Uses the memory currently free (not reserved by PyTorch) so it is
    safe to call after models are already allocated.

    Formula
    -------
    bytes_per_sample = sequence_length × n_input_features × 4  (float32)
    batch_size       = floor(available_bytes × vram_fraction / bytes_per_sample)

    The result is capped at `cap` for gradient stability and rounded down
    to the nearest power of two for CUDA efficiency — though both are
    optional constraints rather than hard requirements.

    Parameters
    ----------
    sequence_length  : timesteps per sample (e.g. 8000)
    n_input_features : feature columns per timestep in the batch tensor
    vram_fraction    : fraction of free VRAM to target (default 0.80 leaves
                       headroom for activations and gradient buffers)
    cap              : hard upper limit on batch size
    gpu              : which GPU to query (should be gpu_comp where gradients live)

    Returns
    -------
    batch_size : int  (always >= 1)
    """
    if not torch.cuda.is_available():
        default = 32
        print(f"[dynamic_batch_size] CUDA not available — defaulting to {default}")
        return default

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    bytes_per_sample = sequence_length * n_input_features * 4   # float32
    raw              = int(free_bytes * vram_fraction / bytes_per_sample)
    raw              = max(1, min(raw, cap))

    # Round down to nearest power of two (optional — comment out if undesired)
    pow2 = 2 ** int(math.floor(math.log2(raw))) if raw >= 1 else 1

    print(
        f"[dynamic_batch_size] GPU={gpu}  "
        f"free={free_bytes/1e9:.1f}GB / total={total_bytes/1e9:.1f}GB  "
        f"raw={raw}  pow2={pow2}  cap={cap}"
    )
    return pow2


# ============================================================================
# SECTION 8: MAIN TRAINING FUNCTION
# ============================================================================

def train_model(
    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir:                   str,
    axis_selection:             str   = 'x',
    downsample_factor:          int   = 1,
    random_seed:                int   = 0,
    # ── Error-pred GRU (frozen, Stage 1) ─────────────────────────────────────
    error_pred_checkpoint:      str   = '',
    pred_hidden_size:           int   = 64,
    pred_num_layers:            int   = 2,
    # ── Compensation model (Stage 2) ─────────────────────────────────────────
    comp_hidden_size:           int   = 64,
    comp_num_layers:            int   = 1,
    comp_dropout:               float = 0.0,
    # ── Loss ─────────────────────────────────────────────────────────────────
    lambda_mlp:                 float = 1e-4,
    lambda_gru:                 float = 1e-5,
    warmup_timesteps:           int   = 0,
    # ── Optimisation ─────────────────────────────────────────────────────────
    learning_rate:              float = 1e-3,
    num_epochs:                 int   = 200,
    batch_size:                 int   = 64,
    gradient_clip_value:        float = 1.0,
    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopping_patience:    int   = 20,
    early_stopping_min_delta:   float = 0.0,
    validation_frequency:       int   = 1,
    # ── I/O ──────────────────────────────────────────────────────────────────
    save_model_path:            Optional[str] = None,
    save_best_only:             bool  = True,
    plot_save_dir:              str   = 'results_comp_v3',
    generate_plots:             bool  = True,
    num_plot_samples:           int   = 3,
    # ── GPU ──────────────────────────────────────────────────────────────────
    gpu_pred:                   str   = 'cuda:0',
    gpu_comp:                   str   = 'cuda:1',
    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri:        str   = 'http://hurco-lambda.hurco.com:8090',
    mlflow_experiment_name:     str   = 'compensation_v3',
    mlflow_run_name:            Optional[str] = None,
    # ── Misc ─────────────────────────────────────────────────────────────────
    verbose:                    int   = 1,
    num_workers:                int   = 4,
    return_history:             bool  = True,
) -> Tuple[ErrorCompensationModel, Optional[Dict], int]:
    """
    Full Phase 2 training loop (Approach C).

    Returns (model, history_dict, seed_used).
    history keys: 'train_loss', 'val_loss'  — mirrors prediction pipeline so
    run_experiment.py works unchanged.
    """
    start_time = time.time()

    # ── Seed ─────────────────────────────────────────────────────────────────
    seed_used = (
        int(np.random.randint(0, 100_000_000)) if random_seed == 0 else random_seed
    )
    torch.manual_seed(seed_used)
    np.random.seed(seed_used)

    # ── Batch size ───────────────────────────────────────────────────────────
    if batch_size is None:
        # Sequence length = X_shared.shape[2] after downsample, but we haven't
        # loaded yet. Use sequence_length param or fall back to a safe default.
        # We resolve this after load_and_subset() below; placeholder for now.
        _resolve_batch_size_later = True
    else:
        _resolve_batch_size_later = False

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.start_run(run_name=mlflow_run_name)

    # ── STEP 1: Datasets ─────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 1: Datasets")
        print("=" * 80)

    # Load two files — pos_shared and ev_shared are shared references (no copy).
    pos_shared, ev_shared = load_two_files(
        data_dir, axis_selection, downsample_factor
    )
    M = pos_shared.shape[0]
    N = ev_shared.shape[1]
    L = pos_shared.shape[1]

    train_indices, val_indices, _ = load_splits(data_dir)

    train_ds = CompPermutationDataset(pos_shared, ev_shared, train_indices)
    val_ds   = CompPermutationDataset(pos_shared, ev_shared, val_indices)

    # Resolve dynamic batch size now that L is known
    if _resolve_batch_size_later:
        batch_size = compute_dynamic_batch_size(
            sequence_length  = L,
            n_input_features = 3,   # [cmd_pos, delta_e, delta_vff] per timestep
            gpu              = gpu_comp,
        )
        print(f"  Dynamic batch_size resolved to: {batch_size}")

    if verbose:
        print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── STEP 2: Model ─────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: Model")
        print("=" * 80)

    model = ErrorCompensationModel(
        error_pred_checkpoint = error_pred_checkpoint,
        pred_hidden_size      = pred_hidden_size,
        pred_num_layers       = pred_num_layers,
        comp_hidden_size      = comp_hidden_size,
        comp_num_layers       = comp_num_layers,
        comp_dropout          = comp_dropout,
        gpu_pred              = gpu_pred,
        gpu_comp              = gpu_comp,
    )

    if verbose:
        print(f"  Frozen error-pred GRU params : {model.get_num_frozen_parameters():,}")
        print(f"  Trainable Stage-2 params     : {model.get_num_trainable_parameters():,}")
        print(f"  GPU pred (frozen) : {gpu_pred}")
        print(f"  GPU comp (train)  : {gpu_comp}")

    # ── STEP 3: Loss ──────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 3: Loss")
        print("=" * 80)

    loss_fn = CompensationLoss(
        mlp              = model.mlp,
        comp_gru         = model.comp_gru,
        lambda_mlp       = lambda_mlp,
        lambda_gru       = lambda_gru,
        warmup_timesteps = warmup_timesteps,
    )

    if verbose:
        print(f"  RMSE + λ_mlp={lambda_mlp} · (||W_mlp||² + ||b||²)"
              f" + λ_gru={lambda_gru} · ||W_comp||²")
        print(f"  Warmup timesteps: {warmup_timesteps}")

    # ── STEP 4: Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    # ── STEP 5: Early stopping ────────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience  = early_stopping_patience,
        min_delta = early_stopping_min_delta,
        verbose   = (verbose >= 1),
    )

    # ── Log config to MLflow ──────────────────────────────────────────────────
    mlflow.log_params({
        'pipeline':                'compensation_v3',
        'axis_selection':          axis_selection,
        'data_dir':                data_dir,
        'downsample_factor':       downsample_factor,
        'random_seed_used':        seed_used,
        'pred_hidden_size':        pred_hidden_size,
        'pred_num_layers':         pred_num_layers,
        'comp_hidden_size':        comp_hidden_size,
        'comp_num_layers':         comp_num_layers,
        'comp_dropout':            comp_dropout,
        'lambda_mlp':              lambda_mlp,
        'lambda_gru':              lambda_gru,
        'warmup_timesteps':        warmup_timesteps,
        'learning_rate':           learning_rate,
        'num_epochs':              num_epochs,
        'batch_size':              batch_size,
        'gradient_clip_value':     gradient_clip_value,
        'early_stopping_patience': early_stopping_patience,
        'frozen_params':           model.get_num_frozen_parameters(),
        'trainable_params':        model.get_num_trainable_parameters(),
        'error_pred_checkpoint':   error_pred_checkpoint,
    }, synchronous=False)

    # ── MLP diagnostic schedule ───────────────────────────────────────────────
    log_epochs = (
        [math.floor(k * num_epochs / 10) for k in range(1, 11)]
        + [num_epochs - 1]
    )
    col_norms_history: List[np.ndarray] = []
    diag_save_dir = Path(plot_save_dir) / "mlp_diagnostics"
    diag_save_dir.mkdir(parents=True, exist_ok=True)

    # ── STEP 6: Training loop ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("STARTING TRAINING (Phase 2 — compensation)")
        print("=" * 80)

    history: Optional[Dict] = {'train_loss': [], 'val_loss': [], 'val_loss_std': []} if return_history else None
    best_val_loss   = float('inf')
    best_val_std    = 0.0
    best_model_state = None
    mlflow.log_param("Training Start Time", datetime.datetime.now().isoformat())

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_parts = _train_epoch(
            model=model, loader=train_loader, loss_fn=loss_fn,
            optimizer=optimizer, clip_value=gradient_clip_value,
            verbose=(verbose >= 2),
        )
        if return_history: history['train_loss'].append(train_parts['rmse'])

        mlflow.log_metrics({
            'train_rmse':    train_parts['rmse'],
            'train_reg_mlp': train_parts['reg_mlp'],
            'train_reg_gru': train_parts['reg_gru'],
            'train_total':   train_parts['total'],
        }, step=epoch, synchronous=False)

        if (epoch + 1) % validation_frequency == 0:
            val_rmse, val_std = _val_epoch(
                model=model, loader=val_loader,
                warmup_timesteps=warmup_timesteps,
            )
            if return_history: history['val_loss'].append(val_rmse)
            if return_history: history['val_loss_std'].append(val_std)

            mlflow.log_metrics({
                'val_rmse':      val_rmse,
                'val_rmse_std':  val_std,
                'best_val_rmse': min(best_val_loss, val_rmse),
            }, step=epoch, synchronous=False)

            if val_rmse < best_val_loss:
                best_val_loss  = val_rmse
                best_val_std   = val_std
                best_model_state = {
                    'mlp':      {k: v.clone() for k, v in model.mlp.state_dict().items()},
                    'comp_gru': {k: v.clone() for k, v in model.comp_gru.state_dict().items()},
                }
                if save_model_path and save_best_only:
                    Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_model_state, save_model_path)

            status = early_stopper(val_rmse)
            if status == 0:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                break

            if verbose >= 1:
                print(
                    f"  Epoch {epoch+1:4d}/{num_epochs}"
                    f"  train={train_parts['rmse']:.5f}"
                    f"  val={val_rmse:.5f} ± {val_std:.5f}"
                    f"  best={best_val_loss:.5f}"
                    f"  [{time.time()-epoch_start:.1f}s]"
                )

        # ── MLP diagnostics ───────────────────────────────────────────────────
        if epoch in log_epochs:
            log_mlp_diagnostics(
                mlp=model.mlp,
                epoch=epoch,
                n_epochs=num_epochs,
                save_dir=str(diag_save_dir),
                col_norms_history=col_norms_history,
            )

        mlflow.log_metrics({
            'epoch_time':           time.time() - epoch_start,
            'early_stop_counter':   early_stopper.counter,
        }, step=epoch, synchronous=False)

    # ── Restore best weights ──────────────────────────────────────────────────
    if best_model_state is not None and save_best_only:
        model.mlp.load_state_dict(best_model_state['mlp'])
        model.comp_gru.load_state_dict(best_model_state['comp_gru'])

    # ── Col-norm evolution plot ───────────────────────────────────────────────
    log_col_norm_evolution(col_norms_history, log_epochs, str(diag_save_dir))

    # ── Log final model ───────────────────────────────────────────────────────
    mlflow.pytorch.log_model(model.mlp,      name="mlp_best")
    mlflow.pytorch.log_model(model.comp_gru, name="comp_gru_best")

    # ── STEP 7: Summary ───────────────────────────────────────────────────────
    total_time = time.time() - start_time
    mlflow.log_param("Training End Time", datetime.datetime.now().isoformat())
    mlflow.log_metric("total_training_minutes", total_time / 60)
    mlflow.log_metrics({
        'final_best_val_rmse': best_val_loss,
        'final_best_val_std':  best_val_std,
    })

    if verbose:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print(f"  Total time   : {total_time / 60:.2f} min")
        print(f"  Best val RMSE: {best_val_loss:.6f} ± {best_val_std:.6f}")
        print("=" * 80)

    # ── STEP 8: Plots ─────────────────────────────────────────────────────────
    if generate_plots:
        visualize_predictions(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            num_samples=num_plot_samples,
            save_dir=plot_save_dir,
            random_seed=seed_used,
        )

    mlflow.end_run()
    return model, history, seed_used


# ============================================================================
# SECTION 9: ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("Error Compensation Training Pipeline — v3 (Approach C)")
    print("Data : 4D tensor (M, N, L, F)")
    print("       M=trajectories, N=VFF runs, L=sequence length, F=features")
    print("Model: frozen error-pred GRU → MLP → concat(h_t, ΔE) → comp GRU → ΔVFF")
    print()
    print("Call train_model(...) to train.")
    print()
    print("Minimal example:")
    print("  from train_val_loop_comp_v3 import train_model")
    print("  model, history = train_model(")
    print("      data_dir               = 'data/comp/',")
    print("      error_pred_checkpoint  = 'checkpoints/error_pred_x.pt',")
    print("      pred_hidden_size       = 64,")
    print("      comp_hidden_size       = 32,")
    print("      lambda_mlp             = 1e-3,")
    print("      lambda_gru             = 1e-5,")
    print("      num_epochs             = 200,")
    print("  )")
