"""
Signal Optimizer for Trajectory Correction
==========================================
Given a trained GRU error-prediction model, finds optimal trajectory x*
that minimizes predicted errors while staying close to the commanded trajectory x.

Optimization problem:
    min_{z} L(x*) = lambda_error * error_term(x*)
                  + lambda_fidelity * fidelity_term(x*)
                  + lambda_smooth * smoothness_term(x*)
    subject to: x* = x + clamp(z, -0.5, 0.5)

Where:
    error_term:      L1 or L2 of |GRU(x*) - GRU(x)|
    fidelity_term:   L1 or L2 of |x* - x|
    smoothness_term: one of {grad_L1, grad_L2, grad2_L1, grad2_L2} of (x* - x)

Hyperparameters searched by Optuna via Ray Tune:
    error_type:      L1 | L2
    fidelity_type:   L1 | L2
    smoothness_type: grad_L1 | grad_L2 | grad2_L1 | grad2_L2
    lambda_error, lambda_fidelity, lambda_smooth
    inner_lr, inner_steps
    init_method:     zeros | gaussian | inverted_error

Validation metric (maximized):
    mean improvement ratio = mean((baseline_RMSE - optimized_RMSE) / baseline_RMSE)
    computed over timesteps [10%, end], averaged over all N signals
"""

import torch
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
import time
import datetime
import os
import sys
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter,  Repeater
from ray.tune.search.optuna.optuna_search import OptunaSearch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Optional

# Import existing modules
from train_val_loop_v10 import TimeSeriesDataset, SequenceModel, AXIS_COLUMNS

# ============================================================================
# CONSTANTS
# ============================================================================
SAVGOL_WINDOW       = 101
SAVGOL_POLY         = 3
MAX_DEVIATION_MM    = 0.5    # physical constraint in mm
WARMUP_FRAC         = 0.10   # fraction of timesteps to skip

# Statistics from combined_statistics.json — used to convert normalized data to mm.
# All .npy data is z-score normalized. To recover physical units:
#   pos_mm    = pos_normalized    * POS_STD[axis]    (mean ≈ 0 for deviation, so omitted)
#   e_scale_mm = e_scale_normalized * E_SCALE_STD[axis]   (mean ≈ 0)
#   z_constraint: MAX_DEVIATION_MM / POS_STD[axis] gives the clamp limit in normalized units
POS_STD = {
    'x': 247.2838848784503,
    'y': 140.93299970896865,
    'z': 112.96077495266569,
}
E_SCALE_STD = {
    'x': 0.0033232292485601647,
    'y': 0.0038927228408261755,
    'z': 0.002672943695112274,
}
ACCEL_STD = {
    'x': 146.27633804618486,
    'y': 142.96117504352614,
    'z': 115.67312792683349,
}
TIMESTEPS     = 8000     # fixed
SAMPLE_SIZE   = 1000     # signals sampled from dataset per trial
MINI_BATCH    = 64       # GPU mini-batch size
N_VAL_PLOTS   = 10       # number of validation plots
GAUSSIAN_CUTOFF_HZ   = 50.0
SIGNAL_SAMPLE_RATE   = 4000.0  # Hz


# ============================================================================
# MODEL LOADING FROM MLFLOW
# ============================================================================

def load_model(
    run_id: str,
    tracking_uri: str,
    model_path: str,
    device: str = 'cuda'
) -> tuple[SequenceModel, dict]:
    """
    Load a trained SequenceModel.

    Architecture params are fetched from MLflow by run_id.
    Weights are loaded directly from model_path via torch.load —
    this handles the model registry path:
        ~/mlartifacts/<exp_id>/models/m-<id>/artifacts/data/model.pth

    Args:
        run_id:       MLflow run ID — used only to fetch architecture params
        tracking_uri: URI of the MLflow tracking server
        model_path:   Direct path to model.pth on disk
        device:       torch device string

    Returns:
        model:  SequenceModel with loaded weights, frozen, on device
        params: dict of run params from MLflow
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Fetch architecture params from MLflow run
    run    = client.get_run(run_id)
    params = run.data.params

    # The .pth file contains the full model object (not just state dict).
    # Load it directly — this is the ground truth for architecture, not MLflow params.
    print(f"Loading weights from: {model_path}")
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(loaded, dict):
        # State dict only — must construct model from MLflow params
        model = SequenceModel(
            model_type    = params['model_type'],
            input_size    = int(params['input_size']),
            hidden_size   = int(params['hidden_size']),
            num_layers    = int(params['num_layers']),
            output_size   = int(params['output_size']),
            dropout       = float(params['dropout']),
            bidirectional = params.get('bidirectional', 'False') == 'True'
        )
        model.load_state_dict(loaded)
        model.to(device)
    else:
        # Full model object — use it directly, architecture is already correct
        model = loaded
        model.to(device)
        # Sync params dict to reflect actual architecture (for the print below)
        params = {**params,
                  'model_type': model.model_type,
                  'hidden_size': str(model.hidden_size),
                  'num_layers': str(model.num_layers)}

    model.eval()

    # Freeze all parameters — model is a fixed oracle, not trained here
    for p in model.parameters():
        p.requires_grad_(False)
    # Keep in train mode so cuDNN allows backward pass through RNN
    # (gradients flow to z, not to model params which are frozen)
    model.train()

    print(f"  Architecture: {params['model_type'].upper()} "
          f"{params['num_layers']} layers x {params['hidden_size']} hidden")
    print(f"  Input size: {params['input_size']}, Output size: {params['output_size']}")
    print(f"  Ready on {device}")

    return model, params


# ============================================================================
# DATA LOADING
# ============================================================================

def load_signal_batch(
    data_file_path: str,
    axis_selection: str,
    target_timesteps: int,
    sample_size: int,
    random_seed: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Load dataset and randomly sample `sample_size` signals.

    The dataset already contains precomputed vel and accel channels.
    We keep both:
      - X_full (pos, vel, accel): used as-is for the baseline GRU call on x
      - X_pos  (pos only):        the channel we optimize; vel/accel for x*
                                  are recomputed via SavGol from optimized pos

    Returns:
        X_pos:  (N, T, 1)  — position channel only
        X_full: (N, T, 3)  — precomputed (pos, vel, accel) for baseline pass
        idxs:   (N,)       — dataset indices of sampled signals
    """
    dataset = TimeSeriesDataset(
        data_file_path       = data_file_path,
        target_timesteps     = target_timesteps,
        axis_selection       = axis_selection,
        include_velocity     = True,
        include_acceleration = True,
        validate_divisibility= True
    )

    total = len(dataset)
    rng   = np.random.default_rng(random_seed)
    idxs  = rng.choice(total, size=min(sample_size, total), replace=False)

    X_list = []
    for i in idxs:
        x, _ = dataset[i]
        X_list.append(x)

    X_full = torch.stack(X_list)   # (N, T, 3) — pos, vel, accel

    return X_full, idxs


# ============================================================================
# KINEMATIC INTEGRATION
# ============================================================================

def integrate_delta_a(
    delta_a: torch.Tensor,              # (B, T, 1) acceleration perturbation
    max_deviation_normalized: float,    # position clamp limit in normalized units
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate δa → δv → δx via cumulative sum with drift correction.

    Without correction, a constant δa causes δx to grow quadratically and
    saturate at the clamp limit — the optimizer exploits this trivial solution.

    Drift correction (applied before integration):
      1. Zero-mean δa so sum(δa) = 0 → δv returns to zero at end of window
         (no net velocity change over the trajectory segment)
      2. After integrating δv, zero-mean δv so sum(δv) = 0 → δx returns to
         zero at end of window (no net position drift)

    This forces the perturbation to be oscillatory rather than drifting,
    keeping δx bounded and making sparsity in δa meaningful.

    After correction, δx is clamped to ±max_deviation_normalized and δv, δa
    are back-derived for full kinematic consistency.

    Returns:
        delta_x:      (B, T, 1) position perturbation, bounded
        delta_v:      (B, T, 1) velocity perturbation, consistent with δx
        delta_a_proj: (B, T, 1) acceleration perturbation, consistent with δx
    """
    # 1. Remove mean from δa so the velocity perturbation has zero net change
    delta_a_zm = delta_a - delta_a.mean(dim=1, keepdim=True)

    # 2. Integrate to δv, then remove mean so position perturbation has zero net drift
    delta_v_raw = torch.cumsum(delta_a_zm, dim=1)
    delta_v_zm  = delta_v_raw - delta_v_raw.mean(dim=1, keepdim=True)

    # 3. Integrate to δx
    delta_x_raw = torch.cumsum(delta_v_zm, dim=1)

    # 4. Clamp δx to feasible set
    delta_x = delta_x_raw.clamp(-max_deviation_normalized, max_deviation_normalized)

    # 5. Back-derive δv and δa from clamped δx for full consistency
    delta_v      = torch.cat([delta_x[:, :1, :],
                               delta_x[:, 1:, :] - delta_x[:, :-1, :]], dim=1)
    delta_a_proj = torch.cat([delta_v[:, :1, :],
                               delta_v[:, 1:, :] - delta_v[:, :-1, :]], dim=1)

    return delta_x, delta_v, delta_a_proj


def build_corrected_input(
    X_full:  torch.Tensor,   # (B, T, 3) original (pos, vel, accel)
    delta_x: torch.Tensor,   # (B, T, 1)
    delta_v: torch.Tensor,   # (B, T, 1)
    delta_a: torch.Tensor,   # (B, T, 1)
) -> torch.Tensor:
    """
    Add kinematic perturbations to original trajectory.

    Returns:
        (B, T, 3) corrected GRU input: (x + δx, v + δv, a + δa)
    """
    x_star = X_full[:, :, 0:1] + delta_x
    v_star = X_full[:, :, 1:2] + delta_v
    a_star = X_full[:, :, 2:3] + delta_a
    return torch.cat([x_star, v_star, a_star], dim=-1)   # (B, T, 3)


# ============================================================================
# INITIALIZATION
# ============================================================================

def bandlimit_gaussian(
    shape:                    tuple,
    cutoff_hz:                float,
    sample_rate:              float,
    device,
    max_deviation_normalized: float,   # clamp limit in normalized position units
) -> torch.Tensor:
    """
    Generate bandlimited Gaussian noise in normalized position units.

    Args:
        shape:                    (B, T, 1)
        cutoff_hz:                maximum frequency in Hz
        sample_rate:              signal sample rate in Hz
        max_deviation_normalized: 3-sigma target in normalized position units
                                  = MAX_DEVIATION_MM / POS_STD[axis]

    Returns:
        (B, T, 1) bandlimited noise, 3-sigma = max_deviation_normalized
    """
    B, T, _ = shape
    noise   = torch.randn(B, T, device=device)

    freqs   = torch.fft.rfftfreq(T, d=1.0/sample_rate).to(device)
    mask    = (freqs <= cutoff_hz).float()
    noise_f = torch.fft.rfft(noise, dim=1)
    noise_f = noise_f * mask.unsqueeze(0)
    noise   = torch.fft.irfft(noise_f, n=T, dim=1)

    std   = noise.std(dim=1, keepdim=True).clamp(min=1e-8)
    noise = noise / std * (max_deviation_normalized / 3.0)

    return noise.unsqueeze(-1)   # (B, T, 1)


def initialize_delta_a(
    X_full:                   torch.Tensor,  # (B, T, 3) original trajectory
    init_method:              str,
    device:                   str,
    max_deviation_normalized: float,         # MAX_DEVIATION_MM / POS_STD[axis]
    e_scale_to_accel_ratio:   float = 1.0,  # E_SCALE_STD[axis] / ACCEL_STD[axis]
    model:                    nn.Module = None,
) -> torch.Tensor:
    """
    Initialize δa (acceleration perturbation) in normalized acceleration units.

    zeros:          δa = 0 — no perturbation
    gaussian:       bandlimited Gaussian noise in acceleration space, scaled so
                    the resulting δx (after double integration) has 3-sigma =
                    max_deviation_normalized. Since cumsum twice amplifies by ~T²/4,
                    we scale δa noise by max_deviation_normalized * 12 / T² (12/T²
                    is the inverse of the variance of double-cumsum of unit noise).
    inverted_error: δa initialised so the resulting δx roughly opposes the
                    predicted error, converted to acceleration units.

    Returns:
        delta_a: (B, T, 1) requires_grad=True, normalized acceleration units
    """
    B, T, _ = X_full.shape

    with torch.no_grad():
        if init_method == 'zeros':
            da_init = torch.zeros(B, T, 1, device=device)

        elif init_method == 'gaussian':
            # Scale noise so double-integrated δx has std ≈ max_deviation_normalized/3
            # Var(cumsum(cumsum(noise))) ≈ T²(T+1)²/144 for uniform noise, approximate
            # as T²/12 per element → std of δx ≈ noise_std * T/sqrt(12)
            # So noise_std = (max_deviation_normalized/3) * sqrt(12) / T
            import math
            noise_std = (max_deviation_normalized / 3.0) * math.sqrt(12.0) / T
            raw = torch.randn(B, T, device=device) * noise_std
            # Bandlimit in frequency domain
            freqs   = torch.fft.rfftfreq(T, d=1.0/SIGNAL_SAMPLE_RATE).to(device)
            mask    = (freqs <= GAUSSIAN_CUTOFF_HZ).float()
            raw_f   = torch.fft.rfft(raw, dim=1)
            raw     = torch.fft.irfft(raw_f * mask.unsqueeze(0), n=T, dim=1)
            da_init = raw.unsqueeze(-1)

        elif init_method == 'inverted_error':
            if model is None:
                raise ValueError("model must be provided for init_method='inverted_error'")
            pred_err = model(X_full)                            # (B, T, 1) normalized e_scale
            # Convert e_scale → acceleration space, negate to oppose error
            da_init  = -pred_err * e_scale_to_accel_ratio
            # Verify resulting δx stays within bounds; clamp δx and re-derive δa
            dv = torch.cumsum(da_init, dim=1)
            dx = torch.cumsum(dv,      dim=1)
            dx_clamped = dx.clamp(-max_deviation_normalized, max_deviation_normalized)
            dv_re = torch.cat([dx_clamped[:, :1, :], dx_clamped[:, 1:, :] - dx_clamped[:, :-1, :]], dim=1)
            da_init = torch.cat([dv_re[:, :1, :], dv_re[:, 1:, :] - dv_re[:, :-1, :]], dim=1)

        else:
            raise ValueError(f"Unknown init_method: {init_method!r}. Choose 'zeros', 'gaussian', or 'inverted_error'.")

    delta_a = da_init.detach().clone().requires_grad_(True)
    return delta_a


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def apply_norm(tensor: torch.Tensor, norm_type: str) -> torch.Tensor:
    """Apply L1 or L2 norm over all elements, return scalar."""
    if norm_type == 'L1':
        return tensor.abs().mean()
    elif norm_type == 'L2':
        return tensor.pow(2).mean().sqrt()
    else:
        raise ValueError(f"Unknown norm: {norm_type}")


def smoothness_loss(
    delta:          torch.Tensor,
    smooth_type:    str
) -> torch.Tensor:
    """
    Compute smoothness penalty on delta = x* - x.

    smooth_type options:
        grad_L1:  L1 of first differences
        grad_L2:  L2 of first differences
        grad2_L1: L1 of second differences
        grad2_L2: L2 of second differences

    Args:
        delta: (B, T, 1)
    """
    if smooth_type in ('grad_L1', 'grad_L2'):
        d1 = delta[:, 1:, :] - delta[:, :-1, :]
        norm_type = 'L1' if smooth_type == 'grad_L1' else 'L2'
        return apply_norm(d1, norm_type)

    elif smooth_type in ('grad2_L1', 'grad2_L2'):
        d1 = delta[:, 1:, :]  - delta[:, :-1, :]
        d2 = d1[:, 1:, :]     - d1[:, :-1, :]
        norm_type = 'L1' if smooth_type == 'grad2_L1' else 'L2'
        return apply_norm(d2, norm_type)

    else:
        raise ValueError(f"Unknown smooth_type: {smooth_type}")


def compute_loss(
    X_full:                   torch.Tensor,  # (B, T, 3) original (pos, vel, accel)
    delta_a:                  torch.Tensor,  # (B, T, 1) acceleration perturbation
    baseline_pred:            torch.Tensor,  # (B, T, 1) GRU(x), normalized e_scale
    model:                    nn.Module,
    warmup_steps:             int,
    error_type:               str,           # L1 | L2 — norm on GRU(x*)
    fidelity_type:            str,           # L1 | L2 — norm on δx
    accel_type:               str,           # L1 | L2 — norm on δa magnitude
    smooth_type:              str,           # L1 | L2 — norm on diff(δa) (accel smoothness)
    lambda_fidelity:          float,
    lambda_accel:             float,
    lambda_smooth:            float,
    max_deviation_normalized: float,
) -> tuple[torch.Tensor, dict]:
    """
    Kinematic-consistent composite loss.

    δa is the optimization variable. δv and δx are derived by cumsum,
    so all three GRU input channels are kinematically consistent.

    Loss = norm_error(GRU(x*))          [drive predicted error to zero]
         + λ_fidelity · norm_fid(δx)    [stay close to commanded path]
         + λ_accel    · norm_accel(δa)  [keep acceleration perturbations small]
         + λ_smooth   · norm_smooth(diff(δa))  [keep δa smooth]

    All terms normalized to be dimensionless and O(1):
      error:    norm(GRU(x*))   / norm(GRU(x))        (0=perfect, 1=no change)
      fidelity: norm(δx)        / max_deviation
      accel:    norm(δa)        / max_deviation
      smooth:   norm(diff(δa))  / max_deviation

    norm is L1 or L2 per term, independently configurable.
    """
    delta_x, delta_v, delta_a_proj = integrate_delta_a(delta_a, max_deviation_normalized)
    gru_input = build_corrected_input(X_full, delta_x, delta_v, delta_a_proj)
    opt_pred  = model(gru_input)

    # Post-warmup slices
    op = opt_pred[:, warmup_steps:, :]
    bp = baseline_pred[:, warmup_steps:, :]
    dx = delta_x[:, warmup_steps:, :]
    da = delta_a[:, warmup_steps:, :]

    # Baseline norm for normalizing the error term (detached)
    with torch.no_grad():
        baseline_norm = apply_norm(bp, error_type).clamp(min=1e-10)

    error_loss    = apply_norm(op, error_type)   / baseline_norm
    fidelity_loss = apply_norm(dx, fidelity_type) / max_deviation_normalized
    accel_loss    = apply_norm(da, accel_type)    / max_deviation_normalized

    # Smoothness: penalize roughness of δa (first differences)
    da_diff    = da[:, 1:, :] - da[:, :-1, :]
    smooth_loss = apply_norm(da_diff, smooth_type) / max_deviation_normalized

    loss = (error_loss
            + lambda_fidelity * fidelity_loss
            + lambda_accel    * accel_loss
            + lambda_smooth   * smooth_loss)

    components = {
        'error':    error_loss.item(),
        'fidelity': fidelity_loss.item(),
        'accel':    accel_loss.item(),
        'smooth':   smooth_loss.item(),
        'total':    loss.item(),
    }
    return loss, components


# ============================================================================
# VALIDATION METRIC
# ============================================================================

def compute_improvement(
    X_full:        torch.Tensor,   # (B, T, 3) original trajectory
    delta_a:       torch.Tensor,   # (B, T, 1) acceleration perturbation
    baseline_pred: torch.Tensor,   # (B, T, 1) GRU(x)
    model:         nn.Module,
    warmup_steps:  int,
    max_deviation_normalized: float,
) -> torch.Tensor:
    """
    Compute per-signal improvement ratio over the post-warmup window.

    improvement = (RMSE(GRU(x)) - RMSE(GRU(x*))) / RMSE(GRU(x))

    Returns:
        (B,) tensor of improvement ratios
    """
    with torch.no_grad():
        delta_x, delta_v, delta_a_proj = integrate_delta_a(delta_a, max_deviation_normalized)
        gru_input = build_corrected_input(X_full, delta_x, delta_v, delta_a_proj)
        opt_pred  = model(gru_input)

    bp = baseline_pred[:, warmup_steps:, 0]
    op = opt_pred[:,     warmup_steps:, 0]

    baseline_rmse  = bp.pow(2).mean(dim=1).sqrt()
    optimized_rmse = op.pow(2).mean(dim=1).sqrt()

    denom = baseline_rmse.clamp(min=1e-10)
    return (baseline_rmse - optimized_rmse) / denom


# ============================================================================
# INNER OPTIMIZATION LOOP (per mini-batch)
# ============================================================================

def init_batch(
    X_full:                   torch.Tensor,   # (B, T, 3) original trajectory
    model:                    nn.Module,
    baseline_pred:            torch.Tensor,   # (B, T, 1) GRU(x) normalized e_scale
    warmup_steps:             int,
    init_method:              str,
    inner_lr:                 float,
    device:                   str,
    max_deviation_normalized: float,
    e_scale_to_accel_ratio:   float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.optim.Optimizer, float]:
    """
    Initialise δa and Adam optimizer for one mini-batch.

    Returns:
        delta_a:   (B, T, 1) acceleration perturbation, requires_grad=True
        X_init:    (B, T, 3) corrected trajectory at init (before any gradient steps)
        optimizer: Adam optimizing delta_a
        init_impr: mean improvement ratio at initialization (scalar)
    """
    delta_a = initialize_delta_a(
        X_full, init_method, device, max_deviation_normalized,
        e_scale_to_accel_ratio=e_scale_to_accel_ratio, model=model,
    )
    with torch.no_grad():
        dx, dv, da_proj = integrate_delta_a(delta_a, max_deviation_normalized)
        X_init    = build_corrected_input(X_full, dx, dv, da_proj)
        init_impr = compute_improvement(X_full, delta_a, baseline_pred, model,
                                        warmup_steps, max_deviation_normalized)
    optimizer = torch.optim.Adam([delta_a], lr=inner_lr)
    return delta_a, X_init.detach(), optimizer, init_impr.mean().item()


def step_batch(
    X_full:                   torch.Tensor,
    delta_a:                  torch.Tensor,
    baseline_pred:            torch.Tensor,
    model:                    nn.Module,
    optimizer:                torch.optim.Optimizer,
    warmup_steps:             int,
    error_type:               str,
    fidelity_type:            str,
    accel_type:               str,
    smooth_type:              str,
    lambda_fidelity:          float,
    lambda_accel:             float,
    lambda_smooth:            float,
    max_deviation_normalized: float,
) -> tuple[float, float]:
    """
    Run one gradient step on δa for a single mini-batch.

    Returns:
        impr:      mean improvement ratio after this step (scalar float)
        step_time: wall time for this step in seconds
    """
    t0 = time.perf_counter()

    optimizer.zero_grad()
    loss, _ = compute_loss(
        X_full, delta_a, baseline_pred, model,
        warmup_steps, error_type, fidelity_type, accel_type, smooth_type,
        lambda_fidelity, lambda_accel, lambda_smooth, max_deviation_normalized,
    )
    loss.backward()
    optimizer.step()

    impr = compute_improvement(X_full, delta_a, baseline_pred, model,
                               warmup_steps, max_deviation_normalized)

    return impr.mean().item(), time.perf_counter() - t0


# ============================================================================
# PLOTTING
# ============================================================================

def plot_optimization_results(
    x_pos:          np.ndarray,   # (T,) commanded position, normalized
    x_init:         np.ndarray,   # (T,) position after init, normalized
    x_star:         np.ndarray,   # (T,) optimized position, normalized
    baseline_pred:  np.ndarray,   # (T,) GRU(x), normalized e_scale units
    opt_pred:       np.ndarray,   # (T,) GRU(x*), normalized e_scale units
    sample_idx:     int,
    save_dir:       str,
    pos_std_mm:     float,        # POS_STD[axis] — to convert normalized deviation → mm
    e_scale_std_mm: float,        # E_SCALE_STD[axis] — to convert normalized e_scale → mm
    mlflow_run:     bool = True
):
    """
    Plot optimization results for a single signal.
    Inputs are in normalized units; all plotted values converted to mm or µm.

    3 subplots:
        1. Commanded x, initialized x_init, and optimized x* (normalized position)
        2. Deviation from commanded: x_init - x and x* - x, converted to mm
        3. Predicted e_scale: GRU(x) and GRU(x*), converted to mm
    """
    T = len(x_pos)
    t = np.arange(T)
    warmup = int(T * WARMUP_FRAC)

    # Denormalize for plotting
    init_deviation_mm = (x_init - x_pos) * pos_std_mm      # normalized → mm
    opt_deviation_mm  = (x_star - x_pos) * pos_std_mm      # normalized → mm
    baseline_pred_mm  = baseline_pred    * e_scale_std_mm  # normalized → mm
    opt_pred_mm       = opt_pred         * e_scale_std_mm  # normalized → mm

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Signal Optimization — Sample {sample_idx}', fontsize=14, fontweight='bold')

    # --- Subplot 1: Commanded, initialized, and optimized trajectory ---
    axes[0].plot(t, x_pos,  'b-',   linewidth=1.2, label='Commanded x',         alpha=0.9)
    axes[0].plot(t, x_init, 'g--',  linewidth=1.0, label='Initialized x_init',  alpha=0.7)
    axes[0].plot(t, x_star, 'r--',  linewidth=1.2, label='Optimized x*',        alpha=0.8)
    axes[0].axvspan(0, warmup, alpha=0.08, color='gray', label='Warmup (excluded)')
    axes[0].set_ylabel('Position (normalized)', fontweight='bold')
    axes[0].set_title('Commanded, Initialized, and Optimized Trajectory')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Subplot 2: Deviations in mm ---
    axes[1].plot(t, init_deviation_mm, 'g-',  linewidth=1.0, label='x_init - x (init deviation)',  alpha=0.7)
    axes[1].plot(t, opt_deviation_mm,  'r--', linewidth=1.2, label='x* - x (opt deviation)',       alpha=0.8)
    axes[1].axhline(0,                 color='k',   linewidth=0.6, linestyle='--', alpha=0.4)
    axes[1].axhline( MAX_DEVIATION_MM, color='red', linewidth=0.8, linestyle=':',  alpha=0.6, label=f'±{MAX_DEVIATION_MM} mm limit')
    axes[1].axhline(-MAX_DEVIATION_MM, color='red', linewidth=0.8, linestyle=':',  alpha=0.6)
    axes[1].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[1].set_ylabel('Deviation from commanded (mm)', fontweight='bold')
    axes[1].set_title('Trajectory Deviations')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # --- Subplot 3: Predicted e_scale errors in mm ---
    axes[2].plot(t, baseline_pred_mm, 'b-',  linewidth=1.2, label='GRU(x) — baseline',   alpha=0.8)
    axes[2].plot(t, opt_pred_mm,      'r--', linewidth=1.2, label='GRU(x*) — optimized', alpha=0.8)
    axes[2].axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
    axes[2].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[2].set_ylabel('Predicted e_scale (mm)', fontweight='bold')
    axes[2].set_xlabel('Timestep', fontweight='bold')
    axes[2].set_title('Baseline vs Optimized Predicted Error')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Stats — in physical units (mm and µm)
    w_base    = baseline_pred_mm[warmup:]
    w_opt     = opt_pred_mm[warmup:]
    base_rmse = np.sqrt(np.mean(w_base ** 2))   # RMSE(GRU(x)) in mm
    opt_rmse  = np.sqrt(np.mean(w_opt  ** 2))   # RMSE(GRU(x*)) in mm
    impr      = (base_rmse - opt_rmse) / (base_rmse + 1e-10)
    max_opt_dev  = np.max(np.abs(opt_deviation_mm))
    max_init_dev = np.max(np.abs(init_deviation_mm))

    stats = (f'Baseline RMSE: {base_rmse*1000:.2f} µm  |  '
             f'Optimized RMSE: {opt_rmse*1000:.2f} µm  |  '
             f'Improvement: {impr*100:.2f}%  |  '
             f'Max opt dev: {max_opt_dev:.4f} mm  |  '
             f'Max init dev: {max_init_dev:.4f} mm')
    axes[2].set_xlabel(f'Timestep\n{stats}', fontweight='bold', fontsize=8)

    plt.tight_layout()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fname = save_path / f'opt_sample_{sample_idx:04d}.png'

    if mlflow_run:
        mlflow.log_figure(fig, f'{save_dir}/opt_sample_{sample_idx:04d}.png')

    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN OBJECTIVE FUNCTION (called by Ray Tune)
# ============================================================================

# These are set in __main__ before ray.init
TRACKING_URI    = None
EXP_ID          = None
DATA_FILE_PATH  = None
MODEL_RUN_ID    = None
MODEL_PATH      = None
AXIS_SELECTION  = 'x'


def objective(trial_params: dict, model_run_id: str = None, model_path: str = None, tracking_uri: str = None, data_file_path: str = None, axis_selection: str = 'x', exp_id: int = 0):
    # Ensure the correct working directory is on the path in Ray workers
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    """
    Ray Tune objective function.

    Runs the full signal optimization over SAMPLE_SIZE signals
    and reports mean improvement ratio to Optuna.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trial_idx = trial_params.get('__trial_index__', 0)

    # Deviation constraint: MAX_DEVIATION_MM specified in mm, converted to normalized units
    # for use in the optimization. All .npy data is z-score normalized.
    axis_key = axis_selection if axis_selection != 'all' else 'x'
    max_deviation_normalized = MAX_DEVIATION_MM / POS_STD[axis_key]
    # Ratio used by inverted_error init to convert normalized e_scale → normalized position
    e_scale_to_pos_ratio = E_SCALE_STD[axis_key] / POS_STD[axis_key]

    # ---- MLflow run ----
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.start_run(experiment_id=str(exp_id), log_system_metrics=False)
    mean_improvement = 0.0   # default if trial fails before completion

    try:
        mlflow.set_tag("run_type",            "signal_optimizer")
        mlflow.set_tag("source_model_run_id", model_run_id)
        mlflow.set_tag("model_path",          model_path)

        logged_params = {k: v for k, v in trial_params.items() if k != '__trial_index__'}
        mlflow.log_params(logged_params)
        mlflow.log_param('max_deviation_mm',          MAX_DEVIATION_MM)
        mlflow.log_param('max_deviation_normalized',  max_deviation_normalized)
        mlflow.log_param('pos_std_mm',                POS_STD[axis_key])

        error_type      = trial_params['error_type']
        fidelity_type   = trial_params['fidelity_type']
        accel_type      = trial_params['accel_type']
        smooth_type     = trial_params['smooth_type']
        lambda_fidelity = trial_params['lambda_fidelity']
        lambda_accel    = trial_params['lambda_accel']
        lambda_smooth   = trial_params['lambda_smooth']
        inner_lr        = trial_params['inner_lr']
        inner_steps     = int(trial_params['inner_steps'])
        init_method     = trial_params['init_method']

        # ---- Load model ----
        model, model_params = load_model(model_run_id, tracking_uri, model_path, device)

        # ---- Load data ----
        # Generate and log random seed for full reproducibility
        trial_seed = int(torch.randint(0, 2**31, (1,)).item())
        mlflow.log_param("trial_seed", trial_seed)

        X_full, sample_idxs = load_signal_batch(
        data_file_path   = data_file_path,
        axis_selection   = axis_selection,
        target_timesteps = TIMESTEPS,
        sample_size      = SAMPLE_SIZE,
        random_seed      = trial_seed
        )

        N            = X_full.shape[0]
        warmup_steps = int(TIMESTEPS * WARMUP_FRAC)
        e_scale_std  = E_SCALE_STD[axis_key]
        # Ratio for inverted_error init: converts normalized e_scale → normalized accel units
        e_scale_to_accel_ratio = E_SCALE_STD[axis_key] / ACCEL_STD[axis_key]

        # ---- Precompute baseline predictions ----
        baseline_preds = []
        with torch.no_grad():
            for start in range(0, N, MINI_BATCH):
                end = min(start + MINI_BATCH, N)
                xb  = X_full[start:end].to(device)   # (B, T, 3)
                baseline_preds.append(model(xb).cpu())
        baseline_all = torch.cat(baseline_preds, dim=0)  # (N, T, 1)

        # ---- Log fixed/infra params ----
        mlflow.log_param("axis_selection",  axis_selection)
        mlflow.log_param("trial_idx",       trial_idx)
        mlflow.log_param("n_signals",       N)
        mlflow.log_param("warmup_steps",    warmup_steps)
        mlflow.log_param("sample_size",     SAMPLE_SIZE)

        # ---- Split into mini-batches and initialise δa + Adam for each ----
        batch_slices = [(s, min(s + MINI_BATCH, N)) for s in range(0, N, MINI_BATCH)]
        batch_states = []
        init_imprs   = []
        for start, end in batch_slices:
            xb = X_full[start:end].to(device)         # (B, T, 3)
            bp = baseline_all[start:end].to(device)   # (B, T, 1)
            delta_a, X_init_b, opt, init_impr = init_batch(
                X_full                   = xb,
                model                    = model,
                baseline_pred            = bp,
                warmup_steps             = warmup_steps,
                init_method              = init_method,
                inner_lr                 = inner_lr,
                device                   = device,
                max_deviation_normalized = max_deviation_normalized,
                e_scale_to_accel_ratio   = e_scale_to_accel_ratio,
            )
            batch_states.append({"X_full": xb, "baseline_pred": bp,
                                  "delta_a": delta_a, "optimizer": opt,
                                  "X_init": X_init_b})
            init_imprs.append(init_impr)

        mean_init_improvement = float(np.mean(init_imprs))
        mlflow.log_metric("init_mean_improvement", mean_init_improvement, step=0, synchronous=False)
        print(f"  Init improvement (step 0): {mean_init_improvement:.4f}")

        # ---- Epoch loop: one gradient step per epoch across all batches ----
        all_step_times = []
        for step in range(inner_steps):
            step_imprs = []
            step_times = []
            for state in batch_states:
                impr, t = step_batch(
                    X_full                   = state["X_full"],
                    delta_a                  = state["delta_a"],
                    baseline_pred            = state["baseline_pred"],
                    model                    = model,
                    optimizer                = state["optimizer"],
                    warmup_steps             = warmup_steps,
                    error_type               = error_type,
                    fidelity_type            = fidelity_type,
                    accel_type               = accel_type,
                    smooth_type              = smooth_type,
                    lambda_fidelity          = lambda_fidelity,
                    lambda_accel             = lambda_accel,
                    lambda_smooth            = lambda_smooth,
                    max_deviation_normalized = max_deviation_normalized,
                )
                step_imprs.append(impr)
                step_times.append(t)

            all_step_times.extend(step_times)
            mean_step_impr = float(np.mean(step_imprs))
            mlflow.log_metric("step_mean_improvement", mean_step_impr, step=step + 1, synchronous=False)

            if step % 10 == 0 or step == inner_steps - 1:
                print(f"  Step {step+1:3d}/{inner_steps} | "
                      f"mean_impr={mean_step_impr:.4f} | "
                      f"avg_step={np.mean(step_times)*1000:.1f}ms")

        # ---- Collect final optimized signals ----
        all_improvements = []
        all_X_star       = []   # (N, T, 3) corrected trajectories
        all_X_init       = []   # (N, T, 3) initialized trajectories
        all_opt_pred     = []   # (N, T, 1)
        for state in batch_states:
            with torch.no_grad():
                dx, dv, da_proj = integrate_delta_a(state["delta_a"], max_deviation_normalized)
                X_star_b   = build_corrected_input(state["X_full"], dx, dv, da_proj)
                opt_pred_b = model(X_star_b)
                impr = compute_improvement(state["X_full"], state["delta_a"],
                                           state["baseline_pred"], model,
                                           warmup_steps, max_deviation_normalized)
            all_improvements.append(impr.cpu())
            all_X_star.append(X_star_b.cpu())
            all_X_init.append(state["X_init"].cpu())
            all_opt_pred.append(opt_pred_b.cpu())

        # ---- Aggregate ----
        all_improvements = torch.cat(all_improvements, dim=0)    # (N,)
        all_X_star       = torch.cat(all_X_star,       dim=0)    # (N, T, 3)
        all_X_init       = torch.cat(all_X_init,       dim=0)    # (N, T, 3)
        all_opt_pred     = torch.cat(all_opt_pred,      dim=0)   # (N, T, 1)

        mean_improvement = all_improvements.mean().item()
        std_improvement  = all_improvements.std().item()

        baseline_rmse_um = float(
        baseline_all[:, warmup_steps:, 0].pow(2).mean().sqrt().item() * e_scale_std * 1000
        )
        opt_rmse_um = float(
        all_opt_pred[:, warmup_steps:, 0].pow(2).mean().sqrt().item() * e_scale_std * 1000
        )

        mlflow.log_metric("mean_improvement",      mean_improvement)
        mlflow.log_metric("std_improvement",       std_improvement)
        mlflow.log_metric("median_improvement",    all_improvements.median().item())
        mlflow.log_metric("pct_improved",          (all_improvements > 0).float().mean().item())
        mlflow.log_metric("init_mean_improvement", mean_init_improvement)
        mlflow.log_metric("baseline_rmse_um",      baseline_rmse_um)
        mlflow.log_metric("optimized_rmse_um",     opt_rmse_um)
        mlflow.log_metric("rmse_reduction_um",     baseline_rmse_um - opt_rmse_um)
        mlflow.log_metric("mean_step_time_ms",     np.mean(all_step_times) * 1000)
        mlflow.log_metric("total_step_time_s",     np.sum(all_step_times))

        # ---- Validation plots (N_VAL_PLOTS random samples) ----
        rng       = np.random.default_rng(trial_seed)
        plot_idxs = rng.choice(N, size=min(N_VAL_PLOTS, N), replace=False)
        plot_dir  = f"plots/trial_{trial_idx}"

        for pi in plot_idxs:
            plot_optimization_results(
                x_pos          = X_full[pi, :, 0].numpy(),
                x_init         = all_X_init[pi, :, 0].numpy(),
                x_star         = all_X_star[pi, :, 0].numpy(),
                baseline_pred  = baseline_all[pi, :, 0].numpy(),
                opt_pred       = all_opt_pred[pi, :, 0].numpy(),
                sample_idx     = int(sample_idxs[pi]),
                save_dir       = plot_dir,
                pos_std_mm     = POS_STD[axis_key],
                e_scale_std_mm = E_SCALE_STD[axis_key],
                mlflow_run     = True
            )

    except Exception as e:
        print(f"  Trial {trial_idx} failed: {e}")
        try:
            mlflow.set_tag("trial_status", f"failed: {type(e).__name__}")
        except Exception:
            pass
        raise

    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass

    tune.report({"mean_improvement": mean_improvement})


# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

SEARCH_CONFIG = {
    "error_type":      tune.choice(['L1', 'L2']),
    "fidelity_type":   tune.choice(['L1', 'L2']),
    "accel_type":      tune.choice(['L1', 'L2']),
    "smooth_type":     tune.choice(['L1', 'L2']),
    "lambda_fidelity": tune.loguniform(1e-4, 1e1),
    "lambda_accel":    tune.loguniform(1e-4, 1e1),
    "lambda_smooth":   tune.loguniform(1e-4, 1e1),
    "inner_lr":        tune.loguniform(1e-8, 1e-6),
    "inner_steps":     tune.randint(10, 101),
    "init_method":     tune.choice(['gaussian', 'inverted_error']),
}

INITIAL_PARAMS = [{
    "error_type":      'L2',
    "fidelity_type":   'L2',
    "accel_type":      'L2',
    "smooth_type":     'L2',
    "lambda_fidelity": 1e-2,
    "lambda_accel":    1e-2,
    "lambda_smooth":   1e-2,
    "inner_lr":        1e-6,
    "inner_steps":     50,
    "init_method":     'gaussian',
}]


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    # ============================================================
    # CONFIGURATION — edit these values directly
    # ============================================================
    TRACKING_URI   = "http://hurco-lambda.hurco.com:8090"
    MODEL_RUN_ID   = "af3f1aaf616843e7b15a9f0637ae6131"
    MODEL_PATH     = "/home/bhatts/LSTM_Hyperopt_Logging/~/mlartifacts/23/models/m-675fe5ff09694de99805a64483630bcc/artifacts/data/model.pth"
    DATA_FILE_PATH = "data/clean_final.npy"
    AXIS_SELECTION = "x"
    NUM_TRIALS     = 100
    NUM_CPUS       = 4
    NUM_GPUS       = 4
    # ============================================================

    # ---- Verify run ID and model.pth before launching Ray ----
    print(f"Verifying model run {MODEL_RUN_ID}...")
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        run    = client.get_run(MODEL_RUN_ID)
        print(f"  Found run: {run.info.run_id} (status: {run.info.status})")
    except mlflow.exceptions.RestException as e:
        print(f"\nERROR: Could not find run '{MODEL_RUN_ID}' at {TRACKING_URI}")
        print(f"  {e}")
        print(f"\nTo list available runs:")
        print(f"  python signal_optimizer.py --list_runs --tracking_uri {TRACKING_URI}")
        sys.exit(1)
    pth = Path(MODEL_PATH)
    if not pth.exists():
        print(f"\nERROR: model.pth not found at: {pth}")
        sys.exit(1)
    MODEL_PATH = str(pth.resolve())
    print(f"  model.pth found: {MODEL_PATH}")

    # ---- MLflow experiment ----
    exp_name = f"signal_opt_{time.strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(exp_name)
    EXP_ID = mlflow.get_experiment_by_name(exp_name).experiment_id
    print(f"MLflow experiment: {exp_name} (ID: {EXP_ID})")

    # ---- Ray + Optuna ----
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0'
    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS,
             include_dashboard=False)

    algo = OptunaSearch(
        points_to_evaluate = INITIAL_PARAMS,
        metric             = "mean_improvement",
        mode               = "max",
    )
    algo = Repeater(algo, repeat=4)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                objective,
                model_run_id   = MODEL_RUN_ID,
                model_path     = MODEL_PATH,
                tracking_uri   = TRACKING_URI,
                data_file_path = DATA_FILE_PATH,
                axis_selection = AXIS_SELECTION,
                exp_id         = EXP_ID
            ),
            resources={"gpu": 1, "cpu": 1}
        ),
        param_space=SEARCH_CONFIG,
        tune_config=tune.TuneConfig(
            metric      = "mean_improvement",
            mode        = "max",
            search_alg  = algo,
            num_samples = NUM_TRIALS
        )
    )

    results = tuner.fit()

    best = results.get_best_result(metric="mean_improvement", mode="max")
    print("\n" + "="*60)
    print("BEST TRIAL")
    print("="*60)
    print(f"Mean improvement: {best.metrics['mean_improvement']:.4f}")
    print(f"Config: {best.config}")
    print("="*60)