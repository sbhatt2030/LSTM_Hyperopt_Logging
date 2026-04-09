"""
Signal Optimizer — Acceleration-Space Trajectory Correction
============================================================

Optimization variable: δa(t), shape (B, T, 1) per batch.

Kinematic recovery (inside computation graph):
    δv[k] = cumsum(δa)[k]
    δx[k] = cumsum(δv)[k]      (= z(t), position deviation)

Perturbed LSTM inputs (kinematically consistent by construction):
    x*[k] = x[k] + δx[k]
    v*[k] = v[k] + δv[k]
    a*[k] = a[k] + δa[k]

Loss:
    L = λ1 · L_fidelity + λ2 · L_disp + λ3 · L_smooth

    L_fidelity = (1/T') · ||δx[warmup:] + e(x*, v*, a*)[warmup:]||_2^2
                 Drives realized position x + δx + e toward x.
                 Combines correction and error into one term — they are not
                 independent and separating them creates conflicting gradients.

    L_disp     = (δx[-1])^2
                 Squared final position drift. Prevents the optimizer from
                 exploiting cumulative drift. λ2 >> λ1.

    L_smooth   = (1/T') · ||diff(δa)[warmup:]||_2^2   (optional)
                 Penalizes jerk in δa. Add if δa is empirically spiky.

Hard constraint (post-step projection):
    δx clamped to ±MAX_DEVIATION_MM / POS_STD[axis] after each step.
    Applied in torch.no_grad() — not differentiable.

Evaluation metric (HPO, maximized):
    improvement = ||x + e_original||_2 / ||x + δx + e||_2
    Ratio > 1 means optimization reduced predicted tracking error.

All signals in the optimization are in normalized units (z-score).
Physical unit conversions happen only in plotting and MLflow logging.
MAX_DEVIATION_MM is the single source of truth for the constraint.
"""

import math
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import ray
import torch
import torch.nn as nn
from pathlib import Path
from scipy.signal import butter, filtfilt, savgol_filter
from ray import tune
from ray.tune.search import ConcurrencyLimiter, Repeater
from ray.tune.search.optuna.optuna_search import OptunaSearch

matplotlib.use('Agg')

from train_val_loop_v10 import TimeSeriesDataset, SequenceModel, AXIS_COLUMNS


# ============================================================================
# CONSTANTS
# ============================================================================

# Physical constraint in mm — single source of truth.
# Converted to normalized units once per trial: MAX_DEVIATION_MM / POS_STD[axis]
MAX_DEVIATION_MM  = 100

# Fraction of timesteps excluded from loss and metric (LSTM warmup)
WARMUP_FRAC       = 0.05     # 5% = 400 steps out of 8000

# Data and sampling
TIMESTEPS         = 8000
SAMPLE_SIZE       = 1000     # signals sampled from dataset per trial
MINI_BATCH        = 64       # signals processed together on GPU

# Plotting
N_VAL_PLOTS       = 10

# Initialization parameters
BUTTERWORTH_CUTOFF_HZ = [0.5,40.0]
SIGNAL_SAMPLE_RATE    = 4000.0   # Hz
SAVGOL_WINDOW         = 101
SAVGOL_POLY           = 3

# Normalization statistics from combined_statistics.json
# All .npy data is z-score normalized. Physical units recovered by:
#   pos_mm      = pos_normalized      * POS_STD[axis]   (mean negligible for deviations)
#   e_scale_mm  = e_scale_normalized  * E_SCALE_STD[axis]
#   accel_norm  = accel_physical      / ACCEL_STD[axis]
# Constraint in normalized units: MAX_DEVIATION_MM / POS_STD[axis]
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


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(
    run_id:       str,
    tracking_uri: str,
    model_path:   str,
    device:       str = 'cuda',
) -> tuple[SequenceModel, dict]:
    """
    Load a trained SequenceModel from disk.

    The .pth file is expected to contain the full model object (saved with
    mlflow.pytorch.log_model). Architecture params are also fetched from
    MLflow for logging purposes.

    The model is frozen (requires_grad=False on all params) but kept in
    train() mode so cuDNN allows backward passes through the RNN.
    Gradients flow to delta_a, not to model parameters.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run    = client.get_run(run_id)
    params = run.data.params

    print(f"Loading model from: {model_path}")
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(loaded, dict):
        # State dict only — construct architecture from MLflow params
        model = SequenceModel(
            model_type    = params['model_type'],
            input_size    = int(params['input_size']),
            hidden_size   = int(params['hidden_size']),
            num_layers    = int(params['num_layers']),
            output_size   = int(params['output_size']),
            dropout       = float(params['dropout']),
            bidirectional = params.get('bidirectional', 'False') == 'True',
        )
        model.load_state_dict(loaded)
        model.to(device)
    else:
        # Full model object — architecture is authoritative
        model = loaded
        model.to(device)
        params = {
            **params,
            'model_type':  model.model_type,
            'hidden_size': str(model.hidden_size),
            'num_layers':  str(model.num_layers),
        }

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # train() required for cuDNN RNN backward — gradients only flow to delta_a
    model.train()

    print(f"  {params['model_type'].upper()} "
          f"{params['num_layers']}L x {params['hidden_size']}H  "
          f"in={params['input_size']} out={params['output_size']}  "
          f"device={device}")
    return model, params


# ============================================================================
# DATA LOADING
# ============================================================================

def load_signal_batch(
    data_file_path:   str,
    axis_selection:   str,
    target_timesteps: int,
    sample_size:      int,
    random_seed:      int,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load dataset and randomly sample `sample_size` trajectories.

    Returns:
        X_full: (N, T, 3) normalized (pos, vel, accel) for the selected axis
        idxs:   (N,) dataset indices of sampled signals
    """
    dataset = TimeSeriesDataset(
        data_file_path       = data_file_path,
        target_timesteps     = target_timesteps,
        axis_selection       = axis_selection,
        include_velocity     = True,
        include_acceleration = True,
        validate_divisibility= True,
    )

    total = len(dataset)
    rng   = np.random.default_rng(random_seed)
    idxs  = rng.choice(total, size=min(sample_size, total), replace=False)

    X_list = []
    for i in idxs:
        x, _ = dataset[i]
        X_list.append(x)

    X_full = torch.stack(X_list)   # (N, T, 3)
    return X_full, idxs


# ============================================================================
# KINEMATIC INTEGRATION
# ============================================================================

def integrate_delta_a(
    delta_a:                  torch.Tensor,   # (B, T, 1) normalized accel perturbation
    max_deviation_normalized: float,          # = MAX_DEVIATION_MM / POS_STD[axis]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate delta_a -> delta_v -> delta_x via cumsum.

    delta_x is clamped elementwise to +/-max_deviation_normalized.
    delta_v and delta_a are back-derived from the clamped delta_x so all
    three channels remain kinematically consistent after projection.

    Returns:
        delta_x:      (B, T, 1) position perturbation, clamped
        delta_v:      (B, T, 1) velocity perturbation, consistent with delta_x
        delta_a_proj: (B, T, 1) accel perturbation, consistent with delta_x
    """
    delta_v_raw = torch.cumsum(delta_a,     dim=1)
    delta_x_raw = torch.cumsum(delta_v_raw, dim=1)

    delta_x = delta_x_raw.clamp(-max_deviation_normalized, max_deviation_normalized)

    # Back-derive from clamped delta_x for full kinematic consistency
    delta_v      = torch.cat([delta_x[:, :1, :],
                               delta_x[:, 1:, :] - delta_x[:, :-1, :]], dim=1)
    delta_a_proj = torch.cat([delta_v[:, :1, :],
                               delta_v[:, 1:, :] - delta_v[:, :-1, :]], dim=1)

    return delta_x, delta_v, delta_a_proj


def build_corrected_input(
    X_full:  torch.Tensor,   # (B, T, 3) original normalized (pos, vel, accel)
    delta_x: torch.Tensor,   # (B, T, 1)
    delta_v: torch.Tensor,   # (B, T, 1)
    delta_a: torch.Tensor,   # (B, T, 1)
) -> torch.Tensor:
    """
    Add kinematic perturbations to original trajectory.

    Returns:
        (B, T, 3): (x + delta_x, v + delta_v, a + delta_a)
    """
    return torch.cat([
        X_full[:, :, 0:1] + delta_x,
        X_full[:, :, 1:2] + delta_v,
        X_full[:, :, 2:3] + delta_a,
    ], dim=-1)


# ============================================================================
# INITIALIZATION
# ============================================================================

def _butterworth_lowpass(signal_np: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass filter. Input/output: (T,)."""
    b, a = butter(4, cutoff_hz, btype='bandpass',fs=fs)
    return filtfilt(b, a, signal_np)


def _savgol_diff(signal_np: np.ndarray, window: int, poly: int) -> np.ndarray:
    """First derivative via Savitzky-Golay. Input/output: (T,)."""
    return savgol_filter(signal_np, window_length=window, polyorder=poly, deriv=1)


def initialize_delta_a(
    X_full:                   torch.Tensor,   # (B, T, 3) normalized, on device
    model:                    nn.Module,
    init_method:              str,
    device:                   str,
    max_deviation_normalized: float,
    e_scale_to_accel_ratio:   float,          # E_SCALE_STD[axis] / ACCEL_STD[axis]
) -> torch.Tensor:
    """
    Initialize delta_a in normalized acceleration units.

    inverted_error (preferred):
        1. Predict e = f_LSTM(x, v, a)                [normalized e_scale units]
        2. Low-pass filter e at BUTTERWORTH_CUTOFF_HZ  [remove noise before diff]
        3. Differentiate twice with Savitzky-Golay     [position -> acceleration space]
        4. Scale by E_SCALE_STD / ACCEL_STD            [e_scale -> accel units]
        5. Integrate, clamp delta_x, back-project delta_a [enforce feasibility]

        SavGol is used for differentiation (not finite differences) because it
        fits a local polynomial and is far less sensitive to noise.
        The 40 Hz Butterworth pre-filter removes high-frequency content that
        would be amplified by differentiation.
        This pipeline matches the training distribution.

        Falls back to gaussian if NaN values are produced.

    gaussian (fallback):
        Random Gaussian passed through the same Butterworth filter.
        Scaled so the integrated delta_x has std ~ max_deviation_normalized / 3.

    Returns:
        delta_a: (B, T, 1) requires_grad=True, normalized acceleration units
    """
    B, T, _ = X_full.shape

    with torch.no_grad():
        da_init = None

        if init_method == 'inverted_error':
            try:
                # Step 1: predict error on original trajectory
                e_pred = model(X_full)          # (B, T, 1) normalized e_scale
                e_np   = e_pred[:, :, 0].cpu().numpy()   # (B, T)

                # Steps 2-4: filter then double-differentiate each signal
                da_np = np.zeros_like(e_np)
                for b in range(B):
                    filtered  = _butterworth_lowpass(e_np[b], BUTTERWORTH_CUTOFF_HZ, SIGNAL_SAMPLE_RATE)
                    d1        = _savgol_diff(filtered, SAVGOL_WINDOW, SAVGOL_POLY)
                    d2        = _savgol_diff(d1,       SAVGOL_WINDOW, SAVGOL_POLY)
                    da_np[b]  = d2 * e_scale_to_accel_ratio

                da_init = torch.tensor(da_np, dtype=torch.float32, device=device).unsqueeze(-1)

                if torch.isnan(da_init).any():
                    raise ValueError("NaN in inverted_error init")

                # Step 5: integrate, clamp delta_x, back-project to keep feasible
                dv = torch.cumsum(da_init, dim=1)
                dx = torch.cumsum(dv,      dim=1)
                dx_clamped = dx.clamp(-max_deviation_normalized, max_deviation_normalized)
                dv_re  = torch.cat([dx_clamped[:, :1, :],
                                    dx_clamped[:, 1:, :] - dx_clamped[:, :-1, :]], dim=1)
                da_init = torch.cat([dv_re[:, :1, :],
                                     dv_re[:, 1:, :] - dv_re[:, :-1, :]], dim=1)

            except Exception as ex:
                print(f"  inverted_error init failed ({ex}), falling back to gaussian")
                da_init = None

        if da_init is None:
            # Gaussian fallback: band-limited noise, scaled by integrated delta_x std
            noise_np = np.random.randn(B, T).astype(np.float32)
            da_np    = np.zeros_like(noise_np)
            for b in range(B):
                da_np[b] = _butterworth_lowpass(noise_np[b], BUTTERWORTH_CUTOFF_HZ, SIGNAL_SAMPLE_RATE)
            da_init = torch.tensor(da_np, device=device).unsqueeze(-1)
            dv      = torch.cumsum(da_init, dim=1)
            dx      = torch.cumsum(dv,      dim=1)
            dx_std  = dx.std().clamp(min=1e-8)
            da_init = da_init * (max_deviation_normalized / 3.0) / dx_std

    delta_a = da_init.detach().clone().requires_grad_(True)
    return delta_a


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def compute_loss(
    X_full:                   torch.Tensor,   # (B, T, 3) original trajectory
    delta_a:                  torch.Tensor,   # (B, T, 1) accel perturbation
    baseline_pred:            torch.Tensor,   # (B, T, 1) f_LSTM(x, v, a)
    model:                    nn.Module,
    warmup_steps:             int,
    lambda_fidelity:          float,
    lambda_disp:              float,
    lambda_smooth:            float,
    max_deviation_normalized: float,
) -> tuple[torch.Tensor, dict]:
    """
    Three-term loss over the current delta_a.

    L_fidelity = (1/T') * ||delta_x[warmup:] + e(x*, v*, a*)[warmup:]||_2^2
        The sum delta_x + e is how far the realized position deviates from the
        commanded trajectory: machine executes x + delta_x with predicted error
        e, so realized = x + delta_x + e. Minimizing ||delta_x + e|| drives
        realized toward x. Collapsing error and position penalty into one term
        avoids the conflicting gradients that arise from treating them separately.

    L_disp = (delta_x[-1])^2
        Squared final position drift — a single scalar, not a norm over time.
        lambda_disp >> lambda_fidelity to make this effectively binding and
        prevent the optimizer from exploiting one-directional cumulative drift.

    L_smooth = (1/T') * ||diff(delta_a)[warmup:]||_2^2
        Penalizes jerk in delta_a. Keeps LSTM inputs in-distribution by
        preventing high-frequency acceleration content. Set lambda_smooth=0
        to disable and add only if delta_a is empirically spiky.

    All per-time terms divided by T' for trajectory-length invariance.
    """
    delta_x, delta_v, delta_a_proj = integrate_delta_a(delta_a, max_deviation_normalized)
    gru_input = build_corrected_input(X_full, delta_x, delta_v, delta_a_proj)
    e_opt     = model(gru_input)   # (B, T, 1) predicted error on corrected trajectory

    # Post-warmup slices
    dx = delta_x[:, warmup_steps:, :]   # (B, T', 1)
    e  = e_opt[:,   warmup_steps:, :]   # (B, T', 1)
    da = delta_a[:, warmup_steps:, :]   # (B, T', 1)

    # Fidelity: ||delta_x + e||^2 / T'
    fidelity_loss = (dx + e).pow(2).mean()

    # Displacement: squared final delta_x value (scalar per signal, averaged over batch)
    disp_loss = delta_x[:, -1, :].pow(2).mean()

    # Smoothness: ||diff(delta_a)||^2 / T'
    da_diff     = da[:, 1:, :] - da[:, :-1, :]
    smooth_loss = da_diff.pow(2).mean()

    loss = (lambda_fidelity * fidelity_loss
            + lambda_disp   * disp_loss
            + lambda_smooth * smooth_loss)

    components = {
        'fidelity':         fidelity_loss.item(),
        'disp':             disp_loss.item(),
        'smooth':           smooth_loss.item(),
        'total':            loss.item(),
        'contrib_fidelity': (lambda_fidelity * fidelity_loss).item(),
        'contrib_disp':     (lambda_disp     * disp_loss).item(),
        'contrib_smooth':   (lambda_smooth   * smooth_loss).item(),
    }
    return loss, components


# ============================================================================
# EVALUATION METRIC
# ============================================================================

def compute_improvement(
    X_full:                   torch.Tensor,   # (B, T, 3)
    delta_a:                  torch.Tensor,   # (B, T, 1)
    baseline_pred:            torch.Tensor,   # (B, T, 1) f_LSTM(x, v, a)
    model:                    nn.Module,
    warmup_steps:             int,
    max_deviation_normalized: float,
) -> torch.Tensor:
    """
    Per-signal improvement ratio (HPO objective, maximized).

    improvement = ||x + e_original||_2 / ||x + delta_x + e||_2

    x is included so the metric penalizes both large uncorrected errors
    and large corrections that don't meaningfully help.

    Ratio > 1: optimization reduced predicted tracking error.
    Ratio = 1: no change.
    Ratio < 1: optimization made things worse.

    Returns:
        (B,) tensor of improvement ratios
    """
    with torch.no_grad():
        delta_x, delta_v, delta_a_proj = integrate_delta_a(delta_a, max_deviation_normalized)
        gru_input = build_corrected_input(X_full, delta_x, delta_v, delta_a_proj)
        e_opt     = model(gru_input)   # (B, T, 1)

    x   = X_full[:, warmup_steps:, 0]         # (B, T')
    e0  = baseline_pred[:, warmup_steps:, 0]  # (B, T')
    dx  = delta_x[:, warmup_steps:, 0]        # (B, T')
    e1  = e_opt[:, warmup_steps:, 0]          # (B, T')

    original_norm  = (x + e0).pow(2).mean(dim=1).sqrt()       # (B,)
    corrected_norm = (x + dx + e1).pow(2).mean(dim=1).sqrt()  # (B,)

    denom = corrected_norm.clamp(min=1e-10)
    return original_norm / denom   # (B,)


# ============================================================================
# PER-BATCH OPTIMIZATION STATE
# ============================================================================

def init_batch(
    X_full:                   torch.Tensor,   # (B, T, 3) on device
    model:                    nn.Module,
    baseline_pred:            torch.Tensor,   # (B, T, 1) on device
    warmup_steps:             int,
    init_method:              str,
    inner_lr:                 float,
    device:                   str,
    max_deviation_normalized: float,
    e_scale_to_accel_ratio:   float,
) -> tuple[torch.Tensor, torch.Tensor, torch.optim.Optimizer, float]:
    """
    Initialise delta_a and Adam optimizer for one mini-batch.
    Also computes the improvement at step 0 for MLflow logging.

    Returns:
        delta_a:   (B, T, 1) requires_grad=True
        X_init:    (B, T, 3) corrected trajectory at initialization (for plotting)
        optimizer: Adam on [delta_a]
        init_impr: mean improvement ratio before any gradient steps (scalar)
    """
    delta_a = initialize_delta_a(
        X_full, model, init_method, device,
        max_deviation_normalized, e_scale_to_accel_ratio,
    )
    with torch.no_grad():
        dx, dv, da_proj = integrate_delta_a(delta_a, max_deviation_normalized)
        X_init    = build_corrected_input(X_full, dx, dv, da_proj)
        init_impr = compute_improvement(
            X_full, delta_a, baseline_pred, model, warmup_steps, max_deviation_normalized,
        )
    optimizer = torch.optim.Adam([delta_a], lr=inner_lr)
    return delta_a, X_init.detach(), optimizer, init_impr.mean().item()


def step_batch(
    X_full:                   torch.Tensor,
    delta_a:                  torch.Tensor,
    baseline_pred:            torch.Tensor,
    model:                    nn.Module,
    optimizer:                torch.optim.Optimizer,
    warmup_steps:             int,
    lambda_fidelity:          float,
    lambda_disp:              float,
    lambda_smooth:            float,
    max_deviation_normalized: float,
) -> tuple[float, dict, float]:
    """
    Run one gradient step on delta_a for a single mini-batch.

    Returns:
        impr:       mean improvement ratio after this step
        components: loss breakdown dict (raw terms + weighted contributions)
        step_time:  wall time in seconds
    """
    t0 = time.perf_counter()

    optimizer.zero_grad()
    loss, components = compute_loss(
        X_full, delta_a, baseline_pred, model,
        warmup_steps, lambda_fidelity, lambda_disp, lambda_smooth,
        max_deviation_normalized,
    )
    loss.backward()
    optimizer.step()

    impr = compute_improvement(
        X_full, delta_a, baseline_pred, model,
        warmup_steps, max_deviation_normalized,
    )
    return impr.mean().item(), components, time.perf_counter() - t0


# ============================================================================
# PLOTTING
# ============================================================================

def plot_optimization_results(
    x_pos:          np.ndarray,
    x_init:         np.ndarray,
    x_star:         np.ndarray,
    baseline_pred:  np.ndarray,
    opt_pred:       np.ndarray,
    delta_a:        np.ndarray,   # (T,) optimized accel perturbation, normalized
    sample_idx:     int,
    save_dir:       str,
    pos_std_mm:     float,
    e_scale_std_mm: float,
    accel_std:      float,        # ACCEL_STD[axis] — converts normalized accel -> mm/s^2
    warmup_steps:   int,
    mlflow_run:     bool = True,
):
    """
    4-subplot figure. All inputs in normalized units; physical conversions here.

    Subplot 1: Commanded, initialized, optimized position (normalized)
    Subplot 2: Position deviations (x_init - x, x* - x) in mm with +-limit lines
    Subplot 3: Predicted e_scale baseline vs optimized in mm
    Subplot 4: Optimized delta_a in mm/s^2

    Stats annotation: RMSE in um and max deviations in mm.
    """
    T      = len(x_pos)
    t      = np.arange(T)
    warmup = warmup_steps

    # Denormalize to physical units
    init_dev_mm      = (x_init - x_pos) * pos_std_mm
    opt_dev_mm       = (x_star - x_pos) * pos_std_mm
    baseline_pred_mm = baseline_pred * e_scale_std_mm
    opt_pred_mm      = opt_pred      * e_scale_std_mm
    delta_a_physical = delta_a       * accel_std        # mm/s^2

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f'Signal Optimization — Sample {sample_idx}', fontsize=14, fontweight='bold')

    # ---- Subplot 1: Trajectories (normalized position) ----
    axes[0].plot(t, x_pos,  'b-',  lw=1.2, label='Commanded x',        alpha=0.9)
    axes[0].plot(t, x_init, 'g--', lw=1.0, label='Initialized x_init', alpha=0.7)
    axes[0].plot(t, x_star, 'r--', lw=1.2, label='Optimized x*',       alpha=0.8)
    axes[0].axvspan(0, warmup, alpha=0.08, color='gray', label='Warmup (excluded)')
    axes[0].set_ylabel('Position (normalized)', fontweight='bold')
    axes[0].set_title('Commanded, Initialized, and Optimized Trajectory')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # ---- Subplot 2: Deviations in mm ----
    axes[1].plot(t, init_dev_mm, 'g-',  lw=1.0, label='x_init - x', alpha=0.7)
    axes[1].plot(t, opt_dev_mm,  'r--', lw=1.2, label='x* - x',     alpha=0.8)
    axes[1].axhline(0,                 color='k',   lw=0.6, ls='--', alpha=0.4)
    axes[1].axhline( MAX_DEVIATION_MM, color='red', lw=0.8, ls=':',  alpha=0.6,
                    label=f'+/-{MAX_DEVIATION_MM} mm limit')
    axes[1].axhline(-MAX_DEVIATION_MM, color='red', lw=0.8, ls=':',  alpha=0.6)
    axes[1].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[1].set_ylabel('Deviation from commanded (mm)', fontweight='bold')
    axes[1].set_title('Position Deviations')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # ---- Subplot 3: Predicted e_scale in mm ----
    axes[2].plot(t, baseline_pred_mm, 'b-',  lw=1.2, label='f_LSTM(x)  baseline',  alpha=0.8)
    axes[2].plot(t, opt_pred_mm,      'r--', lw=1.2, label='f_LSTM(x*) optimized', alpha=0.8)
    axes[2].axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    axes[2].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[2].set_ylabel('Predicted e_scale (mm)', fontweight='bold')
    axes[2].set_title('Baseline vs Optimized Predicted Error')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # ---- Subplot 4: delta_a in mm/s^2 ----
    axes[3].plot(t, delta_a_physical, 'm-', lw=1.0, label='δa (optimized)', alpha=0.8)
    axes[3].axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    axes[3].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[3].set_ylabel('δa (mm/s²)', fontweight='bold')
    axes[3].set_title('Optimized Acceleration Perturbation')
    axes[3].legend(loc='upper right', fontsize=9)
    axes[3].grid(True, alpha=0.3)

    # Stats annotation (post-warmup only, physical units)
    wb            = baseline_pred_mm[warmup:]
    wo            = opt_pred_mm[warmup:]
    base_rmse_um  = np.sqrt(np.mean(wb ** 2)) * 1000
    opt_rmse_um   = np.sqrt(np.mean(wo ** 2)) * 1000
    impr_pct      = (base_rmse_um - opt_rmse_um) / (base_rmse_um + 1e-10) * 100
    max_opt_dev   = np.max(np.abs(opt_dev_mm))
    max_init_dev  = np.max(np.abs(init_dev_mm))

    stats = (f'Baseline RMSE: {base_rmse_um:.2f} um  |  '
             f'Optimized RMSE: {opt_rmse_um:.2f} um  |  '
             f'Improvement: {impr_pct:.1f}%  |  '
             f'Max opt dev: {max_opt_dev:.4f} mm  |  '
             f'Max init dev: {max_init_dev:.4f} mm')
    axes[3].set_xlabel(f'Timestep\n{stats}', fontsize=8)

    plt.tight_layout()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fname = save_path / f'opt_sample_{sample_idx:04d}.png'

    if mlflow_run:
        mlflow.log_figure(fig, f'{save_dir}/opt_sample_{sample_idx:04d}.png')

    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# OBJECTIVE (called by Ray Tune)
# ============================================================================

def objective(
    trial_params:   dict,
    model_run_id:   str = None,
    model_path:     str = None,
    tracking_uri:   str = None,
    data_file_path: str = None,
    axis_selection: str = 'x',
    exp_id:         int = 0,
):
    """
    Ray Tune objective. Optimizes delta_a for SAMPLE_SIZE signals and
    reports mean_improvement to Optuna (maximized).

    Loop structure: epoch-first, batch-second.
        Initialise delta_a and Adam for every batch.
        for step in n_iterations:
            for batch in batches:
                one gradient step on that batch's delta_a

    Each signal has its own delta_a and Adam optimizer and is optimized
    independently. The epoch-first structure ensures the step-wise improvement
    curve in MLflow is meaningful (all signals at step k before any reach k+1).

    MLflow run lifecycle: start_run -> try/finally -> end_run guaranteed even
    on KeyboardInterrupt or if the run is deleted from the UI mid-trial.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    trial_idx = trial_params.get('__trial_index__', 0)

    # ---- Unit conversions ----
    # All optimization in normalized units.
    # MAX_DEVIATION_MM is the single physical-unit source of truth.
    axis_key                 = axis_selection if axis_selection != 'all' else 'x'
    max_deviation_normalized = MAX_DEVIATION_MM / POS_STD[axis_key]
    e_scale_to_accel_ratio   = E_SCALE_STD[axis_key] / ACCEL_STD[axis_key]
    e_scale_std              = E_SCALE_STD[axis_key]

    # ---- Unpack hyperparameters ----
    lambda_fidelity = trial_params['lambda_fidelity']
    lambda_disp     = trial_params['lambda_disp']
    lambda_smooth   = trial_params['lambda_smooth']
    inner_lr        = trial_params['inner_lr']
    n_iterations    = int(trial_params['n_iterations'])
    init_method     = trial_params['init_method']

    warmup_steps = int(TIMESTEPS * WARMUP_FRAC)

    # ---- MLflow run ----
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.start_run(experiment_id=str(exp_id), log_system_metrics=False)
    mean_improvement = 0.0   # default reported if trial fails before completing

    try:
        mlflow.set_tag('run_type',            'signal_optimizer')
        mlflow.set_tag('source_model_run_id', model_run_id)
        mlflow.set_tag('model_path',          model_path)

        logged_params = {k: v for k, v in trial_params.items() if k != '__trial_index__'}
        mlflow.log_params(logged_params)
        mlflow.log_param('max_deviation_mm',         MAX_DEVIATION_MM)
        mlflow.log_param('max_deviation_normalized', max_deviation_normalized)
        mlflow.log_param('pos_std_mm',               POS_STD[axis_key])
        mlflow.log_param('axis_selection',           axis_selection)
        mlflow.log_param('trial_idx',                trial_idx)
        mlflow.log_param('warmup_steps',             warmup_steps)
        mlflow.log_param('sample_size',              SAMPLE_SIZE)

        # ---- Load model ----
        model, model_params = load_model(model_run_id, tracking_uri, model_path, device)

        # ---- Load data ----
        trial_seed = int(torch.randint(0, 2**31, (1,)).item())
        mlflow.log_param('trial_seed', trial_seed)

        X_full, sample_idxs = load_signal_batch(
            data_file_path   = data_file_path,
            axis_selection   = axis_selection,
            target_timesteps = TIMESTEPS,
            sample_size      = SAMPLE_SIZE,
            random_seed      = trial_seed,
        )
        N = X_full.shape[0]
        mlflow.log_param('n_signals', N)

        # ---- Precompute baseline predictions ----
        baseline_preds = []
        with torch.no_grad():
            for start in range(0, N, MINI_BATCH):
                end = min(start + MINI_BATCH, N)
                xb  = X_full[start:end].to(device)
                baseline_preds.append(model(xb).cpu())
        baseline_all = torch.cat(baseline_preds, dim=0)   # (N, T, 1)

        # ---- Initialise delta_a and Adam for every batch ----
        batch_slices = [(s, min(s + MINI_BATCH, N)) for s in range(0, N, MINI_BATCH)]
        batch_states = []
        init_imprs   = []

        for start, end in batch_slices:
            xb = X_full[start:end].to(device)
            bp = baseline_all[start:end].to(device)
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
            batch_states.append({
                'X_full':        xb,
                'baseline_pred': bp,
                'delta_a':       delta_a,
                'optimizer':     opt,
                'X_init':        X_init_b,
            })
            init_imprs.append(init_impr)

        mean_init_improvement = float(np.mean(init_imprs))
        # Log at step 0 so init sits on the same curve as per-step improvement
        mlflow.log_metric('step_mean_improvement', mean_init_improvement,
                          step=0, synchronous=False)
        print(f"  Init improvement: {mean_init_improvement:.4f}")

        # ---- Epoch loop: one gradient step per epoch across all batches ----
        all_step_times = []

        for step in range(n_iterations):
            step_imprs      = []
            step_times      = []
            step_components = []

            for state in batch_states:
                impr, components, t = step_batch(
                    X_full                   = state['X_full'],
                    delta_a                  = state['delta_a'],
                    baseline_pred            = state['baseline_pred'],
                    model                    = model,
                    optimizer                = state['optimizer'],
                    warmup_steps             = warmup_steps,
                    lambda_fidelity          = lambda_fidelity,
                    lambda_disp              = lambda_disp,
                    lambda_smooth            = lambda_smooth,
                    max_deviation_normalized = max_deviation_normalized,
                )
                step_imprs.append(impr)
                step_times.append(t)
                step_components.append(components)

            all_step_times.extend(step_times)
            mean_step_impr = float(np.mean(step_imprs))
            # step+1 so curve runs: 0 (init), 1, 2, ..., n_iterations
            mlflow.log_metric('step_mean_improvement', mean_step_impr,
                              step=step + 1, synchronous=False)
            for key in step_components[0]:
                mlflow.log_metric(
                    f'loss_{key}',
                    float(np.mean([c[key] for c in step_components])),
                    step=step + 1,
                    synchronous=False,
                )

            if step % 20 == 0 or step == n_iterations - 1:
                print(f"  Step {step+1:4d}/{n_iterations} | "
                      f"mean_impr={mean_step_impr:.4f} | "
                      f"avg_step={np.mean(step_times)*1000:.1f}ms")

        # ---- Collect final optimized signals ----
        all_improvements = []
        all_X_star       = []   # (N, T, 3) corrected trajectories
        all_X_init       = []   # (N, T, 3) initialized trajectories
        all_opt_pred     = []   # (N, T, 1) predicted error on x*
        all_delta_a      = []   # (N, T, 1) optimized acceleration perturbation

        for state in batch_states:
            with torch.no_grad():
                dx, dv, da_proj = integrate_delta_a(state['delta_a'], max_deviation_normalized)
                X_star_b        = build_corrected_input(state['X_full'], dx, dv, da_proj)
                opt_pred_b      = model(X_star_b)
                impr            = compute_improvement(
                    state['X_full'], state['delta_a'], state['baseline_pred'],
                    model, warmup_steps, max_deviation_normalized,
                )
            all_improvements.append(impr.cpu())
            all_X_star.append(X_star_b.cpu())
            all_X_init.append(state['X_init'].cpu())
            all_opt_pred.append(opt_pred_b.cpu())
            all_delta_a.append(state['delta_a'].detach().cpu())

        all_improvements = torch.cat(all_improvements, dim=0)    # (N,)
        all_X_star       = torch.cat(all_X_star,       dim=0)    # (N, T, 3)
        all_X_init       = torch.cat(all_X_init,       dim=0)    # (N, T, 3)
        all_opt_pred     = torch.cat(all_opt_pred,      dim=0)   # (N, T, 1)
        all_delta_a      = torch.cat(all_delta_a,       dim=0)   # (N, T, 1)

        mean_improvement = all_improvements.mean().item()
        std_improvement  = all_improvements.std().item()

        # RMSE in um: normalized e_scale -> mm -> um
        baseline_rmse_um = float(
            baseline_all[:, warmup_steps:, 0].pow(2).mean().sqrt().item() * e_scale_std * 1000
        )
        opt_rmse_um = float(
            all_opt_pred[:, warmup_steps:, 0].pow(2).mean().sqrt().item() * e_scale_std * 1000
        )

        mlflow.log_metric('mean_improvement',      mean_improvement)
        mlflow.log_metric('std_improvement',       std_improvement)
        mlflow.log_metric('median_improvement',    all_improvements.median().item())
        mlflow.log_metric('pct_improved',          (all_improvements > 0).float().mean().item())
        mlflow.log_metric('init_mean_improvement', mean_init_improvement)
        mlflow.log_metric('baseline_rmse_um',      baseline_rmse_um)
        mlflow.log_metric('optimized_rmse_um',     opt_rmse_um)
        mlflow.log_metric('rmse_reduction_um',     baseline_rmse_um - opt_rmse_um)
        mlflow.log_metric('mean_step_time_ms',     np.mean(all_step_times) * 1000)
        mlflow.log_metric('total_step_time_s',     np.sum(all_step_times))

        # ---- Validation plots ----
        rng       = np.random.default_rng(trial_seed)
        plot_idxs = rng.choice(N, size=min(N_VAL_PLOTS, N), replace=False)
        plot_dir  = f'plots/trial_{trial_idx}'

        for pi in plot_idxs:
            plot_optimization_results(
                x_pos          = X_full[pi, :, 0].numpy(),
                x_init         = all_X_init[pi, :, 0].numpy(),
                x_star         = all_X_star[pi, :, 0].numpy(),
                baseline_pred  = baseline_all[pi, :, 0].numpy(),
                opt_pred       = all_opt_pred[pi, :, 0].numpy(),
                delta_a        = all_delta_a[pi, :, 0].numpy(),
                sample_idx     = int(sample_idxs[pi]),
                save_dir       = plot_dir,
                pos_std_mm     = POS_STD[axis_key],
                e_scale_std_mm = E_SCALE_STD[axis_key],
                accel_std      = ACCEL_STD[axis_key],
                warmup_steps   = warmup_steps,
                mlflow_run     = True,
            )

    except Exception as e:
        print(f"  Trial {trial_idx} failed: {type(e).__name__}: {e}")
        try:
            mlflow.set_tag('trial_status', f'failed: {type(e).__name__}')
        except Exception:
            pass
        raise

    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass

    tune.report({'mean_improvement': mean_improvement})


# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

SEARCH_CONFIG = {
    'lambda_fidelity': tune.loguniform(1e-3, 1e0),
    'lambda_disp':     tune.loguniform(1e6,  1e7),
    'lambda_smooth':   tune.loguniform(1e0, 1e5),
    'inner_lr':        tune.loguniform(1e-7, 1e-5),
    'n_iterations':    tune.randint(100, 201),
    'init_method':     tune.choice(['inverted_error', 'gaussian']),
}

INITIAL_PARAMS = [{
    'lambda_fidelity': 0.1,
    'lambda_disp':     1e6,
    'lambda_smooth':   1e3,
    'inner_lr':        1e-7,
    'n_iterations':    100,
    'init_method':     'inverted_error',
}]


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':

    # ============================================================
    # CONFIGURATION — edit these values directly
    # ============================================================
    TRACKING_URI   = 'http://hurco-lambda.hurco.com:8090'
    MODEL_RUN_ID   = 'af3f1aaf616843e7b15a9f0637ae6131'
    MODEL_PATH     = '/home/bhatts/LSTM_Hyperopt_Logging/~/mlartifacts/23/models/m-675fe5ff09694de99805a64483630bcc/artifacts/data/model.pth'
    DATA_FILE_PATH = 'data/clean_final.npy'
    AXIS_SELECTION = 'x'
    NUM_TRIALS     = 100
    NUM_CPUS       = 4
    NUM_GPUS       = 4
    # ============================================================

    # ---- Verify model before launching Ray ----
    print(f'Verifying model run {MODEL_RUN_ID}...')
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        run    = client.get_run(MODEL_RUN_ID)
        print(f'  Found run: {run.info.run_id} (status: {run.info.status})')
    except mlflow.exceptions.RestException as e:
        print(f'\nERROR: Could not find run {MODEL_RUN_ID!r} at {TRACKING_URI}')
        print(f'  {e}')
        sys.exit(1)
    pth = Path(MODEL_PATH)
    if not pth.exists():
        print(f'\nERROR: model.pth not found at: {pth}')
        sys.exit(1)
    MODEL_PATH = str(pth.resolve())
    print(f'  model.pth found: {MODEL_PATH}')

    # ---- MLflow experiment ----
    exp_name = f'signal_opt_{time.strftime("%Y%m%d_%H%M%S")}'
    mlflow.set_experiment(exp_name)
    EXP_ID = mlflow.get_experiment_by_name(exp_name).experiment_id
    print(f'MLflow experiment: {exp_name} (ID: {EXP_ID})')

    # ---- Ray + Optuna ----
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0'
    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS, include_dashboard=False)

    algo = OptunaSearch(
        points_to_evaluate = INITIAL_PARAMS,
        metric             = 'mean_improvement',
        mode               = 'max',
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
                exp_id         = EXP_ID,
            ),
            resources={'gpu': 1, 'cpu': 1},
        ),
        param_space = SEARCH_CONFIG,
        tune_config = tune.TuneConfig(
            metric      = 'mean_improvement',
            mode        = 'max',
            search_alg  = algo,
            num_samples = NUM_TRIALS,
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric='mean_improvement', mode='max')
    print('\n' + '='*60)
    print('BEST TRIAL')
    print('='*60)
    print(f"Mean improvement: {best.metrics['mean_improvement']:.4f}")
    print(f'Config: {best.config}')
    print('='*60)