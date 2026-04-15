"""
Signal Optimizer v5 — Position-Space Trajectory Correction
===========================================================

Changes from v4:
    Optimization variable changed from δa(t) to δx(t).
    δv and δa are derived from δx via finite differences:

        δv[t] = δx[t] - δx[t-1]   (prepend 0)
        δa[t] = δv[t] - δv[t-1]   (prepend 0)

    This eliminates the cumsum amplification problem entirely — gradients
    flow directly to δx without passing through two integrations.

    Initialization rewritten:
        'inverted_error': convert GRU prediction to mm, then to pos_norm,
            clamp to ±δ_max. No differentiation, no saturation.
        'gaussian':       band-limited noise as target δx, peak-normalized
            to fill_frac * δ_max. No cumsum involved.
        'zeros':          δx = 0, clean diagnostic baseline.

Loss (unchanged from v4):
    L = L_error + λ_x·L_x + λ_v·L_v + λ_a·L_a + λ_TV·L_TV

    L_error = (1/T') · ||δx_escale[warmup:] + f_GRU(x*)[warmup:]||²
        δx converted to e_scale_norm via pos_to_escale before summing.

    L_x  = mean( ReLU(|δx(t)| / δ_max - 1)² )
    L_v  = mean( ReLU(|v(t) + δv(t)| / v_max_norm - 1)² )
    L_a  = mean( ReLU(|a(t) + δa(t)| / a_max_norm - 1)² )
    L_TV = mean( |δv(t)| )   [= ||δx||_TV = total variation of δx]

Kinematic limits (physical → normalized per trial):
    δ_max  = MAX_DEVIATION_MM / POS_STD[axis]
    v_max  = MAX_VEL_MM_S     / VEL_STD[axis]
    a_max  = MAX_ACCEL_MM_S2  / ACCEL_STD[axis]

    Physical values:
        MAX_DEVIATION_MM = 0.5 mm
        MAX_VEL_MM_S     = 32000 mm/min = 533.33 mm/s
        MAX_ACCEL_MM_S2  = 8820000 mm/min² = 2450 mm/s²

Perturbed LSTM inputs (kinematically consistent):
    x*[t] = x[t] + δx[t]
    v*[t] = v[t] + δv[t]
    a*[t] = a[t] + δa[t]

Evaluation metric (HPO, maximized):
    improvement = ||x_escale + e_original||₂ / ||x_escale + δx_escale + e||₂

All signals in normalized units. Physical conversions only in plotting/logging.
"""

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
from scipy.signal import butter, filtfilt
from ray import tune
from ray.tune.search import ConcurrencyLimiter, Repeater
from ray.tune.search.optuna.optuna_search import OptunaSearch

matplotlib.use('Agg')

from train_val_loop_v10 import TimeSeriesDataset, SequenceModel, AXIS_COLUMNS


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_DEVIATION_MM = 0.5           # mm
MAX_VEL_MM_S     = 32000 / 60   # mm/s  (32000 mm/min)
MAX_ACCEL_MM_S2  = 8_820_000 / 3600  # mm/s²  (8,820,000 mm/min²)

WARMUP_FRAC      = 0          # 5% = 400 steps out of 8000

TIMESTEPS        = 8000
SAMPLE_SIZE      = 1000
MINI_BATCH       = 64
N_VAL_PLOTS      = 10

# Gaussian init bandpass filter
BUTTERWORTH_CUTOFF_HZ = [0.5, 40.0]
SIGNAL_SAMPLE_RATE    = 4000.0   # Hz

# Normalization statistics from combined_statistics.json
POS_STD = {
    'x': 247.2838848784503,
    'y': 140.93299970896865,
    'z': 112.96077495266569,
}
VEL_STD = {
    'x': 30.507020763512948,
    'y': 29.115152697380314,
    'z': 18.466082945337515,
}
ACCEL_STD = {
    'x': 146.27633804618486,
    'y': 142.96117504352614,
    'z': 115.67312792683349,
}
E_SCALE_STD = {
    'x': 0.0033232292485601647,
    'y': 0.0038927228408261755,
    'z': 0.002672943695112274,
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
    Frozen but kept in train() mode so cuDNN allows backward passes.
    Gradients flow to delta_x only.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run    = client.get_run(run_id)
    params = run.data.params

    print(f"Loading model from: {model_path}")
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(loaded, dict):
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
    model.train()  # required for cuDNN RNN backward

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
        X_full: (N, T, 3) normalized (pos, vel, accel)
        idxs:   (N,) dataset indices
    """
    dataset = TimeSeriesDataset(
        data_file_path        = data_file_path,
        target_timesteps      = target_timesteps,
        axis_selection        = axis_selection,
        include_velocity      = True,
        include_acceleration  = True,
        validate_divisibility = True,
    )
    total = len(dataset)
    rng   = np.random.default_rng(random_seed)
    idxs  = rng.choice(total, size=min(sample_size, total), replace=False)
    X_list = [dataset[i][0] for i in idxs]
    return torch.stack(X_list), idxs


# ============================================================================
# KINEMATICS  (delta_x is the optimization variable)
# ============================================================================

def derive_delta_v_a(
    delta_x: torch.Tensor,   # (B, T, 1) position perturbation
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derive velocity and acceleration perturbations from delta_x via
    finite differences. Prepends zero so all outputs are length T.

        δv[0] = δx[0],    δv[t] = δx[t] - δx[t-1]
        δa[0] = δv[0],    δa[t] = δv[t] - δv[t-1]

    Returns:
        delta_v: (B, T, 1)
        delta_a: (B, T, 1)
    """
    delta_v = torch.cat([delta_x[:, :1, :],
                         delta_x[:, 1:, :] - delta_x[:, :-1, :]], dim=1)
    delta_a = torch.cat([delta_v[:, :1, :],
                         delta_v[:, 1:, :] - delta_v[:, :-1, :]], dim=1)
    return delta_v, delta_a


def build_corrected_input(
    X_full:  torch.Tensor,   # (B, T, 3) original normalized (pos, vel, accel)
    delta_x: torch.Tensor,   # (B, T, 1)
    delta_v: torch.Tensor,   # (B, T, 1)
    delta_a: torch.Tensor,   # (B, T, 1)
) -> torch.Tensor:
    """Kinematically consistent corrected trajectory. Returns (B, T, 3)."""
    return torch.cat([
        X_full[:, :, 0:1] + delta_x,
        X_full[:, :, 1:2] + delta_v,
        X_full[:, :, 2:3] + delta_a,
    ], dim=-1)


# ============================================================================
# INITIALIZATION  (all methods produce delta_x directly)
# ============================================================================

def _butterworth_bandpass(signal_np: np.ndarray, cutoff_hz, fs: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass filter. Input/output: (T,)."""
    b, a = butter(4, cutoff_hz, btype='bandpass', fs=fs)
    return filtfilt(b, a, signal_np)


def initialize_delta_x(
    X_full:                   torch.Tensor,   # (B, T, 3) on device
    model:                    nn.Module,
    init_method:              str,            # 'inverted_error' | 'gaussian' | 'zeros'
    device:                   str,
    max_deviation_normalized: float,          # δ_max in pos_norm units
    e_scale_std:              float,          # E_SCALE_STD[axis] in mm
    pos_std:                  float,          # POS_STD[axis] in mm
    fill_frac:                float = 0.8,    # fraction of δ_max to use for gaussian
) -> torch.Tensor:
    """
    Initialize delta_x (the optimization variable) in pos_norm units.

    inverted_error:
        Use GRU prediction as the initial correction target.
        e_pred is in e_scale_norm units → convert to mm → negate (we want
        to cancel the error) → convert to pos_norm → clamp to ±δ_max.
        No differentiation, no cumsum, no saturation possible.

        e_mm        = e_pred * e_scale_std          [mm]
        target_dx   = -e_mm / pos_std               [pos_norm]
        delta_x     = clamp(target_dx, ±δ_max)

    gaussian:
        Band-limited noise as target δx, peak-normalized to fill_frac * δ_max.
        Smooth by construction (bandpass filtered), well-bounded by design.

    zeros:
        δx = 0 everywhere. Clean diagnostic baseline — improvement from step 0
        is purely due to gradient steps with no init bias.

    Returns: (B, T, 1) requires_grad=True, pos_norm units
    """
    B, T, _ = X_full.shape

    with torch.no_grad():

        if init_method == 'zeros':
            dx_init = torch.zeros(B, T, 1, device=device)

        elif init_method == 'inverted_error':
            try:
                e_pred  = model(X_full)                      # (B, T, 1) e_scale_norm
                e_mm    = e_pred[:, :, 0] * e_scale_std      # (B, T) mm
                # Negate: we want to shift position to cancel the predicted error
                target  = -e_mm / pos_std                    # (B, T) pos_norm
                dx_init = target.clamp(
                    -max_deviation_normalized,
                     max_deviation_normalized,
                ).unsqueeze(-1)                              # (B, T, 1)

                if torch.isnan(dx_init).any():
                    raise ValueError("NaN in inverted_error init")

            except Exception as ex:
                print(f"  inverted_error init failed ({ex}), falling back to gaussian")
                init_method = 'gaussian'

        if init_method == 'gaussian':
            noise_np = np.random.randn(B, T).astype(np.float32)
            dx_np    = np.zeros_like(noise_np)
            for b in range(B):
                dx_np[b] = _butterworth_bandpass(
                    noise_np[b], BUTTERWORTH_CUTOFF_HZ, SIGNAL_SAMPLE_RATE
                )
            dx_t  = torch.tensor(dx_np, device=device)        # (B, T)
            # Peak-normalize then scale to fill_frac * δ_max
            peak  = dx_t.abs().max(dim=1).values.clamp(min=1e-8)   # (B,)
            dx_t  = dx_t / peak.unsqueeze(1) * (max_deviation_normalized * fill_frac)
            dx_init = dx_t.unsqueeze(-1)                       # (B, T, 1)

    return dx_init.detach().clone().requires_grad_(True)


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def _relu_quadratic_penalty(
    signal: torch.Tensor,   # (B, T, 1)
    limit:  float,
) -> torch.Tensor:
    """
    Soft constraint: mean( ReLU(|signal| / limit - 1)² ).
    Zero inside [-limit, limit], quadratic outside. Dimensionless.
    """
    return torch.relu(signal.abs() / limit - 1.0).pow(2).mean()


def compute_loss(
    X_full:                   torch.Tensor,   # (B, T, 3)
    delta_x:                  torch.Tensor,   # (B, T, 1) optimization variable
    model:                    nn.Module,
    warmup_steps:             int,
    max_deviation_normalized: float,
    v_max_normalized:         float,
    a_max_normalized:         float,
    pos_to_escale:            float,          # POS_STD / E_SCALE_STD
    lambda_x:                 float,
    lambda_v:                 float,
    lambda_a:                 float,
    lambda_tv:                float,
) -> tuple[torch.Tensor, dict]:
    """
    L = L_error + λ_x·L_x + λ_v·L_v + λ_a·L_a + λ_TV·L_TV

    L_error: realized tracking error in e_scale_norm (post-warmup).
    L_x:     soft position constraint on δx.
    L_v:     soft velocity constraint on corrected v.
    L_a:     soft accel constraint on corrected a.
    L_TV:    total variation of δx (= L1 of δv), sparsity prior.
    """
    delta_v, delta_a = derive_delta_v_a(delta_x)

    gru_input = build_corrected_input(X_full, delta_x, delta_v, delta_a)
    e_opt     = model(gru_input)   # (B, T, 1) e_scale_norm

    # ---- Error: post-warmup, δx converted to e_scale_norm before summing ----
    dx_w       = delta_x[:, warmup_steps:, :]          # (B, T', 1) pos_norm
    e_w        = e_opt[:,   warmup_steps:, :]           # (B, T', 1) e_scale_norm
    error_loss = (dx_w * pos_to_escale + e_w).pow(2).mean()

    # ---- Position: |δx| ≤ δ_max ----
    pos_loss = _relu_quadratic_penalty(delta_x, max_deviation_normalized)

    # ---- Velocity: |v + δv| ≤ v_max ----
    vel_loss = _relu_quadratic_penalty(
        X_full[:, :, 1:2] + delta_v, v_max_normalized
    )

    # ---- Acceleration: |a + δa| ≤ a_max ----
    accel_loss = _relu_quadratic_penalty(
        X_full[:, :, 2:3] + delta_a, a_max_normalized
    )

    # ---- TV: L1 norm of δv = total variation of δx ----
    tv_loss = delta_v.pow(2).mean()

    loss = (error_loss
            + lambda_x  * pos_loss
            + lambda_v  * vel_loss
            + lambda_a  * accel_loss
            + lambda_tv * tv_loss)

    return loss, {
        'error': error_loss.item(),
        'pos':   pos_loss.item(),
        'vel':   vel_loss.item(),
        'accel': accel_loss.item(),
        'tv':    tv_loss.item(),
        'total': loss.item(),
    }


# ============================================================================
# EVALUATION METRIC
# ============================================================================

def compute_improvement(
    X_full:        torch.Tensor,   # (B, T, 3)
    delta_x:       torch.Tensor,   # (B, T, 1)
    baseline_pred: torch.Tensor,   # (B, T, 1) f_GRU(x)
    model:         nn.Module,
    warmup_steps:  int,
    pos_to_escale: float,
) -> torch.Tensor:
    """
    improvement = ||x_escale + e_original||₂ / ||x_escale + δx_escale + e||₂

    All terms in e_scale_norm. Ratio > 1 means error reduced.
    Returns (B,) tensor.
    """
    with torch.no_grad():
        delta_v, delta_a = derive_delta_v_a(delta_x)
        gru_input = build_corrected_input(X_full, delta_x, delta_v, delta_a)
        e_opt     = model(gru_input)   # (B, T, 1)

    x_escale  = X_full[:, warmup_steps:, 0] * pos_to_escale    # (B, T')
    dx_escale = delta_x[:, warmup_steps:, 0] * pos_to_escale   # (B, T')
    e0        = baseline_pred[:, warmup_steps:, 0]              # (B, T')
    e1        = e_opt[:, warmup_steps:, 0]                      # (B, T')

    original_norm  = (x_escale + e0).pow(2).mean(dim=1).sqrt()
    corrected_norm = (x_escale + dx_escale + e1).pow(2).mean(dim=1).sqrt()
    return original_norm / corrected_norm.clamp(min=1e-10)


# ============================================================================
# PER-BATCH OPTIMIZATION STATE
# ============================================================================

def init_batch(
    X_full:                   torch.Tensor,
    model:                    nn.Module,
    baseline_pred:            torch.Tensor,
    warmup_steps:             int,
    init_method:              str,
    inner_lr:                 float,
    device:                   str,
    max_deviation_normalized: float,
    e_scale_std:              float,
    pos_std:                  float,
    pos_to_escale:            float,
) -> tuple[torch.Tensor, torch.Tensor, torch.optim.Optimizer, float]:
    """
    Initialise delta_x and Adam for one mini-batch.

    Returns:
        delta_x:   (B, T, 1) requires_grad=True
        X_init:    (B, T, 3) corrected trajectory at initialization
        optimizer: Adam on [delta_x]
        init_impr: mean improvement before any gradient steps
    """
    delta_x = initialize_delta_x(
        X_full, model, init_method, device,
        max_deviation_normalized, e_scale_std, pos_std,
    )
    with torch.no_grad():
        delta_v, delta_a = derive_delta_v_a(delta_x)
        X_init    = build_corrected_input(X_full, delta_x, delta_v, delta_a)
        init_impr = compute_improvement(
            X_full, delta_x, baseline_pred, model, warmup_steps, pos_to_escale,
        )
    optimizer = torch.optim.Adam([delta_x], lr=inner_lr)
    return delta_x, X_init.detach(), optimizer, init_impr.mean().item()


def step_batch(
    X_full:                   torch.Tensor,
    delta_x:                  torch.Tensor,
    model:                    nn.Module,
    optimizer:                torch.optim.Optimizer,
    warmup_steps:             int,
    max_deviation_normalized: float,
    v_max_normalized:         float,
    a_max_normalized:         float,
    pos_to_escale:            float,
    lambda_x:                 float,
    lambda_v:                 float,
    lambda_a:                 float,
    lambda_tv:                float,
    print_grad_stats:         bool = False,
) -> tuple[dict, float]:
    """
    One gradient step on delta_x. Returns (loss_components, step_time).
    No gradient scaling needed — delta_x gradients are direct, no cumsum.
    """
    t0 = time.perf_counter()

    optimizer.zero_grad()
    loss, components = compute_loss(
        X_full, delta_x, model, warmup_steps,
        max_deviation_normalized, v_max_normalized, a_max_normalized,
        pos_to_escale, lambda_x, lambda_v, lambda_a, lambda_tv,
    )
    loss.backward()

    if print_grad_stats and delta_x.grad is not None:
        g = delta_x.grad.abs()
        print(f"    grad  max={g.max().item():.3e}  mean={g.mean().item():.3e}")

    optimizer.step()
    return components, time.perf_counter() - t0


# ============================================================================
# PLOTTING
# ============================================================================

def plot_optimization_results(
    x_pos:          np.ndarray,
    x_init:         np.ndarray,
    x_star:         np.ndarray,
    baseline_pred:  np.ndarray,
    opt_pred:       np.ndarray,
    sample_idx:     int,
    save_dir:       str,
    pos_std_mm:     float,
    e_scale_std_mm: float,
    warmup_steps:   int,
    mlflow_run:     bool = True,
):
    """3-subplot figure in physical units."""
    T, t, warmup = len(x_pos), np.arange(len(x_pos)), warmup_steps

    init_dev_mm      = (x_init - x_pos) * pos_std_mm
    opt_dev_mm       = (x_star - x_pos) * pos_std_mm
    baseline_pred_mm = baseline_pred * e_scale_std_mm
    opt_pred_mm      = opt_pred      * e_scale_std_mm

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Signal Optimization — Sample {sample_idx}',
                 fontsize=14, fontweight='bold')

    axes[0].plot(t, x_pos,  'b-',  lw=1.2, label='Commanded x',        alpha=0.9)
    axes[0].plot(t, x_init, 'g--', lw=1.0, label='Initialized x_init', alpha=0.7)
    axes[0].plot(t, x_star, 'r--', lw=1.2, label='Optimized x*',       alpha=0.8)
    axes[0].axvspan(0, warmup, alpha=0.08, color='gray', label='Warmup')
    axes[0].set_ylabel('Position (normalized)', fontweight='bold')
    axes[0].set_title('Commanded, Initialized, and Optimized Trajectory')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, init_dev_mm, 'g-',  lw=1.0, label='x_init - x', alpha=0.7)
    axes[1].plot(t, opt_dev_mm,  'r--', lw=1.2, label='x* - x',     alpha=0.8)
    axes[1].axhline(0,                 color='k',   lw=0.6, ls='--', alpha=0.4)
    #axes[1].axhline( MAX_DEVIATION_MM, color='red', lw=0.8, ls=':',  alpha=0.6,
    #                label=f'+/-{MAX_DEVIATION_MM} mm limit')
    #axes[1].axhline(-MAX_DEVIATION_MM, color='red', lw=0.8, ls=':',  alpha=0.6)
    axes[1].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[1].set_ylabel('Deviation from commanded (mm)', fontweight='bold')
    axes[1].set_title('Position Deviations')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, baseline_pred_mm, 'b-',  lw=1.2,
                 label='f_GRU(x)  baseline',  alpha=0.8)
    axes[2].plot(t, opt_pred_mm,      'r--', lw=1.2,
                 label='f_GRU(x*) optimized', alpha=0.8)
    axes[2].axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    axes[2].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[2].set_ylabel('Predicted e_scale (mm)', fontweight='bold')
    axes[2].set_title('Baseline vs Optimized Predicted Error')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    wb           = baseline_pred_mm[warmup:]
    wo           = opt_pred_mm[warmup:]
    base_rmse_um = np.sqrt(np.mean(wb ** 2)) * 1000
    opt_rmse_um  = np.sqrt(np.mean(wo ** 2)) * 1000
    impr_pct     = (base_rmse_um - opt_rmse_um) / (base_rmse_um + 1e-10) * 100
    max_dev      = np.max(np.abs(opt_dev_mm))

    axes[2].set_xlabel(
        f'Timestep\nBaseline RMSE: {base_rmse_um:.2f} um  |  '
        f'Optimized RMSE: {opt_rmse_um:.2f} um  |  '
        f'Improvement: {impr_pct:.1f}%  |  '
        f'Max dev: {max_dev:.4f} mm',
        fontsize=8
    )
    plt.tight_layout()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fname = save_path / f'opt_sample_{sample_idx:04d}.png'
    if mlflow_run:
        mlflow.log_figure(fig, f'{save_dir}/opt_sample_{sample_idx:04d}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# OBJECTIVE
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
    Ray Tune objective. Optimizes delta_x for SAMPLE_SIZE signals.
    Reports mean_improvement (maximized).

    Searched hyperparameters:
        lambda_x, lambda_v, lambda_a, lambda_tv  — loss weights
        inner_lr                                  — Adam lr for delta_x
        n_iterations                              — gradient steps

    Fixed (FIXED_TRIAL_PARAMS):
        init_method, print_grad_stats
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    trial_params = {**FIXED_TRIAL_PARAMS, **trial_params}

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    trial_idx = trial_params.get('__trial_index__', 0)

    # ---- Normalized kinematic limits ----
    axis_key                 = axis_selection if axis_selection != 'all' else 'x'
    max_deviation_normalized = MAX_DEVIATION_MM / POS_STD[axis_key]
    v_max_normalized         = MAX_VEL_MM_S    / VEL_STD[axis_key]
    a_max_normalized         = MAX_ACCEL_MM_S2 / ACCEL_STD[axis_key]
    pos_to_escale            = POS_STD[axis_key] / E_SCALE_STD[axis_key]
    e_scale_std              = E_SCALE_STD[axis_key]
    pos_std                  = POS_STD[axis_key]

    # ---- Unpack trial params ----
    lambda_x         = float(trial_params['lambda_x'])
    lambda_v         = float(trial_params['lambda_v'])
    lambda_a         = float(trial_params['lambda_a'])
    lambda_tv        = float(trial_params['lambda_tv'])
    inner_lr         = float(trial_params['inner_lr'])
    n_iterations     = int(trial_params['n_iterations'])
    init_method      = trial_params.get('init_method',      'inverted_error')
    print_grad_stats = bool(trial_params.get('print_grad_stats', False))

    warmup_steps = int(TIMESTEPS * WARMUP_FRAC)

    # ---- MLflow ----
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.start_run(experiment_id=str(exp_id), log_system_metrics=False)
    mean_improvement = 0.0

    try:
        mlflow.set_tag('run_type',            'signal_optimizer_v5')
        mlflow.set_tag('source_model_run_id', model_run_id)

        logged = {k: v for k, v in trial_params.items() if k != '__trial_index__'}
        mlflow.log_params(logged)
        mlflow.log_params({
            'max_deviation_mm':         MAX_DEVIATION_MM,
            'max_vel_mm_s':             MAX_VEL_MM_S,
            'max_accel_mm_s2':          MAX_ACCEL_MM_S2,
            'max_deviation_normalized': max_deviation_normalized,
            'v_max_normalized':         v_max_normalized,
            'a_max_normalized':         a_max_normalized,
            'pos_to_escale':            pos_to_escale,
            'pos_std_mm':               pos_std,
            'axis_selection':           axis_selection,
            'trial_idx':                trial_idx,
            'warmup_steps':             warmup_steps,
            'sample_size':              SAMPLE_SIZE,
        })

        # ---- Load model ----
        model, _ = load_model(model_run_id, tracking_uri, model_path, device)

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

        # ---- Baseline predictions ----
        baseline_preds = []
        with torch.no_grad():
            for s in range(0, N, MINI_BATCH):
                xb = X_full[s:min(s + MINI_BATCH, N)].to(device)
                baseline_preds.append(model(xb).cpu())
        baseline_all = torch.cat(baseline_preds, dim=0)   # (N, T, 1)

        # ---- Initialise delta_x and Adam per batch ----
        batch_slices = [(s, min(s + MINI_BATCH, N)) for s in range(0, N, MINI_BATCH)]
        batch_states = []
        init_imprs   = []

        for start, end in batch_slices:
            xb = X_full[start:end].to(device)
            bp = baseline_all[start:end].to(device)
            delta_x, X_init_b, opt, init_impr = init_batch(
                X_full                   = xb,
                model                    = model,
                baseline_pred            = bp,
                warmup_steps             = warmup_steps,
                init_method              = init_method,
                inner_lr                 = inner_lr,
                device                   = device,
                max_deviation_normalized = max_deviation_normalized,
                e_scale_std              = e_scale_std,
                pos_std                  = pos_std,
                pos_to_escale            = pos_to_escale,
            )
            batch_states.append({
                'X_full':        xb,
                'baseline_pred': bp,
                'delta_x':       delta_x,
                'optimizer':     opt,
                'X_init':        X_init_b,
            })
            init_imprs.append(init_impr)

        mean_init_improvement = float(np.mean(init_imprs))
        mlflow.log_metric('step_mean_improvement', mean_init_improvement,
                          step=0, synchronous=False)
        print(f"  Init improvement: {mean_init_improvement:.4f}")

        # ---- Optimization loop ----
        all_step_times = []

        for step in range(n_iterations):
            step_imprs      = []
            step_times      = []
            step_components = {k: [] for k in ('error','pos','vel','accel','tv','total')}

            for state in batch_states:
                components, t = step_batch(
                    X_full                   = state['X_full'],
                    delta_x                  = state['delta_x'],
                    model                    = model,
                    optimizer                = state['optimizer'],
                    warmup_steps             = warmup_steps,
                    max_deviation_normalized = max_deviation_normalized,
                    v_max_normalized         = v_max_normalized,
                    a_max_normalized         = a_max_normalized,
                    pos_to_escale            = pos_to_escale,
                    lambda_x                 = lambda_x,
                    lambda_v                 = lambda_v,
                    lambda_a                 = lambda_a,
                    lambda_tv                = lambda_tv,
                    print_grad_stats         = print_grad_stats,
                )
                with torch.no_grad():
                    impr = compute_improvement(
                        state['X_full'], state['delta_x'], state['baseline_pred'],
                        model, warmup_steps, pos_to_escale,
                    )
                step_imprs.append(impr.mean().item())
                step_times.append(t)
                for k, v in components.items():
                    step_components[k].append(v)

            all_step_times.extend(step_times)
            mean_step_impr = float(np.mean(step_imprs))

            mlflow.log_metric('step_mean_improvement', mean_step_impr,
                              step=step + 1, synchronous=False)
            for k, vals in step_components.items():
                mlflow.log_metric(f'loss_{k}', float(np.mean(vals)),
                                  step=step + 1, synchronous=False)

            if step % 20 == 0 or step == n_iterations - 1:
                print(f"  Step {step+1:4d}/{n_iterations} | "
                      f"impr={mean_step_impr:.4f} | "
                      f"err={np.mean(step_components['error']):.3e} | "
                      f"pos={np.mean(step_components['pos']):.3e} | "
                      f"vel={np.mean(step_components['vel']):.3e} | "
                      f"accel={np.mean(step_components['accel']):.3e} | "
                      f"tv={np.mean(step_components['tv']):.3e} | "
                      f"t={np.mean(step_times)*1000:.1f}ms")

        # ---- Collect final results ----
        all_improvements = []
        all_X_star       = []
        all_X_init       = []
        all_opt_pred     = []

        for state in batch_states:
            with torch.no_grad():
                delta_v, delta_a = derive_delta_v_a(state['delta_x'])
                X_star_b   = build_corrected_input(
                    state['X_full'], state['delta_x'], delta_v, delta_a
                )
                opt_pred_b = model(X_star_b)
                impr       = compute_improvement(
                    state['X_full'], state['delta_x'], state['baseline_pred'],
                    model, warmup_steps, pos_to_escale,
                )
            all_improvements.append(impr.cpu())
            all_X_star.append(X_star_b.cpu())
            all_X_init.append(state['X_init'].cpu())
            all_opt_pred.append(opt_pred_b.cpu())

        all_improvements = torch.cat(all_improvements, dim=0)
        all_X_star       = torch.cat(all_X_star,       dim=0)
        all_X_init       = torch.cat(all_X_init,       dim=0)
        all_opt_pred     = torch.cat(all_opt_pred,      dim=0)

        mean_improvement = all_improvements.mean().item()

        baseline_rmse_um = float(
            baseline_all[:, warmup_steps:, 0].pow(2).mean().sqrt().item()
            * e_scale_std * 1000
        )
        opt_rmse_um = float(
            all_opt_pred[:, warmup_steps:, 0].pow(2).mean().sqrt().item()
            * e_scale_std * 1000
        )

        mlflow.log_metrics({
            'mean_improvement':      mean_improvement,
            'std_improvement':       all_improvements.std().item(),
            'median_improvement':    all_improvements.median().item(),
            'pct_improved':          (all_improvements > 1.0).float().mean().item(),
            'init_mean_improvement': mean_init_improvement,
            'baseline_rmse_um':      baseline_rmse_um,
            'optimized_rmse_um':     opt_rmse_um,
            'rmse_reduction_um':     baseline_rmse_um - opt_rmse_um,
            'mean_step_time_ms':     float(np.mean(all_step_times)) * 1000,
            'total_step_time_s':     float(np.sum(all_step_times)),
        })

        # ---- Plots ----
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
                sample_idx     = int(sample_idxs[pi]),
                save_dir       = plot_dir,
                pos_std_mm     = pos_std,
                e_scale_std_mm = e_scale_std,
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
# SEARCH SPACE
# ============================================================================

SEARCH_CONFIG = {
    'lambda_x':     tune.loguniform(1e0,  1e6),
    'lambda_v':     tune.loguniform(1e0,  1e6),
    'lambda_a':     tune.loguniform(1e0,  1e6),
    'lambda_tv':    tune.loguniform(1e-3, 1e3),
    'inner_lr':     tune.loguniform(1e-5, 1e-1),
    'n_iterations': tune.randint(100, 1001),
    'init_method':  tune.choice(['inverted_error','zeros','gaussian']),
}

INITIAL_PARAMS = [{
    'lambda_x':     1e4,
    'lambda_v':     1e2,
    'lambda_a':     1e2,
    'lambda_tv':    1e0,
    'inner_lr':     1e-3,
    'n_iterations': 200,
    'init_method':      'zeros',
}]

FIXED_TRIAL_PARAMS = {
    #'init_method':      'inverted_error',
    'print_grad_stats': False,
}


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':

    # ============================================================
    # CONFIGURATION
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

    exp_name = f'signal_opt_v5_{time.strftime("%Y%m%d_%H%M%S")}'
    mlflow.set_experiment(exp_name)
    EXP_ID = mlflow.get_experiment_by_name(exp_name).experiment_id
    print(f'MLflow experiment: {exp_name} (ID: {EXP_ID})')

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