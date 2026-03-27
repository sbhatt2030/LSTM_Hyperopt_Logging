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
    init_method:     gaussian | inverted_error

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

# Position std per axis (from combined_statistics.json) for normalization
# MAX_DEVIATION in normalized units = MAX_DEVIATION_MM / pos_std
POS_STD = {
    'x': 247.2838848784503,
    'y': 140.93299970896865,
    'z': 112.96077495266569,
}
TIMESTEPS     = 8000     # fixed
SAMPLE_SIZE   = 1000     # signals sampled from dataset per trial
MINI_BATCH    = 64       # GPU mini-batch size
N_VAL_PLOTS   = 10       # number of validation plots
GAUSSIAN_CUTOFF_HZ   = 50.0
SIGNAL_SAMPLE_RATE   = 1000.0  # Hz — adjust if different


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

    model = SequenceModel(
        model_type    = params['model_type'],
        input_size    = int(params['input_size']),
        hidden_size   = int(params['hidden_size']),
        num_layers    = int(params['num_layers']),
        output_size   = int(params['output_size']),
        dropout       = float(params['dropout']),
        bidirectional = params.get('bidirectional', 'False') == 'True'
    )

    # The .pth file contains the full model object (not just state dict)
    print(f"Loading weights from: {model_path}")
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both full model and state dict
    if isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        # Full model object saved — extract its state dict
        model.load_state_dict(loaded.state_dict())

    model.to(device)
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

    X_full = torch.stack(X_list)   # (N, T, 3)  — pos, vel, accel precomputed
    X_pos  = X_full[:, :, 0:1]    # (N, T, 1)  — position only, the optimizable signal

    return X_pos, X_full, idxs


# ============================================================================
# SAVGOL DIFFERENTIABLE WRAPPER
# ============================================================================

def savgol_torch(x: torch.Tensor, window: int = SAVGOL_WINDOW, poly: int = SAVGOL_POLY) -> torch.Tensor:
    """
    Apply Savitzky-Golay filter as a differentiable 1D convolution.

    SavGol is a linear FIR filter — we precompute the coefficients once
    and apply as a grouped conv1d so gradients flow back to x.

    Args:
        x:      (B, T, 1) tensor
        window: filter window length (must be odd)
        poly:   polynomial order

    Returns:
        (B, T, 1) smoothed tensor
    """
    from scipy.signal import savgol_coeffs

    coeffs = savgol_coeffs(window, poly, deriv=0)
    coeffs_tensor = torch.tensor(
        coeffs, dtype=x.dtype, device=x.device
    ).flip(0).view(1, 1, -1)   # (out_channels=1, in_channels=1, kernel_size)

    # (B, T, 1) -> (B, 1, T) for conv1d
    x_t   = x.permute(0, 2, 1)
    pad   = window // 2
    x_pad = torch.nn.functional.pad(x_t, (pad, pad), mode='reflect')
    out   = torch.nn.functional.conv1d(x_pad, coeffs_tensor, groups=1)
    return out.permute(0, 2, 1)   # back to (B, T, 1)


def compute_velocity(x: torch.Tensor) -> torch.Tensor:
    """First derivative via SavGol. Returns (B, T, 1)."""
    from scipy.signal import savgol_coeffs
    coeffs = savgol_coeffs(SAVGOL_WINDOW, SAVGOL_POLY, deriv=1)
    coeffs_tensor = torch.tensor(
        coeffs, dtype=x.dtype, device=x.device
    ).flip(0).view(1, 1, -1)
    x_t   = x.permute(0, 2, 1)
    pad   = SAVGOL_WINDOW // 2
    x_pad = torch.nn.functional.pad(x_t, (pad, pad), mode='reflect')
    out   = torch.nn.functional.conv1d(x_pad, coeffs_tensor, groups=1)
    return out.permute(0, 2, 1)


def compute_acceleration(x: torch.Tensor) -> torch.Tensor:
    """Second derivative via SavGol. Returns (B, T, 1)."""
    from scipy.signal import savgol_coeffs
    coeffs = savgol_coeffs(SAVGOL_WINDOW, SAVGOL_POLY, deriv=2)
    coeffs_tensor = torch.tensor(
        coeffs, dtype=x.dtype, device=x.device
    ).flip(0).view(1, 1, -1)
    x_t   = x.permute(0, 2, 1)
    pad   = SAVGOL_WINDOW // 2
    x_pad = torch.nn.functional.pad(x_t, (pad, pad), mode='reflect')
    out   = torch.nn.functional.conv1d(x_pad, coeffs_tensor, groups=1)
    return out.permute(0, 2, 1)


def build_gru_input(x_pos: torch.Tensor) -> torch.Tensor:
    """
    Build the 3-channel input (pos, vel, accel) expected by the GRU.

    Args:
        x_pos: (B, T, 1) position signal

    Returns:
        (B, T, 3) tensor
    """
    vel   = compute_velocity(x_pos)
    accel = compute_acceleration(x_pos)
    return torch.cat([x_pos, vel, accel], dim=-1)   # (B, T, 3)


# ============================================================================
# INITIALIZATION
# ============================================================================

def bandlimit_gaussian(shape: tuple, cutoff_hz: float, sample_rate: float, device, max_deviation: float = MAX_DEVIATION_MM) -> torch.Tensor:
    """
    Generate bandlimited Gaussian noise via FFT low-pass filtering.

    Args:
        shape:         (B, T, 1)
        cutoff_hz:     maximum frequency in Hz
        sample_rate:   signal sample rate in Hz
        max_deviation: clamp limit in normalized units

    Returns:
        (B, T, 1) bandlimited noise, NOT yet clamped
    """
    B, T, _ = shape
    noise   = torch.randn(B, T, device=device)

    # FFT, zero out above cutoff, IFFT
    freqs   = torch.fft.rfftfreq(T, d=1.0/sample_rate).to(device)
    mask    = (freqs <= cutoff_hz).float()
    noise_f = torch.fft.rfft(noise, dim=1)
    noise_f = noise_f * mask.unsqueeze(0)
    noise   = torch.fft.irfft(noise_f, n=T, dim=1)

    # Normalise to unit std then scale to fit within ±0.5mm
    std = noise.std(dim=1, keepdim=True).clamp(min=1e-8)
    noise = noise / std * (max_deviation / 3.0)   # 3-sigma within bounds

    return noise.unsqueeze(-1)   # (B, T, 1)


def initialize_z(
    x_pos:         torch.Tensor,
    model:         nn.Module,
    init_method:   str,
    device:        str,
    max_deviation: float = MAX_DEVIATION_MM
) -> torch.Tensor:
    """
    Initialize the perturbation variable z.

    gaussian:        bandlimited Gaussian noise
    inverted_error:  z = -GRU(x), so first step tries to cancel the error

    Returns:
        z: (B, T, 1) requires_grad=True tensor
    """
    B, T, _ = x_pos.shape

    with torch.no_grad():
        if init_method == 'gaussian':
            z_init = bandlimit_gaussian(
                (B, T, 1), GAUSSIAN_CUTOFF_HZ, SIGNAL_SAMPLE_RATE, device,
                max_deviation=max_deviation
            )

        elif init_method == 'inverted_error':
            gru_in  = build_gru_input(x_pos)
            pred_err = model(gru_in)          # (B, T, 1)
            z_init   = -pred_err.clone()
            z_init   = z_init.clamp(-max_deviation, max_deviation)

        else:
            raise ValueError(f"Unknown init_method: {init_method}")

    z = z_init.detach().clone().requires_grad_(True)
    return z


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
    x_pos:           torch.Tensor,   # (B, T, 1) commanded
    z:               torch.Tensor,   # (B, T, 1) perturbation
    baseline_pred:   torch.Tensor,   # (B, T, 1) GRU(x) precomputed
    model:           nn.Module,
    warmup_steps:    int,
    error_type:      str,
    fidelity_type:   str,
    smooth_type:     str,
    lambda_error:    float,
    lambda_fidelity: float,
    lambda_smooth:   float,
) -> tuple[torch.Tensor, dict]:
    """
    Compute composite loss for current z.

    Returns:
        loss:       scalar tensor with gradients
        components: dict of individual loss values (detached floats)
    """
    # Build optimized signal
    x_star = x_pos + z                              # (B, T, 1)
    delta  = z                                       # = x* - x

    # Build GRU input and get prediction
    gru_in   = build_gru_input(x_star)              # (B, T, 3)
    opt_pred = model(gru_in)                         # (B, T, 1)

    # Apply warmup mask — exclude first warmup_steps
    bp  = baseline_pred[:, warmup_steps:, :]
    op  = opt_pred[:,      warmup_steps:, :]
    xp  = x_pos[:,         warmup_steps:, :]
    xsp = x_star[:,        warmup_steps:, :]
    dp  = delta[:,         warmup_steps:, :]

    # Error term: how much did predicted error change
    err_loss  = apply_norm(op - bp,  error_type)

    # Fidelity term: how far did we deviate from commanded
    fid_loss  = apply_norm(xsp - xp, fidelity_type)

    # Smoothness term
    smo_loss  = smoothness_loss(dp, smooth_type)

    loss = (lambda_error    * err_loss
          + lambda_fidelity * fid_loss
          + lambda_smooth   * smo_loss)

    components = {
        'error':     err_loss.item(),
        'fidelity':  fid_loss.item(),
        'smoothness': smo_loss.item(),
        'total':      loss.item()
    }

    return loss, components


# ============================================================================
# VALIDATION METRIC
# ============================================================================

def compute_improvement(
    x_pos:        torch.Tensor,   # (B, T, 1)
    x_star:       torch.Tensor,   # (B, T, 1)
    baseline_pred:torch.Tensor,   # (B, T, 1) GRU(x)
    opt_pred:     torch.Tensor,   # (B, T, 1) GRU(x*)
    warmup_steps: int
) -> torch.Tensor:
    """
    Compute per-signal improvement ratio.

    baseline_RMSE  = RMSE(GRU(x))              over [warmup:end]
    optimized_RMSE = RMSE(x - x* - GRU(x*))   over [warmup:end]
    improvement    = (baseline_RMSE - optimized_RMSE) / baseline_RMSE

    Returns:
        (B,) tensor of improvement ratios
    """
    bp = baseline_pred[:, warmup_steps:, 0]   # (B, T')
    xp = x_pos[:,        warmup_steps:, 0]
    xs = x_star[:,       warmup_steps:, 0]
    op = opt_pred[:,     warmup_steps:, 0]

    baseline_rmse  = bp.pow(2).mean(dim=1).sqrt()                # (B,)
    optimized_rmse = (xp - xs - op).pow(2).mean(dim=1).sqrt()   # (B,)

    # Avoid division by zero
    denom = baseline_rmse.clamp(min=1e-10)
    improvement = (baseline_rmse - optimized_rmse) / denom       # (B,)

    return improvement


# ============================================================================
# INNER OPTIMIZATION LOOP (per mini-batch)
# ============================================================================

def optimize_batch(
    x_pos:           torch.Tensor,   # (B, T, 1) on device
    model:           nn.Module,
    baseline_pred:   torch.Tensor,   # (B, T, 1) GRU(x), precomputed
    warmup_steps:    int,
    inner_steps:     int,
    inner_lr:        float,
    error_type:      str,
    fidelity_type:   str,
    smooth_type:     str,
    lambda_error:    float,
    lambda_fidelity: float,
    lambda_smooth:   float,
    init_method:     str,
    device:          str,
    max_deviation:   float = MAX_DEVIATION_MM,
    verbose:         bool  = False
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """
    Run inner optimization loop for a mini-batch of signals.

    Returns:
        x_star:      (B, T, 1) optimized signals
        opt_pred:    (B, T, 1) GRU(x*)
        step_times:  list of per-step wall times in seconds
    """
    # Initialize perturbation
    z = initialize_z(x_pos, model, init_method, device, max_deviation)  # (B, T, 1)

    optimizer = torch.optim.Adam([z], lr=inner_lr)
    step_times = []

    for step in range(inner_steps):
        t0 = time.perf_counter()

        optimizer.zero_grad()
        loss, components = compute_loss(
            x_pos, z, baseline_pred, model,
            warmup_steps, error_type, fidelity_type, smooth_type,
            lambda_error, lambda_fidelity, lambda_smooth
        )
        loss.backward()
        optimizer.step()

        # Projected gradient — clamp z back to feasible set
        with torch.no_grad():
            z.clamp_(-max_deviation, max_deviation)
            x_star_step   = x_pos + z
            opt_pred_step = model(build_gru_input(x_star_step))
            impr = compute_improvement(x_pos, x_star_step, baseline_pred, opt_pred_step, warmup_steps)
        mlflow.log_metric("step_mean_improvement", impr.mean().item(), step=step, synchronous=False)

        step_time = time.perf_counter() - t0
        step_times.append(step_time)

        if verbose and (step % 10 == 0 or step == inner_steps - 1):
            print(f"    step {step+1:3d}/{inner_steps} | "
                  f"total={components['total']:.6f} "
                  f"err={components['error']:.6f} "
                  f"fid={components['fidelity']:.6f} "
                  f"smo={components['smoothness']:.6f} "
                  f"({step_time*1000:.1f}ms)")

    # Final optimized signal
    with torch.no_grad():
        x_star = x_pos + z
        gru_in  = build_gru_input(x_star)
        opt_pred = model(gru_in)

    return x_star.detach(), opt_pred.detach(), step_times


# ============================================================================
# PLOTTING
# ============================================================================

def plot_optimization_results(
    x_pos:         np.ndarray,   # (T,) commanded position
    x_star:        np.ndarray,   # (T,) optimized position
    baseline_pred: np.ndarray,   # (T,) GRU(x)
    opt_pred:      np.ndarray,   # (T,) GRU(x*)
    sample_idx:    int,
    save_dir:      str,
    mlflow_run:    bool = True
):
    """
    Plot optimization results for a single signal.

    3 subplots:
        1. Commanded x and optimized x*
        2. Initial predicted error GRU(x) and optimized predicted error GRU(x*)
        3. Deviation signal x* - x
    """
    T = len(x_pos)
    t = np.arange(T)
    warmup = int(T * WARMUP_FRAC)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Signal Optimization — Sample {sample_idx}', fontsize=14, fontweight='bold')

    # --- Subplot 1: Commanded vs Optimized ---
    axes[0].plot(t, x_pos,  'b-',  linewidth=1.2, label='Commanded x',   alpha=0.8)
    axes[0].plot(t, x_star, 'r--', linewidth=1.2, label='Optimized x*',  alpha=0.8)
    axes[0].axvspan(0, warmup, alpha=0.08, color='gray', label='Warmup (excluded)')
    axes[0].set_ylabel('Position (normalized)', fontweight='bold')
    axes[0].set_title('Commanded vs Optimized Trajectory')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Subplot 2: Predicted errors ---
    axes[1].plot(t, baseline_pred, 'g-',  linewidth=1.2, label='GRU(x) — initial pred error',   alpha=0.8)
    axes[1].plot(t, opt_pred,      'm--', linewidth=1.2, label='GRU(x*) — optimized pred error', alpha=0.8)
    axes[1].axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
    axes[1].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[1].set_ylabel('Predicted Error (normalized)', fontweight='bold')
    axes[1].set_title('Initial vs Optimized Predicted Error')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # --- Subplot 3: Deviation ---
    deviation = x_star - x_pos
    axes[2].plot(t, deviation, 'orange', linewidth=1.2, label='x* - x (deviation)')
    axes[2].axhline(0,              color='k',   linewidth=0.6, linestyle='--', alpha=0.4)
    axes[2].axhline( MAX_DEVIATION_MM, color='red', linewidth=0.8, linestyle=':',  alpha=0.6, label=f'±{MAX_DEVIATION_MM}mm limit (normalized)')
    axes[2].axhline(-MAX_DEVIATION_MM, color='red', linewidth=0.8, linestyle=':',  alpha=0.6)
    axes[2].axvspan(0, warmup, alpha=0.08, color='gray')
    axes[2].set_ylabel('Deviation (mm)', fontweight='bold')
    axes[2].set_xlabel('Timestep', fontweight='bold')
    axes[2].set_title('Deviation Signal (x* - x)')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Stats annotation
    w_base  = baseline_pred[warmup:]
    w_opt   = opt_pred[warmup:]
    w_x     = x_pos[warmup:]
    w_xs    = x_star[warmup:]
    base_rmse = np.sqrt(np.mean(w_base**2))
    opt_rmse  = np.sqrt(np.mean((w_x - w_xs - w_opt)**2))
    impr      = (base_rmse - opt_rmse) / (base_rmse + 1e-10)
    max_dev   = np.max(np.abs(deviation))

    stats = (f'Baseline RMSE: {base_rmse:.6f}  |  '
             f'Optimized RMSE: {opt_rmse:.6f}  |  '
             f'Improvement: {impr*100:.2f}%  |  '
             f'Max deviation: {max_dev:.4f}mm')
    fig.text(0.5, 0.005, stats, ha='center', fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fname = save_path / f'opt_sample_{sample_idx:04d}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')

    if mlflow_run:
        # log_artifact uploads the file to the run's artifact store,
        # organised under a subfolder matching save_dir
        mlflow.log_artifact(str(fname), artifact_path=save_dir)

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

    # Compute MAX_DEVIATION in normalized units for this axis
    axis_key = axis_selection if axis_selection != 'all' else 'x'
    max_deviation = MAX_DEVIATION_MM / POS_STD[axis_key]

    # ---- MLflow run ----
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.start_run(experiment_id=str(exp_id), log_system_metrics=False)

    # Tag this run so it can be filtered separately from training runs
    mlflow.set_tag("run_type",            "signal_optimizer")
    mlflow.set_tag("source_model_run_id", model_run_id)
    mlflow.set_tag("model_path",          model_path)

    # Log all hyperparams
    logged_params = {k: v for k, v in trial_params.items() if k != '__trial_index__'}
    mlflow.log_params(logged_params)
    mlflow.log_param('max_deviation_normalized', max_deviation)
    mlflow.log_param('max_deviation_mm', MAX_DEVIATION_MM)

    error_type      = trial_params['error_type']
    fidelity_type   = trial_params['fidelity_type']
    smooth_type     = trial_params['smooth_type']
    lambda_error    = trial_params['lambda_error']
    lambda_fidelity = trial_params['lambda_fidelity']
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

    X_pos, X_full, sample_idxs = load_signal_batch(
        data_file_path  = data_file_path,
        axis_selection  = axis_selection,
        target_timesteps= TIMESTEPS,
        sample_size     = SAMPLE_SIZE,
        random_seed     = trial_seed
    )

    N          = X_pos.shape[0]
    warmup_steps = int(TIMESTEPS * WARMUP_FRAC)

    # ---- Precompute baseline predictions ----
    # Use precomputed vel/accel from dataset directly — no SavGol needed here
    baseline_preds = []
    with torch.no_grad():
        for start in range(0, N, MINI_BATCH):
            end    = min(start + MINI_BATCH, N)
            xb     = X_full[start:end].to(device)   # (B, T, 3) precomputed pos/vel/accel
            bp     = model(xb)
            baseline_preds.append(bp.cpu())
    baseline_all = torch.cat(baseline_preds, dim=0)  # (N, T, 1)

    # ---- Run optimization in mini-batches ----
    all_improvements = []
    all_x_star       = []
    all_opt_pred     = []
    all_step_times   = []

    total_signals_done = 0

    for start in range(0, N, MINI_BATCH):
        end = min(start + MINI_BATCH, N)
        B   = end - start

        xb   = X_pos[start:end].to(device)     # (B, T, 1)
        bp   = baseline_all[start:end].to(device)

        batch_start = time.perf_counter()

        x_star_b, opt_pred_b, step_times = optimize_batch(
            x_pos           = xb,
            model           = model,
            baseline_pred   = bp,
            warmup_steps    = warmup_steps,
            inner_steps     = inner_steps,
            inner_lr        = inner_lr,
            error_type      = error_type,
            fidelity_type   = fidelity_type,
            smooth_type     = smooth_type,
            lambda_error    = lambda_error,
            lambda_fidelity = lambda_fidelity,
            lambda_smooth   = lambda_smooth,
            init_method     = init_method,
            device          = device,
            max_deviation   = max_deviation,
            verbose         = False
        )

        batch_time = time.perf_counter() - batch_start
        all_step_times.extend(step_times)

        # Compute improvement for this batch
        impr = compute_improvement(xb, x_star_b, bp, opt_pred_b, warmup_steps)
        all_improvements.append(impr.cpu())
        all_x_star.append(x_star_b.cpu())
        all_opt_pred.append(opt_pred_b.cpu())

        total_signals_done += B
        avg_step_ms = np.mean(step_times) * 1000

        # Log per-batch timing
        mlflow.log_metric("batch_time_s",       batch_time,   step=total_signals_done)
        mlflow.log_metric("avg_step_time_ms",   avg_step_ms,  step=total_signals_done)

        print(f"  Batch {start}:{end} | "
              f"mean_impr={impr.mean().item():.4f} | "
              f"batch_time={batch_time:.2f}s | "
              f"avg_step={avg_step_ms:.1f}ms")

    # ---- Aggregate results ----
    all_improvements = torch.cat(all_improvements, dim=0)   # (N,)
    all_x_star       = torch.cat(all_x_star,       dim=0)   # (N, T, 1)
    all_opt_pred     = torch.cat(all_opt_pred,      dim=0)   # (N, T, 1)

    mean_improvement = all_improvements.mean().item()
    std_improvement  = all_improvements.std().item()

    mlflow.log_metric("mean_improvement",     mean_improvement)
    mlflow.log_metric("std_improvement",      std_improvement)
    mlflow.log_metric("median_improvement",   all_improvements.median().item())
    mlflow.log_metric("pct_improved",         (all_improvements > 0).float().mean().item())
    mlflow.log_metric("mean_step_time_ms",    np.mean(all_step_times) * 1000)
    mlflow.log_metric("total_step_time_s",    np.sum(all_step_times))

    # ---- Validation plots (N_VAL_PLOTS random samples) ----
    rng       = np.random.default_rng(trial_seed)
    plot_idxs = rng.choice(N, size=min(N_VAL_PLOTS, N), replace=False)
    plot_dir  = f"plots/trial_{trial_idx}"

    for pi in plot_idxs:
        plot_optimization_results(
            x_pos         = X_pos[pi, :, 0].numpy(),
            x_star        = all_x_star[pi, :, 0].numpy(),
            baseline_pred = baseline_all[pi, :, 0].numpy(),
            opt_pred      = all_opt_pred[pi, :, 0].numpy(),
            sample_idx    = int(sample_idxs[pi]),
            save_dir      = plot_dir,
            mlflow_run    = True
        )

    mlflow.end_run()

    tune.report({"mean_improvement": mean_improvement})


# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

SEARCH_CONFIG = {
    "error_type":      tune.choice(['L1', 'L2']),
    "fidelity_type":   tune.choice(['L1', 'L2']),
    "smooth_type":     tune.choice(['grad_L1', 'grad_L2', 'grad2_L1', 'grad2_L2']),
    "lambda_error":    tune.loguniform(1e-4, 1e1),
    "lambda_fidelity": tune.loguniform(1e-4, 1e1),
    "lambda_smooth":   tune.loguniform(1e-4, 1e1),
    "inner_lr":        tune.loguniform(1e-4, 1e-1),
    "inner_steps":     tune.randint(10, 101),
    "init_method":     tune.choice(['gaussian', 'inverted_error']),
}

INITIAL_PARAMS = [{
    "error_type":      'L2',
    "fidelity_type":   'L2',
    "smooth_type":     'grad_L2',
    "lambda_error":    1e-2,
    "lambda_fidelity": 1e-2,
    "lambda_smooth":   1e-2,
    "inner_lr":        1e-3,
    "inner_steps":     50,
    "init_method":     'inverted_error',
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

    algo  = OptunaSearch(points_to_evaluate=INITIAL_PARAMS)
    algo = Repeater(algo, repeat=4)
    algo  = ConcurrencyLimiter(algo, max_concurrent=4)

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