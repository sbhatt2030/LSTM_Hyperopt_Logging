"""
Training Pipeline for Time Series Error Prediction v9 (Simplified Regularization)
Works with preprocessed data from data_cleaning_pipeline.py

Expected Input Data Format:
    - Shape: (n_samples, timesteps, 12)
    - Columns 0-2: pos_x, pos_y, pos_z (normalized, micron-rounded)
    - Columns 3-5: vel_x, vel_y, vel_z (normalized, pre-computed)
    - Columns 6-8: accel_x, accel_y, accel_z (normalized, pre-computed)
    - Columns 9-11: e_scale_x/y/z (normalized, filtered)

Model Configurations:
    - Min model: axis_selection='x', errors auto=[9]
      Input: pos_x (+ optional vel_x, accel_x)
      Output: e_scale_x
    
    - Max model: axis_selection='all', errors auto=[9,10,11]
      Input: pos/vel/accel for all axes (9 features if all enabled)
      Output: All 3 error signals

Configurable input features, comprehensive loss functions, model validation

Simplified Regularization (v9):
- Frobenius norm for Jacobian dy/dx (replaces spectral norm via power iteration)
- Hidden state derivative ||dh_t/dt|| (replaces spectral radius of W_hh)
- ~50% less code, faster, more stable

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
# from constants_simplified import (
#     NumericalStability,
#     RegularizationDefaults,
#     DataSchema,
#     DataFiltering
# )


# ============================================================================
# DATASET MODULE
# ============================================================================
# Column indices (must match data_cleaning_pipeline.py output)
COLUMN_STRUCTURE = {
    'pos_x': 0, 'pos_y': 1, 'pos_z': 2,
    'vel_x': 3, 'vel_y': 4, 'vel_z': 5,
    'accel_x': 6, 'accel_y': 7, 'accel_z': 8,
    'e_scale_x': 9, 'e_scale_y': 10, 'e_scale_z': 11
}

AXIS_COLUMNS = {
    'x': {'pos': 0, 'vel': 3, 'accel': 6, 'e_scale': 9},
    'y': {'pos': 1, 'vel': 4, 'accel': 7, 'e_scale': 10},
    'z': {'pos': 2, 'vel': 5, 'accel': 8, 'e_scale': 11}
}

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data_file_path: str,
        target_timesteps: int,
        axis_selection: str,  # 'x', 'y', 'z', or 'all'
        downsample_factor: int = 1,
        include_velocity: bool = True,
        include_acceleration: bool = True,
        motion_threshold: float = 0.9,
        min_position_change: float = 0.0001,
        dtype: torch.dtype = torch.float32,
        validate_divisibility: bool = True
    ):
        self.target_timesteps = target_timesteps
        self.axis_selection = axis_selection.lower()
        self.downsample_factor = downsample_factor
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.motion_threshold = motion_threshold
        self.min_position_change = min_position_change
        self.dtype = dtype
        
        # Validate axis selection
        if self.axis_selection not in ['x', 'y', 'z', 'all']:
            raise ValueError(f"axis_selection must be 'x', 'y', 'z', or 'all', got '{axis_selection}'")
        
        # Load data (preprocessed by data_cleaning_pipeline.py)
        print(f"Loading preprocessed data from {data_file_path}...")
        data = np.load(data_file_path)
        print(f"Loaded data shape: {data.shape}")
        print("  Expected format: (samples, timesteps, 12 features)")
        print("  Features: pos(3), vel(3), accel(3), e_scale(3)")
        # Validate data format
        if data.shape[2] != 12:
            raise ValueError(
                f"Data has {data.shape[2]} features, expected {12}. "
                f"Ensure data is from data_cleaning_pipeline.py"
            )
        
        original_timesteps = data.shape[1]
        
        # Apply downsampling BEFORE reshaping
        if downsample_factor > 1:
            print(f"Downsampling by factor of {downsample_factor}...")
            data = data[:, ::downsample_factor, :]
            downsampled_timesteps = data.shape[1]
            print(f"Downsampled from {original_timesteps} to {downsampled_timesteps} timesteps")
        else:
            downsampled_timesteps = original_timesteps
        
        # Validate divisibility
        if validate_divisibility and downsampled_timesteps % target_timesteps != 0:
            raise ValueError(
                f"target_timesteps ({target_timesteps}) must divide {downsampled_timesteps} cleanly. "
                f"{downsampled_timesteps} % {target_timesteps} = {downsampled_timesteps % target_timesteps}"
            )
        
        # Reshape
        n_windows = downsampled_timesteps // target_timesteps
        print(f"Reshaping to {target_timesteps} timesteps ({n_windows} windows per sample)...")
        m, _, n_features = data.shape
        data_reshaped = data.reshape(m * n_windows, target_timesteps, n_features)
        print(f"Reshaped data shape: {data_reshaped.shape}")
        
        # Filter by motion: keep windows where enough timesteps have non-negligible velocity.
        # Uses commanded velocity columns (3-5) as a proxy for the removed has_motion flag.
        # A timestep is "moving" when the RMS velocity across the relevant axes exceeds
        # motion_threshold is re-interpreted here as the *minimum fraction of timesteps*
        # that must be moving (i.e. windows with too little motion are dropped).
        print(f"Filtering samples with <{motion_threshold*100:.0f}% moving timesteps (velocity-based)...")
        if self.axis_selection == 'all':
            vel_cols = [3, 4, 5]
        else:
            vel_cols = [AXIS_COLUMNS[self.axis_selection]['vel']]

        vel_data = data_reshaped[:, :, vel_cols]                        # (samples, timesteps, n_vel)
        vel_magnitude = np.sqrt(np.sum(vel_data ** 2, axis=-1))         # (samples, timesteps)
        # Use the 99th-percentile of all velocities as a normalisation reference so the
        # threshold is data-relative rather than unit-dependent.
        vel_scale = np.percentile(vel_magnitude, 99) + 1e-8
        moving_fraction = np.mean(vel_magnitude / vel_scale > 0.01, axis=1)  # (samples,)
        motion_mask = moving_fraction >= (1.0 - motion_threshold)

        n_before = len(data_reshaped)
        data_filtered = data_reshaped[motion_mask]
        n_after_motion = len(data_filtered)
        print(f"  Removed {n_before - n_after_motion} samples ({(n_before - n_after_motion)/n_before*100:.1f}%)")
        
        # Filter by position change
        print(f"Filtering samples with position change < {min_position_change} mm...")
        if self.axis_selection == 'all':
            # Use total 3D distance
            pos_data = data_filtered[:, :, 0:3]  # All position columns
            total_variation = np.sum(np.sqrt(np.sum(np.diff(pos_data, axis=1)**2, axis=2)), axis=1)
        else:
            # Use single axis position
            pos_col_idx = AXIS_COLUMNS[self.axis_selection]['pos']
            position_data = data_filtered[:, :, pos_col_idx]
            total_variation = np.sum(np.abs(np.diff(position_data, axis=1)), axis=1)
        
        position_mask = total_variation >= min_position_change
        
        n_before = len(data_filtered)
        data_filtered = data_filtered[position_mask]
        n_after_position = len(data_filtered)
        print(f"  Removed {n_before - n_after_position} samples ({(n_before - n_after_position)/n_before*100:.1f}%)")
        
        # Build feature indices based on configuration
        feature_indices, feature_names, error_indices = self._build_feature_indices(
            self.axis_selection, include_velocity, include_acceleration
        )
        
        # Extract input features (NO COMPUTATION, just column selection)
        self.X = data_filtered[:, :, feature_indices].astype(np.float32)
        
        # Extract error columns as targets (automatically determined)
        self.y = data_filtered[:, :, error_indices].astype(np.float32)
        
        # Store configuration
        self.feature_names = feature_names
        self.error_indices = error_indices
        self.num_input_features = len(feature_names)
        self.num_output_features = len(error_indices)
        
        print("\nFinal dataset:")
        print(f"  Samples: {len(self.X)}")
        print(f"  Axis selection: {self.axis_selection.upper()}")
        print(f"  Input features: {self.num_input_features} {feature_names}")
        print(f"  Output features: {self.num_output_features} (error columns {error_indices})")
        print(f"  X shape per sample: ({target_timesteps}, {self.num_input_features})")
        print(f"  y shape per sample: ({target_timesteps}, {self.num_output_features})")
        print(f"  Total retention: {len(self.X)/(m*n_windows)*100:.1f}% of windowed samples")
    
    def _build_feature_indices(
        self, 
        axis_selection: str,
        include_velocity: bool,
        include_acceleration: bool
    ) -> Tuple[List[int], List[str], List[int]]:
        """
        Build list of column indices to extract as input features and error targets.
        
        Args:
            axis_selection: 'x', 'y', 'z', or 'all'
            include_velocity: Whether to include velocity features
            include_acceleration: Whether to include acceleration features
        
        Returns:
            feature_indices: List of column indices to extract for inputs
            feature_names: List of feature names for display
            error_indices: List of column indices for error targets
        """
        feature_indices = []
        feature_names = []
        error_indices = []
        
        if axis_selection == 'all':
            # All three axes
            for axis in ['x', 'y', 'z']:
                # Position (always included)
                feature_indices.append(AXIS_COLUMNS[axis]['pos'])
                feature_names.append(f'pos_{axis}')
                
                # Velocity (optional)
                if include_velocity:
                    feature_indices.append(AXIS_COLUMNS[axis]['vel'])
                    feature_names.append(f'vel_{axis}')
                
                # Acceleration (optional)
                if include_acceleration:
                    feature_indices.append(AXIS_COLUMNS[axis]['accel'])
                    feature_names.append(f'accel_{axis}')
                
                # Errors (always for output)
                error_indices.append(AXIS_COLUMNS[axis]['e_scale'])
        else:
            # Single axis
            axis = axis_selection
            
            # Position (always included)
            feature_indices.append(AXIS_COLUMNS[axis]['pos'])
            feature_names.append(f'pos_{axis}')
            
            # Velocity (optional)
            if include_velocity:
                feature_indices.append(AXIS_COLUMNS[axis]['vel'])
                feature_names.append(f'vel_{axis}')
            
            # Acceleration (optional)
            if include_acceleration:
                feature_indices.append(AXIS_COLUMNS[axis]['accel'])
                feature_names.append(f'accel_{axis}')
            
            # Errors (always for output)
            error_indices = [AXIS_COLUMNS[axis]['e_scale']]
        
        return feature_indices, feature_names, error_indices
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=self.dtype)
        y = torch.tensor(self.y[idx], dtype=self.dtype)
        return X, y

def split_dataset(
    dataset: Dataset,
    train_fraction: float = 8/9,
    random_seed: int = 0,
    verbose: bool = True
):
    """Split dataset into training and validation subsets."""
    total_size = len(dataset)
    
    # Handle random seed
    if random_seed == 0:
        seed_used = np.random.randint(0, 100000000)
        if verbose:
            print(f"Generated random seed: {seed_used}")
    else:
        seed_used = random_seed
        if verbose:
            print(f"Using specified seed: {seed_used}")
    
    # Set seed
    torch.manual_seed(seed_used)
    np.random.seed(seed_used)
    
    # Calculate split sizes
    train_size = int(train_fraction * total_size)
    val_size = total_size - train_size
    
    # Create random permutation
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    if verbose:
        print("\nDataset split:")
        print(f"  Total samples: {total_size:,}")
        print(f"  Train samples: {train_size:,} ({train_size/total_size*100:.1f}%)")
        print(f"  Val samples: {val_size:,} ({val_size/total_size*100:.1f}%)")
    
    return train_dataset, val_dataset, seed_used

# ============================================================================
# MODEL MODULE
# ============================================================================
class SequenceModel(nn.Module):
    """LSTM or GRU model for sequence-to-sequence prediction."""
    
    def __init__(
        self,
        model_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super(SequenceModel, self).__init__()
        
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # Validate model type
        if self.model_type not in ['lstm', 'gru']:
            raise ValueError(f"model_type must be 'lstm' or 'gru', got '{model_type}'")
        
        # Select RNN type
        rnn_class = nn.LSTM if self.model_type == 'lstm' else nn.GRU
        
        # Create RNN
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        # Output layer
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(rnn_output_size, output_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        """Forward pass."""
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Input feature size mismatch. Expected {self.input_size}, got {x.shape[-1]}"
            )
        
        rnn_out, _ = self.rnn(x)
        batch_size, seq_len, hidden_dim = rnn_out.shape
        rnn_out_reshaped = rnn_out.contiguous().view(-1, hidden_dim)
        output = self.fc(rnn_out_reshaped)
        output = output.view(batch_size, seq_len, self.output_size)
        return output
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def initialize_model(
    model_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    output_size: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
    device: str = 'cuda',
    verbose: bool = True
):
    """Initialize and validate a sequence model."""
    model = SequenceModel(
        model_type=model_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
        bidirectional=bidirectional
    )
    # Move to device
    model = model.to(device)
    
    if verbose:
        print(f"\nModel: {model_type.upper()}")
        print(f"  Architecture: {num_layers} layers, {hidden_size} hidden units, {input_size} input features")
        print(f"  Total parameters: {model.get_num_parameters():,}")
        print(f"  Device: {device}")
    
    return model
def compute_jacobian_frobenius_penalty(
    model: nn.Module,
    inputs: torch.Tensor,
    accel_noise_std: float = 0.146,
    axis_selection: str = 'x',
    include_velocity: bool = True,
    include_acceleration: bool = True,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Lipschitz penalty that enforces smoothness.
    
    FULLY NORMALIZED: Result is independent of batch size, sequence length,
    input dimension, and output dimension. This makes the penalty comparable
    across different model architectures and data shapes.
    
    Normalization: Divides by (batch * seq_len * input_dim * output_dim)
    
    Returns tensor with gradients flowing to model parameters.
    """
    batch_size, seq_len, num_features = inputs.shape
    device = inputs.device
    num_axes = 3 if axis_selection == 'all' else 1
    
    # Generate perturbation
    delta_accel = torch.randn(
        batch_size, seq_len, num_axes, device=device
    ) * accel_noise_std
    
    delta_vel = torch.cumsum(delta_accel, dim=1)
    delta_pos = torch.cumsum(delta_vel, dim=1)
    
    delta = torch.zeros_like(inputs)
    
    if axis_selection == 'all':
        feature_idx = 0
        delta[:, :, 0:3] = delta_pos
        feature_idx = 3
        if include_velocity:
            delta[:, :, feature_idx:feature_idx+3] = delta_vel
            feature_idx += 3
        if include_acceleration:
            delta[:, :, feature_idx:feature_idx+3] = delta_accel
    else:
        feature_idx = 0
        delta[:, :, feature_idx] = delta_pos[:, :, 0]
        feature_idx += 1
        if include_velocity:
            delta[:, :, feature_idx] = delta_vel[:, :, 0]
            feature_idx += 1
        if include_acceleration:
            delta[:, :, feature_idx] = delta_accel[:, :, 0]
    
    # Get predictions
    y = model(inputs).detach()
    output_dim = y.shape[-1]
    
    y_pert = model(inputs + delta)
    
    # Compute penalty
    numerator = (y_pert - y).pow(2).sum(dim=-1)
    
    with torch.no_grad():
        denominator = delta.pow(2).sum(dim=-1) + eps
    
    ratio = numerator / denominator
    
    # FULL NORMALIZATION: divide by all dimensions
    # This makes the penalty model/sequence agnostic
    penalty = ratio.sum() / (batch_size * seq_len * num_features * output_dim)
    
    return penalty
def compute_hidden_derivative_penalty(
    hidden_states: torch.Tensor,
    max_penalty: float = 100.0
) -> torch.Tensor:
    """
    Penalize temporal derivative ||dh_t/dt|| of hidden states.
    
    FULLY NORMALIZED: Result is independent of batch size, sequence length,
    and hidden dimension. This makes the penalty comparable across different
    model architectures and sequence lengths.
    
    Normalization: Divides by (batch * (seq_len - 1) * hidden_dim)
    This is equivalent to computing the mean squared temporal derivative
    per hidden state element (like MSE).
    
    Args:
        hidden_states: Hidden states from RNN (batch, seq_len, hidden_size)
        max_penalty: Maximum penalty value to prevent explosion
    
    Returns:
        Penalty term for temporal smoothness (normalized, clamped)
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Compute temporal differences: h_t - h_{t-1}
    dh_dt = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    
    # Frobenius norm squared, fully normalized
    # This is the average squared derivative per element
    penalty = torch.sum(dh_dt ** 2) / (batch_size * (seq_len - 1) * hidden_size)
    
    # Clamp to prevent explosion
    penalty = torch.clamp(penalty, 0.0, max_penalty)
    
    return penalty
def compute_max_abs_error_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Max absolute error over time per signal per output, squared,
    averaged over signals and outputs.

    Shape: predictions/targets (B, T, O)
    Returns scalar.
    """
    abs_err = (predictions - targets).abs()   # (B, T, O)
    max_err = abs_err.amax(dim=1)             # (B, O) — max over time
    return max_err.pow(2).mean()              # scalar


def extract_hidden_states_lstm(
    model: nn.Module,
    inputs: torch.Tensor,
    return_all_layers: bool = False
) -> torch.Tensor:
    """
    Extract hidden states from LSTM forward pass.
    
    Args:
        model: SequenceModel with LSTM
        inputs: Input tensor (batch, seq_len, input_dim)
        return_all_layers: If True, return hidden states from all layers
    
    Returns:
        Hidden states (batch, seq_len, hidden_size) or list of them
    """
    # Forward through RNN to get hidden states
    if hasattr(model, 'rnn'):
        rnn = model.rnn
        
        if isinstance(rnn, nn.LSTM):
            # LSTM returns (output, (h_n, c_n))
            # output contains hidden states for all timesteps
            rnn_output, _ = rnn(inputs)
            
            # rnn_output has shape (batch, seq_len, num_directions * hidden_size)
            return rnn_output
        
        elif isinstance(rnn, nn.GRU):
            # GRU returns (output, h_n)
            rnn_output, _ = rnn(inputs)
            return rnn_output
        
        else:
            raise ValueError(f"Unsupported RNN type: {type(rnn)}")
    
    else:
        raise ValueError("Model does not have 'rnn' attribute")
def compute_l2_regularization(model: nn.Module) -> torch.Tensor:
    """Compute L2 regularization on all model parameters."""
    l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_reg += torch.sum(param ** 2)
    return l2_reg
class CompositeLoss(nn.Module):
    """
    Simplified composite loss - L2 handled by optimizer.
    
    Components:
    1. Base loss (MSE or MAE)
    2. Jacobian Frobenius norm
    3. Hidden state derivative
    
    Note: L2 regularization via optimizer.weight_decay
    """
    
    def __init__(
        self,
        base_loss: str = 'mse',
        jacobian_weight: float = 0.0,
        hidden_derivative_weight: float = 0.0,
        max_abs_weight: float = 0.0,
        warmup_timesteps: int = 10,
        jacobian_every_n_batches: int = 1,
        hidden_derivative_every_n_batches: int = 1,
        track_l2: bool = False
    ):
        super().__init__()
        
        # Base loss
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss()
        else:
            raise ValueError(f"base_loss must be 'mse' or 'mae', got '{base_loss}'")
        
        self.jacobian_weight = jacobian_weight
        self.hidden_derivative_weight = hidden_derivative_weight
        self.max_abs_weight = max_abs_weight
        self.warmup_timesteps = warmup_timesteps
        
        # Computation schedules
        self.jacobian_every_n_batches = jacobian_every_n_batches
        self.hidden_derivative_every_n_batches = hidden_derivative_every_n_batches
        
        # Tracking
        self.track_l2 = track_l2
        
        # Caches
        self._cached_jacobian_penalty = 0.0
        self._cached_hidden_derivative_penalty = 0.0
        
    def reset_cache(self):
        """Reset cached regularization values. Call at start of each epoch."""
        self._cached_jacobian_penalty = 0.0
        self._cached_hidden_derivative_penalty = 0.0
    
    def needs_input_gradients(self) -> bool:
        """Check if input gradients are needed (only for Jacobian)."""
        return self.jacobian_weight > 0

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        batch_idx: int = 0
    ) -> tuple:
        """Compute composite loss. Returns (loss, components_dict)."""

        # Discard warmup timesteps
        if self.warmup_timesteps > 0:
            predictions = predictions[:, self.warmup_timesteps:, :]
            targets     = targets[:,     self.warmup_timesteps:, :]
            if inputs is not None:
                inputs = inputs[:, self.warmup_timesteps:, :]
            if hidden_states is not None:
                hidden_states = hidden_states[:, self.warmup_timesteps:, :]

        # Base loss
        base_loss_value = self.base_loss(predictions, targets)
        loss = base_loss_value

        # Track raw penalty values
        jacobian_penalty_raw     = 0.0
        hidden_deriv_penalty_raw = 0.0
        max_abs_penalty_raw      = 0.0

        # Jacobian Frobenius norm
        if self.jacobian_weight > 0:
            if inputs is None:
                raise ValueError("inputs required when jacobian_weight > 0")
            should_compute = (batch_idx % self.jacobian_every_n_batches == 0)
            if should_compute:
                jacobian_penalty = compute_jacobian_frobenius_penalty(model, inputs)
                self._cached_jacobian_penalty = jacobian_penalty.item()
            else:
                jacobian_penalty = self._cached_jacobian_penalty
            jacobian_penalty_raw = self._cached_jacobian_penalty
            loss = loss + self.jacobian_weight * jacobian_penalty

        # Hidden state derivative penalty
        if self.hidden_derivative_weight > 0:
            if hidden_states is not None:
                should_compute = (batch_idx % self.hidden_derivative_every_n_batches == 0)
                if should_compute:
                    hidden_penalty = compute_hidden_derivative_penalty(hidden_states)
                    self._cached_hidden_derivative_penalty = hidden_penalty.item() if torch.is_tensor(hidden_penalty) else hidden_penalty
                else:
                    hidden_penalty = self._cached_hidden_derivative_penalty
                hidden_deriv_penalty_raw = self._cached_hidden_derivative_penalty
                loss = loss + self.hidden_derivative_weight * hidden_penalty

        # Max absolute error penalty
        if self.max_abs_weight > 0:
            max_abs_penalty = compute_max_abs_error_loss(predictions, targets)
            max_abs_penalty_raw = max_abs_penalty.item()
            loss = loss + self.max_abs_weight * max_abs_penalty

        components = {
            'base_loss':            base_loss_value.item(),
            'jacobian_penalty':     jacobian_penalty_raw,
            'hidden_deriv_penalty': hidden_deriv_penalty_raw,
            'max_abs_penalty':      max_abs_penalty_raw,
            'jacobian_contrib':     self.jacobian_weight          * jacobian_penalty_raw,
            'hidden_deriv_contrib': self.hidden_derivative_weight * hidden_deriv_penalty_raw,
            'max_abs_contrib':      self.max_abs_weight           * max_abs_penalty_raw,
            'total':                loss.item(),
        }

        return loss, components

def create_loss_function(
    base_loss: str = 'mse',
    jacobian_weight: float = 0.0,
    jacobian_every_n_batches: int = 1,
    hidden_derivative_weight: float = 0.0,
    hidden_derivative_every_n_batches: int = 1,
    max_abs_weight: float = 0.0,
    warmup_timesteps: int = 0,
    track_l2: bool = True,
    verbose: bool = True
) -> CompositeLoss:
    """Factory function to create a composite loss function."""
    loss_fn = CompositeLoss(
        base_loss=base_loss,
        jacobian_weight=jacobian_weight,
        hidden_derivative_weight=hidden_derivative_weight,
        max_abs_weight=max_abs_weight,
        warmup_timesteps=warmup_timesteps,
        jacobian_every_n_batches=jacobian_every_n_batches,
        hidden_derivative_every_n_batches=hidden_derivative_every_n_batches,
        track_l2=track_l2
    )
    
    if verbose:
        print("\nLoss Function Configuration:")
        print(f"  Base loss: {base_loss.upper()}")
        if jacobian_weight > 0 or hidden_derivative_weight > 0:
            print(f"  Regularization: Jacobian={jacobian_weight}, Hidden Derivative={hidden_derivative_weight}")
        if max_abs_weight > 0:
            print(f"  Max abs error weight: {max_abs_weight}")
        print(f"  L2 regularization: via optimizer.weight_decay")
        if warmup_timesteps > 0:
            print(f"  Warmup timesteps: {warmup_timesteps} (discarded from loss)")
    
    return loss_fn
# ============================================================================
# TRAINING & VALIDATION MODULES
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip_value: Optional[float] = None,
    use_composite_loss: bool = False,
    verbose: int = 1
):
    """Train model for one epoch. Returns (avg_loss, avg_components)."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    # Accumulators for component logging
    accum = {
        'base_loss': 0.0, 'jacobian_penalty': 0.0, 'hidden_deriv_penalty': 0.0,
        'max_abs_penalty': 0.0, 'jacobian_contrib': 0.0,
        'hidden_deriv_contrib': 0.0, 'max_abs_contrib': 0.0,
    }
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        
        # Extract hidden states if needed
        if use_composite_loss and loss_function.hidden_derivative_weight > 0:
            hidden_states = extract_hidden_states_lstm(model, X_batch)
        else:
            hidden_states = None
        
        if use_composite_loss:
            loss, components = loss_function(
                predictions, y_batch, model, X_batch, hidden_states, batch_idx
            )
            for k in accum:
                accum[k] += components.get(k, 0.0)
        else:
            loss = loss_function(predictions, y_batch)
        
        loss.backward()
        
        if gradient_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if verbose >= 2 and (batch_idx + 1) % 10 == 0:
            print(f"    Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in accum.items()} if use_composite_loss else {}

    if verbose >= 1:
        print(f"  Train Loss: {avg_loss:.6f}")
    
    return avg_loss, avg_components
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: Callable,
    device: str,
    use_composite_loss: bool = False,
    warmup_timesteps: int = 0,
    max_abs_weight: float = 0.0,
    verbose: int = 1
):
    """
    Validate model for one epoch.

    When max_abs_weight=0: returns pure MSE (Stage 1 — unchanged behaviour).
    When max_abs_weight>0: returns MSE + max_abs_weight * max_abs_error (Stage 2).
    'best val loss' always refers to this composite so existing tooling is unaffected.

    Returns: (composite, std_mse, avg_mse, avg_max_abs)
    """
    model.eval()
    mse_loss_fn  = nn.MSELoss()
    mse_losses   = []
    max_abs_losses = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            if warmup_timesteps > 0:
                predictions = predictions[:, warmup_timesteps:, :]
                y_batch     = y_batch[:,     warmup_timesteps:, :]
            mse_losses.append(mse_loss_fn(predictions, y_batch).item())
            if max_abs_weight > 0:
                max_abs_losses.append(compute_max_abs_error_loss(predictions, y_batch).item())

    avg_mse     = float(np.mean(mse_losses))
    std_mse     = float(np.std(mse_losses, ddof=1)) if len(mse_losses) > 1 else 0.0
    avg_max_abs = float(np.mean(max_abs_losses)) if max_abs_losses else 0.0
    composite   = avg_mse + max_abs_weight * avg_max_abs

    if verbose >= 1:
        if max_abs_weight > 0:
            print(f"  Val MSE: {avg_mse:.6f} ± {std_mse:.6f} | "
                  f"Val MaxAbs: {avg_max_abs:.6f} | Composite: {composite:.6f}")
        else:
            print(f"  Val Loss (MSE): {avg_mse:.6f} ± {std_mse:.6f}")

    return composite, std_mse, avg_mse, avg_max_abs

# ============================================================================
# EARLY STOPPING MODULE
# ============================================================================
class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return 2
        
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"  Val loss improved: {self.best_loss:.6f} to {val_loss:.6f}")
            self.best_loss = val_loss
            self.counter = 0
            return 2
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")
                self.early_stop = True
                return 0
            
            return 1

# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

def plot_predictions(
    model: nn.Module,
    dataset: Dataset,
    indices: list,
    device: str,
    save_dir: str = 'results',
    dataset_name: str = 'train',
    error_names: List[str] = ['e_scale'],
    warmup_timesteps: int = 0,
):
    """Plot predictions: position input, error signals, and deltas only.
    All plots and stats are computed over post-warmup timesteps only."""
    model.eval()
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {len(indices)} plots for {dataset_name} set...")
    
    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(indices):
            X, y_true = dataset[sample_idx]
            X_batch = X.unsqueeze(0).to(device)
            
            y_pred = model(X_batch)
            
            X = X.cpu().numpy()
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.squeeze(0).cpu().numpy()

            # Slice to post-warmup region
            w = warmup_timesteps
            position = X[w:, 0]
            y_true   = y_true[w:, :]
            y_pred   = y_pred[w:, :]
            delta    = y_pred - y_true
            num_outputs = y_true.shape[-1]

            n_rows = 2 + num_outputs  # position + one row per output + delta
            fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))
            fig.suptitle(f'{dataset_name.capitalize()} Sample {sample_idx} - Predictions vs Actual'
                         + (f' (warmup={w} steps excluded)' if w > 0 else ''),
                        fontsize=16, fontweight='bold')

            timesteps = np.arange(w, w + len(position))

            # Plot 1: Position Input
            axes[0].plot(timesteps, position, 'b-', linewidth=1.5, label='Position')
            axes[0].set_ylabel('Position (normalized)', fontsize=11, fontweight='bold')
            axes[0].set_title('Input: Position Signal', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc='upper right')

            # One subplot per output signal
            colors_true = ['g', 'c', 'm']
            colors_pred = ['r', 'orange', 'purple']
            for i in range(num_outputs):
                label = error_names[i] if i < len(error_names) else f'output_{i}'
                axes[i + 1].plot(timesteps, y_true[:, i],
                                 f'{colors_true[i % len(colors_true)]}-',
                                 linewidth=1.5, label=f'Actual {label}', alpha=0.7)
                axes[i + 1].plot(timesteps, y_pred[:, i],
                                 f'{colors_pred[i % len(colors_pred)]}--',
                                 linewidth=1.5, label=f'Predicted {label}', alpha=0.7)
                axes[i + 1].set_ylabel(f'{label} (normalized)', fontsize=11, fontweight='bold')
                axes[i + 1].set_title(f'Output {i + 1}: {label}', fontsize=12)
                axes[i + 1].grid(True, alpha=0.3)
                axes[i + 1].legend(loc='upper right')

            # Final plot: Prediction Errors (Delta)
            delta_ax = axes[-1]
            delta_colors = ['purple', 'orange', 'brown']
            for i in range(num_outputs):
                label = error_names[i] if i < len(error_names) else f'output_{i}'
                delta_ax.plot(timesteps, delta[:, i],
                              color=delta_colors[i % len(delta_colors)],
                              linewidth=1.5, label=f'Δ{label}', alpha=0.7)
            delta_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            delta_ax.set_xlabel('Timestep', fontsize=11, fontweight='bold')
            delta_ax.set_ylabel('Prediction Error (Δ)', fontsize=11, fontweight='bold')
            delta_ax.set_title('Prediction Error (Predicted - Actual)', fontsize=12)
            delta_ax.grid(True, alpha=0.3)
            delta_ax.legend(loc='upper right')

            # Stats computed over post-warmup region only
            stats_lines = []
            for i in range(num_outputs):
                label = error_names[i] if i < len(error_names) else f'output_{i}'
                mae_i  = np.mean(np.abs(delta[:, i]))
                rmse_i = np.sqrt(np.mean(delta[:, i] ** 2))
                stats_lines.append(f'MAE {label}={mae_i:.6f}  RMSE {label}={rmse_i:.6f}')
            stats_text = '\n'.join(stats_lines)

            fig.text(0.99, 0.01, stats_text, ha='right', va='bottom', 
                    fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            filename = f'{dataset_name}_sample_{sample_idx:04d}.png'
            filepath = save_path / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            mlflow.log_figure(fig, str(filepath))
            
            if (plot_idx + 1) % 5 == 0:
                print(f"  Saved {plot_idx + 1}/{len(indices)} plots...")
    
    print(f"Ã¢Å“â€œ All {len(indices)} plots saved to '{save_dir}/'")
    
def visualize_predictions(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str,
    num_samples: int = 10,
    save_dir: str = 'results',
    random_seed: int = 42,
    error_names: List[str] = ['e_scale'],
    warmup_timesteps: int = 0,
):
    """Visualize predictions on both train and validation sets."""
    print("\n" + "="*80)
    print("GENERATING PREDICTION VISUALIZATIONS")
    print("="*80)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    train_size = len(train_dataset)
    train_indices = np.random.choice(train_size, size=min(num_samples, train_size), 
                                     replace=False).tolist()
    
    val_size = len(val_dataset)
    val_indices = np.random.choice(val_size, size=min(num_samples, val_size), 
                                   replace=False).tolist()
    
    print(f"Selected {len(train_indices)} random samples from training set")
    print(f"Selected {len(val_indices)} random samples from validation set")
    
    plot_predictions(
        model=model,
        dataset=train_dataset,
        indices=train_indices,
        device=device,
        save_dir=save_dir,
        dataset_name='train',
        error_names=error_names,
        warmup_timesteps=warmup_timesteps,
    )
    
    plot_predictions(
        model=model,
        dataset=val_dataset,
        indices=val_indices,
        device=device,
        save_dir=save_dir,
        dataset_name='val',
        error_names=error_names,
        warmup_timesteps=warmup_timesteps,
    )
    
    print("\n" + "="*80)
    print(f"VISUALIZATION COMPLETE - All plots saved to '{save_dir}/'")
    print("="*80)

# ============================================================================
# MASTER TRAINING ORCHESTRATOR
# ============================================================================

def train_model(
    # Data parameters
    data_file_path: str,
    target_timesteps: int,
    axis_selection: str = 'x',
    downsample_factor: int = 1,
    include_velocity: bool = True,
    include_acceleration: bool = True,
    motion_threshold: float = 0.9,
    min_position_change: float = 0.0001,
    train_val_split: float = 8/9,
    random_seed: int = 0,
    
    # Model parameters
    model_type: str = 'lstm',
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,
    bidirectional: bool = False,
    load_model_path: Optional[str] = None,
    load_model_strict: bool = True,
    
    # Loss function parameters
    base_loss: str = 'mse',
    l2_weight: float = 0.0,
    jacobian_weight: float = 0.0,
    jacobian_every_n_batches: int = 1,
    hidden_derivative_weight: float = 0.0,
    hidden_derivative_every_n_batches: int = 1,
    max_abs_weight: float = 0.0,
    
    # Training parameters
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    gradient_clip_value: Optional[float] = None,
    device: str = 'cuda',

    #Logging
    mlflow_experiment_id:int = 0,
    tracking_uri:str = "http://hurco-lambda.hurco.com:8090",
    
    # Early stopping parameters
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.000001,
    
    # Validation (always MSE) parameters
    validation_frequency: int = 1,
    
    # DataLoader parameters
    num_workers: int = 0,
    
    # Output parameters
    verbose: int = 1,
    return_history: bool = True,
    save_model_path: Optional[str] = None,
    save_best_only: bool = True,
    
    # Visualization parameters
    generate_plots: bool = True,
    num_plot_samples: int = 10,
    plot_save_dir: str = 'results',
    error_names: List[str] = ['e_scale']
) -> Tuple[nn.Module, Optional[Dict[str, list]], int]:
    """
    Master training function for time series error prediction.
    
    Returns:
        model: Trained model
        history: Dict with 'train_loss' and 'val_loss' lists (if return_history=True)
        seed_used: Random seed that was used
    """
    
    ####MAKE MLFLOW RUN
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.start_run(experiment_id=str(mlflow_experiment_id), log_system_metrics=True)
    
    if verbose >= 1:
        print("="*80)
        print("TRAINING PIPELINE")
        print("="*80)
    
    start_time = time.time()
    
    # ========================================================================
    # STEP 1: CREATE DATASET
    # ========================================================================
    if verbose >= 1:
        print("\n" + "-"*80)
        print("STEP 1: Loading and preparing dataset")
        print("-"*80)
        
    dataset = TimeSeriesDataset(
        data_file_path=data_file_path,
        target_timesteps=target_timesteps,
        axis_selection=axis_selection,
        downsample_factor=downsample_factor,
        include_velocity=include_velocity,
        include_acceleration=include_acceleration,
        motion_threshold=motion_threshold,
        min_position_change=min_position_change,
        dtype=torch.float32,
        validate_divisibility=True
    )
    
    input_size = dataset.num_input_features
    output_size = dataset.num_output_features
    

    
    if verbose >= 1:
        print(f"Input size: {input_size} (features: {dataset.feature_names})")
        print(f"Output size: {output_size} (errors: {error_names})")
    
    
    dataset_stats = {
        'total_samples': len(dataset),
        'num_input_features': input_size,
        'feature_names': dataset.feature_names,
        'num_output_features': output_size
    }
    # ========================================================================
    # STEP 2: SPLIT DATASET
    # ========================================================================
    if verbose >= 1:
        print("\n" + "-"*80)
        print("STEP 2: Splitting into train/validation sets")
        print("-"*80)
    
    train_dataset, val_dataset, seed_used = split_dataset(
        dataset=dataset,
        train_fraction=train_val_split,
        random_seed=random_seed,
        verbose=(verbose >= 1)
    )
    
    
    split_info = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'train_val_split': train_val_split,
        'random_seed_used': seed_used
    }

    
    # ========================================================================
    # STEP 3: CREATE DATA LOADERS
    # ========================================================================
    if verbose >= 1:
        print("\n" + "-"*80)
        print("STEP 3: Creating data loaders")
        print("-"*80)
    
    if num_workers > 0 and os.name == 'nt':
        print(f"Warning: num_workers={num_workers} on Windows may cause issues. Consider setting to 0.")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    
    if verbose >= 1:
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    dataloader_config = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'train_batches': len(train_loader),
        'val_batches': len(val_loader)
    }
    
    # ========================================================================
    # STEP 4: INITIALIZE MODEL
    # ========================================================================
    if verbose >= 1:
        print("\n" + "-"*80)
        print("STEP 4: Initializing model")
        print("-"*80)
    
    try:
        model = initialize_model(
            model_type=model_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            bidirectional=bidirectional,
            device=device,
            verbose=(verbose >= 1)
        )
        
        # Load pretrained weights if specified
        if load_model_path is not None:
            if verbose >= 1:
                print(f"\nLoading pretrained model from: {load_model_path}")
            
            try:
                loaded = torch.load(load_model_path, map_location=device,weights_only=False)
                if isinstance(loaded, dict):
                    model.load_state_dict(loaded)
                else:
                    model.load_state_dict(loaded.state_dict())

                    model.to(device)
                
                if verbose >= 1:
                    print(f"Successfully loaded model weights")
                    if not load_model_strict:
                        print(f"Non-strict loading - some weights may not match")
            
            except FileNotFoundError:
                raise FileNotFoundError(f"Model file not found: {load_model_path}")
            except RuntimeError as e:
                if "size mismatch" in str(e) or "Missing key" in str(e):
                    print(f"\nModel architecture mismatch!")
                    print(f"  Error: {e}")
                    print(f"\n  The saved model doesn't match current architecture:")
                    print(f"    Current: {model_type.upper()} with {num_layers} layers, {hidden_size} hidden units")
                    print(f"  Try:")
                    print(f"    - Check if model_type, hidden_size, num_layers match saved model")
                    print(f"    - Set load_model_strict=False to load partial weights")
                    raise
                else:
                    raise
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nÃ¢Å“â€” CUDA out of memory error. Try:")
            print(f"  - Reducing batch_size (current: {batch_size})")
            print(f"  - Reducing hidden_size (current: {hidden_size})")
            print(f"  - Reducing num_layers (current: {num_layers})")
            raise
        else:
            raise
    
    
    model_config = {
        'model_type': model_type,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'output_size': output_size,
        'dropout': dropout,
        'bidirectional': bidirectional,
        'total_parameters': model.get_num_parameters(),
        'load_model_path': load_model_path,
        'load_model_strict': load_model_strict
    }
    
    # ========================================================================
    # STEP 5: SETUP LOSS FUNCTION
    # ========================================================================
    if verbose >= 1:
        print("\n" + "-"*80)
        print("STEP 5: Setting up loss function")
        print("-"*80)
    warmup_timesteps =None
    resolved_warmup = warmup_timesteps if warmup_timesteps is not None else int(0.05 * target_timesteps)
    use_composite_loss = True  # always CompositeLoss — handles warmup in both regularized and plain paths

    if use_composite_loss and (l2_weight > 0 or jacobian_weight > 0 or hidden_derivative_weight > 0 or max_abs_weight > 0):
        loss_fn = create_loss_function(
            base_loss=base_loss,
            # l2_weight=l2_weight,
            jacobian_weight=jacobian_weight,
            hidden_derivative_weight=hidden_derivative_weight,
            max_abs_weight=max_abs_weight,
            jacobian_every_n_batches=jacobian_every_n_batches,
            hidden_derivative_every_n_batches=hidden_derivative_every_n_batches,
            warmup_timesteps=resolved_warmup,
            verbose=(verbose >= 1)
        )
    else:
        loss_fn = create_loss_function(
            base_loss=base_loss,
            warmup_timesteps=resolved_warmup,
            verbose=(verbose >= 1)
        )

    loss_config = {
        'base_loss': base_loss,
        'use_composite_loss': use_composite_loss,
        'l2_weight': l2_weight,
        'jacobian_weight': jacobian_weight,
        'hidden_derivative_weight': hidden_derivative_weight,
        'max_abs_weight': max_abs_weight,
        'jacobian_every_n_batches': jacobian_every_n_batches,
        'hidden_derivative_every_n_batches': hidden_derivative_every_n_batches,
        'warmup_timesteps': resolved_warmup,
    }
    
    # ========================================================================
    # STEP 6: SETUP OPTIMIZER
    # ========================================================================
    if verbose >= 1:
        print("\n" + "-"*80)
        print("STEP 6: Setting up optimizer")
        print("-"*80)
    
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer_type_lower = optimizer_type.lower()
    
    optimizer_map = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
        'adamw': torch.optim.AdamW
    }
    
    if optimizer_type_lower not in optimizer_map:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Supported: {list(optimizer_map.keys())}")
    
    optimizer = optimizer_map[optimizer_type_lower](
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_weight,
        **optimizer_kwargs
    )
    
    if verbose >= 1:
        print(f"Optimizer: {optimizer_type.upper()}, Learning rate: {learning_rate}")
        if optimizer_kwargs:
            print(f"Additional args: {optimizer_kwargs}")

    optimizer_config = {
        'optimizer_type': optimizer_type,
        'learning_rate': learning_rate,
        'optimizer_kwargs': optimizer_kwargs,
        'gradient_clip_value': gradient_clip_value
    }
    # ========================================================================
    # STEP 7: SETUP EARLY STOPPING
    # ========================================================================
    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        verbose=(verbose >= 1)
    )
    
    early_stop_config = {
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_min_delta': early_stopping_min_delta
    }
    
    training_config = {
        'num_epochs': num_epochs,
        'validation_frequency': validation_frequency,
        'device': device,
        'verbose': verbose
    }
    
    data_config = {
        'data_file_path': data_file_path,
        'target_timesteps': target_timesteps,
        'axis_selection': axis_selection,
        'downsample_factor': downsample_factor,
        'include_velocity': include_velocity,
        'include_acceleration': include_acceleration,
        'motion_threshold': motion_threshold,
        'min_position_change': min_position_change
    }
    
    viz_config = {
        'generate_plots': generate_plots,
        'num_plot_samples': num_plot_samples,
        'plot_save_dir': plot_save_dir,
        'error_names': error_names
    }
    
    all_config = {
        **data_config,
        **dataset_stats,
        **split_info,
        **model_config,
        **loss_config,
        **optimizer_config,
        **training_config,
        **early_stop_config,
        **dataloader_config,
        **viz_config
    }

    ###It might be worth it to just log this one table instead of all of the above tables as this one is collating all of the individual ones. 
    mlflow.log_params(all_config, synchronous=False) ### I can also write this to a json in the artificat store instead of the param table if that's easier - NR
    
    # ========================================================================
    # STEP 8: TRAINING LOOP
    # ========================================================================
    if verbose >= 1:
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
    
    history = {'train_loss': [], 'val_loss': []} if return_history else None
    best_val_loss = float('inf')
    best_model_state = None  # Track best model state dict
    
    # #LOG_ONCE - Training start time
    training_start_time = datetime.datetime.now().isoformat()
    mlflow.log_param("Training Start Time", training_start_time)
    best_epoch = 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        
        # Reset cached regularization values at start of each epoch
        # if use_composite_loss:
        #     #loss_fn.reset_cache()
        #     pass
        if verbose >= 1:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
        
        # Training
        train_loss, train_components = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_function=loss_fn,
            optimizer=optimizer,
            device=device,
            gradient_clip_value=gradient_clip_value,
            use_composite_loss=use_composite_loss,
            verbose=verbose
        )
        
        if return_history:
            history['train_loss'].append(train_loss)
        
        # #LOG_PER_EPOCH - Training loss and components
        mlflow.log_metric("Train loss", train_loss, step=epoch, synchronous=False)
        if train_components:
            mlflow.log_metric("train_base_loss",            train_components['base_loss'],            step=epoch, synchronous=False)
            mlflow.log_metric("train_jacobian_penalty",     train_components['jacobian_penalty'],     step=epoch, synchronous=False)
            mlflow.log_metric("train_hidden_deriv_penalty", train_components['hidden_deriv_penalty'], step=epoch, synchronous=False)
            mlflow.log_metric("train_max_abs_penalty",      train_components['max_abs_penalty'],      step=epoch, synchronous=False)
            mlflow.log_metric("train_jacobian_contrib",     train_components['jacobian_contrib'],     step=epoch, synchronous=False)
            mlflow.log_metric("train_hidden_deriv_contrib", train_components['hidden_deriv_contrib'], step=epoch, synchronous=False)
            mlflow.log_metric("train_max_abs_contrib",      train_components['max_abs_contrib'],      step=epoch, synchronous=False)
        epoch_train_loss = train_loss
        
        # Validation
        val_loss = None
        if (epoch + 1) % validation_frequency == 0:
            val_loss, val_loss_std, val_mse, val_max_abs = validate_epoch(
                model=model,
                val_loader=val_loader,
                loss_function=loss_fn,
                device=device,
                use_composite_loss=use_composite_loss,
                warmup_timesteps=resolved_warmup,
                max_abs_weight=max_abs_weight,
                verbose=verbose
            )
            
            if return_history:
                history['val_loss'].append(val_loss)
            
            # #LOG_PER_EPOCH - Validation loss
            epoch_val_loss = val_loss
            mlflow.log_metric("Val loss",        epoch_val_loss, step=epoch, synchronous=False)
            mlflow.log_metric("val_loss_std",    val_loss_std,   step=epoch, synchronous=False)
            mlflow.log_metric("val_mse",         val_mse,        step=epoch, synchronous=False)
            mlflow.log_metric("val_max_abs",     val_max_abs,    step=epoch, synchronous=False)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_std = val_loss_std
                
                # Save best model
                if save_model_path is not None:
                    best_model_state = model.state_dict().copy()
                    
                    # Create directory if needed
                    save_path = Path(save_model_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save immediately if save_best_only=True
                    if save_best_only:
                        torch.save(best_model_state, save_model_path)
                        if verbose >= 1:
                            print(f"Saved best model to {save_model_path}")
            
            # #LOG_PER_EPOCH - Best validation loss so far
            epoch_best_val_loss = best_val_loss
            mlflow.log_metric("best val loss", epoch_best_val_loss, step = epoch, synchronous=False)
            mlflow.log_metric("best val std", best_val_std, step = epoch, synchronous=False)
            
            # Early stopping check
            status = early_stopper(val_loss)
            if status == 0:
                if verbose >= 1:
                    print(f"\nTraining stopped at epoch {epoch + 1}")
                break
            elif status == 2:
                #LOG_UPDATE keep best model and store epoch+1
                #mlflow.pytorch.log_model(model, name= f"model_epoch_{epoch+1}")  ###Instead of constantly saving the best model. It might be worth it to write the best model to disk on top of it self, then save only once to mlflow. 
                best_epoch = epoch
                ###Writing the model to MLFLOW is relatively expensive so cutting that down might help. 
                #write to desk
            
        epoch_time = time.time() - epoch_start_time
        
        # #LOG_PER_EPOCH - Epoch time
        epoch_time_seconds = epoch_time
        mlflow.log_metric("Epoch Time", epoch_time_seconds, step=epoch, synchronous=False)
        
        # #LOG_PER_EPOCH - Early stopping counter
        epochs_without_improvement = early_stopper.counter
        mlflow.log_metric("Early Stopping Counter", epochs_without_improvement, step=epoch, synchronous=False)
        if verbose >= 1:
            print(f"  Epoch time: {epoch_time:.2f}s")
    if best_model_state:
        model.load_state_dict(best_model_state)  # restore best weights onto model before logging
    mlflow.pytorch.log_model(model, name="model_best")  
    # ========================================================================
    # STEP 9: TRAINING SUMMARY
    # ========================================================================
    total_time = time.time() - start_time
    
    # #LOG_ONCE - Training end time and duration
    training_end_time = datetime.datetime.now().isoformat()
    mlflow.log_param("Training End Time", training_end_time, synchronous=False)
    total_training_time_minutes = total_time / 60
    final_best_val_loss = best_val_loss
    
    if verbose >= 1:
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total training time: {total_training_time_minutes:.2f} minutes")
        print(f"Best validation loss: {final_best_val_loss:.6f}")
        print(f"Random seed used: {seed_used}")
        print("="*80)
    
    # #LOG_ONCE - Add final training results to config
    all_config['training_start_time'] = training_start_time
    all_config['training_end_time'] = training_end_time
    all_config['total_training_time_minutes'] = total_training_time_minutes
    all_config['best_val_loss'] = final_best_val_loss
    
    # ========================================================================
    # STEP 10: SAVE FINAL MODEL
    # ========================================================================
    if save_model_path is not None:
        save_path = Path(save_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_best_only:
            # Already saved during training, just confirm
            if verbose >= 1:
                print(f"Best model saved to {save_model_path}")
                print(f"  (Best val loss: {best_val_loss:.6f})")
        else:
            # Save both best and final
            if best_model_state is not None:
                best_path = save_path.parent / f"{save_path.stem}_best{save_path.suffix}"
                torch.save(model, best_path)
                if verbose >= 1:
                    print(f"Saved best model to {best_path}")
                    print(f"(Best val loss: {best_val_loss:.6f})")
            
            final_path = save_path.parent / f"{save_path.stem}_final{save_path.suffix}"
            torch.save(model, final_path)
            if verbose >= 1:
                print(f"Saved final model to {final_path}")
                if history and len(history['val_loss']) > 0:
                    print(f"  (Final val loss: {history['val_loss'][-1]:.6f})")
            mlflow.pytorch.log_model(model, name="model_final")
    
    # ========================================================================
    # STEP 10: GENERATE VISUALIZATIONS
    # ========================================================================
    if generate_plots:
        visualize_predictions(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            num_samples=num_plot_samples,
            save_dir=plot_save_dir,
            random_seed=seed_used,
            error_names=error_names,
            warmup_timesteps=resolved_warmup,
        )
    mlflow.end_run()
    return model, history, seed_used


if __name__ == "__main__":
    print("Training Pipeline v10")
    print("See logging_guide.py for logging documentation")
    print("See test_suite.py for testing framework")