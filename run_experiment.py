"""
Simple experiment runner - returns only final validation loss.
Designed to work with external hyperparameter tuning and logging systems.

FIXED VERSION: Properly filters out metadata columns from experiment.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import time

def _get_train_fn(pipeline: str):
    if pipeline == 'prediction':
        from train_val_loop_v10 import train_model
    elif pipeline == 'compensation_v3':
        from train_val_loop_comp_v3 import train_model
    else:
        raise ValueError(
            f"Unknown pipeline: '{pipeline}'. "
            f"Valid options: 'prediction', 'compensation_v3'."
        )
    return train_model


# Keys accepted by each pipeline's train_model.
# Any config key NOT in this set is silently dropped before the call,
# preventing "unexpected keyword argument" crashes when shared or
# pipeline-specific keys don't exist in the other pipeline's signature.
_VALID_KEYS = {
    'prediction': {
        'data_file_path', 'target_timesteps', 'axis_selection',
        'downsample_factor', 'include_velocity', 'include_acceleration',
        'motion_threshold', 'min_position_change', 'train_val_split',
        'random_seed', 'run_purpose',
        'model_type', 'hidden_size', 'num_layers', 'dropout', 'bidirectional',
        'load_model_path', 'load_model_strict',
        'base_loss', 'l2_weight', 'jacobian_weight', 'jacobian_every_n_batches',
        'hidden_derivative_weight', 'hidden_derivative_every_n_batches',
        'max_abs_weight', 'warmup_timesteps',
        'num_epochs', 'batch_size', 'learning_rate', 'optimizer_type',
        'optimizer_kwargs', 'gradient_clip_value', 'device',
        'mlflow_experiment_id', 'tracking_uri',
        'early_stopping_patience', 'early_stopping_min_delta',
        'validation_frequency', 'num_workers',
        'verbose', 'return_history', 'save_model_path', 'save_best_only',
        'generate_plots', 'num_plot_samples', 'plot_save_dir', 'error_names',
    },
    'compensation_v3': {
        'data_dir',
        'axis_selection', 'downsample_factor', 'random_seed',
        'error_pred_checkpoint', 'pred_hidden_size', 'pred_num_layers',
        'comp_hidden_size', 'comp_num_layers', 'comp_dropout',
        'lambda_mlp', 'lambda_gru', 'warmup_timesteps',
        'learning_rate', 'num_epochs', 'batch_size', 'gradient_clip_value',
        'early_stopping_patience', 'early_stopping_min_delta',
        'validation_frequency',
        'save_model_path', 'save_best_only',
        'plot_save_dir', 'generate_plots', 'num_plot_samples',
        'gpu_pred', 'gpu_comp',
        'mlflow_tracking_uri', 'mlflow_experiment_name', 'mlflow_run_name',
        'mlflow_experiment_id',
        'verbose', 'num_workers', 'return_history',
    },
}


def _filter_config(config: dict, pipeline: str) -> dict:
    """Drop any keys not accepted by this pipeline's train_model."""
    valid = _VALID_KEYS.get(pipeline, set())
    dropped = [k for k in config if k not in valid]
    if dropped:
        print(f"  [config filter] dropping {len(dropped)} key(s) not valid "
              f"for '{pipeline}': {dropped}")
    return {k: v for k, v in config.items() if k in valid}


def load_config(config_path='config.json'):
    """Load shared configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def flatten_config(config, pipeline: str = 'prediction'):
    """
    Flatten the new pipeline-aware config structure.

    Merge order (right wins):
        shared  <  pipeline-specific section

    Keys starting with '_' (e.g. _comment) are dropped.
    Unknown pipeline names fall back to shared-only.
    """
    flat = {}
    # shared defaults
    for k, v in config.get('shared', {}).items():
        if not k.startswith('_'):
            flat[k] = v
    # pipeline-specific overrides
    for k, v in config.get(pipeline, {}).items():
        if not k.startswith('_'):
            flat[k] = v
    # Legacy flat format (old config.json has no 'shared' key)
    if 'shared' not in config:
        for section, params in config.items():
            if isinstance(params, dict):
                flat.update(params)
    return flat


def load_experiments(csv_path='experiment.csv'):
    """Load experiment configurations from CSV."""
    df = pd.read_csv(csv_path)
    return df


def merge_configs(base_config, experiment_row):
    """
    Merge base config with experiment-specific settings.
    
    CSV values override config.json values.
    NaN values in CSV are ignored (config.json value used).
    Metadata columns (experiment_name, notes) are excluded.
    
    Args:
        base_config: Dict from config.json (flattened)
        experiment_row: Single row from experiment.csv (as Series)
    
    Returns:
        Merged config dict with only valid training parameters
    """
    config = base_config.copy()
    
    # Columns that are metadata, not training parameters
    # These should NOT be passed to train_model()
    metadata_columns = {'experiment_name', 'notes'}
    
    # Convert experiment row to dict, dropping NaN values and metadata
    exp_dict = experiment_row.to_dict()
    exp_dict = {k: v for k, v in exp_dict.items() 
                if pd.notna(v) and k not in metadata_columns}
    
    # Update config with experiment values
    config.update(exp_dict)
    
    return config


def run_experiment(csv_path: str, config_path: str, experiment_idx: int, extra_configs: dict = {}, pipeline: str = 'prediction') -> float:
    """
    Simple entry point for running a single experiment.
    
    Args:
        csv_path: Path to experiment.csv file
        config_path: Path to config.json file  
        experiment_idx: Row index in CSV (0-based)
        is_hyper: booleen flag to set mlflow behavior. defaults to false
    
    Returns:
        final_val_loss: Final validation loss (float)
        
    Raises:
        ValueError: If experiment_idx is out of range
        RuntimeError: If training fails
        
    Example:
        >>> final_loss = run_experiment('experiment.csv', 'config.json', 0)
        >>> print(f"Final validation loss: {final_loss:.6f}")
    """
    # Load configurations
    base_config = flatten_config(load_config(config_path), pipeline=pipeline)
    experiments_df = load_experiments(csv_path)
    
    # Validate index
    if experiment_idx < -1 or experiment_idx >= len(experiments_df):
        raise ValueError(
            f"experiment_idx={experiment_idx} out of range. "
            f"CSV has {len(experiments_df)} experiments (indices 0-{len(experiments_df)-1})"
        )
    if experiment_idx != -1:
        # Get experiment row
        experiment_row = experiments_df.iloc[experiment_idx]
        exp_name = experiment_row.get('experiment_name', f'experiment_{experiment_idx}')
        # Merge configurations (this now filters out metadata)
        config = merge_configs(base_config, experiment_row)
    else:
        experiment_row = -1
        exp_name = "HyperParam_Opt"
        config = base_config
    
    # Configure for minimal output
    config['verbose'] = 1  # Disable detailed logging
    config['generate_plots'] = True  # No visualization
    config['return_history'] = True  # Need history to get val_loss

    for key,value in extra_configs.items():
        config[key] = value
    try:
        # Train model
        train_model = _get_train_fn(pipeline)
        config = _filter_config(config, pipeline)
        result = train_model(**config)
        model, history, seed = result if len(result) == 3 else (*result, None)
        
        # Get final validation loss
        # All pipelines use 'val_loss' as the key (compensation_v3 also uses this).
        val_history = history.get('val_loss') if history else None
        if val_history and len(val_history) > 0:
            final_val_loss = val_history[-1]
        else:
            raise RuntimeError(
                f"No validation loss returned for experiment '{exp_name}'. "
                f"Check that validation_frequency > 0 in config and return_history=True."
            )
        
        # Verify it's a valid number
        if np.isnan(final_val_loss) or np.isinf(final_val_loss):
            raise RuntimeError(
                f"Invalid validation loss: {final_val_loss} (NaN or Inf)"
            )
        
        return float(final_val_loss)
        
    except TypeError as e:
        # This catches "unexpected keyword argument" errors
        if "unexpected keyword argument" in str(e):
            # Extract which parameter caused the issue
            import re
            match = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if match:
                bad_param = match.group(1)
                raise RuntimeError(
                    f"Training failed for experiment '{exp_name}': "
                    f"Parameter '{bad_param}' is in CSV but not valid for train_model(). "
                    f"Add it to metadata_columns in merge_configs() or remove from CSV."
                ) from e
        raise RuntimeError(
            f"Training failed for experiment '{exp_name}': {str(e)}"
        ) from e
    
    except Exception as e:
        raise RuntimeError(
            f"Training failed for experiment '{exp_name}': {str(e)}"
        ) from e


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python run_experiment.py <experiment.csv> <config.json> <experiment_idx>")
        print("\nExample:")
        print("  python run_experiment.py experiment.csv config.json 0")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    config_path = sys.argv[2]
    experiment_idx = int(sys.argv[3])
    
    try:
        final_val_loss = run_experiment(csv_path, config_path, experiment_idx)
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {experiment_idx} COMPLETE")
        print(f"{'='*60}")
        print(f"Final Validation Loss: {final_val_loss:.6f}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)