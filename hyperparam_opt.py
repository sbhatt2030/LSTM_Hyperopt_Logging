"""
Hyperparameter optimisation — prediction and compensation_v3 pipelines.

Configuration model
-------------------
Everything lives in one PARAM_TABLE per pipeline. Each entry is:

    "param_name": (kind, value_or_range, initial)

    kind = "fixed"  -> passed to every trial, used to filter MLflow history
    kind = "search" -> searched by Optuna; initial seeds the TPE warm-start
    kind = "trial"  -> passed every trial, NOT used to filter MLflow history
                       (infrastructure settings: device, verbose, etc.)

FIXED_PARAMS, SEARCH_SPACE, INITIAL_POINT, and TRIAL_PARAMS are all derived
from the table automatically. There is no separate place to keep in sync.

Checkpoint warm-starting
------------------------
Set CHECKPOINT["path"] to a saved model file to lock the architecture and
warm-start weights. Any param listed in CHECKPOINT["locked"] that is currently
"search" in the table is automatically moved to fixed. Set path=None to
train from scratch.

Multi-GPU (compensation_v3)
---------------------------
Each compensation trial needs two GPUs (gpu_pred + gpu_comp). Ray sets
CUDA_VISIBLE_DEVICES per-trial so inside each trial process the two GPUs are
always visible as cuda:0 and cuda:1.
  4 GPUs + compensation_v3: MAX_CONCURRENT = 2  (2 trials x 2 GPUs)
  4 GPUs + prediction:      MAX_CONCURRENT = 4  (4 trials x 1 GPU)
"""

import math
import os
import sys
import time

import mlflow
import ray
from mlflow.tracking import MlflowClient
from ray import tune
from ray.tune.search import ConcurrencyLimiter, Repeater
from ray.tune.search.optuna.optuna_search import OptunaSearch

from run_experiment import run_experiment

# ============================================================================
# TOP-LEVEL SWITCH  -- only thing you need to change when swapping pipelines
# ============================================================================

# 'prediction'      -> 1 GPU per trial
# 'compensation_v3' -> 2 GPUs per trial
PIPELINE = 'compensation_v3'

TRACKING_URI = "http://hurco-lambda.hurco.com:8090"

# ============================================================================
# PARAM TABLES
#
# Each entry:  "name": (kind, value_or_dist, initial)
#
#   kind = "fixed"  locked; every trial gets this value; MLflow history filter
#   kind = "search" Optuna searches this; initial seeds the TPE
#   kind = "trial"  every trial gets this value; NOT in MLflow filter
#
# POW2 params: the search space stores the exponent (e.g. 5 -> 2**5 = 32).
# Add the param name to POW2_PARAMS below.
# ============================================================================

PARAM_TABLE_PREDICTION = {
    # ── Fixed ─────────────────────────────────────────────────────────────────
    "data_file_path":           ("fixed",  "data/complete_error_prediction_data.npy", None),
    "target_timesteps":         ("fixed",  8000,                                      None),
    "downsample_factor":        ("fixed",  1,                                         None),
    "axis_selection":           ("fixed",  "x",                                       None),
    "base_loss":                ("fixed",  "mse",                                     None),
    "jacobian_weight":          ("fixed",  0,                                         None),
    "hidden_derivative_weight": ("fixed",  0,                                         None),
    "model_type":               ("fixed",  "gru",                                     None),
    "num_layers":               ("fixed",  2,                                         None),
    "hidden_size":              ("fixed",  32,                                        None),
    "run_purpose":              ("fixed",  "prediction",                              None),
    # ── Searched ──────────────────────────────────────────────────────────────
    "l2_weight":                ("search", tune.loguniform(1e-6, 1e-2),               5e-4),
    "dropout":                  ("search", tune.uniform(0.0, 0.5),                    0.1 ),
    "learning_rate":            ("search", tune.loguniform(1e-6, 1e-1),               5e-3),
    # Uncomment to search architecture (also remove from fixed above):
    # "hidden_size":            ("search", tune.randint(4, 8),                        5   ),  # pow2
    # "num_layers":             ("search", tune.randint(1, 5),                        2   ),
    # ── Trial (infra) ─────────────────────────────────────────────────────────
    "batch_size":               ("trial",  None,                                      None),  # None -> dynamic
    "device":                   ("trial",  "cuda",                                    None),
    "verbose":                  ("trial",  0,                                         None),
    "generate_plots":           ("trial",  False,                                     None),
}

PARAM_TABLE_COMPENSATION = {
    # ── Fixed ─────────────────────────────────────────────────────────────────
    "data_path":                ("fixed",  "data/comp_dataset.npy",                  None),
    "error_pred_checkpoint":    ("fixed",  "checkpoints/error_pred_x.pt",            None),
    "axis_selection":           ("fixed",  "x",                                      None),
    "downsample_factor":        ("fixed",  1,                                        None),
    "pred_hidden_size":         ("fixed",  32,                                       None),
    "pred_num_layers":          ("fixed",  2,                                        None),
    "comp_num_layers":          ("fixed",  1,                                        None),
    "run_purpose":              ("fixed",  "compensation_v3",                        None),
    # ── Searched ──────────────────────────────────────────────────────────────
    "lambda_mlp":               ("search", tune.loguniform(1e-5, 1e-1),              1e-4),
    "lambda_gru":               ("search", tune.loguniform(1e-6, 1e-3),              1e-5),
    "learning_rate":            ("search", tune.loguniform(1e-4, 1e-2),              1e-3),
    "comp_hidden_size":         ("search", tune.randint(4, 8),                       5   ),  # pow2: 2**5=32
    # Uncomment to search depth:
    # "comp_num_layers":        ("search", tune.randint(1, 4),                       1   ),
    # ── Trial (infra) ─────────────────────────────────────────────────────────
    "batch_size":               ("trial",  None,                                     None),  # None -> dynamic
    "gpu_pred":                 ("trial",  "cuda:0",                                 None),  # within CUDA_VISIBLE_DEVICES
    "gpu_comp":                 ("trial",  "cuda:1",                                 None),
    "verbose":                  ("trial",  0,                                        None),
    "generate_plots":           ("trial",  False,                                    None),
}

PARAM_TABLE = (
    PARAM_TABLE_COMPENSATION if PIPELINE == 'compensation_v3'
    else PARAM_TABLE_PREDICTION
)

# Params stored as exponents in the search space (actual value = 2**exponent).
POW2_PARAMS = {"hidden_size", "comp_hidden_size"}

# ============================================================================
# CHECKPOINT
#
# path=None  -> train from scratch
# path=str   -> load weights; params in "locked" are moved from search->fixed
# ============================================================================

CHECKPOINT = {
    "path": None,
    # Values to lock when a checkpoint is loaded. Only params currently
    # "search" in the table are affected; others are ignored silently.
    "locked": {
        "hidden_size":      32,   # prediction (actual value, not exponent)
        "comp_hidden_size": 32,   # compensation_v3 (actual value, not exponent)
    },
}

# ============================================================================
# RAY TUNE SETTINGS
# ============================================================================

NUM_SAMPLES    = 100
REPEAT         = 2
GPUS_PER_TRIAL = 2 if PIPELINE == 'compensation_v3' else 1
MAX_CONCURRENT = 2 if PIPELINE == 'compensation_v3' else 4

# ============================================================================
# DERIVED CONFIG  (computed from the table — do not edit below this line)
# ============================================================================

def _derive(
    table:      dict,
    checkpoint: dict,
) -> tuple[dict, dict, dict, dict]:
    """
    Derive FIXED_PARAMS, SEARCH_SPACE, INITIAL_POINT, TRIAL_PARAMS from table.

    Checkpoint locking moves architecture params from search -> fixed using
    the value in checkpoint["locked"], and injects load_model_path into trial.
    """
    fixed   = {k: v for k, (kind, v, _) in table.items() if kind == "fixed"}
    search  = {k: v for k, (kind, v, _) in table.items() if kind == "search"}
    initial = {k: i for k, (kind, _, i) in table.items() if kind == "search"}
    trial   = {k: v for k, (kind, v, _) in table.items() if kind == "trial"}

    if checkpoint["path"] is not None:
        for param, locked_val in checkpoint["locked"].items():
            if param in search:
                # Store exponent in fixed if this is a pow2 param,
                # so MLflow filtering compares apples to apples.
                fixed[param] = locked_val
                search.pop(param)
                initial.pop(param, None)
                print(f"  [checkpoint] '{param}' moved search -> fixed = {locked_val}")
        trial["load_model_path"]   = checkpoint["path"]
        trial["load_model_strict"] = True

    return fixed, search, initial, trial


FIXED_PARAMS, SEARCH_SPACE, INITIAL_POINT, TRIAL_PARAMS = _derive(PARAM_TABLE, CHECKPOINT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_search_params(raw: dict) -> dict:
    """Apply 2** transform to pow2 params before passing to train_model."""
    return {
        k: (2 ** int(v) if k in POW2_PARAMS else v)
        for k, v in raw.items()
    }


def _get_bounds(search_space: dict) -> dict:
    return {
        k: (v.lower, v.upper)
        for k, v in search_space.items()
        if hasattr(v, "lower") and hasattr(v, "upper")
    }


def _within_bounds(point: dict, bounds: dict) -> bool:
    return all(
        low <= float(point[k]) <= high
        for k, (low, high) in bounds.items()
        if k in point
    )


def _params_match_fixed(run_params: dict, fixed: dict, tol: float = 1e-6) -> bool:
    for k, v in fixed.items():
        if k not in run_params:
            return False
        try:
            if abs(float(run_params[k]) - float(v)) > tol:
                return False
        except (ValueError, TypeError):
            if str(run_params[k]) != str(v):
                return False
    return True


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(hyperopt_params: dict, exp_id: int, pipeline: str):
    """
    Ray Tune objective — called once per trial.

    Config priority (right wins):
        config.json < FIXED_PARAMS < TRIAL_PARAMS < searched params
    """
    searched = {k: v for k, v in hyperopt_params.items() if k != "__trial_index__"}
    resolved = _resolve_search_params(searched)

    extra_configs = {
        **FIXED_PARAMS,
        **TRIAL_PARAMS,
        **resolved,
        "mlflow_experiment_id": exp_id,
        "save_model_path": f"models/exp_{hyperopt_params['__trial_index__']}",
    }

    final_val_loss = run_experiment(
        "experiment.csv", "config.json", -1,
        extra_configs, pipeline=pipeline,
    )
    print(f"loss: {final_val_loss}")
    tune.report({"best val loss": final_val_loss})


# ── MLflow warm-start ─────────────────────────────────────────────────────────

def load_history_from_mlflow(
    search_keys: set,
    metric:      str = "best val loss",
) -> tuple[list[dict], list[float]]:
    """
    Pull completed MLflow runs matching FIXED_PARAMS for TPE warm-starting.
    POW2_PARAMS are converted back to exponent form to match the search space.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)

    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in client.search_experiments()],
        filter_string=f"metrics.`{metric}` > 0",
        order_by=["start_time ASC"],
    )

    points, rewards, skipped = [], [], 0
    for run in runs:
        params  = run.data.params
        metrics = run.data.metrics

        if metric not in metrics:
            skipped += 1
            continue
        if not _params_match_fixed(params, FIXED_PARAMS):
            skipped += 1
            continue

        try:
            point = {}
            for k in search_keys:
                if k not in params:
                    raise KeyError(k)
                raw = float(params[k])
                # MLflow stores actual value (e.g. 32); convert back to exponent
                point[k] = math.log2(raw) if k in POW2_PARAMS else raw
            points.append(point)
            rewards.append(float(metrics[metric]))
        except (KeyError, ValueError):
            skipped += 1
            continue

    print(f"Loaded {len(points)} historical runs from MLflow (skipped {skipped})")
    return points, rewards


# ── Main ──────────────────────────────────────────────────────────────────────

def main(exp_id: int):
    print(f"\nHyperParam_Opt")
    print(f"  pipeline       : {PIPELINE}")
    print(f"  gpus_per_trial : {GPUS_PER_TRIAL}")
    print(f"  max_concurrent : {MAX_CONCURRENT}")
    print(f"  checkpoint     : {CHECKPOINT['path']}")
    print(f"  fixed params   : {list(FIXED_PARAMS.keys())}")
    print(f"  searched params: {list(SEARCH_SPACE.keys())}")
    print(f"  initial point  : {INITIAL_POINT}")
    print(f"  trial params   : {list(TRIAL_PARAMS.keys())}")

    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    search_keys = set(SEARCH_SPACE.keys())
    bounds      = _get_bounds(SEARCH_SPACE)

    # Load and filter MLflow history for TPE warm-start
    points, rewards = load_history_from_mlflow(search_keys)
    filtered_points, filtered_rewards = [], []
    for pt, rw in zip(points, rewards):
        stripped = {k: pt[k] for k in search_keys if k in pt}
        if len(stripped) != len(search_keys):
            continue
        if not _within_bounds(stripped, bounds):
            continue
        filtered_points.append(stripped)
        filtered_rewards.append(rw)
    print(f"\nUsable historical points after bound filtering: {len(filtered_points)}")

    # Seed Optuna TPE
    if filtered_points:
        points_to_evaluate = filtered_points
        evaluated_rewards  = filtered_rewards
    else:
        if _within_bounds(INITIAL_POINT, bounds):
            print("No MLflow history — seeding with INITIAL_POINT")
            points_to_evaluate = [INITIAL_POINT]
        else:
            print("Warning: INITIAL_POINT outside current search bounds — starting cold")
            points_to_evaluate = None
        evaluated_rewards = None

    algo = OptunaSearch(
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        mode="min",
        metric="best val loss",
    )
    algo = Repeater(algo, repeat=REPEAT)
    algo = ConcurrencyLimiter(algo, max_concurrent=MAX_CONCURRENT)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective, exp_id=exp_id, pipeline=PIPELINE),
            resources={"gpu": GPUS_PER_TRIAL, "cpu": 1},
        ),
        param_space=SEARCH_SPACE,
        tune_config=tune.TuneConfig(
            metric="best val loss",
            mode="min",
            search_alg=algo,
            num_samples=NUM_SAMPLES,
        ),
    )
    results = tuner.fit()
    print(results)


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    exp_name = time.asctime()
    mlflow.set_experiment(exp_name)
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        print("MLflow experiment not found — exiting")
        sys.exit(1)

    mlflow.enable_system_metrics_logging()
    print(f"MLflow experiment: {exp_name} (ID: {exp.experiment_id})")

    main(exp_id=exp.experiment_id)
