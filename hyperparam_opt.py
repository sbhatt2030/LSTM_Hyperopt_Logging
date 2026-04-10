import mlflow
import os
import sys
import math
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter, Repeater
from ray.tune.search.optuna.optuna_search import OptunaSearch
from mlflow.tracking import MlflowClient
import time
from run_experiment import run_experiment

# ============================================================================
# ✏️  CONFIGURE EVERYTHING HERE — nothing else should need editing
# ============================================================================

TRACKING_URI = "http://hurco-lambda.hurco.com:8090"

# --- Fixed params: passed to every trial AND used to filter MLflow history.
# Changing any of these makes historical runs incomparable, so they act as
# the filter criteria when loading warm-start points from MLflow.
FIXED_PARAMS = {
    "data_file_path":           "/home/bhatts/LSTM_Hyperopt_Logging/data/clean_final.npy",
    "target_timesteps":         8000,
    "downsample_factor":        1,
    "axis_selection":           "x",
    "model_type":               "gru",
    "num_layers":               4,
    "base_loss":                "mse",
    "max_abs_weight":           1,
    "jacobian_weight":          18.55961201319177,
    "hidden_derivative_weight": 22971.9626260835,
}

# --- Trial params: passed to every trial but NOT used to filter MLflow history.
# These are housekeeping / infrastructure settings that don't affect the loss.
TRIAL_PARAMS = {
    "batch_size": 128,
    "device":     "cuda",
    "verbose":    0,
}

# --- Warm-start checkpoint.
# Set LOAD_MODEL_PATH to None to train from scratch.
# When set, hidden_size is fixed to CHECKPOINT_HIDDEN_SIZE (must match the
# checkpoint architecture) and is NOT searched.
LOAD_MODEL_PATH        = "/home/bhatts/LSTM_Hyperopt_Logging/~/mlartifacts/133/models/m-bb5e6f113e3c4628a89939fa0d4b0bcb/artifacts/data/model.pth"
LOAD_MODEL_STRICT      = True
CHECKPOINT_HIDDEN_SIZE = 64   # ignored when LOAD_MODEL_PATH is None

# --- Search space.
# hidden_size uses tune.randint over exponents — actual value = 2**n.
# e.g. randint(4, 8) searches over {16, 32, 64, 128}.
# Remove hidden_size when warm-starting (architecture is locked to checkpoint).
SEARCH_SPACE = {
    "l2_weight":     tune.loguniform(1e-6, 1e-5),
    "dropout":       tune.uniform(0, 0.2),
    "learning_rate": tune.loguniform(1e-6, 1e-5),
    #"jacobian_weight":tune.loguniform(1e-1, 50),
    #"hidden_derivative_weight": tune.loguniform(1e-2, 5e4),
    #"max_abs_weight": tune.loguniform(0.1,10)

}

# --- Hand-picked fallback starting point.
# Used only when no usable MLflow history exists.
# hidden_size here is the exponent (e.g. 6 → 2**6 = 64).
INITIAL_POINT = {
    "l2_weight":     5e-6,
    "dropout":       0.15,
    "learning_rate": 1e-6,
    #"jacobian_weight": 1,
    #"hidden_derivative_weight": 1e2,
    #"max_abs_weight":           2.0

}

# --- Ray Tune settings ---
NUM_SAMPLES    = 50
REPEAT         = 2
MAX_CONCURRENT = 5
PIPELINE = 'compensation' 

# ============================================================================
# INTERNALS — no need to edit below this line
# ============================================================================

# Params stored as exponents in the search space and in MLflow history,
# transformed via 2** before being passed to the model.
POW2_PARAMS = {"hidden_size"}


def _resolve_search_params(raw: dict) -> dict:
    """Apply pow2 transform to any params that need it, pass the rest through."""
    return {
        k: (2 ** int(v) if k in POW2_PARAMS else v)
        for k, v in raw.items()
    }


def _checkpoint_params() -> dict:
    """Checkpoint-related overrides, or empty dict if training from scratch."""
    if LOAD_MODEL_PATH is None:
        return {}
    return {
        "hidden_size":       CHECKPOINT_HIDDEN_SIZE,
        "load_model_path":   LOAD_MODEL_PATH,
        "load_model_strict": LOAD_MODEL_STRICT,
    }


def _fixed_params_for_filtering() -> dict:
    """
    The params historical MLflow runs must match to be comparable.
    = FIXED_PARAMS + hidden_size when warm-starting (architecture is locked).
    TRIAL_PARAMS excluded — they don't affect loss comparability.
    """
    fixed = {**FIXED_PARAMS}
    if LOAD_MODEL_PATH is not None:
        fixed["hidden_size"] = CHECKPOINT_HIDDEN_SIZE
    return fixed


def _get_bounds(search_space: dict) -> dict:
    """Extract numeric (low, high) bounds for filtering historical points."""
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


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(hyperopt_params, exp_id: int, pipeline: str):
    """Ray Tune objective — called once per trial."""
    searched = {k: v for k, v in hyperopt_params.items() if k != "__trial_index__"}
    resolved = _resolve_search_params(searched)

    # Priority (right wins): config.json < FIXED_PARAMS < TRIAL_PARAMS
    #                        < checkpoint < searched params
    extra_configs = {
        **FIXED_PARAMS,
        **TRIAL_PARAMS,
        **_checkpoint_params(),
        **resolved,
        "mlflow_experiment_id": exp_id,
        "save_model_path": f"models/exp_{hyperopt_params['__trial_index__']}",
    }

    final_val_loss = run_experiment("experiment.csv", "config.json", -1, extra_configs, pipeline=pipeline)
    print(f"loss: {final_val_loss}")
    tune.report({"best val loss": final_val_loss})


# ── MLflow warm-start history ─────────────────────────────────────────────────

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


def load_history_from_mlflow(
    search_keys: set,
    metric: str = "best val loss",
) -> tuple[list[dict], list[float]]:
    """
    Pull completed MLflow runs that match _fixed_params_for_filtering().
    Returns only the keys currently being searched. POW2_PARAMS are converted
    back to exponent form so they live in the same space as the search distribution.
    """
    client         = MlflowClient(tracking_uri=TRACKING_URI)
    fixed_to_check = _fixed_params_for_filtering()

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

        if not _params_match_fixed(params, fixed_to_check):
            skipped += 1
            continue

        try:
            point = {}
            for k in search_keys:
                if k not in params:
                    raise KeyError(k)
                raw = float(params[k])
                # MLflow stores actual value (e.g. 64); convert back to exponent
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
    print("HyperParam_Opt started")
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    search_keys = set(SEARCH_SPACE.keys())
    bounds      = _get_bounds(SEARCH_SPACE)
    print(f"Search space:  {search_keys}")
    print(f"Fixed params:  {FIXED_PARAMS}")
    print(f"Trial params:  {TRIAL_PARAMS}")
    print(f"Checkpoint:    {LOAD_MODEL_PATH}")

    # 1. Load MLflow history and filter to current search bounds
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
    print(f"Usable historical points after bound filtering: {len(filtered_points)}")

    # 2. Seed Optuna
    if filtered_points:
        points_to_evaluate = filtered_points
        evaluated_rewards  = filtered_rewards
    else:
        if _within_bounds(INITIAL_POINT, bounds):
            print("No MLflow history — seeding with INITIAL_POINT")
            points_to_evaluate = [INITIAL_POINT]
        else:
            print("Warning: INITIAL_POINT is outside current bounds — starting cold")
            points_to_evaluate = None
        evaluated_rewards = None

    # 3. Build search algorithm
    algo = OptunaSearch(
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        mode="min",
        metric="best val loss",
    )
    algo = Repeater(algo, repeat=REPEAT)
    algo = ConcurrencyLimiter(algo, max_concurrent=MAX_CONCURRENT)

    # 4. Launch
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective, exp_id=exp_id, pipeline=PIPELINE),
            resources={"gpu": 1, "cpu": 1},
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