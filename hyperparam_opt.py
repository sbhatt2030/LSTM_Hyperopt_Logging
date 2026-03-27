import mlflow
import os
import sys
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter, Repeater
from ray.tune.search.optuna.optuna_search import OptunaSearch
from mlflow.tracking import MlflowClient
import time
import pandas as pd
from run_experiment import run_experiment
import math

FIXED_PARAMS = {
    "target_timesteps": 800,
    "downsample_factor": 10,
    "base_loss": "mse",
    "hidden_derivative_weight": 0.0,
    "jacobian_weight": 0.0,
    "model_type": "gru",
}

# ── PRETRAINED MODEL & FIXED ARCHITECTURE ─────────────────────────────────────
# Every trial warm-starts from this checkpoint. Set to None to train from scratch.
LOAD_MODEL_PATH   = "/home/bhatts/LSTM_Hyperopt_Logging/~/mlartifacts/87/models/m-bde0ac59b28946ef8f6144a182dd9c9d/artifacts/data/model.pth"   # ← edit this
LOAD_MODEL_STRICT = True   # False = partial load if architecture ever differs
FIXED_HIDDEN_SIZE = 64     # ← match the pretrained model's hidden_size
FIXED_NUM_LAYERS  = 2      # ← match the pretrained model's num_layers
# ─────────────────────────────────────────────────────────────────────────────

def params_match_fixed(run_params: dict, fixed: dict, tol: float = 1e-6) -> bool:
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


def objective(hyperopt_params):
    extra_configs = {k: v for k, v in hyperopt_params.items() if k != '__trial_index__'}
    extra_configs['save_model_path'] = f'models/exp_{hyperopt_params["__trial_index__"]}'
    extra_configs['data_file_path'] = 'data/complete_error_prediction_data.npy'
    extra_configs['verbose'] = 0
    extra_configs['downsample_factor'] = 10
    extra_configs['target_timesteps'] = 800
    extra_configs['device'] = 'cuda'
    extra_configs['mlflow_experiment_id'] = ExpID
    extra_configs['batch_size'] = 2048
    extra_configs['base_loss'] = "mse"
    extra_configs['hidden_derivative_weight'] = 0
    extra_configs['jacobian_weight'] = 0
    extra_configs['model_type'] = 'gru'
    extra_configs['hidden_size'] = FIXED_HIDDEN_SIZE
    extra_configs['num_layers']  = FIXED_NUM_LAYERS
    if LOAD_MODEL_PATH is not None:
        extra_configs['load_model_path']   = LOAD_MODEL_PATH
        extra_configs['load_model_strict'] = LOAD_MODEL_STRICT
    final_val_loss = run_experiment('experiment.csv', 'config.json', -1, extra_configs)
    print("loss: " + str(final_val_loss))
    tune.report({"best val loss": final_val_loss})


def build_search_config(csv_path: str = "search.csv") -> dict:
    """Build Ray Tune search config from search.csv."""
    search_df = pd.read_csv(csv_path)
    search_config = {}
    for _, row in search_df.iterrows():
        name = str(row["Name"])
        dist = str(row["type"])
        if dist == "uniform":
            search_config[name] = tune.uniform(float(row["1"]), float(row["2"]))
        elif dist == "loguniform":
            search_config[name] = tune.loguniform(float(row["1"]), float(row["2"]))
        elif dist == "choice":
            search_config[name] = tune.choice([str(row["1"]), str(row["2"])])
        elif dist == "randint":
            search_config[name] = tune.randint(int(row["1"]), int(row["2"]))
        elif dist == "randn":
            search_config[name] = tune.randn(float(row["1"]), float(row["2"]))
    return search_config


def get_distribution_bounds(search_config: dict) -> dict:
    """Extract (low, high) bounds from Ray Tune search space for validation."""
    bounds = {}
    for k, v in search_config.items():
        if hasattr(v, 'lower') and hasattr(v, 'upper'):
            bounds[k] = (v.lower, v.upper)
    return bounds


def is_within_bounds(point: dict, bounds: dict) -> bool:
    for k, (low, high) in bounds.items():
        if k in point:
            if not (low <= float(point[k]) <= high):
                return False
    return True


def load_history_from_mlflow(tracking_uri: str, metric: str = "best val loss") -> tuple[list[dict], list[float]]:
    """
    Pull all completed runs from MLflow and return them in the format
    OptunaSearch expects: (points_to_evaluate, evaluated_rewards).
    
    Only returns runs that have ALL the hyperparameters in search_config,
    and have a valid recorded metric.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # The params OptunaSearch is searching over — must match search_config keys.
    # hidden_size and num_layers are now fixed (not searched), so excluded here
    # so that older runs that logged them can still be loaded as history.
    SEARCH_PARAMS = [
        "l2_weight", "dropout", "learning_rate"
    ]
    
    all_experiments = client.search_experiments()
    all_run_ids = [e.experiment_id for e in all_experiments]
    
    runs = client.search_runs(
        experiment_ids=all_run_ids,
        filter_string=f"metrics.`{metric}` > 0",  # only completed runs
        order_by=["start_time ASC"]
    )
    
    points = []
    rewards = []
    skipped = 0
    
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        
        # Skip if metric missing
        if metric not in metrics:
            skipped += 1
            continue
        
        # Skip if any search param is missing from this run
        if not all(p in params for p in SEARCH_PARAMS):
            skipped += 1
            continue
 
        # Skip runs that used different fixed settings (different architecture,
        # loss type, etc.) — their rewards aren't comparable to this search
        fixed_to_check = {
            **FIXED_PARAMS,
            'hidden_size': FIXED_HIDDEN_SIZE,
            'num_layers':  FIXED_NUM_LAYERS,
        }
        if not params_match_fixed(params, fixed_to_check):
            skipped += 1
            continue
        
        try:
            point = {
                "l2_weight":    float(params["l2_weight"]),
                "dropout":      float(params["dropout"]),
                "learning_rate":float(params["learning_rate"]),
            }
            loss = float(metrics[metric])
            points.append(point)
            rewards.append(loss)
        except (ValueError, KeyError):
            skipped += 1
            continue
    
    print(f"Loaded {len(points)} historical runs from MLflow (skipped {skipped})")
    return points, rewards


def main():
    print("HyperParam_Opt started")
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = "0"

    # --- Step 1: Build search config ---
    search_config = build_search_config("search.csv")
    search_keys = set(search_config.keys())
    bounds = get_distribution_bounds(search_config)
    print(f"Search space keys: {search_keys}")
    print(f"Bounds: {bounds}")

    # --- Step 2: Load and filter history from MLflow ---
    points, rewards = load_history_from_mlflow(
        "http://hurco-lambda.hurco.com:8090"
    )

    filtered_points = []
    filtered_rewards = []
    for point, reward in zip(points, rewards):
        stripped = {k: point[k] for k in search_keys if k in point}
        if len(stripped) != len(search_keys):
            continue  # missing a key after stripping
        if not is_within_bounds(stripped, bounds):
            continue
        filtered_points.append(stripped)
        filtered_rewards.append(reward)

    print(f"After filtering to current bounds: {len(filtered_points)} usable historical points")

    # --- Step 3: Hand-picked starting point ---
    new_point = {
        "l2_weight": 1e-3,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    }

    if filtered_points:
        # Use history — new_point not needed
        points_to_evaluate = filtered_points
        evaluated_rewards = filtered_rewards
    else:
        # No usable history — validate and seed with hand-picked point
        if not is_within_bounds(new_point, bounds):
            print("Warning: hand-picked new_point is out of current bounds — skipping it")
            points_to_evaluate = None
        else:
            points_to_evaluate = [new_point]
        evaluated_rewards = None

    # --- Step 4: Build search algorithm ---
    algo = OptunaSearch(
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        mode="min",
        metric="best val loss",
    )
    algo = Repeater(algo, repeat=4)
    algo = ConcurrencyLimiter(algo, max_concurrent=5)

    # --- Step 5: Run tuner ---
    num_samples = 300
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective),
            resources={"gpu": 1, "cpu": 1},
        ),
        param_space=search_config,
        tune_config=tune.TuneConfig(
            metric="best val loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
    )
    results = tuner.fit()
    print(results)


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    uri = "http://hurco-lambda.hurco.com:8090"
    mlflow.set_tracking_uri(uri)
    ExpName = time.asctime()
    mlflow.set_experiment(ExpName)
    exp = mlflow.get_experiment_by_name(ExpName)
    if exp:
        ExpID = exp.experiment_id  # type: ignore
    else:
        print("experiment not found")
        sys.exit(0)
    mlflow.enable_system_metrics_logging()
    print(f"Training on: {uri} with experiment id {ExpID}")
    main()