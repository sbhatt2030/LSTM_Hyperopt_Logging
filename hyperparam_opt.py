
import mlflow
import os
import sys
import ray
from ray import  tune
from ray.tune.search import ConcurrencyLimiter, Repeater
from ray.tune.search.optuna.optuna_search import OptunaSearch
from mlflow.tracking import MlflowClient
import time
import pandas as pd
from run_experiment import run_experiment
import math

FIXED_PARAMS = {
    "target_timesteps": 8000,
    "downsample_factor": 1,
    "base_loss": "mse",
    "hidden_derivative_weight": 0.0,
    "jacobian_weight": 0.0,
    "model_type": "gru",
}

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
    #print(hyperopt_params)
    extra_configs = {k:v for k,v in hyperopt_params.items() if k != '__trial_index__'}
    extra_configs['save_model_path'] = f'models/exp_{hyperopt_params['__trial_index__']}'
    extra_configs['data_file_path'] = 'data/complete_error_prediction_data.npy'
    extra_configs['verbose'] = 0
    extra_configs['downsample_factor'] = 1
    extra_configs['target_timesteps'] = 8000
    extra_configs['device'] = 'cuda'
    extra_configs['mlflow_experiment_id'] = ExpID
    extra_configs['batch_size'] = 1024
    extra_configs['base_loss'] = "mse"
    extra_configs['hidden_derivative_weight'] = 0
    extra_configs['jacobian_weight'] = 0
    extra_configs['model_type'] = 'gru'
    final_val_loss = run_experiment('experiment.csv', 'config.json', -1, extra_configs)
    print("loss: " + str(final_val_loss))
    tune.report({"mean_test_inst_loss": final_val_loss})
    return
 
def load_history_from_mlflow(tracking_uri: str, metric: str = "best val loss") -> tuple[list[dict], list[float]]:
    """
    Pull all completed runs from MLflow and return them in the format
    OptunaSearch expects: (points_to_evaluate, evaluated_rewards).
    
    Only returns runs that have ALL the hyperparameters in search_config,
    and have a valid recorded metric.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # The params OptunaSearch is searching over — must match search_config keys
    SEARCH_PARAMS = [
        "base_loss", "l2_weight", "jacobian_weight", "hidden_derivative_weight",
        "model_type", "hidden_size", "num_layers", "dropout", "learning_rate"
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
        
        try:
            point = {
                "base_loss":                  str(params["base_loss"]),
                "l2_weight":                  float(params["l2_weight"]),
                "jacobian_weight":            float(params["jacobian_weight"]),
                "hidden_derivative_weight":   float(params["hidden_derivative_weight"]),
                "model_type":                 str(params["model_type"]),
                # hidden_size was stored as 2^x in objective(), reverse it
                "hidden_size":                int(round(math.log2(float(params["hidden_size"])))),
                "num_layers":                 int(params["num_layers"]),
                "dropout":                    float(params["dropout"]),
                "learning_rate":              float(params["learning_rate"]),
            }
            loss = float(metrics[metric])
            points.append(point)
            rewards.append(loss)
        except (ValueError, KeyError):
            skipped += 1
            continue
    
    print(f"Loaded {len(points)} historical runs from MLflow (skipped {skipped})")
    return points, rewards

def get_distribution_bounds(search_config: dict) -> dict:
    """Extract (low, high) bounds from Ray Tune search space for validation."""
    bounds = {}
    for k, v in search_config.items():
        if hasattr(v, 'lower') and hasattr(v, 'upper'):  # uniform, loguniform, randint
            bounds[k] = (v.lower, v.upper)
    return bounds

def is_within_bounds(point: dict, bounds: dict) -> bool:
    for k, (low, high) in bounds.items():
        if k in point:
            if not (low <= point[k] <= high):
                return False
    return True
def main():
    print("HyperParam_Opt started")
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = "0"

    # --- Step 1: Build search_config from search.csv first ---
    # Must be done before processing history so we know which keys are valid
    SearchDf = pd.read_csv("search.csv")
    search_config = {}
    for index in SearchDf.index:
        k = SearchDf.loc[index]
        if str(k["type"]) == "uniform":
            search_config[str(k["Name"])] = tune.uniform(float(k["1"]), float(k["2"]))
        if str(k["type"]) == "loguniform":
            search_config[str(k["Name"])] = tune.loguniform(float(k["1"]), float(k["2"]))
        if str(k["type"]) == "choice":
            search_config[str(k["Name"])] = tune.choice([str(k["1"]), str(k["2"])])
        if str(k["type"]) == "randint":
            search_config[str(k["Name"])] = tune.randint(int(k["1"]), int(k["2"]))
        if str(k["type"]) == "randn":
            search_config[str(k["Name"])] = tune.randn(float(k["1"]), float(k["2"]))

    search_keys = set(search_config.keys())
    print(f"Search space keys: {search_keys}")
    search_keys = set(search_config.keys())
    bounds = get_distribution_bounds(search_config)
    print(f"Search space keys: {search_keys}")
    print(f"Bounds: {bounds}")
    # --- Step 2: Load history from MLflow and strip to search keys ---
    points, rewards = load_history_from_mlflow("http://hurco-lambda.hurco.com:8090")

    filtered_points = []
    filtered_rewards = []
    for point, reward in zip(points, rewards):
        if not search_keys.issubset(point.keys()):
            continue
        stripped = {k: point[k] for k in search_keys}
        if not is_within_bounds(stripped, bounds):  # <-- drop out-of-bounds
            continue
        filtered_points.append(stripped)
        filtered_rewards.append(reward)

    print(f"After stripping to search keys: {len(filtered_points)} usable historical points")

    # --- Step 3: Append hand-picked starting point (no reward yet) ---
    new_point = {
        "l2_weight": 1e-3,
        "hidden_size": 5,
        "num_layers": 3,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    }

    if not filtered_points:
        # DB is empty or no matching runs — just seed with hand-picked point
        points_to_evaluate = [new_point]
        evaluated_rewards = None
    else:
        # Append hand-picked point at end; it has no reward so stays outside evaluated_rewards
        points_to_evaluate = filtered_points
        evaluated_rewards = filtered_rewards  # length = filtered_points only, not new_point

    # --- Step 4: Build search algorithm ---
    algo = OptunaSearch(
        points_to_evaluate=points_to_evaluate,
        evaluated_rewards=evaluated_rewards,
        mode="min",
        metric="mean_test_inst_loss"
    )
    algo = Repeater(algo, repeat=4)
    algo = ConcurrencyLimiter(algo, max_concurrent=5)

    # --- Step 5: Run tuner ---
    num_samples = 50
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective),
            resources={"gpu": 4, "cpu": 4}
        ),
        param_space=search_config,
        tune_config=tune.TuneConfig(
            metric="mean_test_inst_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples
        )
    )
    results = tuner.fit()
    print(results) 

if __name__ == "__main__":
    ray.init(include_dashboard=False)
    
    uri = "http://hurco-lambda.hurco.com:8090"
    mlflow.set_tracking_uri(uri)
    ExpName = time.asctime()
    mlflow.set_experiment(ExpName)
    if mlflow.get_experiment_by_name(ExpName):
      ExpID = mlflow.get_experiment_by_name(ExpName).experiment_id # type: ignore
    else:
       print("experiment not found")
       sys.exit(0)
    mlflow.enable_system_metrics_logging()
    print(f"Training on: {uri} with experiment id {ExpID}")
    main()