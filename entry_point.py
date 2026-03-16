"""
Entry point for running experiments.
Simple interface: provide CSV, config, and experiment index → get final validation loss.
"""

from run_experiment import run_experiment
import mlflow
import time

def init_mlflow(Tracking_URI: str) -> int:
    """
    initializes the mlflow tracking, expiriment and starts system metrics logging    
    Args:
        Tracking_URI: URI for the tracking database
    Returns:
        Nothing
    """
    mlflow.set_tracking_uri(Tracking_URI)
    print(Tracking_URI)
    exp_name = time.asctime()
    mlflow.set_experiment(exp_name)
    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id #type: ignore
    print(f"Experiment ID: {exp_id}")
    mlflow.enable_system_metrics_logging()
    return exp_id

exp_id = init_mlflow("http://localhost:8090")
extra_configs = {'mlflow_experiment_id' : exp_id}
# Example: Run first experiment in CSV
final_val_loss = run_experiment('experiment.csv', 'config.json', 0, extra_configs)
print(f"Experiment 0 final validation loss: {final_val_loss:.6f}")
final_val_loss = run_experiment('experiment.csv', 'config.json', 1, extra_configs)
print(f"Experiment 0 final validation loss: {final_val_loss:.6f}")
#%%
final_val_loss = run_experiment('experiment.csv', 'config.json', 2 , extra_configs)
print(f"Experiment 0 final validation loss: {final_val_loss:.6f}")

final_val_loss = run_experiment('experiment.csv', 'config.json', 3 , extra_configs)
print(f"Experiment 0 final validation loss: {final_val_loss:.6f}")
#%%
final_val_loss = run_experiment('experiment.csv', 'config.json', 4 , extra_configs)
print(f"Experiment 0 final validation loss: {final_val_loss:.6f}")

final_val_loss = run_experiment('experiment.csv', 'config.json', 5 , extra_configs)
print(f"Experiment 0 final validation loss: {final_val_loss:.6f}") 
#%%
final_val_loss = run_experiment('experiment.csv', 'config.json', 6 , extra_configs)
print(f"Experiment 0 final validation loss: {final_val_loss:.6f}")