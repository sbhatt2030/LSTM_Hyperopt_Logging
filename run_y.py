from run_experiment import run_experiment
import mlflow

mlflow.set_tracking_uri("http://hurco-lambda.hurco.com:8090")
mlflow.set_experiment("stage2_yz_finetuning")
mlflow.enable_system_metrics_logging()
exp_id = mlflow.get_experiment_by_name("stage2_yz_finetuning").experiment_id

run_experiment("experiment_updated.csv", "config.json", 0, {"mlflow_experiment_id": exp_id}, pipeline="prediction")