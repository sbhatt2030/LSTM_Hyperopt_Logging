
import mlflow
import os
import sys
import ray
from ray import  tune
from ray.tune.search import ConcurrencyLimiter, Repeater
from ray.tune.search.optuna.optuna_search import OptunaSearch
import time
import pandas as pd
from run_experiment import run_experiment

 
 
def objective(hyperopt_params):
    #print(hyperopt_params)
    extra_configs = {k:v for k,v in hyperopt_params.items() if k != '__trial_index__'}
    extra_configs['save_model_path'] = f'models/exp_{hyperopt_params['__trial_index__']}'
    extra_configs['data_file_path'] = 'data/clean_temp.npy'
    extra_configs['verbose'] = 0
    extra_configs['target_timesteps'] = 100
    extra_configs['device'] = 'cuda'
    extra_configs['batch_size'] = 64
    extra_configs['mlflow_experiment_id'] = ExpID
    extra_configs['hidden_size'] = 2 ** extra_configs['hidden_size'] 
    final_val_loss = run_experiment('experiment.csv', 'config.json', -1, extra_configs)
    print("loss: " + str(final_val_loss))
    tune.report({"mean_test_inst_loss": final_val_loss})
    return
 

 
def main():
    print("HyperParam_Opt started")
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = "0"
    initial_params = [{"base_loss": 'mse', "l2_weight": 1e-4, "jacobian_weight": 1e-4,  "hidden_derivative_weight" : 1e-4, "model_type": 'lstm', "hidden_size": 3, "num_layers": 2, "dropout": 0, "learning_rate":.001}]
   
    algo = OptunaSearch(points_to_evaluate=initial_params)
    algo = Repeater(algo, repeat=2)
    algo = ConcurrencyLimiter(algo, max_concurrent=5)

    num_samples = 100
    SearchDf = pd.read_csv("search.csv")
    search_config = {}
    for index in SearchDf.index:
        k = SearchDf.loc[index]
        if str(k["type"]) == "uniform":
            search_config[str(k["Name"])] = tune.uniform(float(k["1"]), float(k["2"]))
        if str(k["type"]) == "loguniform":
            search_config[str(k["Name"])] = tune.loguniform(float(k["1"]), float(k["2"]))
        if str(k["type"]) == "choice":
            search_config[str(k["Name"])] = tune.choice([str(k["1"]),str(k["2"])])
        if str(k["type"]) == "randint":
            search_config[str(k["Name"])] = tune.randint(int(k["1"]),int(k["2"]))
        if str(k["type"]) == "randn":
            search_config[str(k["Name"])] = tune.randn(float(k["1"]),float(k["2"])) 

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective), 
            resources= {"gpu": 1 , 'cpu':1}
        ),
        param_space=search_config,
        tune_config=tune.TuneConfig(
            metric="mean_test_inst_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples
        )
    )
    results=tuner.fit()
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