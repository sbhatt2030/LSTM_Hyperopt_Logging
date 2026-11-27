# LSTM Hyperopt Logging

Code for training, tuning and using recurrent (GRU/LSTM) models that **predict and compensate CNC servo following error**. Training runs on a multi-GPU server. Ray Tune and Optuna run the hyperparameter search, and MLflow logs every run.

This repo backs the MSEC papers on data-driven following-error prediction and velocity-feedforward (VFF) compensation:

1. **Error prediction:** a GRU maps commanded position, velocity and acceleration to the measured following error (`e_scale_x/y/z`).
2. **Error compensation:** a second GRU uses the frozen prediction model's hidden states and an error delta (ΔE) to predict the VFF change (ΔVFF) that cancels the error.
3. **Trajectory correction:** a trained prediction model is used inside an optimisation loop to find a modified command signal that minimises predicted error within kinematic limits.

> Paper citations: _add the MSEC paper references / DOIs here._

## Layout

| File | Purpose |
|---|---|
| `train_val_loop_v10.py` | Training pipeline for **error prediction** (`pipeline="prediction"`). Defines `SequenceModel`, `TimeSeriesDataset`, `EarlyStopping`, and the Jacobian and hidden-state-derivative regularisers. |
| `train_val_loop_comp_v3.py` | Training pipeline for **error compensation** (`pipeline="compensation_v3"`). Uses two phases: a frozen prediction GRU, then an MLP and a compensation GRU. Needs 2 GPUs per trial. |
| `run_experiment.py` | Runs one experiment. Reads one row of an experiment CSV, merges it with `config.json`, sends it to the selected pipeline and returns the final validation loss. |
| `hyperparam_opt.py` | Ray Tune + Optuna (TPE) search. You set `PIPELINE` and edit `PARAM_TABLE_*`, where each parameter is `fixed`, `search` or `trial`. Earlier MLflow runs warm-start the search. |
| `entry_point.py`, `run_y.py`, `run_z.py` | Small launchers that run a single experiment-CSV row (e.g. stage-2 y/z fine-tuning). |
| `signal_optimizer_v4.py` | Trajectory correction in position space (δx optimised directly, δv/δa found by finite differences). Its docstring calls it "v5". |
| `normalized_datasate.py` | Streams a large `.npy` dataset and writes per-channel stats plus a normalised copy without loading it all into RAM. |
| `denormalize_results.py` | Pulls `best_val_loss` from MLflow and converts it to physical RMSE (mm / µm) using `combined_statistics.json`. |
| `query_and_plot.py`, `new_file.py` | Query MLflow and draw hyperparameter-vs-loss scatter plots (they use different filters). |
| `signal_inspector.html` | Standalone browser tool for looking at signals. |
| `config.json` | Shared config, plus per-pipeline config. |
| `experiment.csv`, `experiment_updated.csv` | Experiment definitions. Each row is one run. |
| `ranges_for_parameter_search.csv`, `search*.csv` | Search-range notes and phase definitions. |
| `combined_statistics.json` | Normalisation mean/std per channel, used for denormalisation. |
| `mlflow_server.sh` | Starts the MLflow tracking server (port 8090, SQLite backend). |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env.txt          # mlflow, numpy, optuna, pandas, psutil, ray, torch
```

Datasets (`data/*.npy`), models, plots and MLflow stores are **not tracked** (see `.gitignore`). Put the datasets under `data/` at the paths given in `config.json` / the experiment CSV.

## Usage

```bash
# 1. Start the tracking server (on the training host)
bash mlflow_server.sh

# 2. Run a single experiment (row 0 of experiment.csv)
python entry_point.py

# 3. Run a hyperparameter search (set PIPELINE and the PARAM_TABLE in the file first)
python hyperparam_opt.py

# 4. Convert logged losses to physical units
python denormalize_results.py --output results.csv
```

The tracking URI is hard-coded in each script (`TRACKING_URI` / `set_tracking_uri`). Change it there if your server is somewhere else.

## History

Earlier script versions (`signal_optimizer.py` and its v2/v3, `train_val_loop_comp.py` and its v2) were removed because the files above replace them. They are still in the git history.
