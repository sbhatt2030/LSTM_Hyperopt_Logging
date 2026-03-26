"""
Query MLflow for GRU runs on the full dataset and plot hyperparameter scatter plots.
Run this on hurco-lambda where the MLflow server is accessible.

Filters:
  - model_type  = gru
  - hidden_size = 5
  - num_layers  = 2
  - downsample_factor = 1
  - target_timesteps  = 8000
  - data_file_path    = <YOUR_PATH>   ← edit below
"""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRACKING_URI   = "http://hurco-lambda.hurco.com:8090"
DATA_FILE_PATH = "data/complete_error_prediction_data.npy"          # ← change if needed

# Hyperparameters to plot against best_val_loss
HYPERPARAM_COLS = ["dropout", "learning_rate", "l2_weight",
                   "jacobian_weight", "hidden_derivative_weight"]
# ─────────────────────────────────────────────────────────────────────────────

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# ── QUERY ─────────────────────────────────────────────────────────────────────
# Pull ALL runs across all experiments, filter in pandas (avoids MLflow string matching quirks)
all_experiments = client.search_experiments()
print(f"Found {len(all_experiments)} experiments, fetching runs...")

all_runs = []
for exp in all_experiments:
    try:
        batch = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1000,
        )
        all_runs.extend(batch)
    except Exception as e:
        print(f"  Skipping experiment {exp.experiment_id}: {e}")

print(f"Total runs fetched: {len(all_runs)}")

# ── BUILD DATAFRAME ───────────────────────────────────────────────────────────
records = []
for r in all_runs:
    p = r.data.params
    m = r.data.metrics
    records.append({
        "run_id":                   r.info.run_id,
        "best_val_loss":            m.get("best val loss"),
        "model_type":               p.get("model_type", ""),
        "hidden_size":              p.get("hidden_size", ""),
        "num_layers":               p.get("num_layers", ""),
        "downsample_factor":        p.get("downsample_factor", ""),
        "target_timesteps":         p.get("target_timesteps", ""),
        "data_file_path":           p.get("data_file_path", ""),
        "dropout":                  float(p["dropout"])       if "dropout"       in p else None,
        "learning_rate":            float(p["learning_rate"]) if "learning_rate" in p else None,
        "l2_weight":                float(p["l2_weight"])     if "l2_weight"     in p else None,
        "jacobian_weight":          float(p["jacobian_weight"])          if "jacobian_weight"          in p else None,
        "hidden_derivative_weight": float(p["hidden_derivative_weight"]) if "hidden_derivative_weight" in p else None,
    })

df = pd.DataFrame(records)

# ── PRINT UNIQUE VALUES to help debug / confirm what's actually in the DB ─────
print("\n--- Unique values seen in DB (pre-filter) ---")
for col in ["model_type", "hidden_size", "num_layers", "downsample_factor", "target_timesteps"]:
    print(f"  {col}: {sorted(df[col].dropna().unique().tolist())}")
print(f"  data_file_path sample: {df['data_file_path'].dropna().unique()[:5].tolist()}")

# ── FILTER in pandas (string compare, strip whitespace to be safe) ─────────────
mask = (
    (df["model_type"].str.strip()       == "gru") &
    (df["hidden_size"].str.strip()      == "32") &
    (df["num_layers"].str.strip()       == "2") &
    (df["downsample_factor"].str.strip()== "1") &
    (df["target_timesteps"].str.strip() == "8000") &
    (df["data_file_path"].str.strip()   == DATA_FILE_PATH.strip())
)
df = df[mask].dropna(subset=["best_val_loss"]).sort_values("best_val_loss")
print(f"\nRuns after filtering: {len(df)}")
if df.empty:
    raise SystemExit("No runs matched — check the printed unique values above and update DATA_FILE_PATH.")

print("\nTop 10 runs:")
print(df[["best_val_loss", "dropout", "learning_rate", "l2_weight"]].head(10).to_string(index=False))

# ── SCATTER PLOTS ─────────────────────────────────────────────────────────────
valid_cols = [c for c in HYPERPARAM_COLS if c in df.columns and df[c].notna().any()]
n = len(valid_cols)

fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
if n == 1:
    axes = [axes]

cmap = plt.cm.RdYlGn_r
norm = mcolors.LogNorm(vmin=df["best_val_loss"].min(),
                       vmax=df["best_val_loss"].quantile(0.95))

for ax, col in zip(axes, valid_cols):
    x = df[col]
    y = df["best_val_loss"]

    sc = ax.scatter(x, y, c=y, cmap=cmap, norm=norm,
                    alpha=0.7, edgecolors="none", s=40)

    # Log x for weight / lr params
    if col in ("l2_weight", "learning_rate", "jacobian_weight", "hidden_derivative_weight"):
        if (x > 0).all():
            ax.set_xscale("log")

    # Log y for loss
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel("best_val_loss (log)", fontsize=11)
    ax.set_title(f"{col}", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)

    # Annotate best point
    best_idx = y.idxmin()
    ax.annotate(f"  {y[best_idx]:.4f}",
                xy=(x[best_idx], y[best_idx]),
                fontsize=9, color="blue", fontweight="bold")

    # Individual colorbar per axis, same height as the plot
    fig.colorbar(sc, ax=ax, label="best_val_loss")

fig.suptitle(
    f"GRU  hidden=5  layers=2  ds=1  T=8000\n{DATA_FILE_PATH}",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()

out = "hyperparam_scatter.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out}")
plt.show()