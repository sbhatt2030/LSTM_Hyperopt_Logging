"""
Pulls best_val_loss from all MLflow experiments and converts to physical units.

Denormalization:
    MSE_physical = MSE_normalized * std^2
    RMSE_physical = sqrt(MSE_physical)

The std is looked up from combined_statistics.json based on each run's
axis_selection parameter:
    x   -> e_scale_x std
    y   -> e_scale_y std
    z   -> e_scale_z std
    all -> mean variance across e_scale_x/y/z (MSE is averaged over all outputs)

Units: mm -> RMSE also reported in microns

Usage:
    python denormalize_results.py
    python denormalize_results.py --uri http://your-server:8090
    python denormalize_results.py --stats combined_statistics.json
    python denormalize_results.py --output results.csv
"""

import argparse
import json
import math
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

MM_TO_MICRONS = 1000.0
DEFAULT_URI   = "http://hurco-lambda.hurco.com:8090"
DEFAULT_STATS = "combined_statistics.json"


def load_variance_map(stats_path: str) -> dict:
    """
    Build a map of axis_selection -> effective variance for e_scale.
    For 'all', use mean variance across x/y/z since MSE averages over all outputs.
    """
    with open(stats_path) as f:
        stats = json.load(f)

    std = stats["std"]
    var_x = std["e_scale_x"] ** 2
    var_y = std["e_scale_y"] ** 2
    var_z = std["e_scale_z"] ** 2

    return {
        "x":   var_x,
        "y":   var_y,
        "z":   var_z,
        "all": (var_x + var_y + var_z) / 3,  # averaged over 6 outputs (2 per axis)
    }


def denormalize_mse(normalized_mse: float, variance: float) -> dict:
    """Convert normalized MSE to physical units given the effective variance."""
    mse_mm2      = normalized_mse * variance
    rmse_mm      = math.sqrt(mse_mm2)
    rmse_microns = rmse_mm * MM_TO_MICRONS
    return {
        "mse_normalized": normalized_mse,
        "mse_mm2":        mse_mm2,
        "rmse_mm":        rmse_mm,
        "rmse_microns":   rmse_microns,
    }


def fetch_all_runs(client: MlflowClient) -> pd.DataFrame:
    """Fetch every run across every experiment into a single DataFrame."""
    experiments = client.search_experiments()
    print(f"Found {len(experiments)} experiments")

    all_rows = []
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=mlflow.entities.ViewType.ALL,
        )
        print(f"  [{exp.experiment_id}] {exp.name!r:40s} - {len(runs)} runs")
        for run in runs:
            row = {
                "experiment_id":   exp.experiment_id,
                "experiment_name": exp.name,
                "run_id":          run.info.run_id,
                "run_name":        run.info.run_name,
                "status":          run.info.status,
                "start_time":      pd.to_datetime(run.info.start_time, unit="ms"),
                "end_time":        pd.to_datetime(run.info.end_time,   unit="ms"),
            }
            row.update(run.data.params)
            row.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def find_best_val_loss_col(df: pd.DataFrame) -> str:
    """
    Identify the column that holds best_val_loss.
    MLflow stores final metric values; 'best val loss' is logged per epoch
    so the final stored value is the best seen during training.
    """
    candidates = [c for c in df.columns if "best" in c.lower() and "val" in c.lower()]
    if candidates:
        return candidates[0]

    candidates = [c for c in df.columns if "val" in c.lower() and "loss" in c.lower()]
    if candidates:
        print(f"  Warning: no 'best val loss' column found, using {candidates[0]!r}")
        return candidates[0]

    raise ValueError(
        "Could not find a val loss column. "
        f"Available metric columns: {[c for c in df.columns if c.startswith('metric_')]}"
    )


def main(uri: str, stats_path: str, output_path: str):
    # Load variance map from JSON
    variance_map = load_variance_map(stats_path)
    print(f"Loaded normalization stats from {stats_path!r}:")
    for axis, var in variance_map.items():
        print(f"  axis={axis!r}  std={math.sqrt(var)*MM_TO_MICRONS:.4f} um  var={var:.4e} mm^2")

    # Connect to MLflow
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    print(f"\nConnecting to MLflow at {uri}")

    df = fetch_all_runs(client)
    print(f"\nTotal runs fetched: {len(df)}")

    if df.empty:
        print("No runs found. Exiting.")
        return

    best_col = find_best_val_loss_col(df)
    print(f"\nUsing column {best_col!r} as best_val_loss")

    # Convert to numeric, drop runs where it's missing
    df[best_col] = pd.to_numeric(df[best_col], errors="coerce")
    missing = df[best_col].isna().sum()
    if missing:
        print(f"  Dropping {missing} runs with no val loss recorded")
    df = df.dropna(subset=[best_col]).copy()

    # Resolve axis_selection per run, default to 'x' if not logged
    if "axis_selection" in df.columns:
        df["axis_selection"] = df["axis_selection"].fillna("x").str.lower().str.strip()
    else:
        print("  Warning: no axis_selection column found, defaulting all runs to 'x'")
        df["axis_selection"] = "x"

    unknown_axes = set(df["axis_selection"].unique()) - set(variance_map)
    if unknown_axes:
        print(f"  Warning: unknown axis values {unknown_axes}, those runs will be skipped")
        df = df[df["axis_selection"].isin(variance_map)].copy()

    # Denormalize each run using its own axis variance
    def denorm_row(row):
        var = variance_map[row["axis_selection"]]
        return pd.Series(denormalize_mse(row[best_col], var))

    denorm = df.apply(denorm_row, axis=1)
    df = pd.concat([df, denorm], axis=1)

    # Sort best first
    df = df.sort_values("rmse_microns")

    # Summary
    print("\n" + "=" * 60)
    print("TOP 10 RUNS BY RMSE (e_scale, physical units)")
    print("=" * 60)
    display_cols = [
        "experiment_name", "axis_selection", "run_id",
        "mse_normalized", "rmse_mm", "rmse_microns",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("BEST RUN OVERALL")
    print("=" * 60)
    best = df.iloc[0]
    print(f"  Experiment  : {best['experiment_name']}")
    print(f"  Run ID      : {best['run_id']}")
    print(f"  Axis        : {best['axis_selection']}")
    print(f"  MSE (norm)  : {best['mse_normalized']:.6f}")
    print(f"  MSE (mm^2)  : {best['mse_mm2']:.2e}")
    print(f"  RMSE (mm)   : {best['rmse_mm']:.6f}")
    print(f"  RMSE (um)   : {best['rmse_microns']:.4f} microns")
    print("=" * 60)

    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")
    print(f"  {len(df)} runs x {len(df.columns)} columns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denormalize MLflow val losses to physical units")
    parser.add_argument("--uri",    default=DEFAULT_URI,    help="MLflow tracking URI")
    parser.add_argument("--stats",  default=DEFAULT_STATS,  help="Path to combined_statistics.json")
    parser.add_argument("--output", default="experiment_results.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.uri, args.stats, args.output)
    