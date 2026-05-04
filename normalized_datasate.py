"""
normalize_permutation_dataset.py

Computes per-channel mean/std for the permutation dataset (N, 8000, 6)
and writes a normalized copy, without ever loading the full dataset into RAM.

Dataset column layout (6 channels):
  0-2 : Δe_scale  x/y/z
  3-5 : Δvff      x/y/z

Outputs:
  <same dir>/<input_stem>_stats.json        — mean, std, min, max per channel
  <same dir>/<input_stem>_normalized.npy    — normalized float32 dataset
"""

import numpy as np
import json
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    INPUT_FILE  = "data/permutation_dataset_1_5.npy"
    BATCH_SIZE  = 4096*2          # samples per batch — tune to available RAM
    EPS         = 1e-8         # avoid divide-by-zero during normalization

    CHANNEL_NAMES = [
        "delta_escale_x", "delta_escale_y", "delta_escale_z",
        "delta_vff_x",    "delta_vff_y",    "delta_vff_z",
    ]

    # Derived from INPUT_FILE automatically — no need to edit these
    @classmethod
    def stats_file(cls):
        stem = os.path.splitext(cls.INPUT_FILE)[0]
        return stem + "_stats.json"

    @classmethod
    def normalized_file(cls):
        stem = os.path.splitext(cls.INPUT_FILE)[0]
        return stem + "_normalized.npy"


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — Welford streaming mean / variance  +  global min/max
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(data: np.ndarray, cfg: Config) -> dict:
    """
    Single-pass Welford online algorithm over the (N, 8000, 6) array.
    Each 'observation' is one (8000,) time-series flattened per channel,
    so we accumulate across both the N and T=8000 dimensions.

    Returns a dict ready to be serialised as JSON.
    """
    N, T, C = data.shape
    assert C == len(cfg.CHANNEL_NAMES), f"Expected {len(cfg.CHANNEL_NAMES)} channels, got {C}"

    n_batches = int(np.ceil(N / cfg.BATCH_SIZE))

    # Welford accumulators (per channel)
    count   = np.zeros(C, dtype=np.float64)   # total scalar observations
    mean    = np.zeros(C, dtype=np.float64)
    M2      = np.zeros(C, dtype=np.float64)   # sum of squared deviations
    ch_min  = np.full(C,  np.inf, dtype=np.float64)
    ch_max  = np.full(C, -np.inf, dtype=np.float64)

    t0 = time.time()
    print(f"\nPass 1/2 — computing statistics  ({N:,} samples, batch={cfg.BATCH_SIZE})")

    for b in range(n_batches):
        s = b * cfg.BATCH_SIZE
        e = min(s + cfg.BATCH_SIZE, N)
        chunk = data[s:e].astype(np.float64)          # (batch, 8000, 6)

        for c in range(C):
            vals = chunk[:, :, c].ravel()             # (batch*8000,)

            # Welford batch update
            batch_count = vals.size
            batch_mean  = vals.mean()
            batch_var   = vals.var()                  # population var of this batch

            # Chan & Lewis (1979) parallel update
            delta       = batch_mean - mean[c]
            new_count   = count[c] + batch_count
            mean[c]    += delta * batch_count / new_count
            M2[c]      += batch_var * batch_count + delta**2 * count[c] * batch_count / new_count
            count[c]    = new_count

            ch_min[c]   = min(ch_min[c], vals.min())
            ch_max[c]   = max(ch_max[c], vals.max())

        if (b + 1) % max(1, n_batches // 10) == 0 or b == n_batches - 1:
            pct = (b + 1) / n_batches * 100
            print(f"  batch {b+1:>{len(str(n_batches))}}/{n_batches}  ({pct:.0f}%)  "
                  f"elapsed {time.time()-t0:.1f}s")

    std = np.sqrt(M2 / count)

    stats = {
        "dataset_shape": list(data.shape),
        "channel_names": cfg.CHANNEL_NAMES,
        "mean": {cfg.CHANNEL_NAMES[c]: float(mean[c]) for c in range(C)},
        "std":  {cfg.CHANNEL_NAMES[c]: float(std[c])  for c in range(C)},
        "min":  {cfg.CHANNEL_NAMES[c]: float(ch_min[c]) for c in range(C)},
        "max":  {cfg.CHANNEL_NAMES[c]: float(ch_max[c]) for c in range(C)},
    }

    print(f"\n  Done in {time.time()-t0:.1f}s")
    print(f"\n  Per-channel statistics:")
    hdr = f"  {'channel':<22}  {'mean':>14}  {'std':>14}  {'min':>14}  {'max':>14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in range(C):
        nm = cfg.CHANNEL_NAMES[c]
        print(f"  {nm:<22}  {mean[c]:>14.6f}  {std[c]:>14.6f}  "
              f"  {ch_min[c]:>14.6f}  {ch_max[c]:>14.6f}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — normalise and write output
# ─────────────────────────────────────────────────────────────────────────────

def normalize_dataset(data: np.ndarray, stats: dict, output_path: str, cfg: Config):
    """
    Streams through the dataset in batches, applies z-score normalisation
    per channel, writes result to a new .npy via np.memmap.
    """
    N, T, C = data.shape
    n_batches = int(np.ceil(N / cfg.BATCH_SIZE))

    mean_vec = np.array([stats["mean"][n] for n in cfg.CHANNEL_NAMES], dtype=np.float32)
    std_vec  = np.array([stats["std"][n]  for n in cfg.CHANNEL_NAMES], dtype=np.float32)

    print(f"\nPass 2/2 — normalising  →  {output_path}")

    # Pre-allocate output as a memmap so we never hold >1 batch in RAM
    out = np.lib.format.open_memmap(
        output_path,
        mode='w+',
        dtype=np.float32,
        shape=(N, T, C),
    )

    t0 = time.time()

    for b in range(n_batches):
        s = b * cfg.BATCH_SIZE
        e = min(s + cfg.BATCH_SIZE, N)
        chunk = data[s:e].astype(np.float32)          # (batch, T, C)

        # broadcast subtract/divide: mean_vec/std_vec shape (C,) → (1,1,C)
        chunk = (chunk - mean_vec[np.newaxis, np.newaxis, :]) / (
                  std_vec[np.newaxis, np.newaxis, :] + cfg.EPS)

        out[s:e] = chunk
        out.flush()

        if (b + 1) % max(1, n_batches // 10) == 0 or b == n_batches - 1:
            pct = (b + 1) / n_batches * 100
            print(f"  batch {b+1:>{len(str(n_batches))}}/{n_batches}  ({pct:.0f}%)  "
                  f"elapsed {time.time()-t0:.1f}s")

    del out   # flush and close memmap
    print(f"\n  Done in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    stats_file      = cfg.stats_file()
    normalized_file = cfg.normalized_file()

    print("=" * 60)
    print("  Permutation Dataset — Stats & Normalisation")
    print("=" * 60)
    print(f"  Input  : {cfg.INPUT_FILE}")
    print(f"  Stats  : {stats_file}")
    print(f"  Output : {normalized_file}")

    # Memory-map the source (read-only, no RAM load)
    print(f"\nOpening dataset (mmap, read-only)...")
    data = np.load(cfg.INPUT_FILE, mmap_mode='r')
    print(f"  Shape  : {data.shape}  dtype={data.dtype}")
    size_gb = data.nbytes / 1024**3
    print(f"  Size   : {size_gb:.2f} GB")

    # ── Pass 1: stats ────────────────────────────────────────────────────────
    stats = compute_stats(data, cfg)

    print(f"\nSaving stats → {stats_file}")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print("  ✅ Stats saved")

    # ── Pass 2: normalise ────────────────────────────────────────────────────
    normalize_dataset(data, stats, normalized_file, cfg)
    print(f"\n  ✅ Normalized dataset saved")

    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\nSanity check — verifying first batch of normalised output...")
    out_check = np.load(normalized_file, mmap_mode='r')
    sample = out_check[:min(256, out_check.shape[0])].astype(np.float32)
    for c, nm in enumerate(cfg.CHANNEL_NAMES):
        m = sample[:, :, c].mean()
        s = sample[:, :, c].std()
        print(f"  {nm:<22}  mean={m:>8.4f}  std={s:>8.4f}")
    print("\n  (Values near 0 / 1 confirm correct normalisation)")

    print("\n" + "=" * 60)
    print("  All done.")
    print("=" * 60)


if __name__ == "__main__":
    main()