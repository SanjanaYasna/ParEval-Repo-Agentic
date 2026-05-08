#!/usr/bin/env python3
"""
NOTE: GENERARTED BY CLAUDE OPUS 4.6, by inputting my compute_@k.py code from another repository (original naive translation agent) and asking it to adjust for the new .json formats and file output organizatoin 

Compute build@k, run@k, and pass@k metrics from compile agent output directories.

Searches for experiment_metadata_00X.json files under a root directory with structure:
    <root>/<app>/<llm_name>/output_<N>/COMPILATION/experiment_metadata_00X.json

Computes metrics per iteration (epoch), so iteration 1 and iteration 2 are reported
separately. If an output only has iteration 1 but not iteration 2, the sample count
for iteration 2 is reduced accordingly.

Usage:
    python @k.py /path/to/optimize_output_root -o metrics.csv
    python @k.py /path/to/root --k 1 3 5 10 --group-by app llm_name
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from math import comb


ALL_STATUSES = [
    "PARSE_FAIL",
    "BUILD_FAIL",
    "RUNTIME_FAIL",
    "MISSING",
    "VALIDATION_FAIL",
    "PASS",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_dir",
        type=str,
        help="Root directory containing agent outputs structured as "
             "<app>/<llm_name>/output_<N>/COMPILATION/experiment_metadata_00X.json",
    )
    parser.add_argument(
        "-k", "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="K values for build@k, run@k, pass@k (default: 1 3 5).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: prints to stdout).",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        default=["app", "llm_name"],
        help="Fields to group by when computing metrics (default: app llm_name).",
    )
    return parser.parse_args()


def _passk(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased estimator of pass@k."""
    if num_samples < k:
        if num_samples == 0:
            return 0.0
        return float(num_correct > 0)
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))


def derive_flags(status: str) -> Dict[str, bool]:
    """
    From a status string, derive the three success flags:
      - did_build: not BUILD_FAIL and not PARSE_FAIL
      - did_run:   VALIDATION_FAIL or PASS (built and ran, output may differ)
      - did_pass:  only PASS
    """
    return {
        "did_build": status not in ("BUILD_FAIL", "PARSE_FAIL"),
        "did_run": status in ("VALIDATION_FAIL", "PASS"),
        "did_pass": status == "PASS",
    }


def parse_path_metadata(json_path: Path, root: Path) -> Optional[Dict[str, Any]]:
    """
    Extract app, llm_name, output_number, and iteration from file path.

    Supports two directory layouts relative to root:
      5-part: <app>/<llm_name>/output_<N>/COMPILATION/experiment_metadata_<iter>.json
      4-part: <llm_name>/output_<N>/COMPILATION/experiment_metadata_<iter>.json
              (when root IS the app directory)

    Returns None if path doesn't match expected structure.
    """
    try:
        rel = json_path.relative_to(root)
        parts = rel.parts

        # Must end with COMPILATION/experiment_metadata_*.json
        if len(parts) < 4:
            return None

        filename = parts[-1]
        if not filename.startswith("experiment_metadata_") or not filename.endswith(".json"):
            return None

        # Extract iteration from filename
        iter_match = re.match(r"experiment_metadata_(\d+)\.json", filename)
        if not iter_match:
            return None
        iteration = int(iter_match.group(1))

        # Check COMPILATION is the parent
        if parts[-2] != "COMPILATION":
            return None

        # Determine layout based on part count
        if len(parts) == 5:
            # Full layout: app/llm_name/output_N/COMPILATION/file
            app = parts[0]
            llm_name = parts[1]
            output_dir = parts[2]
        elif len(parts) == 4:
            # Root is the app dir: llm_name/output_N/COMPILATION/file
            app = root.name  # use the root directory name as app
            llm_name = parts[0]
            output_dir = parts[1]
        elif len(parts) == 3:
            # Root is the llm dir: output_N/COMPILATION/file
            app = root.parent.name
            llm_name = root.name
            output_dir = parts[0]
        else:
            return None

        # Extract output number
        output_match = re.match(r"output_(\d+)", output_dir)
        if not output_match:
            # Also try output-N format
            output_match = re.match(r"output[-_](\d+)", output_dir)
            if not output_match:
                return None
        output_number = int(output_match.group(1))

        return {
            "app": app,
            "llm_name": llm_name,
            "output_number": output_number,
            "iteration": iteration,
            "file_path": str(json_path),
        }
    except (ValueError, IndexError):
        return None


def gather_entries(root_dir: str) -> List[Dict[str, Any]]:
    """Walk root_dir and collect all experiment_metadata_*.json entries."""
    root = Path(root_dir).resolve()
    entries: List[Dict[str, Any]] = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith("experiment_metadata_") and fname.endswith(".json"):
                full_path = Path(dirpath) / fname
                path_meta = parse_path_metadata(full_path, root)
                if path_meta is None:
                    continue

                # Load the JSON content
                try:
                    with open(full_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  [warn] Skipping {full_path}: {e}", file=sys.stderr)
                    continue

                # Extract status
                overall_status = data.get("overall_status", "BUILD_FAIL")
                flags = derive_flags(overall_status)

                entry = {
                    **path_meta,
                    "status": overall_status,
                    "build_status": data.get("build_status", False),
                    "run_status": data.get("run_status", False),
                    "per_file_status": data.get("per_file_status", {}),
                    "runtime": data.get("run_dictionary", {}).get("runtime", -1)
                              if isinstance(data.get("run_dictionary"), dict) else -1,
                    "input_tokens": data.get("input_tokens", 0),
                    "output_tokens": data.get("output_tokens", 0),
                    "generation_time_s": data.get("generation_time_s", 0),
                    "has_scaling": "scaling" in data,
                    **flags,
                }
                entries.append(entry)

    return entries


def compute_all_metrics(
    df: pd.DataFrame,
    k_values: List[int],
    group_cols: List[str],
) -> pd.DataFrame:
    """Compute build@k, run@k, and pass@k grouped by specified columns + iteration."""

    # Always include iteration in grouping for the per-iteration breakdown
    full_group = group_cols + ["iteration"]

    # Status counts
    status_counts = (
        df.groupby(full_group + ["status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    status_counts.columns.name = None
    for s in ALL_STATUSES:
        if s not in status_counts.columns:
            status_counts[s] = 0
    status_counts["total_samples"] = status_counts[ALL_STATUSES].sum(axis=1)

    # Aggregation for @k computation
    agg_df = df.groupby(full_group).agg(
        num_samples=("status", "count"),
        num_build=("did_build", "sum"),
        num_run=("did_run", "sum"),
        num_pass=("did_pass", "sum"),
    ).reset_index()

    for k in k_values:
        agg_df[f"build@{k}"] = agg_df.apply(
            lambda x: _passk(int(x["num_samples"]), int(x["num_build"]), k), axis=1
        )
        agg_df[f"run@{k}"] = agg_df.apply(
            lambda x: _passk(int(x["num_samples"]), int(x["num_run"]), k), axis=1
        )
        agg_df[f"pass@{k}"] = agg_df.apply(
            lambda x: _passk(int(x["num_samples"]), int(x["num_pass"]), k), axis=1
        )

    # Merge status counts with metrics
    result = pd.merge(status_counts, agg_df, on=full_group, how="outer")

    # Order columns
    metric_cols = [c for c in result.columns if "@" in c]
    order_map = {"build": 0, "run": 1, "pass": 2}
    metric_cols = sorted(metric_cols, key=lambda x: (
        order_map.get(x.split("@")[0], 99), int(x.split("@")[1])
    ))

    ordered_cols = (
        full_group
        + ["num_samples", "total_samples"]
        + ALL_STATUSES
        + metric_cols
    )
    ordered_cols = [c for c in ordered_cols if c in result.columns]

    return result[ordered_cols].sort_values(full_group).reset_index(drop=True)


def compute_cumulative_metrics(
    df: pd.DataFrame,
    k_values: List[int],
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Compute cumulative pass@k: for iteration X, an output "passes" if it
    passed on ANY iteration <= X. This captures the idea that the compile
    agent may fix code across multiple attempts.

    For each (app, llm_name, output_number), track the best status across
    iterations 1..X, then compute @k on the cumulative successes.
    """
    # Priority: higher = better
    priority = {
        "PARSE_FAIL": 0,
        "BUILD_FAIL": 1,
        "RUNTIME_FAIL": 2,
        "MISSING": 3,
        "VALIDATION_FAIL": 4,
        "PASS": 5,
    }

    iterations = sorted(df["iteration"].unique())
    id_cols = ["app", "llm_name", "output_number"]

    cumulative_rows = []

    for max_iter in iterations:
        # For each output, take the BEST status across iterations 1..max_iter
        subset = df[df["iteration"] <= max_iter].copy()
        subset["priority"] = subset["status"].map(priority)

        best = subset.loc[subset.groupby(id_cols)["priority"].idxmax()]

        for _, row in best.iterrows():
            flags = derive_flags(row["status"])
            cumulative_rows.append({
                **{col: row[col] for col in group_cols if col in row.index},
                "app": row["app"],
                "llm_name": row["llm_name"],
                "output_number": row["output_number"],
                "iteration": max_iter,
                "status": row["status"],
                **flags,
            })

    cum_df = pd.DataFrame(cumulative_rows)

    if cum_df.empty:
        return pd.DataFrame()

    return compute_all_metrics(cum_df, k_values, group_cols)


def print_summary(df: pd.DataFrame, entries: List[Dict[str, Any]]) -> None:
    """Print high-level summary statistics."""
    total = len(entries)
    apps = df["app"].nunique()
    models = df["llm_name"].nunique()
    iterations = sorted(df["iteration"].unique())
    outputs_per_iter = df.groupby("iteration")["output_number"].nunique()

    print(f"\n{'='*70}")
    print(f"COMPILE AGENT METRICS SUMMARY")
    print(f"{'='*70}")
    print(f"Total metadata files found: {total}")
    print(f"Apps: {apps} ({', '.join(sorted(df['app'].unique()))})")
    print(f"Models: {models} ({', '.join(sorted(df['llm_name'].unique()))})")
    print(f"Iterations found: {iterations}")
    print(f"Outputs per iteration:")
    for it, count in outputs_per_iter.items():
        print(f"  iteration {it}: {count} outputs")
    print(f"{'='*70}")


def main() -> None:
    args = get_args()
    input_dir = Path(os.path.expanduser(args.input_dir)).resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Error: {input_dir} is not a directory.")

    print(f"Scanning {input_dir} for experiment_metadata_*.json files...")
    entries = gather_entries(str(input_dir))

    if not entries:
        raise SystemExit("Error: No experiment_metadata files found.")

    df = pd.DataFrame(entries)

    # Validate group-by columns
    for col in args.group_by:
        if col not in df.columns:
            raise SystemExit(
                f"Error: group-by column '{col}' not found. "
                f"Available: {sorted(df.columns.tolist())}"
            )

    print_summary(df, entries)

    # Per-iteration metrics (each iteration independently)
    print("\n=== Per-Iteration Metrics (independent) ===")
    per_iter = compute_all_metrics(df, args.k, args.group_by)
    print(per_iter.to_string(index=False))

    # Cumulative metrics (best status up to iteration X)
    print("\n=== Cumulative Metrics (best status through iteration X) ===")
    cumulative = compute_cumulative_metrics(df, args.k, args.group_by)
    if not cumulative.empty:
        print(cumulative.to_string(index=False))
    else:
        print("  (no data)")

    # Token/cost summary per iteration
    print("\n=== Token Usage Summary ===")
    token_summary = df.groupby(args.group_by + ["iteration"]).agg(
        n_outputs=("output_number", "nunique"),
        total_input_tokens=("input_tokens", "sum"),
        total_output_tokens=("output_tokens", "sum"),
        total_generation_time_s=("generation_time_s", "sum"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
    ).reset_index()
    print(token_summary.to_string(index=False))

    # Output CSVs
    if args.output:
        base_path = Path(os.path.expanduser(args.output))
        stem = base_path.stem
        ext = base_path.suffix if base_path.suffix else ".csv"
        parent = base_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        path_per_iter = parent / f"{stem}_per_iteration{ext}"
        path_cumulative = parent / f"{stem}_cumulative{ext}"
        path_tokens = parent / f"{stem}_tokens{ext}"
        path_raw = parent / f"{stem}_raw{ext}"

        per_iter.to_csv(path_per_iter, index=False)
        if not cumulative.empty:
            cumulative.to_csv(path_cumulative, index=False)
        token_summary.to_csv(path_tokens, index=False)
        df.to_csv(path_raw, index=False)

        print(f"\nWrote CSVs to {parent}/")
        print(f"  - {path_per_iter.name}  (per-iteration @k metrics)")
        print(f"  - {path_cumulative.name}  (cumulative best-of @k metrics)")
        print(f"  - {path_tokens.name}  (token usage summary)")
        print(f"  - {path_raw.name}  (all raw entries)")


if __name__ == "__main__":
    main()