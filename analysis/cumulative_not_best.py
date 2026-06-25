import json
import csv
from pathlib import Path
from collections import defaultdict


# Priority: higher = better
STATUS_PRIORITY = {
    "PARSE_FAIL": 0,
    "BUILD_FAIL": 1,
    "RUNTIME_FAIL": 2,
    "MISSING": 3,
    "VALIDATION_FAIL": 4,
    "PASS": 5,
}


def compute_cumulative_results(translate_dir: str, compile_dir: str, output_csv: str):
    """
    Compute cumulative results across translate and compile iterations.
    
    N reflects the number of outputs that have results at that stage:
    - Translate: all outputs from results.json
    - Compile 1: only outputs that went to compile (failed translate OR have compile 1 metadata)
    - Compile 2: only outputs that went to compile 2 (failed compile 1 OR have compile 2 metadata)
    
    Args:
        translate_dir: Path to translate results directory (e.g., /Users/.../hpx_to_legion_repo/)
        compile_dir: Path to compile optimization results directory (e.g., /Users/.../optimize_hpx_to_legion/)
        output_csv: Path to output CSV file
    """
    
    translate_root = Path(translate_dir)
    compile_root = Path(compile_dir)
    
    # Repos and LLMs to process
    REPOS = ["osc_chain_1d", "Barnes_Hut", "heapify", "AsyncSTM"]
    LLMS = ["gpt-5.3-codex", "opus"]
    
    results = []
    
    for repo in REPOS:
        for llm in LLMS:
            print(f"Processing {repo}/{llm}...")
            
            # Track ALL outputs from translate
            all_output_statuses = {}  # output_number -> current_status (for all 20)
            translate_baseline = {}   # output_number -> translate_status (baseline)
            
            # ── 1. Load translate results ─────────────────────────────────
            translate_results_path = translate_root / repo / llm / "results.json"
            if not translate_results_path.exists():
                print(f"  [warn] No translate results at {translate_results_path}")
                continue
                
            with open(translate_results_path) as f:
                translate_data = json.load(f)
            
            # Initialize all outputs with translate status
            for entry in translate_data:
                output_num = entry["output_number"]
                
                # Extract status from overall_status dict
                overall_status = entry.get("overall_status", {})
                if isinstance(overall_status, dict):
                    status = list(overall_status.values())[0] if overall_status else "BUILD_FAIL"
                else:
                    status = overall_status
                
                all_output_statuses[output_num] = status
                translate_baseline[output_num] = status
            
            # Tally after translate (all outputs)
            translate_tally = _tally_statuses(all_output_statuses)
            
            results.append({
                "repo": repo,
                "llm": llm,
                "stage": "translate",
                "n": len(all_output_statuses),  # All outputs from results.json
                "cum_num_better": 0,
                "cum_num_worse": 0,
                **translate_tally
            })
            
            # ── 2. Compile iteration 1: only outputs that have metadata ───
            compile_1_outputs = set()  # Track which outputs have compile 1 results
            
            for output_num in all_output_statuses.keys():
                # Check if this output has compile 1 metadata
                compile_1_path = (
                    compile_root / repo / llm / f"output_{output_num}" / 
                    "COMPILATION" / "experiment_metadata_001.json"
                )
                
                if compile_1_path.exists():
                    compile_1_outputs.add(output_num)
                    
                    with open(compile_1_path) as f:
                        compile_1_data = json.load(f)
                    
                    new_status = compile_1_data.get("overall_status", "BUILD_FAIL")
                    all_output_statuses[output_num] = new_status
            
            # Filter to only outputs with compile 1 results
            compile_1_statuses = {
                num: all_output_statuses[num] for num in compile_1_outputs
            }
            
            # Count better/worse vs translate baseline
            cum_better_1, cum_worse_1 = _count_vs_baseline(compile_1_statuses, translate_baseline)
            
            # Tally after compile 1
            compile_1_tally = _tally_statuses(compile_1_statuses)
            
            results.append({
                "repo": repo,
                "llm": llm,
                "stage": "compile_1",
                "n": len(compile_1_statuses),  # Only outputs with compile 1 metadata
                "num_better": cum_better_1,
                "num_worse": cum_worse_1,
                **compile_1_tally
            })
            
            # ── 3. Compile iteration 2: only outputs that have metadata ───
            compile_2_outputs = set()
            
            for output_num in compile_1_outputs:  # Can only have compile 2 if had compile 1
                compile_2_path = (
                    compile_root / repo / llm / f"output_{output_num}" / 
                    "COMPILATION" / "experiment_metadata_002.json"
                )
                
                if compile_2_path.exists():
                    compile_2_outputs.add(output_num)
                    
                    with open(compile_2_path) as f:
                        compile_2_data = json.load(f)
                    
                    new_status = compile_2_data.get("overall_status", "BUILD_FAIL")
                    all_output_statuses[output_num] = new_status
            
            # Filter to only outputs with compile 2 results
            compile_2_statuses = {
                num: all_output_statuses[num] for num in compile_2_outputs
            }
            
            # Count better/worse vs translate baseline
            cum_better_2, cum_worse_2 = _count_vs_baseline(compile_2_statuses, translate_baseline)
            
            # Tally after compile 2
            compile_2_tally = _tally_statuses(compile_2_statuses)
            
            results.append({
                "repo": repo,
                "llm": llm,
                "stage": "compile_2",
                "n": len(compile_2_statuses),  # Only outputs with compile 2 metadata
                "num_better": cum_better_2,
                "num_worse": cum_worse_2,
                **compile_2_tally
            })
    
    # ── 4. Write CSV ───────────────────────────────────────────────────
    fieldnames = [
        "repo", "llm", "stage", "n", "cum_num_better", "cum_num_worse",
        "PARSE_FAIL", "BUILD_FAIL", "RUNTIME_FAIL", "MISSING", "VALIDATION_FAIL", "PASS"
    ]
    
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nWrote cumulative results to {output_csv}")


def _tally_statuses(statuses_dict: dict) -> dict:
    """Tally status counts from a dict of output_num -> status."""
    tally = defaultdict(int)
    for status in statuses_dict.values():
        tally[status] += 1
    
    return {
        "PARSE_FAIL": tally.get("PARSE_FAIL", 0),
        "BUILD_FAIL": tally.get("BUILD_FAIL", 0),
        "RUNTIME_FAIL": tally.get("RUNTIME_FAIL", 0),
        "MISSING": tally.get("MISSING", 0),
        "VALIDATION_FAIL": tally.get("VALIDATION_FAIL", 0),
        "PASS": tally.get("PASS", 0),
    }


def _count_vs_baseline(current_statuses: dict, baseline_statuses: dict) -> tuple[int, int]:
    """
    Compare current statuses against translate baseline.
    
    Returns:
        (num_better, num_worse) tuple
    """
    num_better = 0
    num_worse = 0
    
    for output_num, current_status in current_statuses.items():
        baseline_status = baseline_statuses[output_num]
        
        current_priority = STATUS_PRIORITY.get(current_status, 0)
        baseline_priority = STATUS_PRIORITY.get(baseline_status, 0)
        
        if current_priority > baseline_priority:
            num_better += 1
        elif current_priority < baseline_priority:
            num_worse += 1
    
    return num_better, num_worse


# ── Usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    compute_cumulative_results(
        translate_dir="/Users/robsonlab/scratch/hpx_to_legion_repo",
        compile_dir="/Users/robsonlab/scratch/optimize_hpx_to_legion",
        output_csv="cumulative_results.csv"
    )