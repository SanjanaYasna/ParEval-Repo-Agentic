#!/usr/bin/env python3
"""
code_only_replay.py

For each output_XX directory under a translations root with the layout:
    <repo>/<model>/output_XX/COMPILATION/llm_response_iter_001.txt
    <repo>/<model>/output_XX/COMPILATION/llm_response_iter_002.txt
    <repo>/<model>/output_XX/COMPILATION/experiment_metadata_001.json
    <repo>/<model>/output_XX/COMPILATION/experiment_metadata_002.json

Parses code from the LLM response, builds it against the GT build file,
runs it, validates outputs, optionally runs scaling, and writes:
    <repo>/<model>/output_XX/code_only_experiment_metadata_001.json
    <repo>/<model>/output_XX/code_only_experiment_metadata_002.json

Usage:
    python code_only_replay.py \
        --results-root /path/to/AsyncSTM/... \
        --target-path /Users/robsonlab/Teetly/ParEval-Repo/targets \
        --system-config /Users/robsonlab/Teetly/ParEval-Repo/config/perlmutter-config.json \
        [--scratch-dir /tmp/scratch] \
        [--enable-scaling] \
        [--dry] \
        [--apps AsyncSTM] \
        [--force-overwrite]
        
NOTE: add the full-code response if you ever want to recompile non code only and update experiment_metadata_001/2.json files
"""

import argparse
import html
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Bring in the helpers from driver_amt / util / build / run
# Adjust sys.path if this script lives outside the ParEval-Repo directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PAREVAL_ROOT = Path("/Users/robsonlab/Teetly/ParEval-Repo")
DRIVERS_DIR    = PAREVAL_ROOT / "src" / "drivers"
if str(DRIVERS_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVERS_DIR))
    
for candidate in [SCRIPT_DIR, PAREVAL_ROOT]:
    if candidate not in sys.path:
        sys.path.insert(0, str(candidate))

from util import setup_tempdir, run_bash, find_config, FileSystemHelper

from driver_amt import (
    build_repo_in_tempdir,
    run_perf_commands,
    run_scaling_analysis,
    _compare_output_files,
    _build_per_file_runtime,
    check_keywords_in_repo,
    SCALING_PROCESSORS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# File suffixes we will overlay from the LLM response into the GT repo.
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".cu",
    ".h", ".hh", ".hpp", ".hxx", ".ipp", ".tpp",
    ".py", ".java", ".rs", ".go",
    ".f", ".f90", ".f03",
}

# Basenames we must NOT take from the LLM — always use GT versions.
BUILD_CONTROL_BASENAMES = {
    "makefile",
    "cmakelists.txt",
    "cmakepresets.json",
    "build.sh",
    "run.sh",
}

# Doc suffixes we skip from LLM response entirely.
DOC_SUFFIXES = {".md", ".rst", ".txt"}

# Patterns that identify a [FileName] header line in an LLM response.
FILE_HEADER_PATTERNS = [
    # [foo.cpp]
    re.compile(r"^\[(?P<name>[^\]\[]+)\]\s*$"),
    # [[ReadMe.md](http://ReadMe.md)]  — markdown link syntax
    re.compile(r"^\[\[(?P<name>[^\]]+)\]\([^)]*\)\]\s*$"),
]

FENCE_START_RE = re.compile(r"^```[A-Za-z0-9_+\-]*\s*$")
FENCE_END_RE   = re.compile(r"^```\s*$")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------

def _sanitize_rel_path(raw: str) -> Optional[str]:
    raw = raw.strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return None
    if any(part == ".." for part in p.parts):
        return None
    return str(p)


def parse_llm_response(text: str) -> Dict[str, str]:
    """
    Parse [FileName] / ```lang ... ``` blocks out of an LLM response.

    Returns {relative_path: file_content}.
    """
    lines = text.splitlines()
    out: Dict[str, str] = {}
    i = 0

    while i < len(lines):
        raw_line = lines[i].strip()
        filename: Optional[str] = None

        for pat in FILE_HEADER_PATTERNS:
            m = pat.match(raw_line)
            if m:
                filename = m.group("name").strip()
                break

        if filename is None:
            i += 1
            continue

        i += 1

        # skip blank lines between header and fence
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines) or not FENCE_START_RE.match(lines[i].strip()):
            # no fence found — the header line was probably something else
            continue

        i += 1  # skip opening fence
        buf: List[str] = []

        while i < len(lines) and not FENCE_END_RE.match(lines[i].strip()):
            buf.append(lines[i])
            i += 1

        if i < len(lines):
            i += 1  # skip closing fence

        safe = _sanitize_rel_path(filename)
        if safe is None:
            continue

        content = html.unescape("\n".join(buf)).rstrip("\n") + "\n"
        out[safe] = content
    # print("PARSED RESPONSE")
    # print(out)
    return out


def _is_code_file(rel_path: str) -> bool:
    """True if the file should be overlaid as generated source code."""
    p = Path(rel_path)
    basename = p.name.lower()
    suffix   = p.suffix.lower()

    if basename in BUILD_CONTROL_BASENAMES:
        return False
    if suffix in DOC_SUFFIXES:
        return False
    # Accept known source extensions
    if suffix in SOURCE_SUFFIXES:
        return True
    # Accept files with no extension only if they look like source
    # (rare, but some repos have extension-less headers)
    return False


# ---------------------------------------------------------------------------
# Temp-dir setup from parsed files (mirrors setup_tempdir for repo/ dirs)
# ---------------------------------------------------------------------------

def setup_tempdir_from_parsed(
    tempdir: str,
    extracted: Dict[str, str],
    gt_repo_path: str,
    gt_build_filename: str,
) -> List[str]:
    """
    1. Copy the entire GT repo into tempdir (gives us GT build file + any
       helper headers / data files the build needs).
    2. Overwrite source/header files with the LLM-generated versions.
    3. Return the list of relative paths actually written.

    We always keep the GT build file regardless of what the LLM produced.
    """
    # Step 1 — seed tempdir with the full GT repo
    for item in os.listdir(gt_repo_path):
        src = os.path.join(gt_repo_path, item)
        dst = os.path.join(tempdir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # Step 2 — overlay LLM-generated code files
    written: List[str] = []
    for rel_path, content in extracted.items():
        if not _is_code_file(rel_path):
            log.debug(f"  [skip] not a code file: {rel_path}")
            continue
        dst = os.path.join(tempdir, rel_path)
        os.makedirs(os.path.dirname(dst) or tempdir, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(rel_path)
        log.debug(f"  [overlay] {rel_path}")

    # Step 3 — ensure GT build file is (re)instated
    gt_build_src = os.path.join(gt_repo_path, gt_build_filename)
    gt_build_dst = os.path.join(tempdir, gt_build_filename)
    if os.path.isfile(gt_build_src):
        shutil.copy2(gt_build_src, gt_build_dst)
        log.debug(f"  [GT build] reinstated {gt_build_filename}")

    return written


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _seed_accounting_fields(existing: dict, iteration: int) -> dict:
    """
    Carry forward token/timing fields from the original metadata.
    Everything else is recomputed.
    """
    return {
        "iteration":          existing.get("iteration", iteration),
        "input_tokens":       existing.get("input_tokens"),
        "output_tokens":      existing.get("output_tokens"),
        "request_count":      existing.get("request_count"),
        "generation_time_s":  existing.get("generation_time_s"),
    }


def _parse_fail_record(existing: dict, iteration: int) -> dict:
    stats = _seed_accounting_fields(existing, iteration)
    stats.update({
        "build_status":    False,
        "run_status":      False,
        "overall_status":  "PARSE_FAIL",
        "per_file_status": existing.get("per_file_status", {}),
        "build_dictionary":        None,
        "run_dictionary":          None,
        "output_files_content":    {},
    })
    return stats


# ---------------------------------------------------------------------------
# Fake args object — driver_amt helpers use args.dry / args.log_build_output etc.
# ---------------------------------------------------------------------------

class _FakeArgs:
    def __init__(self, dry: bool = False):
        self.dry               = dry
        self.log_build_output  = False
        self.log_build_errors  = True
        self.log_run_output    = False
        self.log_run_errors    = True
        self.save_temps        = None
        self.run_timeout       = 1200
        self.target_path       = str(PAREVAL_ROOT / "targets")


# ---------------------------------------------------------------------------
# Core per-output-dir replay
# ---------------------------------------------------------------------------

def replay_output_dir(
    output_dir: Path,
    app: str,
    dest_model: str,
    target_config: dict,
    system_config: dict,
    scratch: str,
    args_dry: bool,
    enable_scaling: bool,
    force_overwrite: bool,
):
    """
    Replays both iterations (001, 002) found under output_dir/COMPILATION.
    Writes code_only_experiment_metadata_00N.json into output_dir.
    """
    comp_dir = output_dir / "COMPILATION"
    if not comp_dir.is_dir():
        return

    # Discover available iterations
    iters = sorted({
        int(m.group(1))
        for p in comp_dir.glob("llm_response_iter_*.txt")
        for m in [re.match(r"llm_response_iter_(\d{3})\.txt$", p.name)]
        if m
    })
    if not iters:
        log.debug(f"  No llm_response_iter_*.txt found under {comp_dir}")
        return

    fake_args = _FakeArgs(dry=args_dry)
    gt_repo_path  = target_config["path"]
    # resolve to absolute using PAREVAL_ROOT prefix (mirrors driver_amt logic)
    if not os.path.isabs(gt_repo_path):
        gt_repo_path = os.path.join(str(PAREVAL_ROOT), gt_repo_path)

    gt_build_filename = target_config.get("build_filename", "Makefile")
    gt_build_path = os.path.join(gt_repo_path, gt_build_filename)
    if not os.path.isfile(gt_build_path):
        log.warning(f"  GT build file not found: {gt_build_path} — skipping {output_dir}")
        return

    ground_truth_files = target_config.get("hpx_ground_truth_file", [])
    output_files       = target_config.get("output_files", [])

    for iteration in iters:
        save_path = comp_dir / f"code_only_experiment_metadata_{iteration:03d}.json"

        if save_path.exists() and not force_overwrite:
            log.info(f"  [skip] {save_path.name} already exists (use --force-overwrite)")
            continue

        llm_path      = comp_dir / f"llm_response_iter_{iteration:03d}.txt"
        existing_meta = _load_json(comp_dir / f"experiment_metadata_{iteration:03d}.json")

        print(f"  iter {iteration:03d}: parsing {llm_path.name}")

        if not llm_path.exists():
            log.warning(f"    [warn] {llm_path.name} missing — skipping")
            continue

        raw_text  = llm_path.read_text(encoding="utf-8", errors="replace")
        extracted = parse_llm_response(raw_text)

        if not extracted:
            log.warning("    [warn] no file blocks parsed from LLM response")
            record = _parse_fail_record(existing_meta, iteration)
            save_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"    [saved] {save_path.name} (PARSE_FAIL)")
            continue

        code_files = {k: v for k, v in extracted.items() if _is_code_file(k)}
        if not code_files:
            log.warning("    [warn] parsed files but none qualify as code overlays")
            record = _parse_fail_record(existing_meta, iteration)
            save_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"    [saved] {save_path.name} (PARSE_FAIL — no code files)")
            continue

        print(f"    parsed {len(code_files)} code file(s): {list(code_files.keys())}")

        # ── keyword check (same guard as driver_amt) ──────────────────────
        # Write to a small temp location just for keyword checking
        with tempfile.TemporaryDirectory(dir=scratch, prefix="kw_check_") as kw_tmp:
            for rel, content in code_files.items():
                dst = os.path.join(kw_tmp, rel)
                os.makedirs(os.path.dirname(dst) or kw_tmp, exist_ok=True)
                with open(dst, "w", encoding="utf-8") as fh:
                    fh.write(content)
            kw_ok = check_keywords_in_repo(kw_tmp, app, dest_model)
            print("KW OK", kw_ok)

        if not kw_ok:
            print("    [info] keyword check failed — INCORRECT_MODEL")
            stats = _seed_accounting_fields(existing_meta, iteration)
            stats.update({
                "build_status":          None,
                "run_status":            None,
                "overall_status":        {f: "INCORRECT_MODEL" for f in output_files}
                                         or {"unknown": "INCORRECT_MODEL"},
                "per_file_status":       {f: "INCORRECT_MODEL" for f in output_files}
                                         or {"unknown": "INCORRECT_MODEL"},
                "build_dictionary":      None,
                "run_dictionary":        None,
                "output_files_content":  {},
            })
            save_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print(f"    [saved] {save_path.name}")
            continue

        # ── build + run in a fresh tempdir ────────────────────────────────
        with tempfile.TemporaryDirectory(dir=scratch,
                                         prefix=f"co_replay_{app}_{output_dir.name}_{iteration:03d}_") as td:
            written = setup_tempdir_from_parsed(td, extracted, gt_repo_path, gt_build_filename)

            if not written:
                log.warning("    [warn] no files written to tempdir after overlay")
                record = _parse_fail_record(existing_meta, iteration)
                save_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                print(f"    [saved] {save_path.name} (PARSE_FAIL — nothing written)")
                continue

            # BUILD
            build_result = build_repo_in_tempdir(target_config, td, fake_args)
            build_ok     = build_result["build_returncode"] == 0

            build_dict = {
                "build_returncode": build_result["build_returncode"],
                "build_stdout":     build_result["build_stdout"],
                "build_stderr":     build_result["build_stderr"],
            }

            print(f"    Build: {'OK' if build_ok else 'FAIL'}")
            if not build_ok:
                per_file = {f: "BUILD_FAIL" for f in output_files} or {"unknown": "BUILD_FAIL"}
                stats = _seed_accounting_fields(existing_meta, iteration)
                stats.update({
                    "build_status":         False,
                    "run_status":           False,
                    "overall_status":       "BUILD_FAIL",
                    "per_file_status":      per_file,
                    "build_dictionary":     build_dict,
                    "run_dictionary":       None,
                    "output_files_content": {},
                })
                save_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
                print(f"    [saved] {save_path.name}")
                continue

            # RUN
            run_result = run_perf_commands(target_config, system_config, td, fake_args)
            all_runs_ok = all(rc == 0 for rc in run_result["run_returncodes"])
            print(f"    Run:   {'OK' if all_runs_ok else 'FAIL'}")

            run_dict = {
                "run_returncodes":      run_result["run_returncodes"],
                "run_stdouts":          run_result["run_stdouts"],
                "run_stderrs":          run_result["run_stderrs"],
                "runtime":              run_result["runtime"],
            }

            # VALIDATE
            num_outputs = len(ground_truth_files) if ground_truth_files else 1
            all_match, per_file_status = _compare_output_files(
                output_files,
                ground_truth_files,
                run_result,
                td,
                num_outputs,
                log_prefix="code_only: ",
                app=app,
            )

            # Determine single overall string (mirrors experiment_metadata style)
            if not build_ok:
                overall = "BUILD_FAIL"
            elif not all_runs_ok:
                overall = "RUNTIME_FAIL"
            elif all_match:
                overall = "PASS"
            else:
                # check for any MISSING
                statuses = set(per_file_status.values())
                if "MISSING" in statuses:
                    overall = "MISSING"
                else:
                    overall = "VALIDATION_FAIL"

            print(f"    Per-file: {per_file_status}")
            print(f"    Overall:  {overall}")

            # Collect output file contents (same as original driver)
            output_files_content: Dict[str, str] = {}
            build_dir_candidate = os.path.join(td, "build")
            check_dir = build_dir_candidate if os.path.isdir(build_dir_candidate) else td
            for fname in output_files:
                fpath = os.path.join(check_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        output_files_content[fname] = open(fpath, encoding="utf-8").read()
                    except Exception:
                        output_files_content[fname] = ""

            # SCALING
            scaling_dict = None
            if (
                enable_scaling
                and overall == "PASS"
                and app.lower() != "asyncstm"      # keep original guard; remove if desired
                and dest_model.lower() in ("hpx", "legion")
                and build_ok
                and all_runs_ok
            ):
                print("    Running scaling analysis...")
                scaling_dict = run_scaling_analysis(
                    target_config, system_config, td, dest_model, fake_args
                )
                print(f"    Scaling: {scaling_dict}")

            # SAVE
            stats = _seed_accounting_fields(existing_meta, iteration)
            stats.update({
                "build_status":          build_ok,
                "run_status":            all_runs_ok,
                "overall_status":        overall,
                "per_file_status":       per_file_status,
                "build_dictionary":      build_dict,
                "run_dictionary":        run_dict,
                "output_files_content":  output_files_content,
            })
            if scaling_dict is not None:
                stats["scaling"] = scaling_dict

            save_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print(f"    [saved] {save_path.name}")


# ---------------------------------------------------------------------------
# Discovery — walk results root
# ---------------------------------------------------------------------------

def iter_output_dirs(results_root: Path):
    """
    Yield (app, model_name, output_dir) for every output_XX directory
    that contains a COMPILATION sub-directory with at least one
    llm_response_iter_XXX.txt file.

    Expected layout:
        results_root/<repo>/<model>/output_XX/COMPILATION/
    """
    for repo_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        app = repo_dir.name
        for model_dir in sorted(p for p in repo_dir.iterdir() if p.is_dir()):
            model_name = model_dir.name
            for output_dir in sorted(
                p for p in model_dir.iterdir()
                if p.is_dir() and re.match(r"output_\d+", p.name)
            ):
                comp_dir = output_dir / "COMPILATION"
                if comp_dir.is_dir() and any(comp_dir.glob("llm_response_iter_*.txt")):
                    yield app, model_name, output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Replay code-only builds from LLM response files."
    )
    parser.add_argument("--results-root", required=True, type=Path,
                        help="Root of the results tree (contains <repo>/<model>/output_XX).")
    parser.add_argument("--target-path", type=str,
                        default=str(PAREVAL_ROOT / "targets"),
                        help="Path to target config JSON files.")
    parser.add_argument("--system-config", type=str,
                        default=str(PAREVAL_ROOT / "config" / "perlmutter-config.json"),
                        help="System config JSON.")
    parser.add_argument("--scratch-dir", type=str, default="/tmp/co_replay_scratch",
                        help="Scratch directory for temp build dirs.")
    parser.add_argument("--apps", nargs="+", type=str,
                        help="Restrict to these app names (case-insensitive).")
    parser.add_argument("--models", nargs="+", type=str,
                        help="Restrict to these model/directory names (substring, case-insensitive).")
    parser.add_argument("--outputs", nargs="+", type=int,
                        help="Restrict to these output numbers (e.g. 1 13 42).")
    parser.add_argument("--iterations", nargs="+", type=int, choices=[1, 2],
                        help="Restrict to these iteration numbers (1 and/or 2).")
    parser.add_argument("--enable-scaling", action="store_true",
                        help="Run scaling analysis when code_only passes (non-AsyncSTM only).")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Overwrite existing code_only_experiment_metadata_*.json files.")
    parser.add_argument("--dry", action="store_true",
                        help="Dry run — parse files but do not build or run.")
    parser.add_argument("--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", type=str.upper)
    parser.add_argument("--full-code", type =bool, store_true = True)
    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(levelname)s %(message)s",
    )

    # Load system config
    system_config: dict = {}
    if os.path.isfile(args.system_config):
        with open(args.system_config, "r") as f:
            system_config = json.load(f)
    else:
        log.warning(f"System config not found at {args.system_config}")

    os.makedirs(args.scratch_dir, exist_ok=True)

    if not args.results_root.is_dir():
        log.error(f"Results root does not exist: {args.results_root}")
        sys.exit(1)

    total = skipped = processed = failed = 0

    for app, model_name, output_dir in iter_output_dirs(args.results_root):

        # ── filters ──────────────────────────────────────────────────────
        if args.apps and app.lower() not in [a.lower() for a in args.apps]:
            continue
        if args.models and not any(m.lower() in model_name.lower() for m in args.models):
            continue
        out_num_m = re.search(r"output_(\d+)", output_dir.name)
        if args.outputs and out_num_m:
            if int(out_num_m.group(1)) not in args.outputs:
                continue

        total += 1

        # ── resolve target config ─────────────────────────────────────────
        # We need dest_model for find_config.  Derive it from the existing
        # experiment_metadata files (first one found wins).
        dest_model = None
        for iter_num in (1, 2):
            meta_path = output_dir / "COMPILATION" / f"experiment_metadata_{iter_num:03d}.json"
            if meta_path.exists():
                m = _load_json(meta_path)
                print("metadata path", meta_path)
                # The new-style metadata doesn't have dest_model; infer from
                # directory name or fall back to "legion".
                dest_model = m.get("dest_model") or _infer_dest_model(model_name)
                break

        if dest_model is None:
            dest_model = _infer_dest_model(model_name)

        try:
            target_config = find_config(app, dest_model, args.target_path)
        except Exception as e:
            log.warning(f"  [warn] find_config failed for {app}/{dest_model}: {e} — skipping {output_dir}")
            skipped += 1
            continue

        # Resolve GT repo path (mirrors driver_amt.py logic)
        gt_repo_path = target_config.get("path", "")
        if not os.path.isabs(gt_repo_path):
            gt_repo_path = os.path.join(str(PAREVAL_ROOT), gt_repo_path)
        if not os.path.isdir(gt_repo_path):
            log.warning(f"  [warn] GT repo not found: {gt_repo_path} — skipping {output_dir}")
            skipped += 1
            continue

        # Patch target_config so helpers pick up the absolute path
        target_config = dict(target_config)
        target_config["path"] = gt_repo_path

        print(f"\n[{app} / {model_name} / {output_dir.name}]")
    
        try:
            replay_output_dir(
                output_dir    = output_dir,
                app           = app,
                dest_model    = dest_model,
                target_config = target_config,
                system_config = system_config,
                scratch       = args.scratch_dir,
                args_dry      = args.dry,
                enable_scaling= args.enable_scaling,
                force_overwrite=args.force_overwrite,
            )
            processed += 1
        except Exception as e:
            log.error(f"  [error] {output_dir}: {e}", exc_info=True)
            failed += 1

    print(f"\nDone. processed={processed}  skipped={skipped}  errors={failed}  total_found={total}")


def _infer_dest_model(model_dir_name: str) -> str:
    """
    Best-effort: extract dest_model from the model directory name.
    e.g. "gpt-5.3-codex" → falls back to "legion" (project default).
    """
    name_lower = model_dir_name.lower()
    if "hpx" in name_lower:
        print("hpx detected")
        return "hpx"
    if "legion" in name_lower:
        print("legion detected")
        return "legion"
    if "kokkos" in name_lower:
        return "kokkos"
    if "openmp" in name_lower:
        return "openmp-offload"
    # default — adjust for your project
    return "legion"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()