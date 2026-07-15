#!/usr/bin/env python3
"""Run the compact native glrmask release benchmark through CFA.

This is intentionally a thin, reproducible wrapper around the benchmark and
correctness infrastructure in constraint-framework-analysis (CFA). It measures
only glrmask's native compiler/runtime. llguidance_native is present only as a
one-pass discrepancy signal; its performance is not summarized.

The caller must:
  1. build/install this checkout's Python FFI in the active Python environment;
  2. install CFA's dependencies, including llguidance;
  3. provide --cfa-root pointing at a CFA checkout containing the workloads.

Example:
  python scripts/run_native_release_benchmark.py \
      --cfa-root ../constraint-framework-analysis \
      --output-dir /tmp/glrmask-native-release-benchmark
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Workload:
    slug: str
    problem_id: str
    category: str
    reason: str
    build_runs: int = 3
    build_timeout_seconds: int = 120
    release_blocker_if_build_fails: bool = False


WORKLOADS: tuple[Workload, ...] = (
    Workload(
        slug="bfcl-008",
        problem_id="bfcl_catalog/size_008/catalog_008_000",
        category="small tool schema",
        reason="Small realistic BFCL-style tool catalog; covers fast native JSON Schema compilation.",
    ),
    Workload(
        slug="bfcl-512",
        problem_id="bfcl_catalog/size_512/catalog_512_000",
        category="large tool schema",
        reason="Large 512-tool BFCL catalog; exercises scaling of native schema compilation.",
    ),
    Workload(
        slug="json-glrm",
        problem_id="grammar_glrm/json/json",
        category="general CFG",
        reason="Checked-in recursive GLRM grammar for full JSON; exercises the general CFG path rather than JSON Schema import.",
    ),
    Workload(
        slug="github-o62060",
        problem_id="jsb/data/Github_hard---o62060",
        category="difficult real-world schema",
        reason="Historically difficult large real-world schema with a long replay, useful for compile and runtime-tail coverage.",
    ),
    Workload(
        slug="vercel",
        problem_id="jsb/data/JsonSchemaStore---vercel",
        category="regression sentinel",
        reason="Historically difficult real-world schema retained as a build-tail regression sentinel.",
        build_runs=1,
        build_timeout_seconds=120,
        release_blocker_if_build_fails=True,
    ),
)

GLRMASK_FRAMEWORK = "glrmask_native"
REFERENCE_FRAMEWORK = "llguidance_native"
SCHEMA_VERSION = 1
DEFAULT_TOTAL_TIMING_RUNS = 51  # run 0 warmup + 50 measured runs
DEFAULT_MEASURED_TIMING_RUNS = DEFAULT_TOTAL_TIMING_RUNS - 1
IMPORTANT_ENV_PREFIXES = ("GLRMASK_", "CFA_")
IMPORTANT_ENV_NAMES = {
    "CC",
    "CFLAGS",
    "CXX",
    "CXXFLAGS",
    "MACOSX_DEPLOYMENT_TARGET",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "RAYON_NUM_THREADS",
    "RUSTFLAGS",
}


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.stdout.strip()


def _git_metadata(path: Path) -> dict[str, Any]:
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=path)
    status = _run_text(["git", "status", "--porcelain"], cwd=path)
    branch = _run_text(["git", "branch", "--show-current"], cwd=path)
    return {
        "path": str(path.resolve()),
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def _sysctl(name: str) -> str | None:
    if shutil.which("sysctl") is None:
        return None
    value = _run_text(["sysctl", "-n", name])
    return value or None


def _cpu_metadata() -> dict[str, Any]:
    cpu_model = None
    if platform.system() == "Darwin":
        cpu_model = _sysctl("machdep.cpu.brand_string")
    if not cpu_model and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "model": cpu_model or platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "physical_cpus": int(_sysctl("hw.physicalcpu") or 0) or None,
    }


def _important_environment() -> dict[str, str]:
    selected: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if key in IMPORTANT_ENV_NAMES or key.startswith(IMPORTANT_ENV_PREFIXES):
            selected[key] = value
    return selected


def _probe_ffi(cfa_root: Path) -> dict[str, Any]:
    code = """
from cfa.adapters.glrmask_adapter import _import_ffi
m = _import_ffi()
print(getattr(m, '__file__', '<unknown>'))
print(getattr(m, '__name__', type(m).__name__))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cfa_root),
        env=_cfa_env(cfa_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "CFA cannot import the glrmask FFI from the selected Python environment:\n"
            + proc.stdout
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {
        "module_file": lines[0] if lines else None,
        "module_name": lines[1] if len(lines) > 1 else None,
    }


def _probe_reference(cfa_root: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", "import llguidance; print(getattr(llguidance, '__version__', 'unknown')); print(llguidance.__file__)"],
        cwd=str(cfa_root),
        env=_cfa_env(cfa_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "llguidance is required for the discrepancy signal in this benchmark:\n" + proc.stdout
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {
        "version": lines[0] if lines else None,
        "module_file": lines[1] if len(lines) > 1 else None,
    }


def _cfa_env(cfa_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(cfa_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def _tee_subprocess(command: Sequence[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    print("$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def _percentile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _stats_seconds(values: Iterable[float | None], *, scale: float = 1.0) -> dict[str, Any] | None:
    cleaned = sorted(float(v) * scale for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not cleaned:
        return None
    return {
        "count": len(cleaned),
        "mean": statistics.fmean(cleaned),
        "p50": _percentile(cleaned, 0.50),
        "p95": _percentile(cleaned, 0.95),
        "p99": _percentile(cleaned, 0.99),
        "p99_9": _percentile(cleaned, 0.999),
        "max": cleaned[-1],
        "min": cleaned[0],
    }


def _flatten_measured_runs(timing: dict[str, Any], key: str) -> list[float]:
    runs = timing.get(key)
    if not isinstance(runs, list) or len(runs) <= 1:
        return []
    values: list[float] = []
    # Run 0 is the explicit warmup/semantic pass. Only runs 1..N feed public stats.
    for run in runs[1:]:
        if not isinstance(run, list):
            continue
        for value in run:
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
    return values


def _load_single_result(path: Path, expected_problem_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path}: missing results list")
    matches = [row for row in results if row.get("problem_id") == expected_problem_id]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one result for {expected_problem_id}, found {len(matches)}")
    return payload, matches[0]


def _load_external_timing_example(
    payload: dict[str, Any],
    main_path: Path,
    expected_problem_id: str,
    example_index: int,
) -> tuple[dict[str, Any] | None, Path | None]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    raw_output = config.get("raw_output") if isinstance(config, dict) else None
    timing_path: Path | None = None
    if isinstance(raw_output, str) and raw_output:
        candidate = Path(raw_output)
        timing_path = candidate if candidate.is_absolute() else (main_path.parent / candidate)
    else:
        candidate = main_path.with_name(f"{main_path.stem}_raw.json")
        if candidate.is_file():
            timing_path = candidate
    if timing_path is None or not timing_path.is_file():
        return None, timing_path

    raw_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    if raw_payload.get("artifact_kind") != "cfa_raw_timing_runs_v1":
        raise ValueError(f"{timing_path}: unsupported raw timing artifact kind")
    results = raw_payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{timing_path}: missing results list")
    matches = [row for row in results if row.get("problem_id") == expected_problem_id]
    if len(matches) != 1:
        raise ValueError(
            f"{timing_path}: expected one raw timing result for {expected_problem_id}, found {len(matches)}"
        )
    examples = matches[0].get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"{timing_path}: missing examples list")
    example_matches = [row for row in examples if row.get("example_index") == example_index]
    if len(example_matches) != 1:
        raise ValueError(
            f"{timing_path}: expected one raw timing example {example_index}, found {len(example_matches)}"
        )
    return example_matches[0], timing_path


def _summarize_workload(workload: Workload, raw_path: Path, command_returncode: int) -> dict[str, Any]:
    payload, result = _load_single_result(raw_path, workload.problem_id)
    frameworks = result.get("frameworks", {})
    glr_meta = frameworks.get(GLRMASK_FRAMEWORK, {}) if isinstance(frameworks, dict) else {}
    ref_meta = frameworks.get(REFERENCE_FRAMEWORK, {}) if isinstance(frameworks, dict) else {}

    build_timeout = bool(glr_meta.get("build_timeout"))
    build_error = glr_meta.get("build_error")
    build_values = [
        float(v)
        for v in glr_meta.get("build_seconds_all_runs", [])
        if isinstance(v, (int, float)) and float(v) < 1_000_000
    ]
    build_stats_ms = _stats_seconds(build_values, scale=1000.0)

    examples = result.get("examples") if isinstance(result.get("examples"), list) else []
    example = examples[0] if examples else None
    correctness: dict[str, Any] = {
        "example_present": example is not None,
        "gold_replay_ok": None,
        "replay_completed": False,
        "correctness_status": "no_example_or_build",
        "expected_valid": None,
        "glrmask_first_reject_idx": None,
        "glrmask_first_commit_reject_idx": None,
        "death_token_idx": None,
        "validity_mismatch_kind": None,
        "reference_available": bool(ref_meta.get("available")),
        "reference_build_error": ref_meta.get("build_error"),
        "reference_first_reject_idx": None,
        "raw_discrepancy_position_count": None,
        "raw_discrepancy_step_count": None,
        "raw_disputed_token_events": None,
        "discrepancy_adjudicated": False,
    }
    runtime: dict[str, Any] | None = None

    if isinstance(example, dict):
        first_reject = example.get("first_reject_idx") or {}
        first_commit_reject = example.get("first_commit_reject_idx") or {}
        expected_valid = example.get("expected_valid")
        glr_reject = first_reject.get(GLRMASK_FRAMEWORK) if isinstance(first_reject, dict) else None
        glr_commit_reject = (
            first_commit_reject.get(GLRMASK_FRAMEWORK)
            if isinstance(first_commit_reject, dict)
            else None
        )
        death = example.get("death_token_idx")
        validity_mismatch = example.get("validity_mismatch_kind")
        replay_completed = bool(
            glr_meta.get("available")
            and glr_reject is None
            and glr_commit_reject is None
            and death is None
            and validity_mismatch is None
        )
        gold_ok: bool | None
        if expected_valid is True:
            gold_ok = replay_completed
        else:
            gold_ok = None
        if not glr_meta.get("available"):
            correctness_status = "build_unavailable"
        elif not replay_completed:
            correctness_status = "replay_failed"
        elif expected_valid is True:
            correctness_status = "valid_gold_replay_passed"
        elif expected_valid is False:
            correctness_status = "invalid_labeled_example_replayed"
        else:
            correctness_status = "unlabeled_example_replay_completed"
        steps = example.get("steps") if isinstance(example.get("steps"), list) else []
        discrepancy_steps = 0
        disputed_events = 0
        for step in steps:
            discrepancy = step.get("discrepancy") if isinstance(step, dict) else None
            if isinstance(discrepancy, dict) and discrepancy.get("has_discrepancy"):
                discrepancy_steps += 1
                disputed_events += int(discrepancy.get("disputed_token_count") or 0)

        ref_reject = first_reject.get(REFERENCE_FRAMEWORK) if isinstance(first_reject, dict) else None
        correctness.update(
            {
                "gold_replay_ok": gold_ok,
                "replay_completed": replay_completed,
                "correctness_status": correctness_status,
                "expected_valid": expected_valid,
                "glrmask_first_reject_idx": glr_reject,
                "glrmask_first_commit_reject_idx": glr_commit_reject,
                "death_token_idx": death,
                "validity_mismatch_kind": validity_mismatch,
                "reference_first_reject_idx": ref_reject,
                "raw_discrepancy_position_count": example.get("raw_discrepancy_position_count"),
                "raw_discrepancy_step_count": discrepancy_steps,
                "raw_disputed_token_events": disputed_events,
            }
        )

        timing_by_framework = example.get("framework_timing") or {}
        glr_timing = timing_by_framework.get(GLRMASK_FRAMEWORK) if isinstance(timing_by_framework, dict) else None
        external_example, timing_artifact_path = _load_external_timing_example(
            payload,
            raw_path,
            workload.problem_id,
            int(example.get("example_index") or 0),
        )
        external_timing = None
        if isinstance(external_example, dict):
            external_by_framework = external_example.get("framework_timing")
            if isinstance(external_by_framework, dict):
                external_timing = external_by_framework.get(GLRMASK_FRAMEWORK)
        timing_runs_source = external_timing if isinstance(external_timing, dict) else glr_timing
        if isinstance(timing_runs_source, dict) and replay_completed:
            mask_values = _flatten_measured_runs(timing_runs_source, "mask_seconds_all_runs")
            commit_values = _flatten_measured_runs(timing_runs_source, "commit_seconds_all_runs")
            tbm_values = _flatten_measured_runs(timing_runs_source, "tbm_seconds_all_runs")
            completed_runs = glr_timing.get("timing_runs") if isinstance(glr_timing, dict) else None
            if completed_runs is None:
                runs = timing_runs_source.get("mask_seconds_all_runs")
                completed_runs = len(runs) if isinstance(runs, list) else 0
            runtime = {
                "warmup_runs_excluded": 1,
                "completed_timing_runs": completed_runs,
                "measured_runs": max(0, int(completed_runs or 0) - 1),
                "token_count": example.get("token_count"),
                "raw_timing_artifact": str(timing_artifact_path) if timing_artifact_path else None,
                "mask_us": _stats_seconds(mask_values, scale=1_000_000.0),
                "commit_us": _stats_seconds(commit_values, scale=1_000_000.0),
                "tbm_us": _stats_seconds(tbm_values, scale=1_000_000.0),
            }

    blocker = bool(
        workload.release_blocker_if_build_fails
        and (build_timeout or build_error or not glr_meta.get("available"))
    )
    if (
        glr_meta.get("available")
        and correctness["example_present"]
        and correctness.get("replay_completed") is False
    ):
        blocker = True

    return {
        "slug": workload.slug,
        "problem_id": workload.problem_id,
        "category": workload.category,
        "reason": workload.reason,
        "command_returncode": command_returncode,
        "result_status": result.get("status"),
        "glrmask_build": {
            "available": bool(glr_meta.get("available")),
            "timeout": build_timeout,
            "error": build_error,
            "all_runs_seconds": build_values,
            "stats_ms": build_stats_ms,
        },
        "correctness": correctness,
        "runtime": runtime,
        "release_blocker": blocker,
        "raw_result": str(raw_path),
        "raw_timing_result": (
            runtime.get("raw_timing_artifact") if isinstance(runtime, dict) else None
        ),
        "cfa_config": payload.get("config"),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    rows = summary["workloads"]
    lines = [
        "# Native glrmask release benchmark",
        "",
        f"Overall status: **{summary['overall_status']}**",
        "",
        "This benchmark measures glrmask native compile, token-mask generation, and token commit performance. ",
        "The llguidance-native pass is used only to expose raw mask disagreements; those disagreements are not adjudicated here and are not treated as comparative performance evidence.",
        "",
        "## Compile/build latency",
        "",
        "| workload | build status | runs | median ms | min ms | max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        build = row["glrmask_build"]
        stats = build["stats_ms"] or {}
        if build["timeout"]:
            status = "TIMEOUT"
        elif build["error"]:
            status = "ERROR"
        elif build["available"]:
            status = "ok"
        else:
            status = "unavailable"
        lines.append(
            f"| `{row['slug']}` | {status} | {stats.get('count', 0)} | {_fmt(stats.get('p50'))} | {_fmt(stats.get('min'))} | {_fmt(stats.get('max'))} |"
        )

    lines += [
        "",
        "## Runtime latency",
        "",
        "Run 0 is excluded as warmup. Percentiles pool all token steps from the remaining repeated full-example traversals.",
        "",
        "| workload | metric | samples | p50 µs | p95 µs | p99 µs | p99.9 µs | max µs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        runtime = row.get("runtime")
        if not runtime:
            continue
        for key, label in (("mask_us", "mask"), ("commit_us", "commit"), ("tbm_us", "mask+commit")):
            stats = runtime.get(key)
            if not stats:
                continue
            lines.append(
                f"| `{row['slug']}` | {label} | {stats['count']} | {_fmt(stats['p50'], 3)} | {_fmt(stats['p95'], 3)} | {_fmt(stats['p99'], 3)} | {_fmt(stats['p99_9'], 3)} | {_fmt(stats['max'], 3)} |"
            )

    lines += [
        "",
        "## Correctness/discrepancy gate",
        "",
        "| workload | glrmask replay status | validity label | reference built | raw discrepancy steps | raw disputed token events | note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        c = row["correctness"]
        note = "raw disagreements not adjudicated"
        if row["glrmask_build"]["timeout"]:
            note = "glrmask build timed out; no runtime result"
        elif row["glrmask_build"]["error"]:
            note = "glrmask build failed; no runtime result"
        elif c.get("replay_completed") is False:
            note = "glrmask replay failed"
        elif c.get("gold_replay_ok") is None:
            note = "checked-in example has no explicit validity label; replay completed"
        lines.append(
            f"| `{row['slug']}` | {c.get('correctness_status')} | {_fmt(c['expected_valid'])} | {_fmt(c['reference_available'])} | {_fmt(c['raw_discrepancy_step_count'], 0)} | {_fmt(c['raw_disputed_token_events'], 0)} | {note} |"
        )

    lines += [
        "",
        "## Workload manifest",
        "",
    ]
    for row in rows:
        lines.append(f"- **`{row['slug']}`** — `{row['problem_id']}`: {row['reason']}")

    lines += [
        "",
        "## Reproduction metadata",
        "",
        f"- glrmask commit: `{summary['metadata']['glrmask']['commit']}`",
        f"- CFA commit: `{summary['metadata']['cfa']['commit']}`",
        f"- Python: `{summary['metadata']['python']['version']}`",
        f"- Rust: `{summary['metadata']['rust']['rustc']}`",
        f"- OS: `{summary['metadata']['platform']['platform']}`",
        f"- CPU: `{summary['metadata']['cpu']['model']}`",
        f"- raw output directory: `{summary['output_dir']}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfa-root", type=Path, required=True, help="Path to constraint-framework-analysis checkout")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh or empty directory for raw results")
    parser.add_argument(
        "--workload",
        action="append",
        choices=[w.slug for w in WORKLOADS],
        help="Run only the selected workload slug; repeatable. Default: full fixed suite.",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=DEFAULT_TOTAL_TIMING_RUNS,
        help="Total glrmask traversals per example, including run 0 warmup (default: 51).",
    )
    parser.add_argument("--quick", action="store_true", help="Smoke mode: one build and 3 total timing runs per workload.")
    parser.add_argument("--fail-on-blocker", action="store_true", help="Exit nonzero after writing results if a release blocker is found.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfa_root = args.cfa_root.resolve()
    output_dir = args.output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]

    if not (cfa_root / "scripts" / "sweep.py").is_file():
        raise SystemExit(f"not a CFA checkout: {cfa_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    raw_dir.mkdir()
    log_dir.mkdir()

    selected_slugs = set(args.workload or [w.slug for w in WORKLOADS])
    selected = [w for w in WORKLOADS if w.slug in selected_slugs]
    total_timing_runs = 3 if args.quick else args.timing_runs
    if total_timing_runs < 2:
        raise SystemExit("--timing-runs must be >= 2 so run 0 can be excluded as warmup")

    ffi = _probe_ffi(cfa_root)
    reference = _probe_reference(cfa_root)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "glrmask": _git_metadata(repo_root),
        "cfa": _git_metadata(cfa_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "rust": {
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": _cpu_metadata(),
        "ffi": ffi,
        "reference": reference,
        "build_mode": "release FFI supplied by caller; CFA glrmask_native adapter",
        "measurement": {
            "source_schema_preprocessing": "disabled: --no-strip-pattern-max-length --no-coerce-one-of-to-any-of",
            "process_isolation": "one fresh CFA sweep subprocess per workload",
            "build_repetitions_default": 1 if args.quick else 3,
            "runtime_total_runs": total_timing_runs,
            "runtime_warmup_runs_excluded": 1,
            "runtime_measured_runs": total_timing_runs - 1,
            "runtime_distribution": "pooled token-step timings from measured full-example traversals",
            "reference_framework": REFERENCE_FRAMEWORK,
            "reference_timing_runs": 0,
            "discrepancy_adjudication": "raw mask disagreements recorded; not adjudicated by this harness",
        },
        "important_environment": _important_environment(),
        "workloads": [asdict(w) for w in selected],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    workload_summaries: list[dict[str, Any]] = []
    env = _cfa_env(cfa_root)
    for index, workload in enumerate(selected, 1):
        print(f"\n=== [{index}/{len(selected)}] {workload.slug}: {workload.problem_id} ===", flush=True)
        raw_path = raw_dir / f"{workload.slug}.json"
        log_path = log_dir / f"{workload.slug}.log"
        build_runs = 1 if args.quick else workload.build_runs
        build_timeout = min(workload.build_timeout_seconds, 15) if args.quick else workload.build_timeout_seconds
        command = [
            sys.executable,
            "-m",
            "scripts.sweep",
            "--problems",
            workload.problem_id,
            "--frameworks",
            GLRMASK_FRAMEWORK,
            REFERENCE_FRAMEWORK,
            "--no-strip-pattern-max-length",
            "--no-coerce-one-of-to-any-of",
            "--max-examples-per-problem",
            "1",
            "--build-timeout-seconds",
            str(build_timeout),
            "--discrepancy-sample-budget",
            "0",
            "--timing-runs",
            f"{GLRMASK_FRAMEWORK}:{total_timing_runs},{REFERENCE_FRAMEWORK}:0",
            "--min-example-time",
            "0",
            "--timing-summary-stats",
            "mean",
            "p50",
            "p95",
            "p99",
            "max",
            "--record-timing-runs",
            "always",
            "--build-runs",
            str(build_runs),
            "--target-build-time",
            "9999",
            "--validity-check",
            "warn",
            "--output",
            str(raw_path),
        ]
        returncode = _tee_subprocess(command, cwd=cfa_root, env=env, log_path=log_path)
        if not raw_path.is_file():
            raise RuntimeError(f"CFA did not produce expected raw result: {raw_path} (exit {returncode})")
        workload_summaries.append(_summarize_workload(workload, raw_path, returncode))

    blockers = [row["slug"] for row in workload_summaries if row["release_blocker"]]
    overall_status = "BLOCKED" if blockers else "READY WITH SPECIFIC CAVEATS"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "overall_status": overall_status,
        "release_blockers": blockers,
        "output_dir": str(output_dir),
        "metadata": metadata,
        "workloads": workload_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_markdown(summary, output_dir / "summary.md")

    print(f"\nWrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Overall status: {overall_status}")
    if blockers:
        print("Release blockers: " + ", ".join(blockers))
    return 1 if blockers and args.fail_on_blocker else 0


if __name__ == "__main__":
    raise SystemExit(main())
