"""Balanced fresh-process comparisons of the same native setup/scaling driver."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jobs-per-worker", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.jobs_per_worker < 1 or args.pairs < 1:
        parser.error("workers, jobs-per-worker and pairs must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    identity = {"started_at": dt.datetime.now().astimezone().isoformat(),
                "max_workers": args.workers, "jobs_per_worker": args.jobs_per_worker, "pairs": args.pairs,
                "binaries": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                             for k, p in binaries.items()},
                "notes": ["Same driver; persistent pools, five weak-scaling trials per row.",
                          "Record physical/logical cores and topology separately; worker count is not core count.",
                          "First workload warmup and pool creation excluded by the driver."]}
    (output / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf8")
    trials = {k: [] for k in binaries}
    for pair in range(args.pairs):
        order = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        for kind in order:
            stem = f"pair-{pair + 1}-{kind}"
            print("RUN", stem, flush=True)
            start = time.perf_counter()
            with (output / (stem + ".json")).open("wb") as out, (output / (stem + ".stderr")).open("wb") as err:
                result = subprocess.run([str(binaries[kind]), str(args.workers), str(args.jobs_per_worker)], stdout=out, stderr=err, timeout=900)
            assert result.returncode == 0, (stem, result.returncode)
            data = json.loads((output / (stem + ".json")).read_text(encoding="utf8"))
            trials[kind].append({(r["case"], r["slots"], r["workers"]): r for r in data["rows"]})
            print("DONE", stem, round(time.perf_counter() - start, 3), flush=True)
    keys = trials["baseline"][0].keys()
    assert all(t.keys() == keys for side in trials.values() for t in side)
    rows = []
    for key in sorted(keys):
        vals = {side: [t[key]["aggregate_ns_per_job"] for t in ts] for side, ts in trials.items()}
        med = {side: statistics.median(v) for side, v in vals.items()}
        rows.append({"case": key[0], "slots": key[1], "workers": key[2],
                     "ns_per_job": med, "trials_ns": vals,
                     "candidate_over_baseline": med["candidate"] / med["baseline"]})
    summary = {"identity": identity, "rows": rows,
               "completed_at": dt.datetime.now().astimezone().isoformat()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    for row in rows:
        if row["workers"] == 1 or (row["slots"] in (0, 4, 128) and row["workers"] == 16):
            print(row["case"], row["slots"], row["workers"], row["ns_per_job"],
                  "ratio", round(row["candidate_over_baseline"], 4), flush=True)


if __name__ == "__main__":
    main()
