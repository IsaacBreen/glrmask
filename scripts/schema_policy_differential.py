#!/usr/bin/env python3
"""Focused official JSON Schema differential checks against a native byte probe.

Run with a Python environment containing jsonschema. Formats use the dialect's
ordinary annotation behavior (no FormatChecker). Unsupported builds are recorded,
not correctness passes. Reported empty languages are recorded without proof.
Exit 1 means a built schema disagreed with the validator or the probe failed.
"""
import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import selectors
import subprocess

from jsonschema import Draft7Validator, Draft202012Validator

FILES = (
    "allOf", "oneOf", "not", "if-then-else", "uniqueItems", "minProperties",
    "maxProperties", "propertyNames", "format", "type", "const",
    "boolean_schema", "dependencies", "dependentRequired",
)
DIALECTS = {
    "draft7": ("http://json-schema.org/draft-07/schema#", Draft7Validator),
    "draft2020-12": ("https://json-schema.org/draft/2020-12/schema", Draft202012Validator),
}


def fixtures(suite):
    for dialect, (uri, validator) in DIALECTS.items():
        for name in FILES:
            path = suite / "tests" / dialect / (name + ".json")
            if not path.exists():
                continue
            for index, group in enumerate(json.loads(path.read_text())):
                raw = group["schema"]
                # Boolean schemas cannot carry a dialect declaration themselves.
                schema = dict(raw, **{"$schema": uri}) if isinstance(raw, dict) else {
                    "$schema": uri, "allOf": [raw],
                }
                yield {
                    "id": f"{dialect}/{name}/{index}", "description": group["description"],
                    "schema": schema, "tests": group["tests"], "validator": validator,
                }
        metadata = [
            ({"type": "string", "default": {"patternProperties": {"a": True, ".*": True}},
              "examples": [{"not": {"uniqueItems": True}}]}, ["ok", 1]),
            ({"type": "integer", "title": "Example", "description": "Annotation",
              "x-custom": {"if": False, "then": False}, "$comment": "retained metadata"}, [1, "x"]),
            ({"type": "string", "format": "made-up"}, ["anything", 2]),
        ]
        for index, (raw, values) in enumerate(metadata):
            schema = dict(raw, **{"$schema": uri})
            yield {"id": f"{dialect}/metadata/{index}", "description": "annotation data",
                   "schema": schema, "tests": [{"data": value} for value in values],
                   "validator": validator}


class Probe:
    def __init__(self, path, mode, log, timeout):
        self.path, self.mode, self.log, self.timeout = path, mode, log, timeout
        self.process = None

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
            self.process = None

    def query(self, schema, values):
        if self.process is None:
            self.process = subprocess.Popen(
                [str(self.path), "--mode", self.mode], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            )
        try:
            self.process.stdin.write(json.dumps({"schema": schema, "cases": values}) + "\n")
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                if not selector.select(self.timeout):
                    raise TimeoutError(f"probe exceeded {self.timeout}s for one schema")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"probe exited with code {self.process.poll()}")
            return json.loads(line)
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            self.close()
            return {"status": "probe_error", "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--mode", choices=("fast-build", "fast-runtime", "both"), default="both")
    parser.add_argument("--suite", type=Path, default=Path(
        "/Users/isaacbreen/Projects2/gcg-paper/datasets/json_schema_test_suite"))
    parser.add_argument("--output", type=Path, default=Path("schema-policy-differential.json"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--filter", default="", help="only fixture IDs containing this substring")
    args = parser.parse_args()
    modes = ("fast-build", "fast-runtime") if args.mode == "both" else (args.mode,)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records, counts, failures = [], {}, []
    with args.output.with_suffix(".stderr.log").open("w") as log:
        for mode in modes:
            probe = Probe(args.probe.resolve(), mode, log, args.timeout)
            count = collections.Counter()
            try:
                for fixture in fixtures(args.suite):
                    if args.filter not in fixture["id"]:
                        continue
                    count["schemas"] += 1
                    cls = fixture.pop("validator")
                    try:
                        cls.check_schema(fixture["schema"])
                        validator = cls(fixture["schema"])
                        expected = [validator.is_valid(test["data"]) for test in fixture["tests"]]
                    except Exception as error:
                        count["fixture_validation_errors"] += 1
                        records.append(dict(fixture, mode=mode, status="fixture_error", error=str(error)))
                        continue
                    mismatched_labels = [index for index, (test, valid) in enumerate(
                        zip(fixture["tests"], expected)) if "valid" in test and test["valid"] != valid]
                    count["official_label_disagreements"] += len(mismatched_labels)
                    result = probe.query(fixture["schema"], [test["data"] for test in fixture["tests"]])
                    record = dict(fixture, mode=mode, expected=expected, result=result,
                                  official_label_disagreements=mismatched_labels)
                    records.append(record)
                    status = result.get("status", "missing_status")
                    count[status] += 1
                    if status == "build_error":
                        kind = result.get("kind", "unknown")
                        count["build_error:" + kind] += 1
                        # A valid witness disproves an empty-language claim.
                        # Malformed claims conflict with check_schema above.
                        # Only explicit unsupported/resource limits are ordinary
                        # coverage observations rather than unexpected failures.
                        if kind == "reported_empty" and any(expected):
                            count["reported_empty_with_valid_witness"] += 1
                            record["build_diagnostic_failure"] = "reported_empty_with_valid_witness"
                            failures.append(record)
                        elif kind == "malformed":
                            count["malformed_valid_schema"] += 1
                            record["build_diagnostic_failure"] = "malformed_valid_schema"
                            failures.append(record)
                        elif kind not in ("unsupported", "resource_limit", "reported_empty"):
                            count["unexpected_build_error"] += 1
                            record["build_diagnostic_failure"] = "unexpected_build_error"
                            failures.append(record)
                        continue
                    if status != "built":
                        count["probe_failures"] += 1
                        failures.append(record)
                        continue
                    actual = result.get("accepts", [])
                    if len(actual) != len(expected):
                        count["probe_failures"] += 1
                        record["protocol_error"] = "accepts length differs from case count"
                        failures.append(record)
                        continue
                    differences = []
                    count["runtime_errors"] += len(result.get("runtime_errors", []))
                    for index, (accepted, valid) in enumerate(zip(actual, expected)):
                        if accepted is None:
                            continue
                        count["judged_cases"] += 1
                        if accepted != valid:
                            kind = "false_accept" if accepted else "false_reject"
                            count[kind] += 1
                            differences.append({"case": index, "kind": kind})
                    if differences or result.get("runtime_errors"):
                        record["differences"] = differences
                        failures.append(record)
                counts[mode] = dict(count)
            finally:
                probe.close()
    report = {
        "jsonschema_version": importlib.metadata.version("jsonschema"),
        "suite": str(args.suite.resolve()), "probe": str(args.probe.resolve()),
        "format_checker": False, "files": FILES, "counts": counts, "records": records,
        "instance_serialization": "Native probe recursively uses comma+space and colon+space "
            "separators, serde_json scalar/key spelling, and preserved input object order. "
            "False rejections are reported unchanged; this does not test every JSON spelling.",
        "interpretation": "Unsupported/resource build errors are coverage observations, "
            "not correctness passes. Reported-empty claims with valid witnesses, malformed "
            "claims for check_schema-valid fixtures, and unexpected build errors are failures. "
            "Reported-empty claims without valid witnesses remain unproven engine claims. "
            "Only built cases with boolean accepts are compared; runtime failures remain distinct.",
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.output.with_suffix(".failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"counts": counts, "failures": len(failures), "output": str(args.output)}, indent=2))
    return int(bool(failures) or any(c.get("fixture_validation_errors", 0) or
                                   c.get("official_label_disagreements", 0) for c in counts.values()))


if __name__ == "__main__":
    raise SystemExit(main())
