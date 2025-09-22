import argparse
import importlib
import json
from typing import Tuple
from pathlib import Path
import sys
import os

from .test_spec import run_test_spec

def main():
    parser = argparse.ArgumentParser(description="Run GSS test specification against an implementation.")
    parser.add_argument(
        "implementation_module",
        help="The Python module containing the GSS implementation (e.g., 'gss_tester.reference_impl')."
    )
    parser.add_argument(
        "implementation_class",
        help="The name of the GSS class within the module (e.g., 'ReferenceGSS')."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to the output JSON file for the results."
    )
    args = parser.parse_args()

    # Enable expensive validation checks in LeveledGSS for testing.
    os.environ['GSS_TESTER_VALIDATE'] = '1'

    try:
        print(f"Importing {args.implementation_class} from {args.implementation_module}...")
        module = importlib.import_module(args.implementation_module)
        gss_class = getattr(module, args.implementation_class)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not load GSS implementation. {e}", file=sys.stderr)
        sys.exit(1)

    print("Running test specification...")
    results = []
    for yielded in run_test_spec(gss_class):
        # Support (state, line) and (state, line, trace) tuples from the test spec.
        if isinstance(yielded, tuple):
            if len(yielded) == 2:
                state, line_no = yielded
                item = {
                    "line": line_no,
                    "state": state
                }
            elif len(yielded) >= 3:
                state, line_no, trace = yielded[0], yielded[1], yielded[2]
                item = {
                    "line": line_no,
                    "state": state,
                    "trace": trace
                }
            else:
                # Unexpected tuple arity; fallback to legacy behavior.
                state, line_no = yielded[0], yielded[1]
                item = {"line": line_no, "state": state}
            results.append(item)

    output_data = {
        "implementation": f"{args.implementation_module}.{args.implementation_class}",
        "results": results
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully ran {len(results)} checks.")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
