import argparse
import importlib
import json
from pathlib import Path
import sys

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

    try:
        print(f"Importing {args.implementation_class} from {args.implementation_module}...")
        module = importlib.import_module(args.implementation_module)
        gss_class = getattr(module, args.implementation_class)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not load GSS implementation. {e}", file=sys.stderr)
        sys.exit(1)

    print("Running test specification...")
    results = []
    for state, line_no in run_test_spec(gss_class):
        results.append({
            "line": line_no,
            "state": state
        })

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
