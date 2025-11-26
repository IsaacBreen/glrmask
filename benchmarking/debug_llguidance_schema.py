import sys
import json
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.grammars.test_schemas import ALL_SCHEMAS

try:
    from llguidance import JsonCompiler
except ImportError:
    print("llguidance not installed")
    sys.exit(1)

def debug_schemas():
    compiler = JsonCompiler()
    
    for name, schema in ALL_SCHEMAS.items():
        print(f"Testing schema: {name}")
        try:
            schema_str = json.dumps(schema)
            compiled = compiler.compile(schema_str)
            print("  Success")
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    debug_schemas()
