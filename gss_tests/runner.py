import json
from test_spec import run_tests


def main():
    results = []
    for state, line_number in run_tests():
        results.append({"gss": state, "line_number": line_number})
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
