import re
import sys
import argparse

def parse_log(filename, top_n):
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    # Regex to capture experiment lines
    # [Det&Sim Experiment] [Context] Config N#x-D#y: NWA=[...] | DWA=[...] -> Time: 1.23s, States: 123
    pattern = re.compile(r'\[Det&Sim Experiment\] \[(.*?)\] Config N#(\d+)-D#(\d+): NWA=\[(.*?)\] \| DWA=\[(.*?)\] -> Time: (.*?), States: (\d+)')
    
    experiments = {}

    for match in pattern.finditer(content):
        context = match.group(1)
        n_idx = int(match.group(2))
        d_idx = int(match.group(3))
        nwa_passes = match.group(4)
        dwa_passes = match.group(5)
        time_str = match.group(6)
        states = int(match.group(7))
        
        # Parse time
        if 'ms' in time_str:
            time_val = float(time_str.replace('ms', '')) / 1000.0
        elif 'µs' in time_str:
            time_val = float(time_str.replace('µs', '')) / 1000000.0
        elif 's' in time_str:
            time_val = float(time_str.replace('s', ''))
        else:
            time_val = 0.0 # Should not happen
            
        if context not in experiments:
            experiments[context] = []
            
        experiments[context].append({
            'n_idx': n_idx,
            'd_idx': d_idx,
            'nwa_passes': nwa_passes,
            'dwa_passes': dwa_passes,
            'time': time_val,
            'states': states
        })

    if not experiments:
        print("No matching experiment lines found in the log.")
        return

    for context, results in experiments.items():
        print(f"--- Context: {context} ---")
        # Sort by states (asc), then time (asc)
        results.sort(key=lambda x: (x['states'], x['time']))
        
        best = results[0]
        print(f"Best Configuration:")
        print(f"  NWA Passes: [{best['nwa_passes']}]")
        print(f"  DWA Passes: [{best['dwa_passes']}]")
        print(f"  Time: {best['time']:.4f}s")
        print(f"  States: {best['states']}")
        print(f"  (Config N#{best['n_idx']}-D#{best['d_idx']})")

        print(f"\nTop {top_n} Configurations:")
        for i, res in enumerate(results[:top_n]):
             print(f"  {i+1}. States: {res['states']}, Time: {res['time']:.4f}s | NWA: [{res['nwa_passes']}] | DWA: [{res['dwa_passes']}]")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Det&Sim Experiment logs.")
    
    # Positional argument for the filename
    parser.add_argument("filename", help="The path to the log file to analyze")
    
    # Optional argument for 'n' (number of top results)
    parser.add_argument("-n", "--number", type=int, default=5, 
                        help="Number of top configurations to display (default: 5)")

    args = parser.parse_args()

    parse_log(args.filename, args.number)