import json
from collections import Counter

def explain_breakdown(filename):
    print(f"--- Breakdown for {filename} ---")
    try:
        with open(filename, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    # Count ranges per weight
    counts = [len(w) for w in weights]
    freqs = Counter(counts)
    
    total_ranges = 0
    total_weights = 0
    
    # Sort by range size
    print(f"{'Size (N)':<10} | {'Freq (Weights)':<15} | {'Subtotal (N * Freq)':<20}")
    print("-" * 55)
    
    sorted_sizes = sorted(freqs.keys())
    
    for size in sorted_sizes:
        count = freqs[size]
        subtotal = size * count
        total_ranges += subtotal
        total_weights += count
        print(f"{size:<10} | {count:<15} | {subtotal:<20}")

    print("-" * 55)
    print(f"{'TOTAL':<10} | {total_weights:<15} | {total_ranges:<20}")
    print("\n")

if __name__ == "__main__":
    explain_breakdown("range_weights_terminal_dwa.json")
    explain_breakdown("range_weights_parser_dwa.json")
