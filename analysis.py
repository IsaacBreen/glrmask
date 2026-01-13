
import json
import math
import sys
from collections import defaultdict

def analyze_weights(filename):
    print(f"--- Analyzing {filename} ---")
    try:
        with open(filename, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    num_unique_weights = len(weights)
    total_ranges = sum(len(w) for w in weights)
    
    # "Atoms" analysis
    # Collect all start and end points
    points = set()
    for w in weights:
        for start, end in w:
            points.add(start)
            points.add(end + 1)
    
    sorted_points = sorted(list(points))
    atoms = []
    if sorted_points:
        for i in range(len(sorted_points) - 1):
            if sorted_points[i] < sorted_points[i+1]:
                atoms.append((sorted_points[i], sorted_points[i+1] - 1))
                
    num_atoms = len(atoms)

    # Entropy analysis (treating weights as symbols)
    # This requires knowing the frequency of each weight, which we don't have from just the unique list.
    # We can only analyze the complexity of the unique set itself.
    
    avg_ranges = total_ranges / num_unique_weights if num_unique_weights > 0 else 0
    
    # Range count frequency analysis
    range_counts = defaultdict(int)
    for w in weights:
        range_counts[len(w)] += 1
        
    print(f"Unique Weights: {num_unique_weights}")
    print(f"Total Ranges (in unique weights): {total_ranges}")
    print(f"Average Ranges per Weight: {avg_ranges:.2f}")
    print(f"Partition Atoms: {num_atoms}")
    print("\nRange Count Frequency (Ranges -> Count):")
    for count in sorted(range_counts.keys()):
        print(f"  {count} ranges: {range_counts[count]}")

    # Visualization: Pie Chart for Contribution (Granular)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        
        counts = sorted(range_counts.keys())
        contributions = {c: c * range_counts[c] for c in counts}
        total_contribution = sum(contributions.values())
        
        # Group small slices into "Other"
        # We want to explode the "Keepers" into individual slices.
        # e.g. Count=6040, Freq=3 -> 3 slices of size 6040.
        
        final_sizes = []
        final_colors = []
        legend_labels = {} # Label -> Color
        
        other_size = 0
        threshold = 0.005 * total_contribution # 0.5% threshold
        always_show = {1, 2, 3}
        
        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        # Generate a color map
        # We need as many colors as there are "Bins" (not slices)
        # We can cycle through a tab20 or similar
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        bin_index = 0
        
        for count, contrib in sorted_contribs:
            if contrib >= threshold or count in always_show:
                freq = range_counts[count]
                # Add 'freq' slices of size 'count'
                # All share the same color
                color = colors[bin_index % len(colors)]
                bin_index += 1
                
                final_sizes.extend([count] * freq)
                final_colors.extend([color] * freq)
                
                legend_labels[f"{count} ranges"] = color
            else:
                other_size += contrib
        
        if other_size > 0:
            final_sizes.append(other_size)
            final_colors.append('#999999') # Grey for Other
            legend_labels["Other"] = '#999999'
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot with white wedges to visualize the "slicing"
        wedges, _ = ax.pie(final_sizes, colors=final_colors, startangle=140, 
                           wedgeprops={"edgecolor":"w", 'linewidth': 0.5})
        
        ax.axis('equal')
        
        # Create a legend
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=color, label=label) for label, color in legend_labels.items()]
        plt.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc="center left", title="Bin Sizes")
        
        plt.title(f'Range Complexity Contribution: {filename}\n(Subdivided by Weight Frequency)')
        
        output_img = filename.replace('.json', '_pie.png')
        plt.savefig(output_img, bbox_inches='tight')
        print(f"\nSaved granular pie chart to {output_img}")
        
    except ImportError:
        print("\nmatplotlib not found, skipping visualization.")

    # Outlier Analysis
    print("\n--- Outlier Analysis ---")
    # Identify large weights (arbitrary threshold or specific request)
    target_sizes = [6053] # Specific request
    
    for i, w in enumerate(weights):
        if len(w) in target_sizes:
            print(f"Weight at index {i} has {len(w)} ranges.")
            # Analyze what it ISN'T (gaps) vs what it IS
            # If ranges are huge, likely it covers most of the space.
            # Let's check the gaps.
            
            # Assuming u32 space or similar? The file name says dwa_i32, so likely i32.
            # But tokens are usually u32 or char based.
            # Let's just print the first few and last few ranges to guess the range.
            print(f"  Ranges (first 5): {w[:5]}")
            print(f"  Ranges (last 5): {w[-5:]}")
            
            # Check for gaps if it looks like a large coverage
            gaps = []
            if len(w) > 1:
                for j in range(len(w) - 1):
                    # Gap between w[j].end and w[j+1].start
                    # w[j] is (start, end) inclusive
                    gap_start = w[j][1] + 1
                    gap_end = w[j+1][0] - 1
                    if gap_start <= gap_end:
                        gaps.append((gap_start, gap_end))
            
            print(f"  Total Gaps detected: {len(gaps)}")
            print(f"  Gaps (first 10): {gaps[:10]}")
            if len(gaps) > 10:
                print(f"  ... and {len(gaps)-10} more gaps.")

if __name__ == "__main__":
    analyze_weights("range_weights_terminal_nwa.json")
    analyze_weights("range_weights_terminal_dwa.json")
    analyze_weights("range_weights_parser_dwa.json")
