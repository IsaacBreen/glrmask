#!/usr/bin/env python3
import re
import argparse
import sys
from collections import Counter, defaultdict

def analyze_log_file(filepath):
    """
    Parses a log file to find pairs of start/end GSS stats and count
    the occurrences of each unique transition.
    """
    start_pattern = re.compile(
        r"Phase1/2-start.*GSSStats {.*?"
        r"unique_nodes: (\d+),.*?"
        r"total_edges: (\d+),.*"
    )
    end_pattern = re.compile(
        r"After processing token .*, number of GSS nodes: (\d+), edges: (\d+)"
    )

    transition_counts = Counter()
    last_start_stats = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                start_match = start_pattern.search(line)
                if start_match:
                    unique_nodes = int(start_match.group(1))
                    total_edges = int(start_match.group(2))
                    last_start_stats = (unique_nodes, total_edges)
                    continue

                end_match = end_pattern.search(line)
                if end_match and last_start_stats:
                    end_nodes = int(end_match.group(1))
                    end_edges = int(end_match.group(2))
                    end_stats = (end_nodes, end_edges)
                    transition = (last_start_stats, end_stats)
                    transition_counts[transition] += 1
                    last_start_stats = None
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)

    return transition_counts

def print_detailed_stats(transition_counts, limit=None):
    """Prints the original detailed view of every unique transition."""
    if not transition_counts:
        print("No relevant log line pairs found.")
        return

    sorted_transitions = sorted(
        transition_counts.items(), key=lambda item: item[1], reverse=True
    )

    if limit:
        sorted_transitions = sorted_transitions[:limit]

    # ... (The pretty printing logic from the previous script)
    w = {'count': 5, 's_nodes': 11, 's_edges': 11, 'e_nodes': 9, 'e_edges': 9, 'status': 7}
    for (start, end), count in sorted_transitions:
        w['count'] = max(w['count'], len(str(count)))
        w['s_nodes'] = max(w['s_nodes'], len(str(start[0])))
        w['s_edges'] = max(w['s_edges'], len(str(start[1])))
        w['e_nodes'] = max(w['e_nodes'], len(str(end[0])))
        w['e_edges'] = max(w['e_edges'], len(str(end[1])))

    header = (f"{'Count':>{w['count']}} | {'Start Nodes':>{w['s_nodes']}} | {'Start Edges':>{w['s_edges']}} | "
              f"{'End Nodes':>{w['e_nodes']}} | {'End Edges':>{w['e_edges']}} | {'Status':<{w['status']}}")
    print(header)
    print("-" * len(header))

    for (start, end), count in sorted_transitions:
        status = "FAILURE" if end[0] == 0 else "Success"
        row = (f"{count:>{w['count']}} | {start[0]:>{w['s_nodes']}} | {start[1]:>{w['s_edges']}} | "
               f"{end[0]:>{w['e_nodes']}} | {end[1]:>{w['e_edges']}} | {status:<{w['status']}}")
        print(row)

def print_summary_stats(transition_counts, sort_by='occurrences', limit=None):
    """Groups transitions by start state and prints an aggregate summary."""
    if not transition_counts:
        print("No relevant log line pairs found.")
        return

    # --- Aggregate data by start_state ---
    summary_data = defaultdict(lambda: {
        'occurrences': 0,
        'failures': 0,
        'outcomes': Counter()
    })

    for (start_state, end_state), count in transition_counts.items():
        summary_data[start_state]['occurrences'] += count
        if end_state[0] == 0:
            summary_data[start_state]['failures'] += count
        else:
            summary_data[start_state]['outcomes'][end_state] += count

    # --- Prepare for sorting and printing ---
    processed_summary = []
    for start_state, data in summary_data.items():
        failure_rate = (data['failures'] / data['occurrences']) * 100
        unique_outcomes = len(data['outcomes']) + (1 if data['failures'] > 0 else 0)

        top_outcome = data['outcomes'].most_common(1)
        top_outcome_str = f"({top_outcome[0][0][0]}, {top_outcome[0][0][1]})" if top_outcome else "N/A"

        processed_summary.append({
            'start_state': start_state,
            'occurrences': data['occurrences'],
            'failures': data['failures'],
            'failure_rate': failure_rate,
            'unique_outcomes': unique_outcomes,
            'top_outcome_str': top_outcome_str
        })

    # --- Sort the data ---
    sort_key_map = {
        'occurrences': lambda x: x['occurrences'],
        'failures': lambda x: x['failures'],
        'failure_rate': lambda x: x['failure_rate'],
        'nodes': lambda x: x['start_state'][0]
    }
    processed_summary.sort(key=sort_key_map[sort_by], reverse=True)

    if limit:
        processed_summary = processed_summary[:limit]

    # --- Print the table ---
    w = {'s_nodes': 11, 's_edges': 11, 'occ': 11, 'fail': 8, 'rate': 10, 'outcomes': 8, 'top': 20}
    header = (f"{'Start Nodes':>{w['s_nodes']}} | {'Start Edges':>{w['s_edges']}} | "
              f"{'Occurrences':>{w['occ']}} | {'Failures':>{w['fail']}} | {'Failure %':>{w['rate']}} | "
              f"{'Outcomes':>{w['outcomes']}} | {'Top Success Outcome':<{w['top']}}")
    print(header)
    print("-" * len(header))

    for item in processed_summary:
        start_nodes, start_edges = item['start_state']
        rate_str = f"{item['failure_rate']:.1f}%"
        row = (f"{start_nodes:>{w['s_nodes']}} | {start_edges:>{w['s_edges']}} | "
               f"{item['occurrences']:>{w['occ']}} | {item['failures']:>{w['fail']}} | {rate_str:>{w['rate']}} | "
               f"{item['unique_outcomes']:>{w['outcomes']}} | {item['top_outcome_str']:<{w['top']}}")
        print(row)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze GLR parser logs to show counts of GSS state transitions.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "logfile", nargs="?", default="./.temp",
        help="Path to the log file to analyze (default: ./.temp)"
    )
    parser.add_argument(
        "--view", choices=['summary', 'detailed'], default='summary',
        help="Set the output view:\n"
             "  summary: (default) Group by start state for a high-level overview.\n"
             "  detailed: Show every unique start->end transition."
    )
    parser.add_argument(
        "--sort-by", choices=['occurrences', 'failures', 'failure_rate', 'nodes'],
        default='occurrences',
        help="Sort the summary view. (default: occurrences)"
    )
    parser.add_argument(
        "-n", "--limit", type=int,
        help="Limit the output to the top N results."
    )
    args = parser.parse_args()

    transition_counts = analyze_log_file(args.logfile)

    if args.view == 'summary':
        print_summary_stats(transition_counts, sort_by=args.sort_by, limit=args.limit)
    else:
        print_detailed_stats(transition_counts, limit=args.limit)

if __name__ == "__main__":
    main()