#!/usr/bin/env python3
import re
import argparse
import sys
from collections import Counter, defaultdict

# ... (The analyze_log_file function is unchanged) ...
def analyze_log_file(filepath):
    """
    Parses a log file to find transitions of (start_state, token, end_state)
    and counts the occurrences of each unique transition.
    """
    # Regex now captures the token character and ID
    start_pattern = re.compile(
        r"Phase1/2-start.*- token '(.+?)' \((\d+)\) -.*GSSStats {.*?"
        r"unique_nodes: (\d+),.*?"
        r"total_edges: (\d+),.*"
    )
    end_pattern = re.compile(
        r"After processing token .*, number of GSS nodes: (\d+), edges: (\d+)"
    )

    transition_counts = Counter()
    last_start_info = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                start_match = start_pattern.search(line)
                if start_match:
                    token_char = start_match.group(1)
                    # Handle the special case of an empty single quote token ''
                    if token_char == "'": token_char = "''"

                    token_id = start_match.group(2)
                    unique_nodes = int(start_match.group(3))
                    total_edges = int(start_match.group(4))

                    last_start_info = {
                        "token": (token_char, token_id),
                        "state": (unique_nodes, total_edges)
                    }
                    continue

                end_match = end_pattern.search(line)
                if end_match and last_start_info:
                    end_nodes = int(end_match.group(1))
                    end_edges = int(end_match.group(2))
                    end_state = (end_nodes, end_edges)

                    # The key for our counter is now a 3-part tuple
                    transition = (last_start_info["state"], last_start_info["token"], end_state)
                    transition_counts[transition] += 1
                    last_start_info = None
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)

    return transition_counts


def print_token_summary(transition_counts, sort_by='occurrences', limit=None):
    """Groups transitions by token and prints an aggregate summary."""
    if not transition_counts:
        print("No relevant log line pairs found.")
        return

    summary_data = defaultdict(lambda: {
        'occurrences': 0, 'failures': 0, 'transitions': Counter()
    })

    for (start, token, end), count in transition_counts.items():
        summary_data[token]['occurrences'] += count
        if end[0] == 0:
            summary_data[token]['failures'] += count
        summary_data[token]['transitions'][(start, end)] += count

    processed = []
    for token, data in summary_data.items():
        failure_rate = (data['failures'] / data['occurrences']) * 100

        # Get the most common transition and unpack its details
        top_trans_tuple = data['transitions'].most_common(1)[0][0]
        start_state, end_state = top_trans_tuple

        processed.append({
            'token': token, 'occurrences': data['occurrences'], 'failures': data['failures'],
            'failure_rate': failure_rate,
            'top_start_nodes': start_state[0], 'top_start_edges': start_state[1],
            'top_end_nodes': end_state[0], 'top_end_edges': end_state[1]
        })

    sort_key = lambda x: x[sort_by]
    processed.sort(key=sort_key, reverse=True)
    if limit: processed = processed[:limit]

    # --- NEW: Adjusted widths and headers for better alignment ---
    w = {'tok': 14, 'id': 3, 'occ': 11, 'fail': 8, 'rate': 10,
         'tsn': 11, 'tse': 11, 'ten': 11, 'tee': 11} # tsn = Top Start Nodes, etc.

    header_top = (f"{'Token':<{w['tok']}} | {'ID':>{w['id']}} | {'Occurrences':>{w['occ']}} | "
                  f"{'Failures':>{w['fail']}} | {'Failure %':>{w['rate']}} | "
                  f"{'Most Common Transition'.center(w['tsn'] + w['tse'] + w['ten'] + w['tee'] + 9)}")

    header_bottom = (f"{'':<{w['tok']}} | {'':>{w['id']}} | {'':>{w['occ']}} | "
                     f"{'':>{w['fail']}} | {'':>{w['rate']}} | "
                     f"{'Start Nodes':>{w['tsn']}} | {'Start Edges':>{w['tse']}} | "
                     f"{'->':^2} | {'End Nodes':>{w['ten']}} | {'End Edges':>{w['tee']}}")

    print(header_top)
    print(header_bottom)
    print("-" * len(header_bottom))

    for item in processed:
        token_char, token_id = item['token']
        rate_str = f"{item['failure_rate']:.1f}%"
        row = (f"{token_char:<{w['tok']}} | {token_id:>{w['id']}} | {item['occurrences']:>{w['occ']}} | "
               f"{item['failures']:>{w['fail']}} | {rate_str:>{w['rate']}} | "
               f"{item['top_start_nodes']:>{w['tsn']}} | {item['top_start_edges']:>{w['tse']}} | "
               f"{'->':^2} | {item['top_end_nodes']:>{w['ten']}} | {item['top_end_edges']:>{w['tee']}}")
        print(row)

# ... (The rest of the script: print_start_state_summary, print_detailed_stats, main are unchanged) ...
def print_start_state_summary(transition_counts, sort_by='occurrences', limit=None):
    """Groups transitions by start state and prints an aggregate summary."""
    if not transition_counts:
        print("No relevant log line pairs found.")
        return

    summary_data = defaultdict(lambda: {
        'occurrences': 0, 'failures': 0, 'outcomes': Counter(), 'tokens': Counter()
    })

    for (start, token, end), count in transition_counts.items():
        summary_data[start]['occurrences'] += count
        summary_data[start]['tokens'][token] += count
        if end[0] == 0:
            summary_data[start]['failures'] += count
        else:
            summary_data[start]['outcomes'][end] += count

    processed = []
    for start, data in summary_data.items():
        failure_rate = (data['failures'] / data['occurrences']) * 100
        top_outcome = data['outcomes'].most_common(1)
        top_outcome_str = f"({top_outcome[0][0][0]}, {top_outcome[0][0][1]})" if top_outcome else "N/A"
        top_token_char = data['tokens'].most_common(1)[0][0][0]
        processed.append({
            'start': start, 'occurrences': data['occurrences'], 'failures': data['failures'],
            'failure_rate': failure_rate, 'top_outcome_str': top_outcome_str, 'top_token': top_token_char
        })

    sort_key = lambda x: x[sort_by] if sort_by != 'nodes' else x['start'][0]
    processed.sort(key=sort_key, reverse=True)
    if limit: processed = processed[:limit]

    w = {'s_nodes': 11, 's_edges': 11, 'occ': 11, 'fail': 8, 'rate': 10, 'top_tok': 9, 'top_out': 20}
    header = (f"{'Start Nodes':>{w['s_nodes']}} | {'Start Edges':>{w['s_edges']}} | {'Occurrences':>{w['occ']}} | "
              f"{'Failures':>{w['fail']}} | {'Failure %':>{w['rate']}} | {'Top Token':<{w['top_tok']}} | {'Top Success Outcome':<{w['top_out']}}")
    print(header)
    print("-" * len(header))

    for item in processed:
        start_nodes, start_edges = item['start']
        rate_str = f"{item['failure_rate']:.1f}%"
        row = (f"{start_nodes:>{w['s_nodes']}} | {start_edges:>{w['s_edges']}} | {item['occurrences']:>{w['occ']}} | "
               f"{item['failures']:>{w['fail']}} | {rate_str:>{w['rate']}} | {item['top_token']:<{w['top_tok']}} | {item['top_outcome_str']:<{w['top_out']}}")
        print(row)

def print_detailed_stats(transition_counts, limit=None):
    """Prints the detailed view of every unique transition, now with tokens."""
    if not transition_counts:
        print("No relevant log line pairs found.")
        return

    sorted_transitions = sorted(transition_counts.items(), key=lambda item: item[1], reverse=True)
    if limit: sorted_transitions = sorted_transitions[:limit]

    w = {'cnt': 5, 'tok': 5, 'id': 3, 's_nodes': 11, 's_edges': 11, 'e_nodes': 9, 'e_edges': 9, 'stat': 7}
    header = (f"{'Count':>{w['cnt']}} | {'Token':<{w['tok']}} | {'ID':>{w['id']}} | {'Start Nodes':>{w['s_nodes']}} | "
              f"{'Start Edges':>{w['s_edges']}} | {'End Nodes':>{w['e_nodes']}} | {'End Edges':>{w['e_edges']}} | {'Status':<{w['stat']}}")
    print(header)
    print("-" * len(header))

    for (start, token, end), count in sorted_transitions:
        status = "FAILURE" if end[0] == 0 else "Success"
        row = (f"{count:>{w['cnt']}} | {token[0]:<{w['tok']}} | {token[1]:>{w['id']}} | {start[0]:>{w['s_nodes']}} | "
               f"{start[1]:>{w['s_edges']}} | {end[0]:>{w['e_nodes']}} | {end[1]:>{w['e_edges']}} | {status:<{w['stat']}}")
        print(row)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze GLR parser logs to show counts of GSS state transitions.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("logfile", nargs="?", default="./.temp", help="Path to the log file (default: ./.temp)")
    parser.add_argument(
        "--view", choices=['token', 'summary', 'detailed'], default='token',
        help="Set the output view:\n"
             "  token:    (default) Group by token to see which are most problematic.\n"
             "  summary:  Group by start state for a high-level overview.\n"
             "  detailed: Show every unique start->token->end transition."
    )
    parser.add_argument(
        "--sort-by", choices=['occurrences', 'failures', 'failure_rate', 'nodes'], default='occurrences',
        help="Sort the summary or token view. (default: occurrences)"
    )
    parser.add_argument("-n", "--limit", type=int, help="Limit the output to the top N results.")
    args = parser.parse_args()

    transition_counts = analyze_log_file(args.logfile)

    if args.view == 'token':
        print_token_summary(transition_counts, sort_by=args.sort_by, limit=args.limit)
    elif args.view == 'summary':
        print_start_state_summary(transition_counts, sort_by=args.sort_by, limit=args.limit)
    else:
        print_detailed_stats(transition_counts, limit=args.limit)

if __name__ == "__main__":
    main()