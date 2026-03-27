#!/usr/bin/env python3
"""Generate terminal-DWA graph SVGs from GLRMASK_DEBUG_PROFILE logs.

Produces honest graph visualization that distinguishes between direct terminal
matches and "future terminal" transitions (where tokens don't actually match the
labeled terminal but position the grammar for it).

Usage:
    python scripts/generate_dwa_graphs.py --log /tmp/profile_*.log --output-dir output/

Or generate logs and graphs in one step:
    python scripts/generate_dwa_graphs.py --schema path/to/schema.json \
        --vocab path/to/gpt2_vocab.json --config open_close --output-dir output/
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict


def generate_profile_log(schema_path, vocab_path, config_name, config_env):
    """Run profiling and capture stderr to a log file."""
    log_path = f"/tmp/profile_{config_name}_{os.path.basename(schema_path).replace('.json', '')}.log"

    env_copy = os.environ.copy()
    for k in ["GLRMASK_NO_OPEN_QUOTE_SPLIT", "GLRMASK_SPLIT_CLOSE_QUOTE"]:
        if k in env_copy:
            del env_copy[k]
    for k, v in config_env.items():
        if v:
            env_copy[k] = v
    env_copy["GLRMASK_DEBUG_PROFILE"] = "1"

    code = f'''
import json, _glrmask as glrmask
with open("{vocab_path}") as f:
    vd = json.load(f)
vocab = glrmask.Vocab.from_dict({{k.encode(): v for k,v in vd.items()}})
with open("{schema_path}") as f:
    schema = f.read()
c = glrmask.Constraint.from_json_schema(schema, vocab)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env_copy, capture_output=True, text=True, timeout=120
    )
    with open(log_path, "w") as f:
        f.write(result.stderr)
    return log_path


def parse_token_map(log_path):
    """Parse [token_map] entries: internal_id -> (original_ids, bytes_repr)."""
    token_map = {}
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r'\[token_map\] internal=(\d+) originals=\[([^\]]*)\] bytes="(.*)"$',
                line,
            )
            if m:
                iid = int(m.group(1))
                originals = [int(x) for x in m.group(2).split(",") if x.strip()]
                bytes_repr = m.group(3)
                token_map[iid] = {"originals": originals, "bytes": bytes_repr}
    return token_map


def parse_terminal_names(log_path):
    """Parse [tokenizer_terminal] entries: expr_id -> name."""
    names = {}
    with open(log_path) as f:
        for line in f:
            m = re.search(r"\[tokenizer_terminal\]\s+expr=(\d+)\s+name=(.*)", line)
            if m:
                eid = int(m.group(1))
                name = m.group(2).strip()
                names[eid] = name
    return names


def parse_weight(weight_line):
    """Parse weight into {tsid: set(token_ids)}."""
    result = {}
    for m in re.finditer(r"(\d+(?:\.\.\=?\d+)?)→\{([^}]+)\}", weight_line):
        tsid_str = m.group(1)
        tids_str = m.group(2)
        if ".." in tsid_str:
            lo, hi = tsid_str.replace("=", "").split("..")
            tsids = range(int(lo), int(hi) + 1)
        else:
            tsids = [int(tsid_str)]
        tids = set()
        for part in tids_str.split(","):
            part = part.strip()
            if ".." in part:
                lo, hi = part.replace("=", "").split("..")
                tids.update(range(int(lo), int(hi) + 1))
            else:
                tids.add(int(part))
        for tsid in tsids:
            result.setdefault(tsid, set()).update(tids)
    return result


def parse_dwa_from_log(log_path):
    """Parse DWA states, transitions, and weights from log."""
    with open(log_path) as f:
        all_lines = f.readlines()

    state_indices = [i for i, l in enumerate(all_lines) if "[terminal_dwa][state]" in l]
    if not state_indices:
        return {}, [], None, {}

    n_states = 0
    for line in all_lines:
        if "[terminal_dwa][summary]" in line:
            m = re.search(r"\bstates=(\d+)", line)
            if m:
                n_states = int(m.group(1))

    last_trial = state_indices[-n_states:] if n_states else state_indices

    states = {}
    transitions = []
    transition_weights = {}  # (src, dst, label) -> weight_dict
    start_state = None

    for si in last_trial:
        line = all_lines[si]
        m = re.search(
            r"\[state\]\s+id=(\d+)(.*?)incoming=(\d+)\s+outgoing=(\d+).*?final=(\S+)",
            line,
        )
        if not m:
            continue

        sid = int(m.group(1))
        is_start = "[START]" in m.group(2)
        is_final = m.group(5) != "none"
        states[sid] = {"is_start": is_start, "is_final": is_final}
        if is_start:
            start_state = sid

        for j in range(si + 1, min(si + 2000, len(all_lines))):
            tline = all_lines[j]
            tm = re.match(r"^\s{4}(\d+)\s+->\s+State\s+(\d+)", tline)
            if tm:
                label = int(tm.group(1))
                target = int(tm.group(2))
                transitions.append((sid, target, label))
                # Parse weight from next line
                if j + 1 < len(all_lines) and "weight:" in all_lines[j + 1]:
                    transition_weights[(sid, target, label)] = parse_weight(
                        all_lines[j + 1]
                    )
            elif "[terminal_dwa]" in tline:
                break
            elif tline.startswith("[glrmask/debug]") and "[terminal_dwa]" not in tline:
                break

    return states, transitions, start_state, transition_weights


def classify_edge(label, terminal_names, weight, token_map):
    """Classify an edge as 'match' or 'future'.

    A 'future' edge is one where the tokens on the edge don't plausibly match
    the terminal pattern. For example, tokens starting with '"' on an edge
    labeled JSON_STRING_CHAR_EXACT_1024 (which excludes '"').

    Returns (classification, sample_tokens).
    """
    tname = terminal_names.get(label, f"T{label}")

    # Collect all internal token IDs from the weight
    all_tids = set()
    for tids in weight.values():
        all_tids.update(tids)

    # Get sample token bytes
    samples = []
    for tid in sorted(all_tids)[:8]:
        info = token_map.get(tid)
        if info:
            samples.append(info["bytes"])

    if not samples:
        return "unknown", []

    # Heuristic classification:
    # 1. If terminal is a single-char literal and all tokens start with that char → match
    # 2. If terminal name contains CHAR_EXACT or CHAR_UPTO (body-only patterns)
    #    and all tokens start with '"' → future (quote terminates the body)
    # 3. If terminal is JSON_STRING_BODY and all tokens start with '"' → future
    # 4. Default: match

    is_body_terminal = any(
        pattern in tname
        for pattern in [
            "CHAR_EXACT",
            "CHAR_UPTO",
            "STRING_BODY",
            "STRING_BOUNDED",
        ]
    )

    all_start_with_quote = all(s.startswith('\\"') or s.startswith('"') for s in samples)

    if is_body_terminal and all_start_with_quote:
        return "future", samples

    # Check for structural literals: if terminal is "}" or "," or "{" etc.
    # and no token starts with that character
    if len(tname) == 1 and tname in '{}[],':
        terminal_char = tname
        if not any(s.startswith(terminal_char) or s.startswith(f"\\{terminal_char}") for s in samples):
            return "future", samples

    return "match", samples


def shorten_terminal(name):
    """Shorten terminal name for graph labels."""
    if len(name) > 40:
        name = name[:37] + "..."
    name = name.replace("\\", "\\\\").replace('"', '\\"')
    return name


def escape_dot(s):
    """Escape a string for use in dot labels."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def generate_dot_svg(
    states, transitions, start_state, transition_weights,
    terminal_names, token_map, title, output_dir, filename
):
    """Generate SVG with honest edge annotations."""
    if not states:
        return None

    # Merge parallel edges and classify
    edge_info = defaultdict(lambda: {"labels": [], "classes": [], "all_samples": []})
    for src, dst, label in transitions:
        tname = terminal_names.get(label, f"T{label}")
        tname_short = shorten_terminal(tname)
        weight = transition_weights.get((src, dst, label), {})
        classification, samples = classify_edge(label, terminal_names, weight, token_map)
        key = (src, dst)
        edge_info[key]["labels"].append((tname_short, classification))
        edge_info[key]["classes"].append(classification)
        edge_info[key]["all_samples"].extend(samples)

    lines = []
    lines.append("digraph DWA {")
    lines.append(f'  label="{escape_dot(title)} ({len(states)} states, {len(transitions)} transitions)";')
    lines.append('  labelloc="t";')
    lines.append("  fontsize=16;")
    lines.append('  fontname="Helvetica";')
    lines.append("  rankdir=LR;")
    lines.append('  node [fontname="Helvetica", fontsize=11];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')
    lines.append("")

    if start_state is not None:
        lines.append("  __start [shape=point, width=0.15];")
        lines.append(f"  __start -> S{start_state};")
        lines.append("")

    for sid in sorted(states.keys()):
        info = states[sid]
        attrs = [f'label="S{sid}"']
        if info["is_final"]:
            attrs += ['shape=doublecircle', 'color="#006600"', "penwidth=2"]
        else:
            attrs.append("shape=circle")
        if info["is_start"]:
            attrs += ["style=filled", 'fillcolor="#ddeeff"']
        lines.append(f'  S{sid} [{", ".join(attrs)}];')

    lines.append("")

    for (src, dst), info in sorted(edge_info.items()):
        labels_with_class = info["labels"]
        has_future = any(c == "future" for _, c in labels_with_class)

        # Build label text
        label_parts = []
        for tname, cls in labels_with_class[:4]:
            if cls == "future":
                label_parts.append(f"[future] {tname}")
            else:
                label_parts.append(tname)
        if len(labels_with_class) > 4:
            label_parts.append(f"+{len(labels_with_class) - 4} more")

        label_text = "\\n".join(label_parts)

        # Add sample tokens for future edges
        if has_future and info["all_samples"]:
            unique_samples = list(dict.fromkeys(info["all_samples"]))[:3]
            sample_str = ", ".join(escape_dot(s) for s in unique_samples)
            if len(info["all_samples"]) > 3:
                sample_str += f", +{len(info['all_samples']) - 3}"
            label_text += f"\\n(tokens: {sample_str})"

        attrs = [f'label="{label_text}"']

        if has_future:
            attrs.append("style=dashed")
            attrs.append('color="#999999"')
            attrs.append('fontcolor="#666666"')

        if src == dst:
            attrs += ["dir=both", "arrowtail=none"]

        lines.append(f'  S{src} -> S{dst} [{", ".join(attrs)}];')

    lines.append("}")

    dot_content = "\n".join(lines)
    dot_path = f"/tmp/{filename}.dot"
    svg_path = os.path.join(output_dir, f"{filename}.svg")

    with open(dot_path, "w") as f:
        f.write(dot_content)

    result = subprocess.run(
        ["dot", "-Tsvg", dot_path, "-o", svg_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  dot error: {result.stderr}")
        return None

    return svg_path


CONFIGS = {
    "no_split": {
        "GLRMASK_NO_OPEN_QUOTE_SPLIT": "1",
        "GLRMASK_SPLIT_CLOSE_QUOTE": "",
    },
    "open_only": {
        "GLRMASK_NO_OPEN_QUOTE_SPLIT": "",
        "GLRMASK_SPLIT_CLOSE_QUOTE": "",
    },
    "close_only": {
        "GLRMASK_NO_OPEN_QUOTE_SPLIT": "1",
        "GLRMASK_SPLIT_CLOSE_QUOTE": "1",
    },
    "open_close": {
        "GLRMASK_NO_OPEN_QUOTE_SPLIT": "",
        "GLRMASK_SPLIT_CLOSE_QUOTE": "1",
    },
}


def process_log(log_path, title, output_dir, filename):
    """Process a single log file into a graph."""
    token_map = parse_token_map(log_path)
    terminal_names = parse_terminal_names(log_path)
    states, transitions, start_state, weights = parse_dwa_from_log(log_path)

    print(f"  States: {len(states)}, Transitions: {len(transitions)}, Start: {start_state}")
    print(f"  Token map entries: {len(token_map)}")

    # Count future edges
    future_count = 0
    for src, dst, label in transitions:
        w = weights.get((src, dst, label), {})
        cls, _ = classify_edge(label, terminal_names, w, token_map)
        if cls == "future":
            future_count += 1
    print(f"  Future edges: {future_count}/{len(transitions)}")

    svg_path = generate_dot_svg(
        states, transitions, start_state, weights,
        terminal_names, token_map, title, output_dir, filename
    )
    return svg_path


def main():
    parser = argparse.ArgumentParser(description="Generate terminal-DWA graph SVGs")
    parser.add_argument("--log", nargs="+", help="Pre-existing profile log file(s)")
    parser.add_argument("--schema", help="Schema file to profile")
    parser.add_argument("--vocab", help="Vocab JSON file")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), help="Single config to run")
    parser.add_argument("--all-configs", action="store_true", help="Run all 4 configs")
    parser.add_argument("--output-dir", default=".", help="Output directory for SVGs")
    parser.add_argument("--title", help="Graph title prefix")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.log:
        for log_path in args.log:
            basename = os.path.basename(log_path).replace(".log", "")
            title = args.title or basename
            print(f"\n{'='*50}")
            print(f"Processing {log_path}...")
            svg = process_log(log_path, title, args.output_dir, f"dwa_{basename}")
            if svg:
                print(f"  SVG: {svg}")

    elif args.schema and args.vocab:
        configs_to_run = CONFIGS if args.all_configs else {args.config or "open_only": CONFIGS.get(args.config or "open_only")}
        schema_name = os.path.basename(args.schema).replace(".json", "")

        for config_name, config_env in configs_to_run.items():
            print(f"\n{'='*50}")
            print(f"Generating {schema_name}/{config_name}...")
            log_path = generate_profile_log(args.schema, args.vocab, config_name, config_env)
            print(f"  Log: {log_path}")
            title = f"{schema_name} / {config_name.replace('_', ' ').title()}"
            filename = f"dwa_{config_name}_{schema_name}"
            svg = process_log(log_path, title, args.output_dir, filename)
            if svg:
                print(f"  SVG: {svg}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
