#!/usr/bin/env python3
"""Generate DWA graphs (with labeled edges) + paths for kb_814 with correct 3-set vocab splits.

Sets:
  1. Pure non-alnum (NO alphanumeric chars)
  2. Mixed (has alnum AND other chars)
  3. Pure alnum (ONLY alphanumeric chars)

Cases: 1+2+3, 1+2, 1, 2, 3
"""

import json, re, os, subprocess, sys, random, time
from collections import defaultdict, deque
from pathlib import Path

CFA_ROOT = Path("/Users/isaacbreen/Projects2/constraint-framework-analysis")
SCHEMA_PATH = CFA_ROOT / "data/sources/jsonschemabench/data/Kubernetes/kb_814_Normalized.json"
VOCAB_CACHE = CFA_ROOT / ".cache/vocab_cache"
PYTHON = "/Users/isaacbreen/miniforge3/envs/py312/bin/python3"

ALNUM_RE = re.compile(r"[A-Za-z0-9]")
PURE_ALNUM_RE = re.compile(r"^[A-Za-z0-9]+$")

STATE_RE = re.compile(
    r"\[glrmask/debug\]\[terminal_dwa\]\[state\] id=(\d+)( \[START\])? incoming=(\d+) outgoing=(\d+) to_start=(\d+) self_loops=(\d+) final=(.+)$"
)
EDGE_RE = re.compile(r"^\s{4}(\d+)\s+->\s+State\s+(\d+)$")
TOKEN_MAP_RE = re.compile(
    r'\[glrmask/debug\]\[terminal_dwa\]\[token_map\] internal=(\d+) originals=\[([^\]]*)\]'
)
WEIGHT_ENTRY_RE = re.compile(r'(\d+(?:\.\.\=?\d+)?)\s*→\s*\{([^}]*)\}')

INNER_SCRIPT = '''
import json, sys
from pathlib import Path
import _glrmask as glrmask
from cfa.tokenization import build_vocab_info_from_token_bytes

token_bytes_hex = json.loads(Path(sys.argv[1]).read_text())
schema = Path(sys.argv[2]).read_text()
kept = [bytes.fromhex(h) for h in token_bytes_hex]
filtered = build_vocab_info_from_token_bytes(kept)
token_to_id = {tok: tid for tid, tok in filtered.id_to_token_bytes.items()}
vocab = glrmask.Vocab.from_dict(token_to_id)
glrmask.Constraint.from_json_schema(schema, vocab)
id_to_hex = {str(tid): tb.hex() for tid, tb in filtered.id_to_token_bytes.items()}
Path(sys.argv[3]).write_text(json.dumps(id_to_hex))
print(json.dumps({"vocab_size": len(filtered.id_to_token_bytes)}))
'''

# ---- Graph rendering (with edge labels) ----

def parse_log_graph(log_path):
    """Parse log for graph rendering: states, edges with terminal names."""
    lines = Path(log_path).read_text().splitlines()
    terminal_names = {}
    states = {}
    edge_terminals = defaultdict(list)  # (src, dst) -> [terminal_name, ...]
    current_state = None

    for line in lines:
        m = re.match(r'\[glrmask/debug\]\[tokenizer_terminal\] expr=(\d+) name=(.*)', line)
        if m:
            terminal_names[int(m.group(1))] = m.group(2).rstrip('\n\r')
            continue

        m = STATE_RE.search(line)
        if m:
            sid = int(m.group(1))
            states[sid] = {
                'is_start': m.group(2) is not None,
                'final_nonempty': m.group(7).strip() != 'none',
            }
            current_state = sid
            continue

        m = EDGE_RE.match(line)
        if m and current_state is not None:
            tid = int(m.group(1))
            dst = int(m.group(2))
            name = terminal_names.get(tid, f'T{tid}')
            edge_terminals[(current_state, dst)].append(name)
            continue

        if line.startswith('[glrmask/debug][terminal_dwa]'):
            current_state = None

    start_states = [sid for sid, info in states.items() if info['is_start']]
    return terminal_names, states, edge_terminals, start_states[0] if start_states else 0


def format_edge_label(terminal_list):
    """Format terminal names for edge label, grouping AP_KEYs, property keys, and enum values."""
    ap_keys = []
    prop_keys = []
    enum_values = []
    other = []

    SKIP_NAMES = {'JSON_STRING_BODY', 'JSON_INTEGER', 'JSON_NUMBER', 'JSON_BOOL', 'JSON_NULL'}
    # Property key terminals: `name": ` or `name` (with or without close-quote/colon)
    PROP_KEY_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9]*(?:"\s*:\s*)?$')
    # Enum value terminals: `value"` (string literal with close quote)
    ENUM_VAL_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9]*"$')
    # Long regex patterns (abbreviate)
    BODY_REGEX_RE = re.compile(r'^\(\?:.*\)\*')

    for name in terminal_list:
        if re.match(r'OBJ_ORD_\d+_AP_KEY_\d+', name):
            ap_keys.append(name)
        elif BODY_REGEX_RE.match(name):
            # Abbreviate to a short label
            if '": ' in name or '":' in name:
                other.append('ANY_KEY_BODY')
            else:
                other.append('STRING_BODY')
        elif ENUM_VAL_RE.match(name) and name.rstrip('"') not in SKIP_NAMES:
            enum_values.append(name)
        elif PROP_KEY_RE.match(name) and name not in SKIP_NAMES:
            prop_keys.append(name)
        else:
            other.append(name)

    labels = []
    for name in other:
        labels.append(name.replace('\\', '\\\\').replace('"', '\\"'))
    if enum_values:
        if len(enum_values) <= 3:
            for v in enum_values:
                labels.append(v.replace('"', '\\"'))
        else:
            labels.append(f"{len(enum_values)} enum values")
    if prop_keys:
        if len(prop_keys) <= 3:
            for k in prop_keys:
                labels.append(k.replace('"', '\\"'))
        else:
            labels.append(f"{len(prop_keys)} property keys")
    if ap_keys:
        if len(ap_keys) <= 3:
            for k in ap_keys:
                labels.append(k)
        else:
            labels.append(f"{len(ap_keys)} AP_KEYs")

    return '\\n'.join(labels)


def tarjan_scc(nodes, edges):
    index = 0
    stack = []
    onstack = set()
    indices = {}
    lowlink = {}
    sccs = []
    adjacency = defaultdict(list)
    for source, target in edges:
        adjacency[source].append(target)

    def strongconnect(node):
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        onstack.add(node)
        for nxt in adjacency.get(node, []):
            if nxt not in indices:
                strongconnect(nxt)
                lowlink[node] = min(lowlink[node], lowlink[nxt])
            elif nxt in onstack:
                lowlink[node] = min(lowlink[node], indices[nxt])
        if lowlink[node] == indices[node]:
            component = []
            while True:
                item = stack.pop()
                onstack.remove(item)
                component.append(item)
                if item == node:
                    break
            sccs.append(sorted(component))

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    return sccs


def compute_levels(states, edges, start_state):
    nodes = set(states)
    sccs = tarjan_scc(nodes, list(edges))
    component_of = {}
    for cid, comp in enumerate(sccs):
        for node in comp:
            component_of[node] = cid

    dag = defaultdict(set)
    for src, dst in edges:
        sc, dc = component_of[src], component_of[dst]
        if sc != dc:
            dag[sc].add(dc)

    start_c = component_of[start_state]
    reachable = {start_c}
    q = deque([start_c])
    while q:
        c = q.popleft()
        for n in dag.get(c, ()):
            if n not in reachable:
                reachable.add(n)
                q.append(n)

    topo_in = {c: 0 for c in reachable}
    for c in reachable:
        for n in dag.get(c, ()):
            if n in reachable:
                topo_in[n] += 1

    q = deque(sorted(c for c, d in topo_in.items() if d == 0))
    topo = []
    while q:
        c = q.popleft()
        topo.append(c)
        for n in sorted(dag.get(c, ())):
            if n in topo_in:
                topo_in[n] -= 1
                if topo_in[n] == 0:
                    q.append(n)

    levels = {start_c: 0}
    for c in topo:
        base = levels.get(c, 0)
        for n in dag.get(c, ()):
            if n in reachable:
                levels[n] = max(levels.get(n, 0), base + 1)

    max_level = max(levels.values(), default=0)
    for cid in range(len(sccs)):
        if cid not in levels:
            max_level += 1
            levels[cid] = max_level

    return {node: levels[component_of[node]] for node in nodes}


def render_graph(output_dir, case_key, label, states, edge_terminals, start_state, vocab_size):
    """Render DWA graph with terminal names on edges."""
    levels = compute_levels(states, edge_terminals.keys(), start_state)
    groups = defaultdict(list)
    for node, level in levels.items():
        groups[level].append(node)

    lines = []
    lines.append("digraph G {")
    lines.append("  graph [")
    lines.append('    rankdir=LR,')
    lines.append('    bgcolor="white",')
    lines.append('    labelloc=t,')
    lines.append('    labeljust=c,')
    lines.append(f'    label="kb_814 terminal DWA | {label} | vocab {vocab_size}",')
    lines.append('    fontname="Helvetica-Bold",')
    lines.append('    fontsize=20,')
    lines.append('    pad=0.35,')
    lines.append('    nodesep=0.5,')
    lines.append('    ranksep=0.95,')
    lines.append('    splines=true,')
    lines.append('    outputorder=edgesfirst')
    lines.append("  ];")
    lines.append('  node [shape=ellipse, fontname="Helvetica", fontsize=12, margin="0.10,0.06", style="filled"];')
    lines.append('  edge [fontname="Helvetica", fontsize=11, color="#4f5b66", penwidth=1.4, arrowsize=0.7];')
    lines.append('  startdot [shape=circle, width=0.13, height=0.13, fixedsize=true, label="", style=filled, fillcolor="black", color="black"];')

    for level in range(max(groups) + 1 if groups else 0):
        nodes = sorted(groups.get(level, []))
        if not nodes:
            continue
        lines.append("  { rank=same;")
        for node in nodes:
            info = states[node]
            final_text = "final: non-empty" if info["final_nonempty"] else "final: empty"
            label_text = f"S{node}\\n{final_text}"
            if info["final_nonempty"]:
                attrs = 'shape=ellipse, peripheries=2, penwidth=2.2, color="#1b5e20", fillcolor="#f9fff8", fontcolor="#16381c"'
            else:
                attrs = 'shape=ellipse, peripheries=1, penwidth=1.5, color="#4f8fba", fillcolor="#eaf4ff", fontcolor="#16364d"'
            lines.append(f'    n{node} [{attrs}, label="{label_text}"];')
        lines.append("  }")

    lines.append(f'  startdot -> n{start_state} [label="", color="black", penwidth=1.8];')
    for (src, dst), term_list in sorted(edge_terminals.items()):
        elabel = format_edge_label(term_list)
        attrs = [f'label="{elabel}"']
        if src == dst:
            attrs.append('minlen=2')
        lines.append(f'  n{src} -> n{dst} [{", ".join(attrs)}];')
    lines.append("}")

    dot_path = output_dir / f"{case_key}.dot"
    svg_path = output_dir / f"{case_key}.svg"
    dot_path.write_text("\n".join(lines) + "\n")
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)


# ---- Path sampling ----

def parse_range_set(s):
    s = s.strip('{}').strip()
    if not s: return frozenset()
    result = set()
    for part in s.split(','):
        part = part.strip()
        if not part: continue
        if '..=' in part:
            a, b = part.split('..=', 1)
            result.update(range(int(a), int(b) + 1))
        elif '..' in part:
            a, b = part.split('..', 1)
            result.update(range(int(a), int(b)))
        else:
            result.add(int(part))
    return frozenset(result)

def parse_tsid_range(s):
    s = s.strip()
    if '..=' in s:
        a, b = s.split('..=', 1)
        return range(int(a), int(b) + 1)
    elif '..' in s:
        a, b = s.split('..', 1)
        return range(int(a), int(b))
    else:
        return range(int(s), int(s) + 1)

def parse_weight_str(weight_str, max_len=50000):
    if not weight_str or weight_str.strip() == 'none':
        return None
    if len(weight_str) > max_len:
        return None  # skip huge weights to avoid hangs
    result = {}
    for m in WEIGHT_ENTRY_RE.finditer(weight_str):
        tokens = parse_range_set(m.group(2))
        for tsid in parse_tsid_range(m.group(1)):
            if tsid in result:
                result[tsid] = result[tsid] | tokens
            else:
                result[tsid] = tokens
    return result if result else None

def intersect_weights(w1, w2):
    if w1 is None: return w2
    if w2 is None: return w1
    result = {}
    for tsid in set(w1.keys()) & set(w2.keys()):
        common = w1[tsid] & w2[tsid]
        if common:
            result[tsid] = common
    return result if result else {}

def weight_all_tokens(weight):
    if weight is None or weight == {}: return set()
    result = set()
    for tokens in weight.values():
        result.update(tokens)
    return result

def parse_log_paths(log_path):
    terminal_names = {}
    states = {}
    transitions = {}
    internal_to_originals = {}
    current_state = None
    last_edge_key = None

    with open(log_path) as f:
        for line in f:
            line = line.rstrip('\n\r')
            m = re.match(r'\[glrmask/debug\]\[tokenizer_terminal\] expr=(\d+) name=(.*)', line)
            if m:
                terminal_names[int(m.group(1))] = m.group(2).rstrip('\n\r')
                continue
            m = TOKEN_MAP_RE.search(line)
            if m:
                iid = int(m.group(1))
                originals = [int(x) for x in m.group(2).split(',') if x.strip()]
                internal_to_originals[iid] = originals
                continue
            m = STATE_RE.search(line)
            if m:
                sid = int(m.group(1))
                final_str = m.group(7).strip()
                is_final = final_str != 'none'
                states[sid] = {
                    'final': is_final, 'start': m.group(2) is not None,
                    'final_weight': parse_weight_str(final_str) if is_final else None,
                }
                current_state = sid
                last_edge_key = None
                continue
            m = re.match(r'    (\d+) -> State (\d+)', line)
            if m and current_state is not None:
                tid = int(m.group(1))
                ns = int(m.group(2))
                key = (current_state, tid)
                transitions[key] = {'next_state': ns, 'weight': None}
                last_edge_key = key
                continue
            if line.strip().startswith('weight:') and last_edge_key is not None:
                weight_str = line.strip()[len('weight:'):].strip()
                transitions[last_edge_key]['weight'] = parse_weight_str(weight_str)

    start_states = [sid for sid, info in states.items() if info['start']]
    return terminal_names, states, transitions, start_states[0] if start_states else 0, internal_to_originals

def format_token(hex_bytes, tid):
    try:
        tb = bytes.fromhex(hex_bytes)
        text = tb.decode('utf-8', errors='replace')
        escaped = ''
        for ch in text:
            if ch == "'": escaped += "\\'"
            elif ord(ch) >= 0x20 and ord(ch) < 0x7f: escaped += ch
            else: escaped += f'\\x{ord(ch):02x}'
        return f"{tid}:'{escaped}'"
    except:
        return f"{tid}:??"

def sample_paths(states, transitions, start, terminal_names, vocab_map,
                 internal_to_originals, n=500, end_prob=0.05, max_attempts=50000):
    rng = random.Random(42)
    paths = []
    seen = set()
    attempts = 0
    adj = defaultdict(list)
    for (src, tid), info in transitions.items():
        adj[src].append((tid, info['next_state']))

    while len(paths) < n and attempts < max_attempts:
        attempts += 1
        current = start
        path_terms = []
        path_weight = None

        for _ in range(200):
            avail = adj.get(current, [])
            if states[current]['final']:
                if not avail or rng.random() < end_prob: break
            if not avail: break
            tid, ns = rng.choice(avail)
            path_terms.append(f'`{terminal_names.get(tid, f"T{tid}").rstrip(chr(10)+chr(13))}`')
            tw = transitions.get((current, tid), {}).get('weight')
            path_weight = intersect_weights(path_weight, tw)
            current = ns

        if path_terms and states[current]['final']:
            fw = states[current].get('final_weight')
            end_weight = intersect_weights(path_weight, fw)
            path_str = ' → '.join(path_terms)
            if path_str not in seen:
                seen.add(path_str)
                internal_tids = sorted(weight_all_tokens(end_weight))
                original_tids = set()
                for iid in internal_tids:
                    for oid in internal_to_originals.get(iid, [iid]):
                        original_tids.add(oid)
                original_tids = sorted(original_tids)
                wlen = len(original_tids)
                sample = original_tids[:3] if wlen <= 20 else rng.sample(original_tids, min(3, len(original_tids)))
                token_strs = []
                for t in sorted(sample):
                    h = vocab_map.get(str(t))
                    token_strs.append(format_token(h, t) if h else f'{t}:??')
                paths.append({'path': path_str, 'len': len(path_terms),
                              'weight_len': wlen, 'tokens': token_strs})

    paths.sort(key=lambda x: (-x['len'], x['path']))
    return paths

def enumerate_trivial_paths(states, transitions, start, terminal_names, vocab_map, internal_to_originals):
    """For <=3-state DWAs, enumerate all single-step paths."""
    adj = defaultdict(list)
    for (src, tid), info in transitions.items():
        adj[src].append((tid, info['next_state']))
    paths = []
    for tid, dst in adj.get(start, []):
        if not states[dst]['final']: continue
        tname = terminal_names.get(tid, f'T{tid}')
        path_str = f'`{tname.rstrip(chr(10)+chr(13))}`'
        tw = transitions.get((start, tid), {}).get('weight')
        internal_tids = sorted(weight_all_tokens(tw)) if tw else []
        original_tids = set()
        for iid in internal_tids[:10]:
            for oid in internal_to_originals.get(iid, [iid]):
                original_tids.add(oid)
        wlen = len(original_tids) if internal_tids else 0
        sample_t = sorted(original_tids)[:3]
        token_strs = [format_token(vocab_map.get(str(t), ''), t) for t in sample_t]
        paths.append({'path': path_str, 'len': 1, 'weight_len': wlen, 'tokens': token_strs})
    paths.sort(key=lambda x: (-x['weight_len'], x['path']))
    return paths

def write_paths(txt_path, label, states, n_edges, vocab_size, paths):
    with open(txt_path, 'w') as f:
        f.write(f"# kb_814 Terminal DWA Paths — {label}\n")
        f.write(f"# States: {len(states)}, Grouped transitions: {n_edges}, Vocab: {vocab_size}\n")
        nonempty = [p for p in paths if p['weight_len'] > 0]
        empty = [p for p in paths if p['weight_len'] == 0]
        f.write(f"# Sampled: {len(nonempty)} paths with tokens + {len(empty)} paths with empty weight\n#\n")
        idx = 1
        for p in nonempty:
            tstr = ', '.join(p['tokens']) if p['tokens'] else '(empty)'
            wl = p['weight_len']
            tl = f"tokens_all=[{tstr}]" if wl <= 20 else f"tokens_sample=[{tstr}]"
            f.write(f"[{idx:4d}] (len={p['len']:3d}, weight={wl:5d}) {p['path']}  {tl}\n")
            idx += 1
        if empty:
            f.write(f"\n# --- Paths with empty weight (no single token spans the full path) ---\n")
            for p in empty:
                f.write(f"[{idx:4d}] (len={p['len']:3d}) {p['path']}\n")
                idx += 1


# ---- Main ----

def main(output_dir, env_extra=None, env_label=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocab and classify
    sys.path.insert(0, str(CFA_ROOT))
    from cfa.tokenization import load_vocab_info
    vocab_info = load_vocab_info(cache_dir=VOCAB_CACHE)

    set1, set2, set3 = [], [], []
    for tid, tb in vocab_info.id_to_token_bytes.items():
        text = tb.decode("utf-8", errors="replace")
        has_alnum = bool(ALNUM_RE.search(text))
        pure_alnum = bool(PURE_ALNUM_RE.match(text)) if text else False
        if not has_alnum:
            set1.append(tb)
        elif pure_alnum:
            set3.append(tb)
        else:
            set2.append(tb)

    print(f"Set 1 (pure non-alnum): {len(set1)}")
    print(f"Set 2 (mixed): {len(set2)}")
    print(f"Set 3 (pure alnum): {len(set3)}")

    cases = [
        ("set123_full", "1+2+3 (Full Vocab)", set1 + set2 + set3),
        ("set12_no_pure_alnum", "1+2 (No Pure Alnum)", set1 + set2),
        ("set1_pure_nonalnum", "1 (Pure Non-Alnum Only)", set1),
        ("set2_mixed", "2 (Mixed Only)", set2),
        ("set3_pure_alnum", "3 (Pure Alnum Only)", set3),
    ]

    for case_key, label, token_bytes_list in cases:
        full_label = f"{label}{env_label}"
        print(f"\n  {case_key} ({len(token_bytes_list)} tokens)...", flush=True)

        input_path = output_dir / f"{case_key}_input.json"
        input_path.write_text(json.dumps([tb.hex() for tb in token_bytes_list]))

        log_path = output_dir / f"{case_key}.log"
        vocab_path = output_dir / f"{case_key}_vocab.json"
        txt_path = output_dir / f"{case_key}_paths.txt"

        env = os.environ.copy()
        env["GLRMASK_DEBUG_PROFILE"] = "1"
        env["PYTHONPATH"] = str(CFA_ROOT)
        if env_extra:
            env.update(env_extra)

        t0 = time.monotonic()
        proc = subprocess.run(
            [PYTHON, "-c", INNER_SCRIPT, str(input_path), str(SCHEMA_PATH), str(vocab_path)],
            cwd="/Users/isaacbreen/Projects2/glrmask2", env=env,
            capture_output=True, text=True, timeout=120)
        build_time = time.monotonic() - t0

        if proc.returncode != 0:
            print(f"    ERROR: {proc.stderr[-300:]}")
            continue

        log_path.write_text(proc.stderr)
        meta = json.loads(proc.stdout.strip())
        vs = meta['vocab_size']
        print(f"    vocab={vs} build={build_time:.1f}s", end="", flush=True)

        # Graph
        _, gstates, edge_terms, gstart = parse_log_graph(str(log_path))
        render_graph(output_dir, case_key, full_label, gstates, edge_terms, gstart, vs)
        n_grouped = len(edge_terms)
        print(f" states={len(gstates)} edges={n_grouped}", end="", flush=True)

        # Paths
        terminal_names, states, transitions, start, i2o = parse_log_paths(str(log_path))
        n_edges = len(set((s, transitions[(s, t)]['next_state']) for s, t in transitions))

        if len(states) <= 3:
            paths = enumerate_trivial_paths(states, transitions, start, terminal_names,
                                            json.loads(vocab_path.read_text()), i2o)
        else:
            paths = sample_paths(states, transitions, start, terminal_names,
                                 json.loads(vocab_path.read_text()), i2o, n=500)

        write_paths(txt_path, full_label, states, n_edges, vs, paths)
        nonempty = len([p for p in paths if p['weight_len'] > 0])
        print(f" paths={nonempty}+{len(paths)-nonempty}")

    # Convert to PDF and merge
    print("\nConverting to PDFs...", flush=True)
    pdfs = []
    for case_key, _, _ in cases:
        svg = output_dir / f"{case_key}.svg"
        pdf = output_dir / f"{case_key}.pdf"
        if svg.exists():
            subprocess.run(["rsvg-convert", "-f", "pdf", str(svg), "-o", str(pdf)], check=True)
            pdfs.append(str(pdf))

    merged = output_dir / "all_cases.pdf"
    subprocess.run(["/System/Library/Automator/Combine PDF Pages.action/Contents/MacOS/join",
        "-o", str(merged)] + pdfs, check=True)
    print(f"Done! Merged PDF at {merged}")
    return merged


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/kb814_dwa_v2')
    parser.add_argument('--split-close-quote', action='store_true')
    parser.add_argument('--split-colon-space', action='store_true')
    args = parser.parse_args()

    env_extra = {}
    env_label = ""
    if args.split_close_quote:
        env_extra["GLRMASK_SPLIT_CLOSE_QUOTE"] = "1"
        env_label += " [close-split]"
    if args.split_colon_space:
        env_extra["GLRMASK_SPLIT_COLON_SPACE"] = "1"
        env_label += " [colon-split]"

    merged = main(args.output_dir, env_extra=env_extra, env_label=env_label)
