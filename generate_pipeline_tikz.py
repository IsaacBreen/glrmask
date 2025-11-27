#!/usr/bin/env python3
"""
Generate a comprehensive TikZ figure of the full grammar compilation pipeline.
Fixes:
- ValueError in regex parsing (expected 3 values, got 2).
- Blank pages (strict whitespace control).
- Text artifacts (robust minipage usage).
- Styling (Blue=Shift, Red=Reduce).
"""

import json
import re
import subprocess
from typing import Dict, List, Set, Tuple, Any, Optional

# ==============================================================================
# 1. PARSING LOGIC
# ==============================================================================

def escape_tex(s: str) -> str:
    return s.replace('$', '\\$').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&').replace('#', '\\#')

def format_symbol(symbol: Any, term_map: Dict[int, str] = None, as_id: bool = False) -> str:
    if symbol is None: return "$\\varepsilon$"
    if isinstance(symbol, str):
        if symbol.startswith("neg("): return f"$\\neg${symbol[4:-1]}"
        return escape_tex(symbol)
    if isinstance(symbol, int):
        REDUCE_BASE = -2147483648
        GOTO_BASE = 2147483646
        if REDUCE_BASE <= symbol <= REDUCE_BASE + 100: return f"r{symbol - REDUCE_BASE}"
        if symbol >= GOTO_BASE - 100: return "$\\gamma$"
        if as_id: return str(symbol)
        if term_map and symbol in term_map: return escape_tex(term_map[symbol])
        return str(symbol)
    return escape_tex(str(symbol))

def format_weight(w: str) -> str:
    if w == "ALL" or "1844674407" in w: return "ALL"
    return w

def run_dot_layout(nodes: Set[int], edges: List[Tuple], final_states: Set[int], ranksep=0.5, nodesep=0.5) -> Optional[Dict]:
    all_nodes = set(nodes)
    for u, v, _, _ in edges:
        all_nodes.add(u); all_nodes.add(v)

    dot_content = [
        "digraph G {", 
        f"  rankdir=LR; ranksep={ranksep}; nodesep={nodesep};",
        "  margin=0;"
    ]
    for n in sorted(all_nodes):
        shape = "doublecircle" if n in final_states else "circle"
        dot_content.append(f'  {n} [shape={shape}, width=0.5, fixedsize=true];')
    for u, v, _, _ in edges:
        dot_content.append(f'  {u} -> {v};')
    dot_content.append("}")
    
    try:
        proc = subprocess.run(['dot', '-Tplain'], input="\n".join(dot_content), capture_output=True, text=True, check=True)
        layout = {"nodes": {}, "bbox": (0,0,1,1)}
        for line in proc.stdout.splitlines():
            parts = line.split()
            if not parts: continue
            if parts[0] == "graph": layout["bbox"] = (0, 0, float(parts[2]), float(parts[3]))
            elif parts[0] == "node":
                try: layout["nodes"][int(parts[1])] = (float(parts[2]), float(parts[3]))
                except ValueError: pass
        return layout
    except Exception as e:
        return None

def parse_dwa(dwa_str: str) -> Tuple[Set[int], List[Tuple], Set[int]]:
    nodes, edges, finals = set(), [], set()
    curr = None
    for line in dwa_str.strip().split('\n'):
        if line.startswith("State"):
            curr = int(re.search(r"State (\d+):", line).group(1))
            nodes.add(curr)
        elif "->" in line and "final_weight" not in line:
            parts = line.split("->")
            lbl = parts[0].strip()
            sym = int(lbl) if (lbl.lstrip("-").isdigit()) else (None if lbl=="ε" else lbl)
            tgt_part = parts[1].strip()
            tgt = int(re.match(r"(\d+)", tgt_part).group(1))
            nodes.add(tgt)
            w = "ALL"
            if "(weight:" in tgt_part: w = re.search(r"\(weight:\s*([^)]+)\)", tgt_part).group(1)
            if curr is not None: edges.append((curr, tgt, sym, w))
        elif "final_weight:" in line:
            if curr is not None: finals.add(curr)
    return nodes, edges, finals

def parse_tokenizer(data: str) -> Tuple[Set[int], List[Tuple], Set[int]]:
    nodes, edges, finals = set(), [], set()
    blocks = data.split("DFAState")
    state_idx = 0
    for block in blocks:
        if not block.strip(): continue
        curr = state_idx
        nodes.add(curr)
        m_trans = re.search(r"transitions:\s*\{([^}]*)\}", block)
        if m_trans:
            for k, v in re.findall(r"(-?\d+):\s*(\d+)", m_trans.group(1)):
                k, v = int(k), int(v)
                nodes.add(v)
                lbl = "\\$" if k==36 else (chr(k) if 32<=k<=126 and chr(k) not in "$%&_#" else f"0x{k:x}")
                if k in [36, 37, 38, 35, 95]: lbl = "\\" + chr(k)
                edges.append((curr, v, lbl, "ALL"))
        if "finalizers: Bitset { words: []" not in block: finals.add(curr)
        state_idx += 1
    return nodes, edges, finals

def parse_lalr_full(lalr_str: str) -> Dict[int, Dict[int, str]]:
    table = {}
    for s_block in lalr_str.split("StateID"):
        if "Row" not in s_block: continue
        sid = int(re.match(r"\((\d+)\):\s*Row", s_block).group(1))
        actions = {}
        for t, s in re.findall(r"TerminalID\((\d+)\):\s*Shift\(StateID\((\d+)\)\)", s_block):
            actions[int(t)] = f"s{s}"
        for t, nt in re.findall(r"TerminalID\((\d+)\):\s*Reduce\s*\{\s*nonterminal_id:\s*NonTerminalID\((\d+)\)", s_block):
            actions[int(t)] = f"r{nt}"
        table[sid] = actions
    return table

def parse_char_data(char_str: str) -> Tuple[List[Tuple], List[Tuple]]:
    """Parse characterization string for shifts and reduces."""
    shifts = []
    m_s = re.search(r"initial_shifts:\s*\{([^}]*)\}", char_str)
    if m_s: 
        shifts = [(int(s), int(t)) for s, t in re.findall(r"StateID\((\d+)\),\s*StateID\((\d+)\)", m_s.group(1))]
    
    reduces = []
    m_r = re.search(r"initial_reduces:\s*\{([^}]*)\}", char_str)
    if m_r: 
        # FIX: The regex only captures 2 groups, so we iterate over (s, nt) directly
        reduces = [(int(s), int(nt)) for s, nt in re.findall(r"StateID\((\d+)\),\s*\d+,\s*NonTerminalID\((\d+)\)", m_r.group(1))]
        
    return shifts, reduces

# ==============================================================================
# 2. GENERATION LOGIC
# ==============================================================================

def make_graph(name, nodes, edges, finals, scale=1.0, term_map=None, as_id=False, node_size="6mm", font_sz="\\small"):
    layout = run_dot_layout(nodes, edges, finals)
    if not layout: return f"\\begin{{tikzpicture}}\\node[draw,red] {{Layout Failed: {name}}};\\end{{tikzpicture}}"
    
    bb = layout["bbox"]
    cx, cy = bb[2]/2, bb[3]/2
    
    # baseline=(current bounding box.center) is crucial for alignment
    lines = [f"\\begin{{tikzpicture}}[scale={scale}, baseline=(current bounding box.center)]"]
    if name:
        lines.append(f"\\node[anchor=south, font=\\bfseries\\large] at (0, {(bb[3]/2)*2.54 + 0.5}) {{{name}}};")
        
    valid_nodes = set()
    for n in nodes:
        if n in layout["nodes"]:
            valid_nodes.add(n)
            x, y = layout["nodes"][n]
            tx, ty = (x - cx)*2.54, (y - cy)*2.54
            style = "accepting" if n in finals else "state"
            if n == 0: style += ", initial"
            lines.append(f"\\node[{style}, minimum size={node_size}, font={font_sz}] (n{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")

    grouped = {}
    for u, v, sym, w in edges:
        if u in valid_nodes and v in valid_nodes:
            grouped.setdefault((u,v), []).append((sym, w))
            
    for (u, v), labels in grouped.items():
        parts = []
        for sym, w in labels[:3]:
            s = format_symbol(sym, term_map, as_id)
            if format_weight(w) != "ALL": s += f"/{format_weight(w)[:4]}"
            parts.append(s)
        lbl = ", ".join(parts)
        if len(labels) > 3: lbl += "..."
        
        if u == v:
            lines.append(f"\\path[edge] (n{u}) edge[loop above] node[above, font=\\tiny] {{{lbl}}} (n{v});")
        else:
            lines.append(f"\\path[edge] (n{u}) edge node[above, sloped, font=\\tiny] {{{lbl}}} (n{v});")
            
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)

def make_char_component(term_id: int, term_name: str, char_str: str, lalr: Dict) -> str:
    """Generate Minipage with Table (Top) and Styled Graph (Bottom)"""
    
    # 1. Table
    max_st = max(lalr.keys()) if lalr else 8
    tbl_code = [r"\begin{tabular}{|c|c|} \hline \textbf{St} & \textbf{Act} \\ \hline"]
    for s in range(max_st + 1):
        act = lalr.get(s, {}).get(term_id, "-")
        color = ""
        if 's' in act: color = r"\cellcolor{blue!10}"
        elif 'r' in act: color = r"\cellcolor{red!10}"
        tbl_code.append(f"{s} & {color}{act} \\\\")
    tbl_code.append(r"\hline \end{tabular}")
    table_tex = "\n".join(tbl_code)
    
    # 2. Graph
    shifts, reduces = parse_char_data(char_str)
    nodes = set([u for u,v in shifts] + [v for u,v in shifts] + [u for u,nt in reduces] + [0])
    edges = []
    for u, v in shifts: edges.append((u, v, "shift", f"s{v}"))
    for u, nt in reduces: edges.append((u, u, "reduce", f"r{nt}"))
    
    layout_edges = [(u, v, None, None) for u, v, _, _ in edges]
    layout = run_dot_layout(nodes, layout_edges, set(), ranksep=0.3, nodesep=0.3)
    
    graph_tex = ""
    if layout:
        bb = layout["bbox"]
        cx, cy = bb[2]/2, bb[3]/2
        gt = [r"\begin{tikzpicture}[scale=0.75, baseline=(current bounding box.center)]"]
        
        for n in nodes:
            if n in layout["nodes"]:
                x, y = layout["nodes"][n]
                tx, ty = (x-cx)*2.54, (y-cy)*2.54
                gt.append(f"\\node[state, minimum size=5mm, font=\\tiny] (n{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")
        
        for u, v, type_, lbl in edges:
            if u in layout["nodes"] and v in layout["nodes"]:
                if type_ == "shift":
                    gt.append(f"\\draw[->, thick, blue!70!black] (n{u}) -- node[midway, fill=white, inner sep=0.5pt, font=\\tiny, text=blue] {{{lbl}}} (n{v});")
                else:
                    gt.append(f"\\draw[->, thick, red!70!black] (n{u}) to[loop above, min distance=5mm] node[above, font=\\tiny, text=red] {{{lbl}}} (n{u});")
        gt.append(r"\end{tikzpicture}")
        graph_tex = "\n".join(gt)
    else:
        graph_tex = r"\textit{Layout Failed}"

    # Use minipage to stack them safely
    return f"""
    \\begin{{minipage}}{{2.2cm}}
        \\centering
        \\scalebox{0.65}{{ {table_tex} }}
        \\par\\vspace{{5pt}}
        {graph_tex}
        \\par\\vspace{{2pt}}
        \\scriptsize \\textbf{{Char($\\mathcal{{C}}_{{{term_name}}}$)}}
    \\end{{minipage}}
    """

def make_merged_dwa(skel_nodes, skel_edges, skel_finals, term_map) -> str:
    layout = run_dot_layout(skel_nodes, skel_edges, skel_finals, ranksep=2.0)
    if not layout: return r"\begin{tikzpicture}\node{Layout Failed};\end{tikzpicture}"
    
    bb = layout["bbox"]
    cx, cy = bb[2]/2, bb[3]/2
    lines = [
        "\\begin{tikzpicture}[scale=0.9]",
        f"\\node[font=\\bfseries\\large, align=center] at (0, {(bb[3]/2)*2.54 + 1.0}) {{Terminal DWA (Merged)\\\\ \\footnotesize \\itshape (With Embedded Templates)}};"
    ]
    
    valid_n = set()
    for n in skel_nodes:
        if n in layout["nodes"]:
            valid_n.add(n)
            x, y = layout["nodes"][n]
            tx, ty = (x-cx)*2.54, (y-cy)*2.54
            style = "accepting" if n in skel_finals else "state"
            if n == 0: style += ", initial"
            lines.append(f"\\node[{style}, minimum size=8mm] (n{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")
            
    for u, v, sym, w in skel_edges:
        if u in valid_n and v in valid_n:
            ux, uy = layout["nodes"][u]; vx, vy = layout["nodes"][v]
            tux, tuy = (ux-cx)*2.54, (uy-cy)*2.54; tvx, tvy = (vx-cx)*2.54, (vy-cy)*2.54
            lines.append(f"\\path[edge, very thick] (n{u}) edge (n{v});")
            
            mx, my = (tux+tvx)/2, (tuy+tvy)/2
            dx, dy = tvx-tux, tvy-tuy
            l = (dx**2+dy**2)**0.5
            if l > 0:
                nx, ny = -dy/l, dx/l
                mx += nx*0.9; my += ny*0.9
            
            if isinstance(sym, int) and 0 <= sym <= 3:
                name = ["BoxTemplateZero", "BoxTemplateOne", "BoxTemplateTwo", "BoxTemplateThree"][sym]
                lines.append(f"\\node[inner sep=0pt, scale=0.8] at ({mx:.2f},{my:.2f}) {{\\usebox{{\\{name}}}}};")
            else:
                txt = format_symbol(sym, term_map, True)
                lines.append(f"\\node[fill=white, font=\\small, inner sep=1pt] at ({mx:.2f},{my:.2f}) {{{txt}}};")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)

# ==============================================================================
# 3. MAIN
# ==============================================================================

def main():
    try:
        with open("pipeline_artifacts.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: pipeline_artifacts.json not found.")
        return

    term_map = {0: "$", 1: "a", 2: "b", 3: "c"}

    # 1. GENERATE CONTENT
    tn, te, tf = parse_tokenizer(data["tokenizer_dfa"])
    sn, se, sf = parse_dwa(data["skeleton_dwa"])
    fln, fle, flf = parse_dwa(data["flattened_nwa"])
    fin, fie, fif = parse_dwa(data["final_dwa"])
    lalr = parse_lalr_full(data["lalr_table"])

    code_tok = make_graph("Tokenizer DFA", tn, te, tf, scale=0.7)
    code_skel = make_graph("Terminal DWA (Skeleton)", sn, se, sf, scale=0.7, term_map=term_map)
    code_flat = make_graph("Flattened NWA", fln, fle, flf, scale=0.45, as_id=True)
    code_final = make_graph("Final DWA", fin, fie, fif, scale=0.45, as_id=True)
    code_merged = make_merged_dwa(sn, se, sf, term_map)

    # Templates
    templates_mini = {}
    for k in data["template_dfas_all"]:
        tid = int(re.search(r'\d+', k).group())
        n, e, f = parse_dwa(data["template_dfas_all"][k])
        templates_mini[tid] = make_graph("", n, e, f, scale=0.22, node_size="3mm", font_sz="\\tiny")

    # Characterizations (Row)
    sorted_chars = sorted(data["characterizations_all"].items(), key=lambda x: int(re.search(r"TerminalID\((\d+)\)", x[1]).group(1)))
    char_cells = []
    for k, v in sorted_chars:
        tid = int(re.search(r"TerminalID\((\d+)\)", v).group(1))
        name = term_map.get(tid, "?")
        char_cells.append(make_char_component(tid, name, v, lalr))

    code_chars = "\\begin{tabular}{" + "c"*len(char_cells) + "}\n"
    code_chars += " & ".join(char_cells) + "\\\\\n"
    code_chars += "\\end{tabular}"

    # 2. WRITE LATEX
    latex = [
        r"\documentclass[tikz,border=10pt]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern, amsmath, amssymb, colortbl}",
        r"\usetikzlibrary{automata, positioning, arrows.meta, shapes, shadows, fit, calc, backgrounds}",
        "",
        r"% --- COLORS ---",
        r"\definecolor{primary}{RGB}{41,128,185}",
        r"\definecolor{accent}{RGB}{39,174,96}",
        r"\definecolor{dark}{RGB}{52,73,94}",
        r"\definecolor{grammar}{RGB}{155,89,182}",
        r"\definecolor{vocabcolor}{RGB}{241,196,15}",
        r"\definecolor{stagecolor}{RGB}{236,240,241}",
        "",
        r"% --- STYLES ---",
        r"\tikzset{",
        r"  container/.style={rectangle, draw=dark!30, fill=white, rounded corners, drop shadow, inner sep=5pt, align=center},",
        r"  stagearea/.style={draw=dark!20, fill=stagecolor, rounded corners=15pt, inner sep=10pt},",
        r"  grammarbox/.style={rectangle, draw=grammar, very thick, rounded corners, fill=grammar!5, drop shadow},",
        r"  vocabbox/.style={rectangle, draw=vocabcolor, thick, rounded corners, fill=vocabcolor!10, drop shadow, align=center},",
        r"  state/.style={circle, draw=primary, thick, minimum size=6mm, fill=white, text=dark, font=\small},",
        r"  accepting/.style={state, double, double distance=1.5pt, draw=accent, fill=accent!5},",
        r"  initial/.style={fill=primary!15},",
        r"  edge/.style={->, draw=dark!70, semithick, >=Stealth},",
        r"  flowarrow/.style={->, draw=dark!60, line width=1.5pt, dashed, >=Stealth, rounded corners, line cap=round},",
        r"}",
        "",
        r"% --- BOX REGISTERS ---",
    ]

    names = ["BoxTemplateZero", "BoxTemplateOne", "BoxTemplateTwo", "BoxTemplateThree"]
    for n in names: latex.append(f"\\newsavebox{{\\{n}}}")
    for n in ["BoxGrammar", "BoxTokenizer", "BoxSkeleton", "BoxChars", "BoxMerged", "BoxFlat", "BoxFinal"]:
        latex.append(f"\\newsavebox{{\\{n}}}")

    latex.extend([
        r"\begin{document}",
        r"% Note: Using % at end of lines to prevent blank pages from sbox whitespace"
    ])

    def add_box(name, content):
        latex.append(f"\\sbox{{\\{name}}}{{%")
        latex.append(content)
        latex.append(f"}}%")

    for tid in sorted(templates_mini.keys()):
        if tid < 4: add_box(names[tid], templates_mini[tid])

    add_box("BoxGrammar", r"""
\begin{minipage}{3.5cm}
\small \ttfamily
\textbf{Input Grammar:}\\[3pt]
S ::= a "\$";\\
A ::= 'a' b;\\
B ::= 'b' c;\\
C ::= 'c' a $|$ 'c';
\end{minipage}
""")
    add_box("BoxTokenizer", code_tok)
    add_box("BoxSkeleton", code_skel)
    add_box("BoxChars", code_chars)
    add_box("BoxMerged", code_merged)
    add_box("BoxFlat", code_flat)
    add_box("BoxFinal", code_final)

    latex.extend([
        "",
        r"% --- MAIN LAYOUT ---",
        r"\begin{tikzpicture}[node distance=1.5cm and 1.5cm]",
        "",
        r"  % 1. Grammar (Top)",
        r"  \node[grammarbox] (grammar) {\usebox{\BoxGrammar}};",
        "",
        r"  % 2. Columns",
        r"  \node[container, below left=2cm and 1.5cm of grammar] (tokenizer) {\usebox{\BoxTokenizer}};",
        r"  \begin{scope}[on background layer] \node[stagearea, fit=(tokenizer)] (tok_bg) {}; \end{scope}",
        "",
        r"  % Characterizations (Right)",
        r"  \node[container, below right=2cm and 0.5cm of grammar] (chars) {\usebox{\BoxChars}};",
        r"  \begin{scope}[on background layer] \node[stagearea, fit=(chars)] (chars_bg) {}; \end{scope}",
        r"  \node[anchor=south, font=\bfseries\large] at (chars.north) {Below-Zero Characterizations};",
        "",
        r"  % 3. Skeleton",
        r"  \node[container, below=2cm of tok_bg] (skeleton) {\usebox{\BoxSkeleton}};",
        r"  \node[vocabbox, left=1cm of skeleton] (vocab) {\textbf{Vocab}\\(50k)};",
        r"  \draw[flowarrow] (vocab) -- (skeleton);",
        "",
        r"  % 4. Merged",
        r"  \path (skeleton.south) -- (chars.south) coordinate[midway] (mid);",
        r"  \node[container, below=3cm of mid] (merged) {\usebox{\BoxMerged}};",
        "",
        r"  % 5. Flat & Final",
        r"  \node[container, below=2cm of merged] (flat) {\usebox{\BoxFlat}};",
        r"  \node[container, below=2cm of flat] (final) {\usebox{\BoxFinal}};",
        "",
        r"  % Arrows",
        r"  \draw[flowarrow] (grammar) -- ++(0,-1.5) -| node[pos=0.2, fill=white] {Lex} (tok_bg);",
        r"  \draw[flowarrow] (grammar) -- ++(0,-1.5) -| node[pos=0.2, fill=white] {Parse} (chars_bg);",
        r"  \draw[flowarrow] (tok_bg) -- (skeleton);",
        r"  \draw[flowarrow] (skeleton) -- ++(0,-1.5) -| (merged);",
        r"  \draw[flowarrow] (chars_bg) -- ++(0,-1.5) -| (merged);",
        r"  \draw[flowarrow] (merged) -- (flat);",
        r"  \draw[flowarrow] (flat) -- (final);",
        "",
        r"\end{tikzpicture}",
        r"\end{document}"
    ])

    with open("pipeline_full.tex", "w") as f:
        f.write("\n".join(latex))
    print("Generated pipeline_full.tex")

if __name__ == "__main__":
    main()