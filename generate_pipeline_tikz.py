#!/usr/bin/env python3
"""
Generate a comprehensive TikZ figure of the full grammar compilation pipeline.
Architecture: Uses the 'Savebox' pattern for robust layout.
"""

import json
import re
import subprocess
from typing import Dict, List, Set, Tuple, Optional, Any

# ==============================================================================
# 1. PARSING & HELPER LOGIC (Unchanged logic, cleaned up structure)
# ==============================================================================

REDUCE_BASE = -2147483648
GOTO_BASE = 2147483646

def escape_tex(s: str) -> str:
    return s.replace('$', '\\$').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&').replace('#', '\\#')

def format_symbol(symbol: Any, terminal_names: Dict[int, str] = None, as_state_id: bool = False) -> str:
    if symbol is None: return "$\\varepsilon$"
    if isinstance(symbol, str):
        if symbol.startswith("neg("): return f"$\\neg${symbol[4:-1]}"
        return escape_tex(symbol) if '\\' not in symbol else symbol
    if isinstance(symbol, int):
        if symbol <= REDUCE_BASE + 100 and symbol >= REDUCE_BASE: return f"r{symbol - REDUCE_BASE}"
        if symbol >= GOTO_BASE - 100: return "$\\gamma$"
        if as_state_id: return str(symbol)
        if terminal_names and symbol in terminal_names: return escape_tex(terminal_names[symbol])
        return str(symbol)
    return escape_tex(str(symbol))

def format_weight(w: str) -> str:
    if w == "ALL" or "18446744073709551615" in w: return "ALL"
    return w

def run_dot_layout(nodes: Set[int], edges: List[Tuple], final_states: Set[int], ranksep=1.2, nodesep=0.8) -> Dict:
    """Run Graphviz and parse plain output into a layout dict."""
    dot_content = [
        "digraph G {", 
        f"  rankdir=LR; ranksep={ranksep}; nodesep={nodesep};",
        "  margin=0;"
    ]
    for n in sorted(nodes):
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
            elif parts[0] == "node": layout["nodes"][int(parts[1])] = (float(parts[2]), float(parts[3]))
        return layout
    except (subprocess.CalledProcessError, FileNotFoundError):
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
            sym = None if lbl == "ε" else (int(lbl) if lbl.isdigit() or lbl.lstrip('-').isdigit() else lbl)
            tgt_part = parts[1].strip()
            tgt = int(re.match(r"(\d+)", tgt_part).group(1))
            w = re.search(r"\(weight:\s*([^)]+)\)", tgt_part)
            edges.append((curr, tgt, sym, w.group(1).strip() if w else "ALL"))
            nodes.add(tgt)
        elif "final_weight:" in line:
            finals.add(curr)
    return nodes, edges, finals

def parse_tokenizer(data: str) -> Tuple[Set[int], List[Tuple], Set[int]]:
    nodes, edges, finals = set(), [], set()
    idx = 0
    # Simple regex based parser for the specific debug format
    state_blocks = re.split(r"DFAState", data)[1:]
    for block in state_blocks:
        nodes.add(idx)
        trans = re.search(r"transitions: \{([^}]*)\}", block)
        if trans:
            for t in trans.group(1).split(','):
                if ':' in t:
                    k, v = map(int, t.split(':'))
                    lbl = "\\$" if k==36 else (chr(k) if 32<=k<=126 else f"0x{k:x}")
                    if lbl in ['_','%','&','#']: lbl = '\\'+lbl
                    edges.append((idx, v, lbl, "ALL"))
                    nodes.add(v)
        if "finalizers: Bitset { words: []" not in block:
             # Basic heuristic for non-empty finalizers
             finals.add(idx)
        idx += 1
    return nodes, edges, finals

# ==============================================================================
# 2. TIKZ COMPONENT GENERATORS (The "Micro" Scale)
# ==============================================================================

def make_automaton_tikz(name: str, nodes, edges, finals, scale=1.0, 
                        term_map=None, as_id=False, node_size="8mm", font_sz="\\small") -> str:
    layout = run_dot_layout(nodes, edges, finals)
    if not layout: return f"% Failed layout for {name}"
    
    # Calculate centering
    bb = layout["bbox"]
    cx, cy = bb[2]/2, bb[3]/2
    
    lines = [f"\\begin{{tikzpicture}}[scale={scale}, every node/.style={{transform shape}}]"]
    if name:
        lines.append(f"\\node[anchor=south, font=\\bfseries\\Large] at (0, {(bb[3]/2)*2.54 + 0.5}) {{{name}}};")

    # Draw Nodes
    for n, (x, y) in layout["nodes"].items():
        # Convert inches to cm, center at 0,0
        tx, ty = (x - cx)*2.54, (y - cy)*2.54
        style = "accepting" if n in finals else "state"
        if n == 0: style += ", initial"
        lines.append(f"\\node[{style}, minimum size={node_size}] (n{n}) at ({tx:.2f},{ty:.2f}) {{{font_sz} {n}}};")

    # Draw Edges (grouped)
    grouped = {}
    for u, v, sym, w in edges:
        grouped.setdefault((u,v), []).append((sym, w))
    
    for (u, v), labels in grouped.items():
        if u not in layout["nodes"] or v not in layout["nodes"]: continue
        
        lbl_strs = []
        for s, w in labels[:3]:
            txt = format_symbol(s, term_map, as_id)
            if format_weight(w) != "ALL": txt += f"/{format_weight(w)[:5]}"
            lbl_strs.append(txt)
        label = ", ".join(lbl_strs) + (",..." if len(labels)>3 else "")
        
        if u == v:
            lines.append(f"\\path[edge] (n{u}) edge[loop above] node[above, font=\\tiny] {{{label}}} (n{v});")
        else:
            lines.append(f"\\path[edge] (n{u}) edge node[above, sloped, font=\\tiny] {{{label}}} (n{v});")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)

def make_lalr_table(lalr_data: str, term_map, nonterm_map) -> str:
    # Quick parsing of the debug string to get dimensions
    # Note: For production, reuse the detailed parser. Here we do a simplified tabular.
    # We will just generate a static stylistic representation for the visual pipeline.
    
    lines = [
        "\\begin{tikzpicture}",
        "\\node[anchor=south, font=\\bfseries\\large] at (0, 0.2) {LALR(1) Table};",
        "\\node[inner sep=0pt] at (0,0) {",
        "\\scalebox{0.75}{",
        "\\begin{tabular}{c|cccc|ccccc}",
        "\\hline",
        "St & \\textbf{\\$} & \\textbf{a} & \\textbf{b} & \\textbf{c} & \\textit{S} & \\textit{A} & \\textit{B} & \\textit{C} & \\textit{C'} \\\\",
        "\\hline",
        "0 & & s1 & & & & 2 & 3 & & \\\\",
        "1 & & & s4 & & & & & 5 & \\\\",
        "2 & r0 & r0 & r0 & r0 & & & & & \\\\",
        "3 & s6 & & & & & & & & \\\\",
        "4 & & & & s7 & & & & & 8 \\\\",
        "\\vdots & & & \\dots & & & & \\dots & & \\\\",
        "\\hline",
        "\\end{tabular}",
        "}", # end scalebox
        "};",
        "\\end{tikzpicture}"
    ]
    return "\n".join(lines)

def make_merged_dwa(skel_nodes, skel_edges, skel_finals, term_map) -> str:
    layout = run_dot_layout(skel_nodes, skel_edges, skel_finals, ranksep=2.0)
    if not layout: return ""
    bb = layout["bbox"]
    cx, cy = bb[2]/2, bb[3]/2
    scale = 0.9

    lines = [
        f"\\begin{{tikzpicture}}[scale={scale}, every node/.style={{transform shape}}]",
        "\\node[font=\\bfseries\\large, align=center] at (0, " + f"{(bb[3]/2)*2.54 + 1.5}" + r") {Terminal DWA (Merged)\\ \footnotesize \itshape (Edges show Template DFAs)};"
    ]

    # Draw Nodes
    for n, (x, y) in layout["nodes"].items():
        tx, ty = (x - cx)*2.54, (y - cy)*2.54
        style = "accepting" if n in skel_finals else "state"
        if n == 0: style += ", initial"
        lines.append(f"\\node[{style}, minimum size=9mm] (n{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")

    # Draw Edges with Saveboxes
    for u, v, sym, w in skel_edges:
        if u not in layout["nodes"] or v not in layout["nodes"]: continue
        
        # Calculate midpoint for the edge label
        ux, uy = layout["nodes"][u]
        vx, vy = layout["nodes"][v]
        # Convert to tikz coords
        tux, tuy = (ux-cx)*2.54, (uy-cy)*2.54
        tvx, tvy = (vx-cx)*2.54, (vy-cy)*2.54
        
        # Draw the line
        lines.append(f"\\path[edge, very thick] (n{u}) edge (n{v});")
        
        # Calculate label position (midpoint)
        mx, my = (tux+tvx)/2, (tuy+tvy)/2
        
        # Add a slight offset perpendicular to line
        dx, dy = tvx-tux, tvy-tuy
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            nx, ny = -dy/length, dx/length
            mx += nx * 0.8; my += ny * 0.8

        if isinstance(sym, int) and 0 <= sym <= 3: # It's a terminal ID
            # Use the Savebox!
            lines.append(f"\\node[inner sep=0pt, scale=0.8] at ({mx:.2f},{my:.2f}) {{\\usebox{{\\boxT{sym}}}}};")
        else:
            txt = format_symbol(sym, term_map, True)
            lines.append(f"\\node[fill=white, font=\\small, inner sep=1pt] at ({mx:.2f},{my:.2f}) {{{txt}}};")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)

def make_grammar_box(grammar_text):
    return r"""
\begin{minipage}{4cm}
\small \ttfamily
\textbf{Input Grammar:}\\[3pt]
S ::= a "\$";\\
A ::= 'a' b;\\
B ::= 'b' c;\\
C ::= 'c' a $|$ 'c';
\end{minipage}
"""

def make_vocab_box():
    return r"""
\begin{minipage}{3cm}
\centering
\textbf{LLM Vocabulary}\\[2pt]
\footnotesize (50,257 tokens)\\
\tiny e.g., GPT-2/3
\end{minipage}
"""

# ==============================================================================
# 3. MAIN GENERATION ROUTINE
# ==============================================================================

def main():
    try:
        with open("pipeline_artifacts.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: pipeline_artifacts.json not found.")
        return

    term_map = {0: "$", 1: "a", 2: "b", 3: "c"}
    
    # --- 1. Generate Content Strings ---
    
    # Tokenizer
    tok_n, tok_e, tok_f = parse_tokenizer(data["tokenizer_dfa"])
    code_tokenizer = make_automaton_tikz("Tokenizer DFA", tok_n, tok_e, tok_f, scale=0.7, term_map=term_map)
    
    # Skeleton
    skel_n, skel_e, skel_f = parse_dwa(data["skeleton_dwa"])
    code_skeleton = make_automaton_tikz("Terminal DWA (Skeleton)", skel_n, skel_e, skel_f, scale=0.7, term_map=term_map)
    
    # LALR
    code_lalr = make_lalr_table(data["lalr_table"], term_map, {})
    
    # Templates (0..3)
    templates = {}
    for tid_str, dwa_str in data["template_dfas_all"].items():
        # Clean up key "TerminalID(0)" -> 0
        tid = int(re.search(r'\d+', tid_str).group())
        tn, te, tf = parse_dwa(dwa_str)
        # Mini version for edges
        templates[tid] = make_automaton_tikz("", tn, te, tf, scale=0.25, node_size="3mm", font_sz="\\tiny")
        # Full version for display row
        templates[tid+100] = make_automaton_tikz(f"T({term_map.get(tid,'?')})", tn, te, tf, scale=0.5, node_size="6mm", font_sz="\\scriptsize")

    # Merged
    code_merged = make_merged_dwa(skel_n, skel_e, skel_f, term_map)
    
    # Flattened
    flat_n, flat_e, flat_f = parse_dwa(data["flattened_nwa"])
    code_flat = make_automaton_tikz("Flattened NWA", flat_n, flat_e, flat_f, scale=0.45, as_id=True)
    
    # Final
    final_n, final_e, final_f = parse_dwa(data["final_dwa"])
    code_final = make_automaton_tikz("Final DWA", final_n, final_e, final_f, scale=0.45, as_id=True)

    # --- 2. Build LaTeX File ---

    latex = [
        r"\documentclass[tikz,border=20pt]{standalone}",
        r"\usepackage{lmodern, amsmath, amssymb}",
        r"\usetikzlibrary{automata, positioning, arrows.meta, shapes, shadows, fit, calc, backgrounds}",
        "",
        r"% --- COLORS ---",
        r"\definecolor{primary}{RGB}{41,128,185}",
        r"\definecolor{accent}{RGB}{39,174,96}",
        r"\definecolor{dark}{RGB}{52,73,94}",
        r"\definecolor{grammar}{RGB}{155,89,182}",
        r"\definecolor{stagecolor}{RGB}{236,240,241}",
        r"\definecolor{vocabcolor}{RGB}{241,196,15}",
        "",
        r"% --- STYLES ---",
        r"\tikzset{",
        r"  container/.style={rectangle, draw=dark!30, fill=white, rounded corners, drop shadow, inner sep=5pt},",
        r"  stagearea/.style={draw=dark!20, fill=stagecolor, rounded corners=15pt, inner sep=10pt},",
        r"  grammarbox/.style={rectangle, draw=grammar, very thick, rounded corners, fill=grammar!5},",
        r"  vocabbox/.style={rectangle, draw=vocabcolor, thick, rounded corners, fill=vocabcolor!10},",
        r"  state/.style={circle, draw=primary, thick, minimum size=6mm, fill=white, text=dark, font=\scriptsize},",
        r"  accepting/.style={state, double, double distance=1.5pt, draw=accent, fill=accent!5},",
        r"  initial/.style={fill=primary!15},",
        r"  edge/.style={->, draw=dark!70, semithick, >=Stealth},",
        r"  flowarrow/.style={->, draw=dark!60, line width=1.5pt, dashed, >=Stealth, rounded corners},",
        r"}",
        r"\begin{document}",
        "",
        r"% --- SAVEBOX DEFINITIONS (The 'Puzzle Pieces') ---"
    ]

    def add_savebox(name, content):
        latex.append(f"\\newsavebox{{\\{name}}}")
        latex.append(f"\\sbox{{\\{name}}}{{")
        latex.append(content)
        latex.append(f"}}")

    # Order matters slightly for nested boxes (Templates used in Merged)
    
    # 1. Mini Templates (for embedding)
    for tid in sorted(t for t in templates if t < 100):
        add_savebox(f"boxT{tid}", templates[tid])
        
    # 2. Components
    add_savebox("BoxGrammar", make_grammar_box(data["grammar_ebnf"]))
    add_savebox("BoxVocab", make_vocab_box())
    add_savebox("BoxTokenizer", code_tokenizer)
    add_savebox("BoxLALR", code_lalr)
    add_savebox("BoxSkeleton", code_skeleton)
    add_savebox("BoxMerged", code_merged)
    add_savebox("BoxFlat", code_flat)
    add_savebox("BoxFinal", code_final)
    
    # 3. Display Templates (row)
    # Combine them into one horizontal box for layout simplicity
    tpl_row = ["\\begin{tikzpicture}"]
    tpl_row.append(r"\node[font=\bfseries\large] at (0,1.5) {Template DFAs};")
    x = 0
    for tid in sorted(t for t in templates if t >= 100):
        tpl_row.append(f"\\node[inner sep=0pt] at ({x},0) {{{templates[tid]}}};")
        x += 5 # spacing
    tpl_row.append("\\end{tikzpicture}")
    add_savebox("BoxTemplates", "\n".join(tpl_row))

    # --- 3. MAIN LAYOUT (The 'Macro' Scale) ---
    
    latex.extend([
        "",
        r"% --- MAIN LAYOUT SKELETON ---",
        r"\begin{tikzpicture}[node distance=2cm and 2cm]",
        "",
        r"  % 1. TOP: Grammar",
        r"  \node[grammarbox] (grammar) {\usebox{\BoxGrammar}};",
        "",
        r"  % 2. SPLIT: Tokenizer (Left) and LALR (Right)",
        r"  \node[container, below left=2cm and 1cm of grammar] (tokenizer) {\usebox{\BoxTokenizer}};",
        r"  \node[container, below right=2cm and 1cm of grammar] (lalr) {\usebox{\BoxLALR}};",
        r"  % Background grouping",
        r"  \begin{scope}[on background layer]",
        r"    \node[stagearea, fit=(tokenizer)] {};",
        r"    \node[stagearea, fit=(lalr)] {};",
        r"  \end{scope}",
        "",
        r"  % 3. LEFT COLUMN: Skeleton",
        r"  \node[container, below=2.5cm of tokenizer] (skeleton) {\usebox{\BoxSkeleton}};",
        r"  \node[vocabbox, left=1cm of skeleton] (vocab) {\usebox{\BoxVocab}};",
        r"  \begin{scope}[on background layer]",
        r"    \node[stagearea, fit=(skeleton)] (skel_bg) {};",
        r"  \end{scope}",
        "",
        r"  % 4. RIGHT COLUMN: Templates (Approx below LALR)",
        r"  % We align it with Skeleton vertically for symmetry",
        r"  \node[below=2.5cm of lalr] (templates) {\usebox{\BoxTemplates}};",
        "",
        r"  % 5. CENTER: Merged",
        r"  % Calculate a center point below the two columns",
        r"  \path (skeleton.south) -- (templates.south) coordinate[midway] (center_mid);",
        r"  \node[container, below=2cm of center_mid] (merged) {\usebox{\BoxMerged}};",
        "",
        r"  % 6. CENTER: Flattened",
        r"  \node[container, below=2cm of merged] (flat) {\usebox{\BoxFlat}};",
        "",
        r"  % 7. BOTTOM: Final",
        r"  \node[container, below=2cm of flat] (final) {\usebox{\BoxFinal}};",
        "",
        r"  % --- ARROWS & FLOW ---",
        r"  \coordinate (split) at ($(grammar.south) + (0,-1)$);",
        r"  \draw[flowarrow] (grammar) -- (split);",
        r"  \draw[flowarrow] (split) -| node[pos=0.7, fill=white, font=\footnotesize] {Lexing} (tokenizer);",
        r"  \draw[flowarrow] (split) -| node[pos=0.7, fill=white, font=\footnotesize] {Parsing} (lalr);",
        "",
        r"  \draw[flowarrow] (tokenizer) -- (skeleton);",
        r"  \draw[flowarrow] (vocab) |- (skel_bg);",
        r"  \draw[flowarrow] (lalr) -- (templates);",
        "",
        r"  \draw[flowarrow] (skeleton) -- ++(0,-1.5) -| (merged);",
        r"  \draw[flowarrow] (templates) -- ++(0,-1.5) -| (merged);",
        "",
        r"  \draw[flowarrow] (merged) -- node[right, font=\footnotesize] {Flatten} (flat);",
        r"  \draw[flowarrow] (flat) -- node[right, font=\footnotesize] {Determinize} (final);",
        "",
        r"\end{tikzpicture}",
        r"\end{document}"
    ])

    with open("pipeline_full.tex", "w") as f:
        f.write("\n".join(latex))
    
    print("Generated pipeline_full.tex using Savebox Architecture.")

if __name__ == "__main__":
    main()
