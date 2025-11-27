#!/usr/bin/env python3
"""
Generate a comprehensive TikZ figure of the full grammar compilation pipeline.

This generates:
1. Full input grammar (EBNF)
2. Full LALR(1) parse table (as a table)
3. Full Terminal DWA (Skeleton DWA)
4. ALL below-zero characterizations for ALL terminals
5. ALL Template DFAs for ALL terminals
6. Terminal DWA with Template DFAs substituted on edges
7. Full Flattened NWA
8. Full Final DWA

All automata include edge labels with symbols and weights.
Uses UPPERCASE convention for paper (DWA, NWA, DFA, LALR).
"""

import json
import re
import subprocess
from typing import Dict, List, Set, Tuple, Optional, Any

# Special symbol constants (from Rust code)
REDUCE_BASE = -2147483648  # i32::MIN
GOTO_BASE = 2147483646     # i32::MAX - 1


def format_weight(weight_str: str) -> str:
    """Format a weight string for display, using ALL or range notation."""
    if weight_str == "ALL" or "18446744073709551615" in weight_str:
        return "ALL"
    return weight_str


def escape_for_latex_label(s: str) -> str:
    """Escape special characters for use in TikZ node labels."""
    # Handle dollar sign (very important!)
    s = s.replace('$', '\\$')
    # Handle underscores
    s = s.replace('_', '\\_')
    # Handle percent
    s = s.replace('%', '\\%')
    # Handle ampersand
    s = s.replace('&', '\\&')
    # Handle hash
    s = s.replace('#', '\\#')
    return s


def format_symbol(symbol: Any, terminal_names: Dict[int, str] = None) -> str:
    """Format a symbol for edge labels.
    
    Symbols can be:
    - epsilon (None) -> $\\varepsilon$
    - Terminal IDs (0, 1, 2, ...)
    - State IDs for gotos
    - Special reduce/goto markers
    - neg(x) patterns in NWAs
    """
    if symbol is None:
        return "$\\varepsilon$"
    
    if isinstance(symbol, str):
        # Handle neg(x) patterns
        if symbol.startswith("neg("):
            inner = symbol[4:-1]
            try:
                inner_int = int(inner)
                if terminal_names and inner_int in terminal_names:
                    name = escape_for_latex_label(terminal_names[inner_int])
                    return f"$\\neg${name}"
                return f"$\\neg${inner}"
            except:
                return escape_for_latex_label(symbol)
        return escape_for_latex_label(symbol)
    
    # Integer symbols
    if isinstance(symbol, int):
        # Check for special markers
        if symbol <= REDUCE_BASE + 100 and symbol >= REDUCE_BASE:
            # It's a reduce action
            reduce_id = symbol - REDUCE_BASE
            return f"R{reduce_id}"
        elif symbol >= GOTO_BASE - 100 and symbol <= GOTO_BASE + 100:
            # It's a goto action
            if symbol == GOTO_BASE:
                return "GOTO"
            offset = symbol - GOTO_BASE
            return f"G{offset}"
        elif terminal_names and symbol in terminal_names:
            return escape_for_latex_label(terminal_names[symbol])
        else:
            return str(symbol)
    
    return escape_for_latex_label(str(symbol))


def parse_dwa_with_labels(dwa_str: str) -> Tuple[Set[int], List[Tuple[int, int, Any, str]], Dict[int, str]]:
    """Parse a DWA string, extracting nodes, edges with labels, and final weights.
    
    Returns:
        (nodes, edges, final_weights)
        where edges is list of (source, target, symbol, weight)
    """
    nodes = set()
    edges = []
    final_weights = {}
    current_state = None
    
    lines = dwa_str.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("State"):
            match = re.search(r"State (\d+):", line)
            if match:
                current_state = int(match.group(1))
                nodes.add(current_state)
        elif "->" in line and "final_weight" not in line:
            # Parse: symbol -> target (weight: X) or symbol -> target
            parts = line.split("->")
            label_str = parts[0].strip()
            
            # Parse symbol
            if label_str == "ε":
                symbol = None
            elif label_str.startswith("neg("):
                symbol = label_str
            else:
                try:
                    symbol = int(label_str)
                except ValueError:
                    symbol = label_str
            
            # Parse target and optional weight
            target_part = parts[1].strip()
            target_match = re.match(r"(\d+)", target_part)
            if target_match:
                target = int(target_match.group(1))
                
                # Extract weight if present
                weight_match = re.search(r"\(weight:\s*([^)]+)\)", target_part)
                weight = weight_match.group(1).strip() if weight_match else "ALL"
                
                edges.append((current_state, target, symbol, weight))
                nodes.add(target)
        elif "final_weight:" in line:
            weight = line.split(":")[1].strip()
            final_weights[current_state] = weight
            
    return nodes, edges, final_weights


def parse_dwa(dwa_str):
    """Legacy parser for backward compatibility."""
    nodes, edges_with_weights, final_weights = parse_dwa_with_labels(dwa_str)
    edges = [(e[0], e[1], e[2]) for e in edges_with_weights]
    return nodes, edges, final_weights


def run_dot_layout(nodes: Set[int], edges: List[Tuple], final_weights: Dict[int, str], 
                   large: bool = False) -> Optional[str]:
    """Generate layout using Graphviz dot."""
    dot_content = ["digraph G {"]
    dot_content.append('  rankdir=LR;')
    
    if large:
        dot_content.append('  nodesep=1.5;')
        dot_content.append('  ranksep=2.0;')
    else:
        dot_content.append('  nodesep=1.0;')
        dot_content.append('  ranksep=1.2;')
    
    for n in nodes:
        shape = "doublecircle" if n in final_weights else "circle"
        dot_content.append(f'  {n} [shape={shape}, width=0.6, height=0.6, fixedsize=true, fontsize=12];')
        
    for edge in edges:
        u, v = edge[0], edge[1]
        dot_content.append(f'  {u} -> {v};')
        
    dot_content.append("}")
    dot_str = "\n".join(dot_content)
    
    try:
        process = subprocess.Popen(
            ['dot', '-Tplain'], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        stdout, stderr = process.communicate(input=dot_str, timeout=10)
        
        if process.returncode != 0:
            print(f"Graphviz error: {stderr}")
            return None
        return stdout
    except Exception as e:
        print(f"Error running graphviz: {e}")
        return None


def parse_dot_plain(plain_output: str) -> Dict:
    """Parse Graphviz plain output to get node positions."""
    layout = {"nodes": {}, "bbox": (0, 0, 1, 1)}
    
    for line in plain_output.strip().split('\n'):
        parts = line.split()
        if not parts:
            continue
        
        if parts[0] == "graph":
            w, h = float(parts[2]), float(parts[3])
            layout["bbox"] = (0, 0, w, h)
        elif parts[0] == "node":
            n_id = int(parts[1])
            x, y = float(parts[2]), float(parts[3])
            layout["nodes"][n_id] = (x, y)
            
    return layout


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    s = s.replace('\\', '\\textbackslash{}')
    s = s.replace('_', '\\_')
    s = s.replace('{', '\\{')
    s = s.replace('}', '\\}')
    s = s.replace('&', '\\&')
    s = s.replace('%', '\\%')
    s = s.replace('#', '\\#')
    s = s.replace('$', '\\$')
    s = s.replace('~', '\\textasciitilde{}')
    s = s.replace('^', '\\textasciicircum{}')
    return s


def format_edge_label(symbol: Any, weight: str, terminal_names: Dict[int, str] = None) -> str:
    """Format the edge label with symbol and weight."""
    sym_str = format_symbol(symbol, terminal_names)
    weight_str = format_weight(weight)
    
    if weight_str == "ALL":
        return sym_str
    else:
        # Compact weight display
        weight_short = weight_str
        if len(weight_short) > 15:
            weight_short = weight_short[:12] + "..."
        return f"{sym_str}/{weight_short}"


def generate_automaton_tikz_with_labels(
    nodes: Set[int], 
    edges: List[Tuple[int, int, Any, str]], 
    final_weights: Dict[int, str], 
    name: str, 
    pos: Tuple[float, float] = (0, 0), 
    scale: float = 1.0, 
    terminal_names: Dict[int, str] = None,
    show_labels: bool = True,
    compact: bool = False
) -> str:
    """Generate TikZ code for an automaton with labeled edges."""
    
    # Get simple edges for layout
    simple_edges = [(e[0], e[1], e[2]) for e in edges]
    plain = run_dot_layout(nodes, edges, final_weights, large=not compact)
    if not plain:
        return f"% Failed to layout {name}\n"
        
    layout = parse_dot_plain(plain)
    bbox = layout["bbox"]
    cx, cy = bbox[2] / 2, bbox[3] / 2
    
    safe_name = re.sub(r'[^a-zA-Z0-9]', '', name)
    
    tikz = []
    tikz.append(f"\\begin{{scope}}[shift={{({pos[0]},{pos[1]})}}, scale={scale}]")
    if name:
        tikz.append(f"\\node[anchor=south, font=\\bfseries\\large] at (0,{bbox[3]/2 + 0.8}) {{{name}}};")
    
    # Draw nodes
    for n, (x, y) in layout["nodes"].items():
        tx, ty = x - cx, y - cy
        styles = ["state"]
        if n in final_weights:
            styles = ["accepting"]
        if n == 0:
            styles.append("initial")
        style_str = ",".join(styles)
        
        # Add final weight label if present
        if n in final_weights:
            weight = format_weight(final_weights[n])
            if weight != "ALL":
                tikz.append(f"\\node[{style_str}] (n{safe_name}{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")
                tikz.append(f"\\node[font=\\tiny, below=1pt of n{safe_name}{n}] {{{weight}}};")
            else:
                tikz.append(f"\\node[{style_str}] (n{safe_name}{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")
        else:
            tikz.append(f"\\node[{style_str}] (n{safe_name}{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")
    
    # Group edges by (source, target) to handle multiple edges
    edge_groups: Dict[Tuple[int, int], List[Tuple[Any, str]]] = {}
    for u, v, symbol, weight in edges:
        key = (u, v)
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append((symbol, weight))
    
    # Draw edges
    for (u, v), labels in edge_groups.items():
        nu = f"n{safe_name}{u}"
        nv = f"n{safe_name}{v}"
        
        # Format combined label
        if show_labels:
            label_parts = []
            for symbol, weight in labels:
                label_parts.append(format_edge_label(symbol, weight, terminal_names))
            
            # Truncate if too many labels
            if len(label_parts) > 3:
                label_text = ", ".join(label_parts[:3]) + ", ..."
            else:
                label_text = ", ".join(label_parts)
            
            label_text = label_text.replace("_", "\\_")
            
            if u == v:
                tikz.append(f"\\path[edge] ({nu}) edge[loop above] node[font=\\tiny, above] {{{label_text}}} ({nv});")
            else:
                tikz.append(f"\\path[edge] ({nu}) edge node[font=\\tiny, above, sloped] {{{label_text}}} ({nv});")
        else:
            if u == v:
                tikz.append(f"\\path[edge] ({nu}) edge[loop above] ({nv});")
            else:
                tikz.append(f"\\path[edge] ({nu}) edge ({nv});")
    
    tikz.append("\\end{scope}")
    return "\n".join(tikz)


def generate_automaton_tikz(nodes, edges, final_weights, name, pos=(0,0), scale=1.0, large=False):
    """Legacy function for backward compatibility - no edge labels."""
    plain = run_dot_layout(nodes, edges, final_weights, large=large)
    if not plain:
        return f"% Failed to layout {name}\n"
        
    layout = parse_dot_plain(plain)
    bbox = layout["bbox"]
    cx, cy = bbox[2]/2, bbox[3]/2
    
    safe_name = re.sub(r'[^a-zA-Z0-9]', '', name)
    
    tikz = []
    tikz.append(f"\\begin{{scope}}[shift={{({pos[0]},{pos[1]})}}, scale={scale}]")
    tikz.append(f"\\node[anchor=north, font=\\bfseries] at (0,{bbox[3]/2+0.8}) {{{name}}};")
    
    for n, (x, y) in layout["nodes"].items():
        tx, ty = x - cx, y - cy
        style = "state"
        if n in final_weights: style = "accepting"
        if n == 0: style += ",initial"
        tikz.append(f"\\node[{style}] (n{safe_name}{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")
    
    for u, v, label in edges:
        nu = f"n{safe_name}{u}"
        nv = f"n{safe_name}{v}"
        if u == v:
            tikz.append(f"\\path[edge] ({nu}) edge[loop above] ({nv});")
        else:
            tikz.append(f"\\path[edge] ({nu}) edge ({nv});")
    
    tikz.append("\\end{scope}")
    return "\n".join(tikz)


def parse_lalr_table(lalr_str: str) -> Dict:
    """Parse the LALR table string into a structured format."""
    states = {}
    
    # Split by StateID to get each state's content
    state_parts = re.split(r'StateID\((\d+)\):\s*Row\s*\{', lalr_str)
    
    # Skip first empty part
    for i in range(1, len(state_parts), 2):
        state_id = int(state_parts[i])
        if i + 1 < len(state_parts):
            content = state_parts[i + 1]
            # Find the content up to the next state or end
            # We need to handle nested braces
            brace_count = 1
            end_idx = 0
            for j, c in enumerate(content):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = j
                        break
            content = content[:end_idx]
        else:
            content = ""
        
        state_data = {
            "shifts": {},
            "reduces": {},
            "gotos": {},
            "default_reduce": None
        }
        
        # Parse shifts
        shift_pattern = r"TerminalID\((\d+)\):\s*Shift\(StateID\((\d+)\)\)"
        for shift_match in re.finditer(shift_pattern, content):
            term_id = int(shift_match.group(1))
            target = int(shift_match.group(2))
            state_data["shifts"][term_id] = target
        
        # Parse reduces
        reduce_pattern = r"TerminalID\((\d+)\):\s*Reduce\s*\{\s*nonterminal_id:\s*NonTerminalID\((\d+)\),\s*len:\s*(\d+)"
        for red_match in re.finditer(reduce_pattern, content):
            term_id = int(red_match.group(1))
            nt_id = int(red_match.group(2))
            prod_len = int(red_match.group(3))
            state_data["reduces"][term_id] = (nt_id, prod_len)
        
        # Parse default reduce
        def_reduce_pattern = r"default_reduce:\s*Some\(Reduce\s*\{\s*nonterminal_id:\s*NonTerminalID\((\d+)\),\s*len:\s*(\d+)"
        def_match = re.search(def_reduce_pattern, content)
        if def_match:
            state_data["default_reduce"] = (int(def_match.group(1)), int(def_match.group(2)))
        
        # Parse gotos - extract them from the gotos section
        gotos_match = re.search(r"gotos:\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", content)
        if gotos_match:
            gotos_content = gotos_match.group(1)
            goto_pattern = r"NonTerminalID\((\d+)\):\s*Goto\s*\{\s*state_id:\s*(?:Some\(StateID\((\d+)\)\)|None)"
            for goto_match in re.finditer(goto_pattern, gotos_content):
                nt_id = int(goto_match.group(1))
                target = int(goto_match.group(2)) if goto_match.group(2) else None
                state_data["gotos"][nt_id] = target
        
        states[state_id] = state_data
    
    return states


def generate_lalr_table_tikz(lalr_data: Dict, terminal_names: Dict[int, str], 
                             nonterminal_names: Dict[int, str], pos: Tuple[float, float]) -> str:
    """Generate a TikZ table representation of the LALR parse table."""
    
    # Get all terminal and nonterminal IDs
    all_terminals = set()
    all_nonterminals = set()
    for state_data in lalr_data.values():
        all_terminals.update(state_data["shifts"].keys())
        all_terminals.update(state_data["reduces"].keys())
        all_nonterminals.update(state_data["gotos"].keys())
    
    terminals = sorted(all_terminals)
    nonterminals = sorted(all_nonterminals)
    states = sorted(lalr_data.keys())
    
    tikz = []
    tikz.append(f"\\begin{{scope}}[shift={{({pos[0]},{pos[1]})}}]")
    tikz.append("\\node[anchor=north, font=\\bfseries\\large] at (4, 1) {LALR(1) Parse Table};")
    
    # Build table
    col_spec = "c|" + "c" * len(terminals) + "|" + "c" * len(nonterminals)
    
    tikz.append("\\node[anchor=north] at (4, 0) {")
    tikz.append("\\begin{tabular}{" + col_spec + "}")
    tikz.append("\\hline")
    
    # Header row - escape special characters
    header = ["State"]
    for t in terminals:
        name = terminal_names.get(t, f"t{t}")
        name = name.replace('$', '\\$')  # Escape dollar sign
        header.append(name)
    for nt in nonterminals:
        header.append(nonterminal_names.get(nt, f"N{nt}"))
    tikz.append(" & ".join(header) + " \\\\")
    tikz.append("\\hline")
    
    # State rows
    for state in states:
        row = [str(state)]
        state_data = lalr_data[state]
        
        for t in terminals:
            if t in state_data["shifts"]:
                row.append(f"s{state_data['shifts'][t]}")
            elif t in state_data["reduces"]:
                nt, length = state_data["reduces"][t]
                row.append(f"r{nt}")
            elif state_data["default_reduce"]:
                nt, length = state_data["default_reduce"]
                row.append(f"r{nt}")
            else:
                row.append("")
        
        for nt in nonterminals:
            if nt in state_data["gotos"] and state_data["gotos"][nt] is not None:
                row.append(str(state_data["gotos"][nt]))
            else:
                row.append("")
        
        tikz.append(" & ".join(row) + " \\\\")
    
    tikz.append("\\hline")
    tikz.append("\\end{tabular}")
    tikz.append("};")
    tikz.append("\\end{scope}")
    
    return "\n".join(tikz)


def generate_characterization_tikz(char_data: Dict[str, str], pos: Tuple[float, float]) -> str:
    """Generate TikZ representation of below-zero characterizations."""
    
    tikz = []
    tikz.append(f"\\begin{{scope}}[shift={{({pos[0]},{pos[1]})}}]")
    tikz.append("\\node[anchor=north, font=\\bfseries\\large] at (0, 0.5) {Below-Zero Characterizations};")
    
    y_offset = -0.5
    for term_id, char_str in sorted(char_data.items()):
        # Parse key info from characterization
        term_match = re.search(r"terminal:\s*TerminalID\((\d+)\)", char_str)
        term_num = term_match.group(1) if term_match else "?"
        
        # Extract initial shifts
        shift_match = re.search(r"initial_shifts:\s*\{([^}]*)\}", char_str)
        shifts = shift_match.group(1) if shift_match else ""
        
        # Extract initial reduces
        reduce_match = re.search(r"initial_reduces:\s*\{([^}]*)\}", char_str)
        reduces = reduce_match.group(1) if reduce_match else ""
        
        # Format compactly
        shift_pairs = re.findall(r"\(StateID\((\d+)\),\s*StateID\((\d+)\)\)", shifts)
        reduce_tuples = re.findall(r"\(StateID\((\d+)\),\s*\d+,\s*NonTerminalID\((\d+)\)\)", reduces)
        
        shift_str = ", ".join([f"({s},{t})" for s, t in shift_pairs]) if shift_pairs else "∅"
        reduce_str = ", ".join([f"({s},N{n})" for s, n in reduce_tuples]) if reduce_tuples else "∅"
        
        tikz.append(f"\\node[anchor=west, font=\\small] at (-8, {y_offset}) {{")
        tikz.append(f"  \\textbf{{T{term_num}:}} shifts: {{{shift_str}}}, reduces: {{{reduce_str}}}")
        tikz.append("};")
        y_offset -= 0.7
    
    tikz.append("\\end{scope}")
    return "\n".join(tikz)


def main():
    with open("pipeline_artifacts.json", "r") as f:
        data = json.load(f)
    
    # Extract terminal names from grammar
    grammar = data["grammar_ebnf"]
    # For this simple grammar, manually define terminal names based on the grammar
    terminal_names = {
        0: "$",    # End of input
        1: "a",
        2: "b", 
        3: "c"
    }
    
    nonterminal_names = {
        0: "S",
        1: "A",
        2: "B", 
        3: "C",
        4: "C'"
    }
    
    # Parse LALR table
    lalr_data = parse_lalr_table(data["lalr_table"])
    
    # Parse all automata (with labels)
    templates = data["template_dfas_all"]
    char_data = data["characterizations_all"]
    skel_nodes, skel_edges, skel_finals = parse_dwa_with_labels(data["skeleton_dwa"])
    flat_nodes, flat_edges, flat_finals = parse_dwa_with_labels(data["flattened_nwa"])
    final_nodes, final_edges, final_finals = parse_dwa_with_labels(data["final_dwa"])
    
    tex = []
    
    # Document preamble
    tex.append(r"""\documentclass[tikz,border=30pt]{standalone}
\usepackage{lmodern}
\usepackage{tikz}
\usepackage{amsmath}
\usepackage{amssymb}
\usetikzlibrary{automata,positioning,arrows.meta,shapes,shadows,fit,calc}

\definecolor{primary}{RGB}{41,128,185}
\definecolor{accent}{RGB}{39,174,96}
\definecolor{dark}{RGB}{52,73,94}
\definecolor{grammar}{RGB}{155,89,182}
\definecolor{lalr}{RGB}{230,126,34}

\begin{document}
\begin{tikzpicture}[
    >=Stealth,
    font=\sffamily,
    state/.style={
        circle,
        draw=primary,
        very thick,
        minimum size=10mm,
        fill=white,
        text=dark,
        font=\small
    },
    accepting/.style={
        state,
        double,
        double distance=1.5pt,
        draw=accent,
        fill=accent!8
    },
    initial/.style={
        fill=primary!15
    },
    edge/.style={
        ->,
        draw=dark!70,
        thick
    },
    stagebox/.style={
        rectangle,
        draw=dark,
        very thick,
        rounded corners=3pt,
        fill=white,
        drop shadow,
        minimum width=6cm,
        minimum height=2.5cm,
        align=center
    },
    grammarbox/.style={
        rectangle,
        draw=grammar,
        very thick,
        rounded corners=3pt,
        fill=grammar!5,
        align=left,
        font=\ttfamily\small
    },
    flowarrow/.style={
        ->,
        draw=dark,
        line width=2pt,
        dashed
    }
]

""")
    
    # ===================
    # STAGE 0: Input Grammar
    # ===================
    tex.append(r"% ========== STAGE 0: Input Grammar ==========")
    tex.append(r"\node[font=\Huge\bfseries, text=dark] (title) at (0, 2) {Grammar Compilation Pipeline};")
    tex.append("")
    
    # Grammar box
    tex.append(r"\node[grammarbox, below=1cm of title] (grammar) {")
    tex.append(r"  \begin{tabular}{l}")
    tex.append(r"  \textbf{Input Grammar (EBNF):}\\[3pt]")
    for line in grammar.strip().split('\n'):
        line = line.strip()
        if line:
            # Escape special chars and format
            line = line.replace('$', '\\$')  # Escape dollar sign FIRST
            line = line.replace('|', '$|$').replace('"', "``").replace("'", "`")
            tex.append(f"  {line}\\\\")
    tex.append(r"  \end{tabular}")
    tex.append(r"};")
    tex.append("")
    
    # ===================
    # STAGE 1: LALR Parse Table
    # ===================
    tex.append(r"% ========== STAGE 1: LALR(1) Parse Table ==========")
    y_lalr = -6
    tex.append(generate_lalr_table_tikz(lalr_data, terminal_names, nonterminal_names, (0, y_lalr)))
    tex.append("")
    
    # ===================
    # STAGE 2: Below-Zero Characterizations
    # ===================
    tex.append(r"% ========== STAGE 2: Below-Zero Characterizations ==========")
    y_char = y_lalr - 8
    tex.append(generate_characterization_tikz(char_data, (0, y_char)))
    tex.append("")
    
    # ===================
    # STAGE 3: Template DFAs
    # ===================
    tex.append(r"% ========== STAGE 3: Template DFAs ==========")
    y_templates = y_char - 8
    tex.append(f"\\node[font=\\Large\\bfseries, text=dark] at (0, {y_templates + 1}) {{Template DFAs (One Per Terminal)}};")
    
    template_items = list(templates.items())
    num_temps = len(template_items)
    x_spacing = 8
    
    for i, (tid, dwa_str) in enumerate(template_items):
        tnodes, tedges, tfinals = parse_dwa_with_labels(dwa_str)
        tid_num = re.search(r'\d+', tid).group()
        term_name = terminal_names.get(int(tid_num), f"T{tid_num}")
        # Escape $ for LaTeX title
        term_name_escaped = term_name.replace('$', '\\$')
        x_pos = (i - (num_temps - 1) / 2) * x_spacing
        tex.append(generate_automaton_tikz_with_labels(
            tnodes, tedges, tfinals, 
            f"Template DFA for {term_name_escaped}",
            (x_pos, y_templates - 5), 
            scale=0.7, 
            terminal_names=terminal_names,
            show_labels=True,
            compact=True
        ))
    tex.append("")
    
    # ===================
    # STAGE 4: Terminal DWA (Skeleton)
    # ===================
    tex.append(r"% ========== STAGE 4: Terminal DWA (Skeleton) ==========")
    y_skel = y_templates - 16
    tex.append(generate_automaton_tikz_with_labels(
        skel_nodes, skel_edges, skel_finals,
        "Terminal DWA (Skeleton)",
        (0, y_skel),
        scale=0.9,
        terminal_names=terminal_names,
        show_labels=True
    ))
    tex.append("")
    
    # ===================  
    # STAGE 5: Terminal DWA with Template DFAs Substituted
    # ===================
    tex.append(r"% ========== STAGE 5: Terminal DWA with Templates Substituted ==========")
    y_subst = y_skel - 10
    tex.append(f"\\node[font=\\Large\\bfseries, text=dark] at (0, {y_subst + 1}) {{Terminal DWA with Template DFAs on Edges}};")
    tex.append(f"\\node[font=\\small, text=dark!70] at (0, {y_subst + 0.3}) {{(Each edge to a terminal state is replaced by the corresponding Template DFA)}};")
    
    # For this, we show the skeleton but indicate template substitution
    tex.append(generate_automaton_tikz_with_labels(
        skel_nodes, skel_edges, skel_finals,
        "",
        (0, y_subst - 4),
        scale=0.9,
        terminal_names=terminal_names,
        show_labels=True
    ))
    tex.append("")
    
    # ===================
    # STAGE 6: Flattened NWA
    # ===================
    tex.append(r"% ========== STAGE 6: Flattened NWA ==========")
    y_flat = y_subst - 20
    tex.append(generate_automaton_tikz_with_labels(
        flat_nodes, flat_edges, flat_finals,
        "Flattened NWA",
        (0, y_flat),
        scale=0.55,
        terminal_names=terminal_names,
        show_labels=True,
        compact=False
    ))
    tex.append("")
    
    # ===================
    # STAGE 7: Final DWA
    # ===================
    tex.append(r"% ========== STAGE 7: Final DWA ==========")
    y_final = y_flat - 22
    tex.append(generate_automaton_tikz_with_labels(
        final_nodes, final_edges, final_finals,
        "Final DWA",
        (0, y_final),
        scale=0.55,
        terminal_names=terminal_names,
        show_labels=True,
        compact=False
    ))
    tex.append("")
    
    # ===================
    # Flow Arrows
    # ===================
    tex.append(r"% ========== Flow Arrows ==========")
    tex.append(r"\draw[flowarrow] (grammar.south) -- +(0, -0.5);")
    tex.append(f"\\draw[flowarrow] (0, {y_lalr - 3.5}) -- +(0, -1);")
    tex.append(f"\\draw[flowarrow] (0, {y_char - 3.5}) -- +(0, -1);")
    tex.append(f"\\draw[flowarrow] (0, {y_templates - 11}) -- +(0, -1);")
    tex.append(f"\\draw[flowarrow] (0, {y_skel - 5}) -- +(0, -1);")
    tex.append(f"\\draw[flowarrow] (0, {y_subst - 9}) -- +(0, -1);")
    tex.append(f"\\draw[flowarrow] (0, {y_flat - 10}) -- +(0, -1);")
    
    tex.append(r"""
\end{tikzpicture}
\end{document}
""")
    
    output_path = "gcg-paper/paper/figures/pipeline_full.tex"
    with open(output_path, "w") as f:
        f.write("\n".join(tex))
        
    print(f"Generated {output_path}")
    print(f"  - {len(lalr_data)} LALR states")
    print(f"  - {len(char_data)} below-zero characterizations")
    print(f"  - {len(templates)} template DFAs")
    print(f"  - Skeleton DWA: {len(skel_nodes)} states, {len(skel_edges)} edges")
    print(f"  - Flattened NWA: {len(flat_nodes)} states, {len(flat_edges)} edges")
    print(f"  - Final DWA: {len(final_nodes)} states, {len(final_edges)} edges")


if __name__ == "__main__":
    main()
