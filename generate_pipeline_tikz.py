#!/usr/bin/env python3
"""
Generate a comprehensive TikZ figure of the full grammar compilation pipeline.

Layout (from user request):
- Top: Input Grammar
- Split: LEFT = Tokenizer DFA, RIGHT = LALR Parse Table
- LEFT below Tokenizer: Terminal DWA (Skeleton)
- RIGHT below Parse Table: Below-zero Characterizations + Template DFAs
- CENTER (merge): Terminal DWA with Template DFAs substituted on edges (with MINI DFAs!)
- CENTER: Flattened NWA (push transitions visible)
- CENTER: NWA with push transitions resolved
- CENTER: Final DWA

All automata include edge labels with symbols and weights.
Uses UPPERCASE convention for paper (DWA, NWA, DFA, LALR).
"""

import json
import re
import subprocess
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

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
    s = s.replace('$', '\\$')
    s = s.replace('_', '\\_')
    s = s.replace('%', '\\%')
    s = s.replace('&', '\\&')
    s = s.replace('#', '\\#')
    return s


def format_symbol(symbol: Any, terminal_names: Dict[int, str] = None, as_state_id: bool = False) -> str:
    """Format a symbol for edge labels.
    
    If as_state_id=True, always format as state ID (for template DFAs).
    Otherwise, may map to terminal names.
    """
    if symbol is None:
        return "$\\varepsilon$"
    
    if isinstance(symbol, str):
        if symbol.startswith("neg("):
            inner = symbol[4:-1]
            return f"$\\neg${inner}"
        # Don't double-escape strings that already have escapes
        if '\\' in symbol:
            return symbol
        return escape_for_latex_label(symbol)
    
    if isinstance(symbol, int):
        # Check for reduce actions (large negative numbers)
        if symbol <= REDUCE_BASE + 100 and symbol >= REDUCE_BASE:
            reduce_id = symbol - REDUCE_BASE
            return f"R{reduce_id}"
        # Check for goto actions (large positive numbers)
        elif symbol >= GOTO_BASE - 100:
            return "GOTO"
        # Otherwise it's a state ID (or terminal if not as_state_id)
        elif as_state_id:
            return str(symbol)
        elif terminal_names and symbol in terminal_names:
            return escape_for_latex_label(terminal_names[symbol])
        else:
            return str(symbol)
    
    return escape_for_latex_label(str(symbol))


def parse_dwa_with_labels(dwa_str: str) -> Tuple[Set[int], List[Tuple[int, int, Any, str]], Dict[int, str]]:
    """Parse a DWA string, extracting nodes, edges with labels, and final weights."""
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
            parts = line.split("->")
            label_str = parts[0].strip()
            
            if label_str == "ε":
                symbol = None
            elif label_str.startswith("neg("):
                symbol = label_str
            else:
                try:
                    symbol = int(label_str)
                except ValueError:
                    symbol = label_str
            
            target_part = parts[1].strip()
            target_match = re.match(r"(\d+)", target_part)
            if target_match:
                target = int(target_match.group(1))
                weight_match = re.search(r"\(weight:\s*([^)]+)\)", target_part)
                weight = weight_match.group(1).strip() if weight_match else "ALL"
                edges.append((current_state, target, symbol, weight))
                nodes.add(target)
        elif "final_weight:" in line:
            weight = line.split(":")[1].strip()
            final_weights[current_state] = weight
            
    return nodes, edges, final_weights


def parse_tokenizer_dfa(dfa_str: str) -> Tuple[Set[int], List[Tuple[int, int, str, str]], Dict[int, List[int]]]:
    """Parse tokenizer DFA debug output.
    
    Returns (nodes, edges, finalizers) where edges have (source, target, char_label, "ALL")
    and finalizers maps state to list of group IDs.
    """
    nodes = set()
    edges = []
    finalizers = {}
    
    # Parse DFAState entries
    # Pattern: DFAState { transitions: {byte: state, ...}, finalizers: Bitset { words: [...] }, ...
    state_pattern = r"DFAState \{ transitions: \{([^}]*)\}, finalizers: Bitset \{ words: \[([^\]]*)\]"
    
    state_id = 0
    for match in re.finditer(state_pattern, dfa_str):
        nodes.add(state_id)
        transitions_str = match.group(1)
        finalizers_str = match.group(2)
        
        # Parse transitions
        for trans_match in re.finditer(r"(\d+): (\d+)", transitions_str):
            byte_val = int(trans_match.group(1))
            target = int(trans_match.group(2))
            nodes.add(target)
            # Convert byte to char for label
            if byte_val == 36:  # $
                label = "\\$"
            elif 32 <= byte_val <= 126:
                label = chr(byte_val)
                if label in ['$', '_', '%', '&', '#', '{', '}']:
                    label = '\\' + label
            else:
                label = f"0x{byte_val:02x}"
            edges.append((state_id, target, label, "ALL"))
        
        # Parse finalizers
        if finalizers_str.strip():
            words = [int(w) for w in finalizers_str.split(',') if w.strip()]
            final_groups = []
            for word_idx, word in enumerate(words):
                for bit in range(64):
                    if word & (1 << bit):
                        final_groups.append(word_idx * 64 + bit)
            if final_groups:
                finalizers[state_id] = final_groups
        
        state_id += 1
    
    return nodes, edges, finalizers


def run_dot_layout(nodes: Set[int], edges: List[Tuple], final_states: Set[int] = None, 
                   large: bool = False, nodesep: float = None, ranksep: float = None) -> Optional[str]:
    """Generate layout using Graphviz dot."""
    if final_states is None:
        final_states = set()
    
    dot_content = ["digraph G {"]
    dot_content.append('  rankdir=LR;')
    
    # Use provided values, or defaults based on 'large'
    if nodesep is not None:
        dot_content.append(f'  nodesep={nodesep};')
    elif large:
        dot_content.append('  nodesep=1.8;')
    else:
        dot_content.append('  nodesep=1.2;')
    
    if ranksep is not None:
        dot_content.append(f'  ranksep={ranksep};')
    elif large:
        dot_content.append('  ranksep=2.5;')
    else:
        dot_content.append('  ranksep=1.5;')
    
    for n in sorted(nodes):
        shape = "doublecircle" if n in final_states else "circle"
        dot_content.append(f'  {n} [shape={shape}, width=0.4, height=0.4, fixedsize=true, fontsize=8];')
        
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
            return None
        return stdout
    except Exception as e:
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


def format_edge_label(symbol: Any, weight: str, terminal_names: Dict[int, str] = None, 
                      as_state_id: bool = False) -> str:
    """Format the edge label with symbol and weight."""
    sym_str = format_symbol(symbol, terminal_names, as_state_id=as_state_id)
    weight_str = format_weight(weight)
    
    if weight_str == "ALL":
        return sym_str
    else:
        weight_short = weight_str
        if len(weight_short) > 12:
            weight_short = weight_short[:10] + ".."
        return f"{sym_str}/{weight_short}"


def generate_automaton_tikz(
    nodes: Set[int], 
    edges: List[Tuple], 
    final_states: Set[int],
    name: str, 
    pos: Tuple[float, float] = (0, 0), 
    scale: float = 1.0, 
    terminal_names: Dict[int, str] = None,
    show_labels: bool = True,
    as_state_id: bool = False,
    node_size: str = "10mm",
    font_size: str = "\\small"
) -> str:
    """Generate TikZ code for an automaton with labeled edges."""
    
    plain = run_dot_layout(nodes, edges, final_states)
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
    for n in sorted(nodes):
        if n not in layout["nodes"]:
            continue
        x, y = layout["nodes"][n]
        tx, ty = x - cx, y - cy
        styles = ["state"]
        if n in final_states:
            styles = ["accepting"]
        if n == 0:
            styles.append("initial")
        style_str = ",".join(styles)
        tikz.append(f"\\node[{style_str}, minimum size={node_size}] (n{safe_name}{n}) at ({tx:.2f},{ty:.2f}) {{{font_size} {n}}};")
    
    # Group edges by (source, target)
    edge_groups: Dict[Tuple[int, int], List[Tuple[Any, str]]] = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        symbol = edge[2] if len(edge) > 2 else None
        weight = edge[3] if len(edge) > 3 else "ALL"
        key = (u, v)
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append((symbol, weight))
    
    # Draw edges
    for (u, v), labels in edge_groups.items():
        if u not in layout["nodes"] or v not in layout["nodes"]:
            continue
        nu = f"n{safe_name}{u}"
        nv = f"n{safe_name}{v}"
        
        if show_labels:
            label_parts = []
            for symbol, weight in labels[:4]:  # Limit to 4 labels
                label_parts.append(format_edge_label(symbol, weight, terminal_names, as_state_id=as_state_id))
            
            if len(labels) > 4:
                label_text = ", ".join(label_parts) + ", ..."
            else:
                label_text = ", ".join(label_parts)
            
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


def generate_mini_automaton_tikz(
    nodes: Set[int], 
    edges: List[Tuple], 
    final_states: Set[int],
    edge_name: str,
    scale: float = 0.15
) -> str:
    """Generate a MINI automaton for embedding in an edge label.
    
    Returns TikZ scope that can be placed at a position.
    """
    plain = run_dot_layout(nodes, edges, final_states)
    if not plain:
        return ""
        
    layout = parse_dot_plain(plain)


def generate_merged_dwa_with_mini_dfas(
    skel_nodes: Set[int],
    skel_edges: List[Tuple],
    skel_finals: Dict[int, str],
    template_dwas: Dict[int, Tuple],
    terminal_names: Dict[int, str],
    pos: Tuple[float, float],
    scale: float = 0.8
) -> str:
    """Generate the merged Terminal DWA with actual mini template DFA visuals on edges.
    
    This is the "crazy" visualization where edges show miniaturized template DFAs.
    """
    # First, layout the skeleton
    skel_final_set = set(skel_finals.keys())
    plain = run_dot_layout(skel_nodes, skel_edges, skel_final_set)
    if not plain:
        return f"% Failed to layout merged DWA\n"
    
    layout = parse_dot_plain(plain)
    bbox = layout["bbox"]
    cx, cy = bbox[2] / 2, bbox[3] / 2
    
    tikz = []
    tikz.append(f"\\begin{{scope}}[shift={{({pos[0]},{pos[1]})}}, scale={scale}]")
    
    # Draw skeleton nodes
    for n in sorted(skel_nodes):
        if n not in layout["nodes"]:
            continue
        x, y = layout["nodes"][n]
        tx, ty = x - cx, y - cy
        styles = ["state"]
        if n in skel_final_set:
            styles = ["accepting"]
        if n == 0:
            styles.append("initial")
        style_str = ",".join(styles)
        tikz.append(f"\\node[{style_str}, minimum size=8mm] (merged{n}) at ({tx:.2f},{ty:.2f}) {{{n}}};")
    
    # Group edges and prepare mini-DFA positions
    edge_info = []
    for u, v, symbol, weight in skel_edges:
        if u not in layout["nodes"] or v not in layout["nodes"]:
            continue
        
        # Calculate a point 40% along the edge (closer to source) for mini-DFA placement
        x1, y1 = layout["nodes"][u]
        x2, y2 = layout["nodes"][v]
        t = 0.4  # Position along edge (0=source, 1=target)
        mx = x1 + t * (x2 - x1) - cx
        my = y1 + t * (y2 - y1) - cy
        
        # Offset perpendicular to edge direction for label placement
        dx, dy = x2 - x1, y2 - y1
        length = (dx**2 + dy**2) ** 0.5
        if length > 0:
            # Normal vector (perpendicular)
            nx, ny = -dy / length, dx / length
            # Offset the label position above the edge
            offset = 1.2
            mx += nx * offset
            my += ny * offset
        
        edge_info.append((u, v, symbol, weight, mx, my))
    
    # Draw edges with mini-DFAs
    for u, v, symbol, weight, mx, my in edge_info:
        nu = f"merged{u}"
        nv = f"merged{v}"
        
        # Draw the edge
        tikz.append(f"\\path[edge, very thick] ({nu}) edge ({nv});")
        
        # If this symbol is a terminal (0-3), draw a mini template DFA
        if isinstance(symbol, int) and symbol in template_dwas:
            tnodes, tedges, tfinals = template_dwas[symbol]
            tfinal_set = set(tfinals.keys())
            term_name = terminal_names.get(symbol, f"T{symbol}")
            
            # Generate mini-DFA at edge midpoint
            mini_plain = run_dot_layout(tnodes, tedges, tfinal_set, nodesep=0.6, ranksep=0.8)
            if mini_plain:
                mini_layout = parse_dot_plain(mini_plain)
                mini_bbox = mini_layout["bbox"]
                mini_cx, mini_cy = mini_bbox[2] / 2, mini_bbox[3] / 2
                mini_scale = 0.20  # Very small
                
                safe_edge = f"mini{u}to{v}"
                tikz.append(f"\\begin{{scope}}[shift={{({mx:.2f},{my:.2f})}}, scale={mini_scale}]")
                
                # Mini label above
                tikz.append(f"\\node[font=\\tiny\\bfseries, above] at (0, {mini_bbox[3]/2 + 0.5}) {{T({term_name.replace('$', '\\$')})}};")
                
                # Draw mini nodes
                for n in sorted(tnodes):
                    if n not in mini_layout["nodes"]:
                        continue
                    x, y = mini_layout["nodes"][n]
                    tx, ty = x - mini_cx, y - mini_cy
                    style = "state, minimum size=3mm, inner sep=0pt"
                    if n in tfinal_set:
                        style = "accepting, minimum size=3mm, inner sep=0pt"
                    if n == 0:
                        style += ", initial, initial text="
                    tikz.append(f"\\node[{style}] (m{safe_edge}n{n}) at ({tx:.2f},{ty:.2f}) {{}};")
                
                # Draw mini edges (no labels)
                edge_pairs = set()
                for te in tedges:
                    tu, tv = te[0], te[1]
                    if (tu, tv) in edge_pairs:
                        continue
                    edge_pairs.add((tu, tv))
                    if tu not in mini_layout["nodes"] or tv not in mini_layout["nodes"]:
                        continue
                    mnu = f"m{safe_edge}n{tu}"
                    mnv = f"m{safe_edge}n{tv}"
                    if tu == tv:
                        tikz.append(f"\\path[edge, thin] ({mnu}) edge[loop above, looseness=6] ({mnv});")
                    else:
                        tikz.append(f"\\path[edge, thin] ({mnu}) edge ({mnv});")
                
                tikz.append("\\end{scope}")
                
                # Also show weight if not ALL
                weight_str = format_weight(weight)
                if weight_str != "ALL":
                    tikz.append(f"\\node[font=\\tiny, below] at ({mx:.2f},{my - 1:.2f}) {{{weight_str[:10]}}};")
        else:
            # Just draw symbol label for non-terminal edges
            sym_str = format_symbol(symbol, terminal_names, as_state_id=True)
            tikz.append(f"\\node[font=\\small, fill=white] at ({mx:.2f},{my:.2f}) {{{sym_str}}};")
    
    tikz.append("\\end{scope}")
    return "\n".join(tikz)
    bbox = layout["bbox"]
    cx, cy = bbox[2] / 2, bbox[3] / 2
    
    safe_name = re.sub(r'[^a-zA-Z0-9]', '', edge_name)
    
    tikz = []
    tikz.append(f"\\begin{{scope}}[scale={scale}]")
    
    # Draw nodes (very small)
    for n in sorted(nodes):
        if n not in layout["nodes"]:
            continue
        x, y = layout["nodes"][n]
        tx, ty = x - cx, y - cy
        style = "state, minimum size=3mm"
        if n in final_states:
            style = "accepting, minimum size=3mm"
        if n == 0:
            style += ", initial"
        tikz.append(f"\\node[{style}] (m{safe_name}{n}) at ({tx:.2f},{ty:.2f}) {{}};")
    
    # Draw edges (no labels to keep it tiny)
    edge_pairs = set()
    for edge in edges:
        u, v = edge[0], edge[1]
        if (u, v) in edge_pairs:
            continue
        edge_pairs.add((u, v))
        if u not in layout["nodes"] or v not in layout["nodes"]:
            continue
        mu = f"m{safe_name}{u}"
        mv = f"m{safe_name}{v}"
        if u == v:
            tikz.append(f"\\path[edge, thin] ({mu}) edge[loop above, looseness=4] ({mv});")
        else:
            tikz.append(f"\\path[edge, thin] ({mu}) edge ({mv});")
    
    tikz.append("\\end{scope}")
    return "\n".join(tikz)


def parse_lalr_table(lalr_str: str) -> Dict:
    """Parse the LALR table string into a structured format."""
    states = {}
    
    state_parts = re.split(r'StateID\((\d+)\):\s*Row\s*\{', lalr_str)
    
    for i in range(1, len(state_parts), 2):
        state_id = int(state_parts[i])
        if i + 1 < len(state_parts):
            content = state_parts[i + 1]
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
        
        shift_pattern = r"TerminalID\((\d+)\):\s*Shift\(StateID\((\d+)\)\)"
        for shift_match in re.finditer(shift_pattern, content):
            term_id = int(shift_match.group(1))
            target = int(shift_match.group(2))
            state_data["shifts"][term_id] = target
        
        reduce_pattern = r"TerminalID\((\d+)\):\s*Reduce\s*\{\s*nonterminal_id:\s*NonTerminalID\((\d+)\),\s*len:\s*(\d+)"
        for red_match in re.finditer(reduce_pattern, content):
            term_id = int(red_match.group(1))
            nt_id = int(red_match.group(2))
            prod_len = int(red_match.group(3))
            state_data["reduces"][term_id] = (nt_id, prod_len)
        
        def_reduce_pattern = r"default_reduce:\s*Some\(Reduce\s*\{\s*nonterminal_id:\s*NonTerminalID\((\d+)\),\s*len:\s*(\d+)"
        def_match = re.search(def_reduce_pattern, content)
        if def_match:
            state_data["default_reduce"] = (int(def_match.group(1)), int(def_match.group(2)))
        
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
    tikz.append("\\node[anchor=north, font=\\bfseries\\large] at (0, 1) {LALR(1) Parse Table};")
    
    col_spec = "c|" + "c" * len(terminals) + "|" + "c" * len(nonterminals)
    
    tikz.append("\\node[anchor=north] at (0, 0) {")
    tikz.append("\\scalebox{0.8}{")
    tikz.append("\\begin{tabular}{" + col_spec + "}")
    tikz.append("\\hline")
    
    header = ["State"]
    for t in terminals:
        name = terminal_names.get(t, f"t{t}")
        name = name.replace('$', '\\$')
        header.append(f"\\textbf{{{name}}}")
    for nt in nonterminals:
        header.append(f"\\textit{{{nonterminal_names.get(nt, f'N{nt}')}}}")
    tikz.append(" & ".join(header) + " \\\\")
    tikz.append("\\hline")
    
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
    tikz.append("}")
    tikz.append("};")
    tikz.append("\\end{scope}")
    
    return "\n".join(tikz)


def generate_characterization_box(char_data: Dict[str, str], terminal_names: Dict[int, str], 
                                   pos: Tuple[float, float]) -> str:
    """Generate TikZ box showing all below-zero characterizations."""
    
    tikz = []
    tikz.append(f"\\begin{{scope}}[shift={{({pos[0]},{pos[1]})}}]")
    tikz.append("\\node[anchor=north, font=\\bfseries\\large] at (0, 0.5) {Below-Zero Characterizations};")
    
    y_offset = -0.5
    for term_id, char_str in sorted(char_data.items()):
        term_match = re.search(r"terminal:\s*TerminalID\((\d+)\)", char_str)
        term_num = int(term_match.group(1)) if term_match else 0
        term_name = terminal_names.get(term_num, f"T{term_num}")
        term_name_escaped = term_name.replace('$', '\\$')
        
        # Extract key info
        shift_match = re.search(r"initial_shifts:\s*\{([^}]*)\}", char_str)
        shifts = shift_match.group(1) if shift_match else ""
        shift_pairs = re.findall(r"\(StateID\((\d+)\),\s*StateID\((\d+)\)\)", shifts)
        
        reduce_match = re.search(r"initial_reduces:\s*\{([^}]*)\}", char_str)
        reduces = reduce_match.group(1) if reduce_match else ""
        reduce_tuples = re.findall(r"\(StateID\((\d+)\),\s*\d+,\s*NonTerminalID\((\d+)\)\)", reduces)
        
        shift_str = ", ".join([f"({s}$\\to${t})" for s, t in shift_pairs]) if shift_pairs else "$\\emptyset$"
        reduce_str = ", ".join([f"({s},N{n})" for s, n in reduce_tuples]) if reduce_tuples else "$\\emptyset$"
        
        tikz.append(f"\\node[anchor=west, font=\\footnotesize] at (-5, {y_offset}) {{")
        tikz.append(f"  \\textbf{{{term_name_escaped}:}} shifts=\\{{{shift_str}\\}}, reduces=\\{{{reduce_str}\\}}")
        tikz.append("};")
        y_offset -= 0.6
    
    tikz.append("\\end{scope}")
    return "\n".join(tikz)


def main():
    with open("pipeline_artifacts.json", "r") as f:
        data = json.load(f)
    
    # Terminal and nonterminal names
    terminal_names = {0: "$", 1: "a", 2: "b", 3: "c"}
    nonterminal_names = {0: "S", 1: "A", 2: "B", 3: "C", 4: "C'"}
    
    # Parse all data
    lalr_data = parse_lalr_table(data["lalr_table"])
    tokenizer_nodes, tokenizer_edges, tokenizer_finals = parse_tokenizer_dfa(data["tokenizer_dfa"])
    skel_nodes, skel_edges, skel_finals = parse_dwa_with_labels(data["skeleton_dwa"])
    flat_nodes, flat_edges, flat_finals = parse_dwa_with_labels(data["flattened_nwa"])
    final_nodes, final_edges, final_finals = parse_dwa_with_labels(data["final_dwa"])
    
    # Parse template DFAs
    template_dwas = {}
    for tid, dwa_str in data["template_dfas_all"].items():
        nodes, edges, finals = parse_dwa_with_labels(dwa_str)
        term_match = re.search(r'\d+', tid)
        term_id = int(term_match.group()) if term_match else 0
        template_dwas[term_id] = (nodes, edges, finals)
    
    grammar = data["grammar_ebnf"]
    char_data = data["characterizations_all"]
    
    tex = []
    
    # Preamble
    tex.append(r"""\documentclass[tikz,border=20pt]{standalone}
\usepackage{lmodern}
\usepackage{tikz}
\usepackage{amsmath}
\usepackage{amssymb}
\usetikzlibrary{automata,positioning,arrows.meta,shapes,shadows,fit,calc,backgrounds}

\definecolor{primary}{RGB}{41,128,185}
\definecolor{accent}{RGB}{39,174,96}
\definecolor{dark}{RGB}{52,73,94}
\definecolor{grammar}{RGB}{155,89,182}
\definecolor{lalr}{RGB}{230,126,34}
\definecolor{tokenizer}{RGB}{46,204,113}
\definecolor{template}{RGB}{52,152,219}

\begin{document}
\begin{tikzpicture}[
    >=Stealth,
    font=\sffamily,
    state/.style={
        circle,
        draw=primary,
        thick,
        minimum size=8mm,
        fill=white,
        text=dark,
        font=\scriptsize
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
        semithick
    },
    stagebox/.style={
        rectangle,
        draw=dark,
        very thick,
        rounded corners=3pt,
        fill=white,
        drop shadow,
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
        >=Stealth
    },
    splitarrow/.style={
        ->,
        draw=dark,
        line width=1.5pt,
        dashed
    }
]

""")
    
    # Layout constants
    FAR_LEFT_COL = -22  # LLM Vocab
    LEFT_COL = -10      # Tokenizer
    RIGHT_COL = 14
    CENTER_COL = 0
    
    # ===================
    # STAGE 0: LLM Vocab (FAR LEFT) and Input Grammar (TOP CENTER-RIGHT)
    # ===================
    y_grammar = 0
    
    # LLM Vocab box (far left)
    tex.append(r"% ========== FAR LEFT: LLM Vocabulary ==========")
    tex.append(f"\\node[stagebox, fill=tokenizer!10] (vocab) at ({FAR_LEFT_COL}, 0) {{")
    tex.append(r"  \begin{tabular}{c}")
    tex.append(r"  \textbf{LLM Vocabulary}\\[3pt]")
    tex.append(r"  \footnotesize (50,257 tokens)\\")
    tex.append(r"  \tiny e.g., GPT-2/GPT-3")
    tex.append(r"  \end{tabular}")
    tex.append(r"};")
    tex.append("")
    
    # Grammar box (center-right)
    tex.append(r"% ========== TOP: Input Grammar ==========")
    tex.append(f"\\node[grammarbox] (grammar) at ({RIGHT_COL/2}, 0) {{")
    tex.append(r"  \begin{tabular}{l}")
    tex.append(r"  \textbf{Input Grammar (EBNF):}\\[3pt]")
    for line in grammar.strip().split('\n'):
        line = line.strip()
        if line:
            line = line.replace('$', '\\$')
            line = line.replace('|', '$|$').replace('"', "``").replace("'", "`")
            tex.append(f"  {line}\\\\")
    tex.append(r"  \end{tabular}")
    tex.append(r"};")
    tex.append("")
    
    # Split arrows: vocab goes to tokenizer, grammar splits to tokenizer and parser
    y_split = -4
    tex.append(f"\\draw[splitarrow] (vocab.south) -- ({FAR_LEFT_COL}, {y_split});")
    tex.append(f"\\draw[splitarrow] (grammar.south) -- ++(0,-1) -| ({LEFT_COL}, {y_split});")
    tex.append(f"\\draw[splitarrow] (grammar.south) -- ++(0,-1) -| ({RIGHT_COL}, {y_split});")
    tex.append(f"\\node[font=\\footnotesize\\itshape] at ({(FAR_LEFT_COL + LEFT_COL)/2}, {y_split + 1.5}) {{Tokenizer}};")
    tex.append(f"\\node[font=\\footnotesize\\itshape] at ({(LEFT_COL + RIGHT_COL)/2 + 2}, {y_split + 1.5}) {{Parser}};")
    tex.append("")
    
    # ===================
    # LEFT COLUMN: Tokenizer DFA (between FAR_LEFT and LEFT)
    # ===================
    y_tokenizer = -7
    tex.append(r"% ========== LEFT: Tokenizer DFA ==========")
    tokenizer_final_set = set(tokenizer_finals.keys())
    tokenizer_x = (FAR_LEFT_COL + LEFT_COL) / 2  # Between vocab and left column
    tex.append(generate_automaton_tikz(
        tokenizer_nodes, tokenizer_edges, tokenizer_final_set,
        "Tokenizer DFA",
        (tokenizer_x, y_tokenizer),
        scale=0.7,
        show_labels=True,
        node_size="6mm",
        font_size="\\tiny"
    ))
    # Arrow from vocab to tokenizer
    tex.append(f"\\draw[flowarrow] ({FAR_LEFT_COL}, {y_split - 1}) -- ({tokenizer_x - 2}, {y_tokenizer + 2});")
    tex.append("")
    
    # ===================
    # RIGHT COLUMN: LALR Parse Table
    # ===================
    y_lalr = -6
    tex.append(r"% ========== RIGHT: LALR(1) Parse Table ==========")
    tex.append(generate_lalr_table_tikz(lalr_data, terminal_names, nonterminal_names, (RIGHT_COL, y_lalr)))
    tex.append("")
    
    # ===================
    # LEFT COLUMN: Terminal DWA (Skeleton / Precompute1)
    # ===================
    y_terminal_dwa = -18
    tex.append(r"% ========== LEFT: Terminal DWA (Skeleton) ==========")
    skel_final_set = set(skel_finals.keys())
    tex.append(generate_automaton_tikz(
        skel_nodes, skel_edges, skel_final_set,
        "Terminal DWA (Skeleton)",
        (LEFT_COL, y_terminal_dwa),
        scale=0.7,
        terminal_names=terminal_names,
        show_labels=True,
        node_size="6mm",
        font_size="\\tiny"
    ))
    # Arrow from tokenizer to terminal DWA
    tex.append(f"\\draw[flowarrow] ({tokenizer_x}, {y_tokenizer - 3}) -- ({LEFT_COL}, {y_terminal_dwa + 3});")
    tex.append("")
    
    # ===================
    # RIGHT COLUMN: Below-Zero Characterizations
    # ===================
    y_char = -15
    tex.append(r"% ========== RIGHT: Below-Zero Characterizations ==========")
    tex.append(generate_characterization_box(char_data, terminal_names, (RIGHT_COL, y_char)))
    tex.append("")
    
    # ===================
    # RIGHT COLUMN: Template DFAs
    # ===================
    y_templates = -24
    tex.append(r"% ========== RIGHT: Template DFAs ==========")
    tex.append(f"\\node[font=\\bfseries\\large] at ({RIGHT_COL}, {y_templates + 2}) {{Template DFAs}};")
    
    num_templates = len(template_dwas)
    x_spacing = 6  # More spacing between templates
    start_x = RIGHT_COL - (num_templates - 1) * x_spacing / 2
    
    for i, (tid, (tnodes, tedges, tfinals)) in enumerate(sorted(template_dwas.items())):
        term_name = terminal_names.get(tid, f"T{tid}")
        term_name_escaped = term_name.replace('$', '\\$')
        tfinal_set = set(tfinals.keys())
        x_pos = start_x + i * x_spacing
        tex.append(generate_automaton_tikz(
            tnodes, tedges, tfinal_set,
            f"T({term_name_escaped})",
            (x_pos, y_templates - 3),
            scale=0.45,
            show_labels=True,
            as_state_id=True,  # Edge labels are state IDs!
            node_size="5mm",
            font_size="\\tiny"
        ))
    
    # Arrow from LALR to characterizations to templates
    tex.append(f"\\draw[flowarrow] ({RIGHT_COL}, {y_lalr - 4}) -- ({RIGHT_COL}, {y_char + 2});")
    tex.append(f"\\draw[flowarrow] ({RIGHT_COL}, {y_char - 5}) -- ({RIGHT_COL}, {y_templates + 3});")
    tex.append("")
    
    # ===================
    # CENTER: Merge point - Terminal DWA with Template DFAs on Edges
    # ===================
    y_merged = -40
    tex.append(r"% ========== CENTER: Terminal DWA with Template DFAs on Edges ==========")
    tex.append(f"\\node[font=\\bfseries\\large] at ({CENTER_COL}, {y_merged + 4}) {{Terminal DWA with Template DFAs on Edges}};")
    tex.append(f"\\node[font=\\footnotesize\\itshape, text=dark!60] at ({CENTER_COL}, {y_merged + 3.2}) {{(Each terminal edge shows its Template DFA)}};")
    
    # Merge arrows
    tex.append(f"\\draw[splitarrow] ({LEFT_COL}, {y_terminal_dwa - 6}) -- ++(0,-4) -| ({CENTER_COL - 4}, {y_merged + 2});")
    tex.append(f"\\draw[splitarrow] ({RIGHT_COL}, {y_templates - 10}) -- ++(0,-4) -| ({CENTER_COL + 4}, {y_merged + 2});")
    
    # Draw the merged DWA with actual mini-DFAs on edges!
    tex.append(generate_merged_dwa_with_mini_dfas(
        skel_nodes, skel_edges, skel_finals,
        template_dwas, terminal_names,
        (CENTER_COL, y_merged - 4),
        scale=0.9
    ))
    tex.append("")
    
    # ===================
    # CENTER: Flattened NWA
    # ===================
    y_flat = -68
    tex.append(r"% ========== CENTER: Flattened NWA ==========")
    tex.append(f"\\draw[flowarrow] ({CENTER_COL}, {y_merged - 16}) -- ({CENTER_COL}, {y_flat + 12});")
    tex.append(f"\\node[font=\\footnotesize, text=dark!60] at ({CENTER_COL + 5}, {(y_merged - 16 + y_flat + 12)/2}) {{Flatten (inline templates)}};")
    
    flat_final_set = set(flat_finals.keys())
    tex.append(generate_automaton_tikz(
        flat_nodes, flat_edges, flat_final_set,
        "Flattened NWA (with Push Transitions)",
        (CENTER_COL, y_flat),
        scale=0.40,
        terminal_names=None,  # Edge labels are state IDs
        show_labels=True,
        as_state_id=True,
        node_size="5mm",
        font_size="\\tiny"
    ))
    tex.append("")
    
    # ===================
    # CENTER: Final DWA
    # ===================
    y_final = -94
    tex.append(r"% ========== CENTER: Final DWA ==========")
    tex.append(f"\\draw[flowarrow] ({CENTER_COL}, {y_flat - 14}) -- ({CENTER_COL}, {y_final + 12});")
    tex.append(f"\\node[font=\\footnotesize, text=dark!60, align=center] at ({CENTER_COL + 7}, {(y_flat - 14 + y_final + 12)/2}) {{Resolve push transitions,\\\\Determinize \\& Simplify}};")
    
    final_final_set = set(final_finals.keys())
    tex.append(generate_automaton_tikz(
        final_nodes, final_edges, final_final_set,
        "Final DWA",
        (CENTER_COL, y_final),
        scale=0.40,
        terminal_names=None,
        show_labels=True,
        as_state_id=True,
        node_size="5mm",
        font_size="\\tiny"
    ))
    tex.append("")
    
    tex.append(r"""
\end{tikzpicture}
\end{document}
""")
    
    output_path = "gcg-paper/paper/figures/pipeline_full.tex"
    with open(output_path, "w") as f:
        f.write("\n".join(tex))
        
    print(f"Generated {output_path}")
    print(f"  Layout: LEFT (Tokenizer, Terminal DWA), RIGHT (LALR, Chars, Templates), CENTER (merged stages)")
    print(f"  - Tokenizer DFA: {len(tokenizer_nodes)} states")
    print(f"  - {len(lalr_data)} LALR states")
    print(f"  - {len(char_data)} below-zero characterizations")
    print(f"  - {len(template_dwas)} template DFAs")
    print(f"  - Skeleton DWA: {len(skel_nodes)} states, {len(skel_edges)} edges")
    print(f"  - Flattened NWA: {len(flat_nodes)} states, {len(flat_edges)} edges")
    print(f"  - Final DWA: {len(final_nodes)} states, {len(final_edges)} edges")


if __name__ == "__main__":
    main()
