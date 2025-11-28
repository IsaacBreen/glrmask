#!/usr/bin/env python3
"""
Generate TikZ visualization of the grammar compilation pipeline.
Produces a rigid, fixed-size grid layout with orthogonal arrows.
"""

import json
import re
import subprocess
from typing import Dict, List, Set, Tuple, Any, Optional

# ==============================================================================
# LATEX GENERATION - STRUCTURE ONLY
# ==============================================================================

def generate_latex_structure():
    """Generate the macro structure first - components and arrows only."""
    
    latex = [
        r"\documentclass[tikz,border=10pt]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern, amsmath, amssymb, colortbl, graphicx}",
        r"\usetikzlibrary{automata, positioning, arrows.meta, shapes, shadows, fit, calc, backgrounds}",
        "",
        r"% --- COLORS ---",
        r"\definecolor{primary}{RGB}{41,128,185}",
        r"\definecolor{accent}{RGB}{39,174,96}",
        r"\definecolor{dark}{RGB}{52,73,94}",
        r"\definecolor{grammar}{RGB}{155,89,182}",
        r"\definecolor{vocabcolor}{RGB}{241,196,15}",
        r"\definecolor{componentbg}{RGB}{236,240,241}",
        "",
        r"% --- DIMENSIONS (Global 'Constants') ---",
        r"\def\boxwidth{6cm}",
        r"\def\boxheight{4.5cm}",
        r"% Text area is slightly smaller than box to allow padding",
        r"\def\textw{5.6cm}",
        r"\def\texth{4.1cm}",
        "",
        r"% --- MACRO: Content Fitter ---",
        r"% Usage: \node[box] { \fit{Your Long Content Here} };",
        r"% This forces the content to shrink if it exceeds the box size.",
        r"\newcommand{\fit}[1]{%",
        r"  \resizebox{\textw}{!}{%",
        r"    \begin{minipage}[c][\texth][c]{\textw}%",
        r"      \centering\bfseries #1%",
        r"    \end{minipage}%",
        r"  }%",
        r"}",
        "",
        r"% --- STYLES ---",
        r"\tikzset{",
        r"  % The Rigid Box Style",
        r"  box/.style={",
        r"    rectangle,",
        r"    draw=dark!40,",
        r"    fill=componentbg,",
        r"    rounded corners=5pt,",
        r"    % FORCING FIXED SIZE:",
        r"    minimum width=\boxwidth,",
        r"    minimum height=\boxheight,",
        r"    text width=\textw,",
        r"    align=center,",
        r"    font=\bfseries",
        r"  },",
        r"  flowarrow/.style={->, draw=dark!60, line width=2.5pt, >=Stealth},",
        r"  connector/.style={-, draw=dark!60, line width=2.5pt},",
        r"}",
        "",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        "",
        r"  % ====================================================================",
        r"  % SPACING CONFIGURATION",
        r"  % ====================================================================",
        r"  \def\colgap{3cm}      % Horizontal space between columns",
        r"  \def\toprowgap{4cm}   % LARGE vertical space between Inputs and Row 2",
        r"  \def\midrowgap{3cm}   % Vertical space between Row 2 and Row 3",
        "",
        r"  % ====================================================================",
        r"  % ROW 2: The Anchors (Defining the Grid Width)",
        r"  % ====================================================================",
        r"  \node[box] (tokenizer) {\fit{Tokenizer DFA}};",
        r"  \node[box, right=\colgap of tokenizer] (parsetable) {\fit{LALR Parse Table}};",
        "",
        r"  % ====================================================================",
        r"  % ROW 1: Inputs (High above)",
        r"  % ====================================================================",
        r"  ",
        r"  % Calculate center axis of the graph",
        r"  \coordinate (center_axis) at ($(tokenizer.east)!0.5!(parsetable.west)$);",
        r"  ",
        r"  % Place Grammar HIGH above the center axis",
        r"  \node[box, above=\toprowgap of center_axis, anchor=south] (grammar) {\fit{Input Grammar\\(EBNF)}};",
        r"  ",
        r"  % Place Vocab to the left",
        r"  \node[box, left=\colgap of grammar] (vocab) {\fit{LLM Vocab\\(50k tokens)}};",
        "",
        r"  % ====================================================================",
        r"  % ROW 3: Processing",
        r"  % ====================================================================",
        r"  ",
        r"  % Note: Even though 'Templates' has more text, it uses the same rigid 'box' style.",
        r"  \node[box, below=\midrowgap of parsetable] (templates) {\fit{Below-Zero\\Characterizations\\+\\Template DFAs}};",
        r"  ",
        r"  % Terminal DWA below Tokenizer, vertically aligned with Templates",
        r"  \node[box] (terminal_dwa) at (tokenizer |- templates) {\fit{Terminal DWA\\(Precompute1)}};",
        "",
        r"  % ====================================================================",
        r"  % CENTRAL COLUMN: Merged Flow",
        r"  % ====================================================================",
        r"  ",
        r"  % Determine the Y-level for the bracket merge",
        r"  % We look at where the bottom of the previous row is, and go halfway to the next node",
        r"  \path (templates.south) -- ++(0, -\midrowgap) coordinate (next_node_top);",
        r"  \coordinate (merge_bar_y) at ($(templates.south)!0.5!(next_node_top)$);",
        r"  ",
        r"  % Center column nodes",
        r"  \node[box] (dwa_with_templates) at (center_axis |- next_node_top) [anchor=north] {\fit{Terminal DWA\\with\\Template DFAs on Edges}};",
        r"  ",
        r"  \node[box, below=2cm of dwa_with_templates] (flattened_nwa) {\fit{Flattened NWA}};",
        r"  \node[box, below=2cm of flattened_nwa] (resolved_nwa) {\fit{NWA\\(Push Transitions Resolved)}};",
        r"  \node[box, below=2cm of resolved_nwa] (final_dwa) {\fit{Final DWA}};",
        "",
        r"  % ====================================================================",
        r"  % ARROWS",
        r"  % ====================================================================",
        r"  ",
        r"  % --- 1. Grammar -> Tokenizer & Parse Table ---",
        r"  % Calculate a Y-coordinate exactly halfway down the LARGE gap",
        r"  \coordinate (mid_top_gap) at ($(grammar.south)!0.5!(tokenizer.north)$);",
        r"  ",
        r"  % Draw orthogonal arrows using that calculated midpoint",
        r"  \draw[flowarrow] (grammar.south) -- (grammar.south |- mid_top_gap) -| (tokenizer.north);",
        r"  \draw[flowarrow] (grammar.south) -- (grammar.south |- mid_top_gap) -| (parsetable.north);",
        r"  ",
        r"  % --- 2. Vocab -> Terminal DWA ---",
        r"  % Goes down the far left side, then turns right into the box",
        r"  \draw[flowarrow] (vocab.south) |- (terminal_dwa.west);",
        r"  ",
        r"  % --- 3. Vertical Connectors (Row 2 -> Row 3) ---",
        r"  \draw[flowarrow] (tokenizer) -- (terminal_dwa);",
        r"  \draw[flowarrow] (parsetable) -- (templates);",
        r"  ",
        r"  % --- 4. The Bracket Merge (Row 3 -> Central) ---",
        r"  % Left Arm",
        r"  \draw[connector] (terminal_dwa.south) -- (terminal_dwa.south |- merge_bar_y);",
        r"  % Right Arm",
        r"  \draw[connector] (templates.south) -- (templates.south |- merge_bar_y);",
        r"  % Crossbar",
        r"  \draw[connector] (terminal_dwa.south |- merge_bar_y) -- (templates.south |- merge_bar_y);",
        r"  % Down Arrow",
        r"  \draw[flowarrow] (center_axis |- merge_bar_y) -- (dwa_with_templates.north);",
        "",
        r"  % --- 5. Central Column Flow ---",
        r"  \draw[flowarrow] (dwa_with_templates) -- (flattened_nwa);",
        r"  \draw[flowarrow] (flattened_nwa) -- (resolved_nwa);",
        r"  \draw[flowarrow] (resolved_nwa) -- (final_dwa);",
        "",
        r"\end{tikzpicture}",
        r"\end{document}"
    ]
    
    return "\n".join(latex)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    latex_content = generate_latex_structure()
    
    # Writing to a new filename to avoid overwriting your original while testing
    filename = "pipeline_fixed.tex"
    with open(filename, "w") as f:
        f.write(latex_content)
    
    print(f"Generated {filename} with rigid structure and orthogonal arrows")

if __name__ == "__main__":
    main()