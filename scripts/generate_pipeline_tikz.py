#!/usr/bin/env python3
"""
Generate TikZ visualization of the grammar compilation pipeline.
Produces a rigid, fixed-size grid layout with orthogonal arrows.
Now with FULL content in each box.

Can generate:
- Individual components (--component NAME)
- All components (--all-components)
- Full pipeline (default or --full)
"""

import json
import re
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

# ==============================================================================
# CONTENT GENERATORS
# ==============================================================================

def generate_vocab_content():
    """Generate LLM Vocab visualization."""
    return r"""\tiny
\textbf{LLM Vocab}\\
\textbf{(50k tokens)}\\[0.2em]
\begin{tabular}{|c|c|}
\hline
\textbf{ID} & \textbf{Token} \\ \hline
0 & \texttt{<unk>} \\ \hline
1 & \texttt{<s>} \\ \hline
... & ... \\ \hline
40 & \texttt{'('} \\ \hline
41 & \texttt{')'} \\ \hline
43 & \texttt{'+'} \\ \hline
65 & \texttt{'a'} \\ \hline
66 & \texttt{'b'} \\ \hline
... & ... \\ \hline
\end{tabular}"""

def generate_grammar_content():
    """Generate Input Grammar - Expression Grammar (No Right/Hidden-Left Recursion)."""
    return r"""\scriptsize
\textbf{Input Grammar}\\
\textbf{(Expression)}\\[0.4em]
$\begin{array}{rcl}
\texttt{S} &\to& \texttt{E }'\texttt{\$}'\\
\texttt{E} &\to& \texttt{E '+' T}\\
\texttt{E} &\to& \texttt{T}\\
\texttt{T} &\to& \texttt{'a'}\\
\texttt{T} &\to& \texttt{'b'}\\
\texttt{T} &\to& \texttt{'(' E ')'}
\end{array}$"""


def generate_tokenizer_dfa():
    """Generate Tokenizer DFA - Expression Grammar Tokens."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.5cm,
    state/.style={circle, draw, thick, minimum size=0.5cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.5cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (2, 1.5) {Tokenizer DFA};
\node[state, initial, initial text=] (q0) at (0, 0) {0};
\node[accept] (qa) at (2, 1.2) {a};
\node[accept] (qb) at (2, 0.6) {b};
\node[accept] (qplus) at (2, 0) {+};
\node[accept] (qlp) at (2, -0.6) {(};
\node[accept] (qrp) at (2, -1.2) {)};

\draw[->, thick] (q0) -- node[above,font=\tiny] {'a'} (qa);
\draw[->, thick] (q0) -- node[above,pos=0.7,font=\tiny] {'b'} (qb);
\draw[->, thick] (q0) -- node[above,pos=0.7,font=\tiny] {'+'} (qplus);
\draw[->, thick] (q0) -- node[below,pos=0.7,font=\tiny] {'('} (qlp);
\draw[->, thick] (q0) -- node[below,font=\tiny] {')'} (qrp);
\end{tikzpicture}
}"""

def generate_lalr_table():
    """Generate LALR Parse Table - Expression Grammar."""
    return r"""\tiny
\textbf{LALR Parse Table}\\[0.2em]
\begin{tabular}{|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{St} & \textbf{a} & \textbf{b} & \textbf{+} & \textbf{(} & \textbf{)} & \textbf{\$} & \textbf{E} & \textbf{T} \\ \hline
0 & S3 & S4 & & S5 & & & G1 & G2 \\ \hline
1 & & & S6 & & & Acc & & \\ \hline
2 & & & R(E$\to$T) & & R(E$\to$T) & R(E$\to$T) & & \\ \hline
3 & & & R(T$\to$a) & & R(T$\to$a) & R(T$\to$a) & & \\ \hline
4 & & & R(T$\to$b) & & R(T$\to$b) & R(T$\to$b) & & \\ \hline
5 & S3 & S4 & & S5 & & & G7 & G2 \\ \hline
6 & S3 & S4 & & S5 & & & & G8 \\ \hline
7 & & & S6 & & S9 & & & \\ \hline
8 & & & R(E$\to$E+T) & & R(E$\to$E+T) & R(E$\to$E+T) & & \\ \hline
9 & & & R(T$\to$(E)) & & R(T$\to$(E)) & R(T$\to$(E)) & & \\ \hline
\end{tabular}"""

def generate_terminal_dwa():
    """Generate Terminal DWA - Expression Grammar."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.4cm and 1.6cm,
    state/.style={circle, draw, thick, minimum size=0.65cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.65cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (3, 2.2) {Terminal DWA};
\node[font=\tiny] at (3, 1.8) {(with LLM token masks)};
\node[state, fill=blue!15] (q0) at (0, 0) {0};
\node[accept] (qa) at (2.5, 1.5) {'a'};
\node[accept] (qb) at (2.5, 0.5) {'b'};
\node[accept] (qplus) at (2.5, -0.5) {'+'};
\node[accept] (qlp) at (5, 0.7) {'('};
\node[accept] (qrp) at (5, -0.7) {')'};

\draw[->, thick, blue!70] (q0) -- node[above,font=\tiny] {\textbf{'a'}$|$\{65\}} (qa);
\draw[->, thick, red!70] (q0) -- node[above,font=\tiny] {\textbf{'b'}$|$\{66\}} (qb);
\draw[->, thick, green!70] (q0) -- node[above,font=\tiny] {\textbf{'+'}$|$\{43\}} (qplus);
\draw[->, thick, purple!70] (q0) -- node[above,pos=0.7,font=\tiny] {\textbf{'('}$|$\{40\}} (qlp);
\draw[->, thick, orange!70] (q0) -- node[below,pos=0.7,font=\tiny] {\textbf{')'}$|$\{41\}} (qrp);
\end{tikzpicture}
}"""

def generate_below_zero_chars():
    """Generate below-zero characterizations - simplified for expression grammar."""
    return r"""\tiny
\textbf{Below-Zero Characterizations}\\[0.2em]
\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.1cm and 1.3cm,
    state/.style={circle, draw, thick, minimum size=0.5cm, font=\tiny}]

% --- 'a' ---
\node[font=\tiny\bfseries] at (1.5, 2.5) {$\mathcal{C}_{\text{'a'}}$};
\node[anchor=south] at (1.5, 2.7) {
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{R} & \textbf{P} & \textbf{N} \\ \hline
    0 & 0 & S3 \\ \hline
    \end{tabular}
};
\node[state, fill=blue!15] (a0) at (0, 2) {0};
\node[state] (a1) at (2, 2) {S3};
\draw[->, thick, blue!70] (a0) -- node[above,font=\tiny] {R:0} (a1);

% --- 'b' ---
\node[font=\tiny\bfseries] at (5.5, 2.5) {$\mathcal{C}_{\text{'b'}}$};
\node[anchor=south] at (5.5, 2.7) {
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{R} & \textbf{P} & \textbf{N} \\ \hline
    0 & 0 & S4 \\ \hline
    \end{tabular}
};
\node[state, fill=blue!15] (b0) at (4, 2) {0};
\node[state] (b1) at (6, 2) {S4};
\draw[->, thick, red!70] (b0) -- node[above,font=\tiny] {R:0} (b1);

% --- '+' ---
\node[font=\tiny\bfseries] at (1.5, 0.5) {$\mathcal{C}_{\text{'+'}}$};
\node[anchor=south] at (1.5, 0.7) {
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{R} & \textbf{P} & \textbf{N} \\ \hline
    1 & 0 & S6 \\ \hline
    \end{tabular}
};
\node[state, fill=blue!15] (c0) at (0, 0) {0};
\node[state] (c1) at (2, 0) {S6};
\draw[->, thick, green!70] (c0) -- node[above,font=\tiny] {R:1} (c1);

% --- '(' ---
\node[font=\tiny\bfseries] at (5.5, 0.5) {$\mathcal{C}_{\text{'('}}$};
\node[anchor=south] at (5.5, 0.7) {
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{R} & \textbf{P} & \textbf{N} \\ \hline
    0 & 0 & S5 \\ \hline
    \end{tabular}
};
\node[state, fill=blue!15] (d0) at (4, 0) {0};
\node[state] (d1) at (6, 0) {S5};
\draw[->, thick, purple!70] (d0) -- node[above,font=\tiny] {R:0} (d1);

\end{tikzpicture}
}"""

def generate_template_dfas():
    """Generate template DFAs - state IDs for expression grammar."""
    return r"""\tiny
\textbf{Template DFAs}\\
\textbf{(state IDs, push transitions)}\\[0.2em]
\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.1cm and 1.3cm,
    state/.style={circle, draw, thick, minimum size=0.5cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.5cm, font=\tiny}]
% Template 'a'
\node[font=\tiny\bfseries] at (1.5, 1.5) {$\mathcal{T}_{\text{'a'}}$};
\node[state, fill=blue!15] (t0) at (0, 1) {0};
\node[accept] (t1) at (2, 1) {3};
\draw[->, thick, blue!70] (t0) -- node[above,font=\tiny] {S3} (t1);

% Template 'b'
\node[font=\tiny\bfseries] at (5.5, 1.5) {$\mathcal{T}_{\text{'b'}}$};
\node[state, fill=blue!15] (u0) at (4, 1) {0};
\node[accept] (u1) at (6, 1) {4};
\draw[->, thick, red!70] (u0) -- node[above,font=\tiny] {S4} (u1);

% Template '+'
\node[font=\tiny\bfseries] at (1.5, -0.5) {$\mathcal{T}_{\text{'+'}}$};
\node[state, fill=blue!15] (v0) at (0, -1) {0};
\node[accept] (v1) at (2, -1) {6};
\draw[->, thick, green!70] (v0) -- node[above,font=\tiny] {S6} (v1);

% Template '('
\node[font=\tiny\bfseries] at (5.5, -0.5) {$\mathcal{T}_{\text{'('}}$};
\node[state, fill=blue!15] (w0) at (4, -1) {0};
\node[accept] (w1) at (6, -1) {5};
\draw[->, thick, purple!70] (w0) -- node[above,font=\tiny] {S5} (w1);
\end{tikzpicture}
}"""

def generate_dwa_with_templates():
    """Terminal DWA with miniaturized template DFAs - simpler for expression grammar."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=2.5cm and 2cm,
    state/.style={circle, draw, thick, minimum size=0.7cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.7cm, font=\tiny},
    mini/.style={circle, draw, minimum size=0.2cm, font=\tiny, inner sep=0pt}]
\node[font=\bfseries\scriptsize] at (3, 2.5) {DWA with Template DFAs};
\node[font=\tiny] at (3, 2.15) {(miniaturized DFAs on edges)};
\node[state, fill=blue!15] (q0) at (0, 0) {0};
\node[accept] (q1) at (2.5, 1.5) {1};
\node[accept] (q2) at (5, 0) {2};
\node[accept] (q3) at (2.5, -1.5) {3};

% 0 -> 1 ('a')
\draw[->, thick, blue!70, opacity=0.3] (q0) to[bend left=10] (q1);
\node[mini, fill=blue!10] (m00) at (1.0, 1.0) {};
\node[mini] (m01) at (1.7, 1.3) {};
\draw[->, thin] (m00) -- (m01);
\node[font=\tiny, blue!70] at (0.5, 1.5) {$\mathcal{T}_{\text{'a'}}$};

% 0 -> 3 ('b')
\draw[->, thick, red!70, opacity=0.3] (q0) to[bend right=10] (q3);
\node[mini, fill=blue!10] (m10) at (1.0, -1.0) {};
\node[mini] (m11) at (1.7, -1.3) {};
\draw[->, thin] (m10) -- (m11);
\node[font=\tiny, red!70] at (0.5, -1.5) {$\mathcal{T}_{\text{'b'}}$};

% 1 -> 2 ('+')
\draw[->, thick, green!70, opacity=0.3] (q1) -- (q2);
\node[mini, fill=blue!10] (m20) at (3.5, 0.9) {};
\node[mini] (m21) at (4.2, 0.5) {};
\draw[->, thin] (m20) -- (m21);
\node[font=\tiny, green!70] at (3.7, 1.3) {$\mathcal{T}_{\text{'+'}}$};

\end{tikzpicture}
}"""

def generate_flattened_nwa():
    """Flattened NWA - simpler for expression grammar."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.3cm and 1.4cm,
    state/.style={circle, draw, thick, minimum size=0.55cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.55cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (3.5, 2.5) {Flattened NWA};
\node[font=\tiny] at (3.5, 2.15) {(epsilon transitions + weights)};
\node[state, fill=blue!15] (s0) at (0, 0) {0};
\node[accept] (s1) at (2.5, 1.5) {1};
\node[accept] (s2) at (5, 0) {2};
\node[accept] (s3) at (2.5, -1.5) {3};

% 0 -> 1
\draw[->, thick, blue!70] (s0) to[bend left=10] node[pos=0.5, above, font=\tiny] {$\varepsilon|\{65\}$} (s1);

% 0 -> 3  
\draw[->, thick, red!70] (s0) to[bend right=10] node[pos=0.5, below, font=\tiny] {$\varepsilon|\{66\}$} (s3);

% 1 -> 2
\draw[->, thick, green!70] (s1) -- node[above,font=\tiny] {$\varepsilon|\{43\}$} (s2);

% 3 -> 2
\draw[->, thick, green!70] (s3) -- node[below,font=\tiny] {$\varepsilon|\{43\}$} (s2);

\end{tikzpicture}
}"""

def generate_resolved_nwa():
    """NWA with push transitions resolved - simplified."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.3cm and 1.4cm,
    state/.style={circle, draw, thick, minimum size=0.55cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.55cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (3.5, 2.5) {Resolved NWA};
\node[font=\tiny] at (3.5, 2.15) {(push replaced with state edges)};
\node[state, fill=blue!15] (s0) at (0, 0) {0};
\node[accept] (s1) at (2.5, 1.5) {1};
\node[accept] (s2) at (5, 0) {2};
\node[accept] (s3) at (2.5, -1.5) {3};

% 0 -> 1
\draw[->, thick, blue!70] (s0) to[bend left=10] node[pos=0.5, above, font=\tiny] {$\varepsilon|\{65\}$} (s1);

% 0 -> 3
\draw[->, thick, red!70] (s0) to[bend right=10] node[pos=0.5, below, font=\tiny] {$\varepsilon|\{66\}$} (s3);

% 1 -> 2
\draw[->, thick, green!70] (s1) -- node[above,font=\tiny] {$\varepsilon|\{43\}$} (s2);

% 3 -> 2
\draw[->, thick, green!70] (s3) -- node[below,font=\tiny] {$\varepsilon|\{43\}$} (s2);

\end{tikzpicture}
}"""

def generate_final_dwa():
    """Final DWA after determinization - expression grammar."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.8cm and 1.8cm,
    state/.style={circle, draw, thick, minimum size=0.7cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.7cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (3, 2.2) {Final DWA};
\node[font=\tiny] at (3, 1.85) {(determinized \& simplified)};
\node[state, fill=blue!15] (q0) at (0, 0) {0};
\node[state] (q1) at (3, 1.2) {1};
\node[state] (q2) at (3, -1.2) {3};
\node[accept] (q3) at (6, 0) {2};

\draw[->, thick, blue!70] (q0) -- node[above,pos=0.4,font=\tiny] {'a'} (q1);
\draw[->, thick, red!70] (q0) -- node[below,pos=0.4,font=\tiny] {'b'} (q2);
\draw[->, thick, green!70] (q1) -- node[above,font=\tiny] {'+'} (q3);
\draw[->, thick, green!70] (q2) -- node[below,font=\tiny] {'+'} (q3);

\draw[->, thick, gray!70] (q3) to[loop right] node[right,font=\tiny] {$\star$} (q3);
\end{tikzpicture}
}"""


# ==============================================================================
# COMPONENT GENERATORS (for standalone files)
# ==============================================================================

# Map component names to their content generators
COMPONENT_GENERATORS = {
    'vocab': generate_vocab_content,
    'grammar': generate_grammar_content,
    'tokenizer_dfa': generate_tokenizer_dfa,
    'lalr_table': generate_lalr_table,
    'below_zero_chars': generate_below_zero_chars,
    'template_dfas': generate_template_dfas,
    'terminal_dwa': generate_terminal_dwa,
    'dwa_with_templates': generate_dwa_with_templates,
    'flattened_nwa': generate_flattened_nwa,
    'resolved_nwa': generate_resolved_nwa,
    'final_dwa': generate_final_dwa,
}

def generate_standalone_component(component_name: str, content_generator) -> str:
    """Generate a standalone LaTeX document for a component."""
    # For simple table components, wrap in a node
    content = content_generator()
    
    # Determine if content needs wrapping in a node (tables and text)
    needs_node_wrap = component_name in ['vocab', 'grammar', 'lalr_table']
    
    if needs_node_wrap:
        tikz_content = f"  \\node[font=\\bfseries] {{\n{content}\n  }};"
    else:
        # Content is already a tikzpicture or needs minimal wrapping
        tikz_content = content
    
    return f"""\\documentclass[tikz,border=10pt]{{standalone}}
\\input{{shared_styles.tex}}

\\begin{{document}}
\\begin{{tikzpicture}}
{tikz_content}
\\end{{tikzpicture}}
\\end{{document}}
"""

# ==============================================================================
# LATEX GENERATION - STRUCTURE WITH CONTENT
# ==============================================================================

def generate_latex_structure():
    """Generate the complete structure with content."""
    
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
        r"\newlength{\boxwidth}",
        r"\setlength{\boxwidth}{6cm}",
        r"\newlength{\boxheight}",
        r"\setlength{\boxheight}{4.5cm}",
        r"% Text area is slightly smaller than box to allow padding",
        r"\newlength{\textw}",
        r"\setlength{\textw}{5.6cm}",
        r"\newlength{\texth}",
        r"\setlength{\texth}{4.1cm}",
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
        f"  \\node[box] (tokenizer) {{{generate_tokenizer_dfa()}}};",
        f"  \\node[box, right=\\colgap of tokenizer] (parsetable) {{{generate_lalr_table()}}};",
        "",
        r"  % ====================================================================",
        r"  % ROW 1: Inputs (High above)",
        r"  % ====================================================================",
        r"  ",
        r"  % Calculate center axis of the graph",
        r"  \coordinate (center_axis) at ($(tokenizer.east)!0.5!(parsetable.west)$);",
        r"  ",
        r"  % Place Grammar HIGH above the center axis",
        f"  \\node[box, above=\\toprowgap of center_axis, anchor=south] (grammar) {{{generate_grammar_content()}}};",
        r"  ",
        r"  % Place Vocab to the left",
        f"  \\node[box, left=\\colgap of grammar] (vocab) {{{generate_vocab_content()}}};",
        "",
        r"  % ====================================================================",
        r"  % ROW 3 & 4: Processing (Split)",
        r"  % ====================================================================",
        r"  ",
        r"  % Characterizations below Parse Table",
        f"  \\node[box, below=\\midrowgap of parsetable] (characterizations) {{{generate_below_zero_chars()}}};",
        r"  ",
        r"  % Template DFAs below Characterizations",
        f"  \\node[box, below=2cm of characterizations] (template_dfas) {{{generate_template_dfas()}}};",
        r"  ",
        r"  % Terminal DWA below Tokenizer, vertically aligned with Template DFAs",
        f"  \\node[box] (terminal_dwa) at (tokenizer |- template_dfas) {{{generate_terminal_dwa()}}};",
        "",
        r"  % ====================================================================",
        r"  % CENTRAL COLUMN: Merged Flow",
        r"  % ====================================================================",
        r"  ",
        r"  % Determine the Y-level for the bracket merge",
        r"  % We look at where the bottom of the previous row is, and go halfway to the next node",
        r"  \path (template_dfas.south) -- ++(0, -2cm) coordinate (next_node_top);",
        r"  \coordinate (merge_bar_y) at ($(template_dfas.south)!0.5!(next_node_top)$);",
        r"  ",
        r"  % Center column nodes",
        f"  \\node[box] (dwa_with_templates) at (center_axis |- next_node_top) [anchor=north] {{{generate_dwa_with_templates()}}};",
        r"  ",
        f"  \\node[box, below=2cm of dwa_with_templates] (flattened_nwa) {{{generate_flattened_nwa()}}};",
        f"  \\node[box, below=2cm of flattened_nwa] (resolved_nwa) {{{generate_resolved_nwa()}}};",
        f"  \\node[box, below=2cm of resolved_nwa] (final_dwa) {{{generate_final_dwa()}}};",
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
        r"  % --- 3. Vertical Connectors ---",
        r"  \draw[flowarrow] (tokenizer) -- (terminal_dwa);",
        r"  \draw[flowarrow] (parsetable) -- (characterizations);",
        r"  \draw[flowarrow] (characterizations) -- (template_dfas);",
        r"  ",
        r"  % --- 4. The Bracket Merge (Row 4 -> Central) ---",
        r"  % Left Arm",
        r"  \draw[connector] (terminal_dwa.south) -- (terminal_dwa.south |- merge_bar_y);",
        r"  % Right Arm",
        r"  \draw[connector] (template_dfas.south) -- (template_dfas.south |- merge_bar_y);",
        r"  % Crossbar",
        r"  \draw[connector] (terminal_dwa.south |- merge_bar_y) -- (template_dfas.south |- merge_bar_y);",
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
    parser = argparse.ArgumentParser(
        description='Generate TikZ visualizations for the grammar pipeline'
    )
    parser.add_argument(
        '--component',
        type=str,
        choices=list(COMPONENT_GENERATORS.keys()),
        help='Generate a specific component as a standalone file'
    )
    parser.add_argument(
        '--all-components',
        action='store_true',
        help='Generate all components as standalone files'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        default=False,
        help='Generate the full pipeline (default if no other option specified)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='gcg-paper/paper/figures/components',
        help='Output directory for component files'
    )
    
    args = parser.parse_args()
    
    # Default to full pipeline if no specific option
    if not (args.component or args.all_components or args.full):
        args.full = True
    
    components_dir = Path(args.output_dir)
    components_dir.mkdir(parents=True, exist_ok=True)
    
    if args.component:
        # Generate single component
        content = generate_standalone_component(
            args.component,
            COMPONENT_GENERATORS[args.component]
        )
        output_file = components_dir / f"{args.component}.tex"
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"Generated {output_file}")
    
    elif args.all_components:
        # Generate all components
        for name, generator in COMPONENT_GENERATORS.items():
            content = generate_standalone_component(name, generator)
            output_file = components_dir / f"{name}.tex"
            with open(output_file, 'w') as f:
                f.write(content)
            print(f"Generated {output_file}")
        print(f"\n✓ Generated {len(COMPONENT_GENERATORS)} component files in {components_dir}")
    
    if args.full:
        # Generate full pipeline
        latex_content = generate_latex_structure()
        filename = "gcg-paper/paper/figures/pipeline_full.tex"
        with open(filename, "w") as f:
            f.write(latex_content)
        print(f"Generated {filename} with complete content in all boxes")

if __name__ == "__main__":
    main()


# ==============================================================================
# CONTENT GENERATORS
# ==============================================================================

def generate_vocab_content():
    """Generate LLM Vocab visualization."""
    return r"""\tiny
\textbf{LLM Vocab}\\
\textbf{(50k tokens)}\\[0.2em]
\begin{tabular}{|c|c|}
\hline
\textbf{ID} & \textbf{Token} \\ \hline
0 & \texttt{<unk>} \\ \hline
... & ... \\ \hline
65 & \texttt{'a'} \\ \hline
66 & \texttt{'b'} \\ \hline
67 & \texttt{'c'} \\ \hline
68 & \texttt{'d'} \\ \hline
69 & \texttt{'z'} \\ \hline
... & ... \\ \hline
\end{tabular}"""

def generate_grammar_content():
    """Generate Input Grammar - Recursive Cyclic Example."""
    return r"""\scriptsize
\textbf{Input Grammar}\\
\textbf{(Recursive)}\\[0.4em]
$\begin{array}{rcl}
\texttt{S} &\to& \texttt{A }\texttt{'\$'}\\
\texttt{A} &\to& \texttt{'a' B 'z'}\\
\texttt{B} &\to& \texttt{'b' C}\\
\texttt{C} &\to& \texttt{'c' A}\\
\texttt{C} &\to& \texttt{'d'}
\end{array}$"""


def generate_tokenizer_dfa():
    """Generate Tokenizer DFA - Recursive Grammar Tokens."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.5cm,
    state/.style={circle, draw, thick, minimum size=0.5cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.5cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (2, 1.5) {Tokenizer DFA};
\node[state, initial, initial text=] (q0) at (0, 0) {0};
\node[accept] (qa) at (2, 1.2) {a};
\node[accept] (qb) at (2, 0.6) {b};
\node[accept] (qc) at (2, 0) {c};
\node[accept] (qd) at (2, -0.6) {d};
\node[accept] (qz) at (2, -1.2) {z};

\draw[->, thick] (q0) -- node[above,font=\tiny] {'a'} (qa);
\draw[->, thick] (q0) -- node[above,pos=0.7,font=\tiny] {'b'} (qb);
\draw[->, thick] (q0) -- node[above,pos=0.7,font=\tiny] {'c'} (qc);
\draw[->, thick] (q0) -- node[below,pos=0.7,font=\tiny] {'d'} (qd);
\draw[->, thick] (q0) -- node[below,font=\tiny] {'z'} (qz);
\end{tikzpicture}
}"""

def generate_lalr_table():
    """Generate LALR Parse Table - Recursive Grammar."""
    return r"""\tiny
\textbf{LALR Parse Table}\\[0.2em]
\begin{tabular}{|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{St} & \textbf{a} & \textbf{b} & \textbf{c} & \textbf{d} & \textbf{z} & \textbf{\$} & \textbf{A} & \textbf{B} \\ \hline
0 & S1 & & & & & & G5 & \\ \hline
1 & & S2 & & & & & & G6 \\ \hline
2 & & & S3 & S4 & & & & \\ \hline
3 & S1 & & & & & & G7 & \\ \hline
4 & R(C) & R(C) & R(C) & R(C) & R(C) & R(C) & & \\ \hline
5 & & & & & & Acc & & \\ \hline
6 & & & & & S8 & & & \\ \hline
7 & R(C) & & & & R(C) & & & \\ \hline
8 & R(A) & & & & R(A) & & & \\ \hline
\end{tabular}"""

def generate_terminal_dwa():
    """Generate Terminal DWA - Cyclic Grammar Tokens."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.4cm and 1.6cm,
    state/.style={circle, draw, thick, minimum size=0.65cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.65cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (3, 1.8) {Terminal DWA};
\node[font=\tiny] at (3, 1.4) {(with LLM token masks)};
\node[state, fill=blue!15] (q0) at (0, 0) {0};
\node[accept] (qa) at (3, 1.2) {'a'};
\node[accept] (qb) at (3, 0.4) {'b'};
\node[accept] (qc) at (3, -0.4) {'c'};
\node[accept] (qd) at (3, -1.2) {'d'};

\draw[->, thick, blue!70] (q0) -- node[above,font=\tiny] {\textbf{'a'}$|$\{65\}} (qa);
\draw[->, thick, red!70] (q0) -- node[above,font=\tiny] {\textbf{'b'}$|$\{66\}} (qb);
\draw[->, thick, green!70] (q0) -- node[above,font=\tiny] {\textbf{'c'}$|$\{67\}} (qc);
\draw[->, thick, orange!70] (q0) -- node[above,font=\tiny] {\textbf{'d'}$|$\{68\}} (qd);
\end{tikzpicture}
}"""

def generate_below_zero_chars():
    """Generate below-zero characterizations with textual tables."""
    return r"""\tiny
\textbf{Below-Zero Characterizations}\\[0.2em]
\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.1cm and 1.3cm,
    state/.style={circle, draw, thick, minimum size=0.5cm, font=\tiny}]

% --- 'a' (Expects B) ---
\node[font=\tiny\bfseries] at (1.5, 2.5) {$\mathcal{C}_{\text{'a'}}$};
\node[anchor=south] at (1.5, 2.7) {
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{R} & \textbf{P} & \textbf{N} \\ \hline
    0 & 0 & S1 \\ \hline
    \end{tabular}
};
\node[state, fill=blue!15] (a0) at (0, 2) {0};
\node[state] (a1) at (2, 2) {S1};
\draw[->, thick, blue!70] (a0) -- node[above,font=\tiny] {R:0} (a1);

% --- 'b' (Expects C) ---
\node[font=\tiny\bfseries] at (5.5, 2.5) {$\mathcal{C}_{\text{'b'}}$};
\node[anchor=south] at (5.5, 2.7) {
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{R} & \textbf{P} & \textbf{N} \\ \hline
    1 & 0 & S2 \\ \hline
    \end{tabular}
};
\node[state, fill=blue!15] (b0) at (4, 2) {0};
\node[state] (b1) at (6, 2) {S2};
\draw[->, thick, red!70] (b0) -- node[above,font=\tiny] {R:1} (b1);

% --- 'c' (Expects A) ---
\node[font=\tiny\bfseries] at (1.5, 0.5) {$\mathcal{C}_{\text{'c'}}$};
\node[anchor=south] at (1.5, 0.7) {
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{R} & \textbf{P} & \textbf{N} \\ \hline
    2 & 0 & S3 \\ \hline
    \end{tabular}
};
\node[state, fill=blue!15] (c0) at (0, 0) {0};
\node[state] (c1) at (2, 0) {S3};
\draw[->, thick, green!70] (c0) -- node[above,font=\tiny] {R:2} (c1);

% --- 'd' (Reduces C) ---
\node[font=\tiny\bfseries] at (5.5, 0.5) {$\mathcal{C}_{\text{'d'}}$};
\node[anchor=south] at (5.5, 0.7) {
    \begin{tabular}{|l|l|l|}
    \hline
    \textbf{R} & \textbf{P} & \textbf{N} \\ \hline
    2 & 0 & Esc \\ \hline
    \end{tabular}
};
\node[state, fill=blue!15] (d0) at (4, 0) {0};
\node[state] (d1) at (6, 0) {Esc};
\draw[->, dashed, thick, orange!70] (d0) -- node[above,font=\tiny] {R:2,S:C} (d1);

\end{tikzpicture}
}"""

def generate_template_dfas():
    """Generate template DFAs - state IDs only."""
    return r"""\tiny
\textbf{Template DFAs}\\
\textbf{(state IDs, push transitions)}\\[0.2em]
\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.1cm and 1.3cm,
    state/.style={circle, draw, thick, minimum size=0.5cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.5cm, font=\tiny}]
% Template 'a'
\node[font=\tiny\bfseries] at (1.5, 1.5) {$\mathcal{T}_{\text{'a'}}$};
\node[state, fill=blue!15] (t0) at (0, 1) {0};
\node[accept] (t1) at (2, 1) {1};
\draw[->, thick, blue!70] (t0) -- node[above,font=\tiny] {S1} (t1);

% Template 'b'
\node[font=\tiny\bfseries] at (5.5, 1.5) {$\mathcal{T}_{\text{'b'}}$};
\node[state, fill=blue!15] (u0) at (4, 1) {0};
\node[accept] (u1) at (6, 1) {2};
\draw[->, thick, red!70] (u0) -- node[above,font=\tiny] {S2} (u1);

% Template 'c'
\node[font=\tiny\bfseries] at (1.5, -0.5) {$\mathcal{T}_{\text{'c'}}$};
\node[state, fill=blue!15] (v0) at (0, -1) {0};
\node[accept] (v1) at (2, -1) {3};
\draw[->, thick, green!70] (v0) -- node[above,font=\tiny] {S3} (v1);

% Template 'd'
\node[font=\tiny\bfseries] at (5.5, -0.5) {$\mathcal{T}_{\text{'d'}}$};
\node[state, fill=blue!15] (w0) at (4, -1) {0};
\node[accept] (w1) at (6, -1) {4};
\draw[->, thick, orange!70] (w0) -- node[above,font=\tiny] {Esc} (w1);
\end{tikzpicture}
}"""

def generate_dwa_with_templates():
    """Terminal DWA with miniaturized template DFAs on edges."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=2.5cm and 2cm,
    state/.style={circle, draw, thick, minimum size=0.7cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.7cm, font=\tiny},
    mini/.style={circle, draw, minimum size=0.2cm, font=\tiny, inner sep=0pt}]
\node[font=\bfseries\scriptsize] at (3, 2.5) {DWA with Template DFAs};
\node[font=\tiny] at (3, 2.15) {(miniaturized DFAs on edges)};
\node[state, fill=blue!15] (q0) at (0, 0) {0};
\node[accept] (q1) at (2, 1.5) {1};
\node[accept] (q2) at (4, 0) {2};
\node[accept] (q3) at (2, -1.5) {3};
\node[accept] (q4) at (6, 0) {4};

% 0 -> 1 ('a')
\draw[->, thick, blue!70, opacity=0.3] (q0) to[bend left=10] (q1);
\node[mini, fill=blue!10] (m00) at (0.8, 1.0) {};
\node[mini] (m01) at (1.4, 1.3) {};
\draw[->, thin] (m00) -- (m01);
\node[font=\tiny, blue!70] at (0.5, 1.5) {$\mathcal{T}_{\text{'a'}}$};

% 1 -> 2 ('b')
\draw[->, thick, red!70, opacity=0.3] (q1) -- (q2);
\node[mini, fill=blue!10] (m10) at (2.8, 0.9) {};
\node[mini] (m11) at (3.4, 0.5) {};
\draw[->, thin] (m10) -- (m11);
\node[font=\tiny, red!70] at (3.2, 1.2) {$\mathcal{T}_{\text{'b'}}$};

% 2 -> 3 ('c')
\draw[->, thick, green!70, opacity=0.3] (q2) -- (q3);
\node[mini, fill=blue!10] (m20) at (3.4, -0.5) {};
\node[mini] (m21) at (2.8, -0.9) {};
\draw[->, thin] (m20) -- (m21);
\node[font=\tiny, green!70] at (3.5, -1.2) {$\mathcal{T}_{\text{'c'}}$};

% 3 -> 1 ('a' - Cycle)
\draw[->, thick, blue!70, opacity=0.3] (q3) -- (q1);
\node[mini, fill=blue!10] (m30) at (1.8, -0.5) {};
\node[mini] (m31) at (1.8, 0.5) {};
\draw[->, thin] (m30) -- (m31);
\node[font=\tiny, blue!70] at (1.4, 0) {$\mathcal{T}_{\text{'a'}}$};

% 2 -> 4 ('d' - Exit)
\draw[->, thick, orange!70, opacity=0.3] (q2) -- (q4);
\node[mini, fill=blue!10] (m40) at (4.8, 0.2) {};
\node[mini] (m41) at (5.4, 0.2) {};
\draw[->, thin] (m40) -- (m41);
\node[font=\tiny, orange!70] at (5, 0.5) {$\mathcal{T}_{\text{'d'}}$};

\end{tikzpicture}
}"""

def generate_flattened_nwa():
    """Flattened NWA after epsilon-linking template DFAs."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.3cm and 1.4cm,
    state/.style={circle, draw, thick, minimum size=0.55cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.55cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (3.5, 2.5) {Flattened NWA};
\node[font=\tiny] at (3.5, 2.15) {(epsilon transitions + weights)};
\node[state, fill=blue!15] (s0) at (0, 0) {0};
\node[accept] (s1) at (2, 1.5) {1};
\node[accept] (s2) at (4, 0) {2};
\node[accept] (s3) at (2, -1.5) {3};
\node[accept] (s4) at (6, 0) {4};

% 0 -> 1
\draw[->, thick, blue!70] (s0) to[bend left=10] node[pos=0.5, above, font=\tiny] {$\varepsilon|\{65\}$} (s1);

% 1 -> 2
\draw[->, thick, red!70] (s1) -- node[above,font=\tiny] {$\varepsilon|\{66\}$} (s2);

% 2 -> 3
\draw[->, thick, green!70] (s2) -- node[below,font=\tiny] {$\varepsilon|\{67\}$} (s3);

% 3 -> 1 (Cycle)
\draw[->, thick, blue!70] (s3) -- node[left,font=\tiny] {$\varepsilon|\{65\}$} (s1);

% 2 -> 4 (Exit)
\draw[->, thick, orange!70] (s2) -- node[above,font=\tiny] {$\varepsilon|\{68\}$} (s4);

\end{tikzpicture}
}"""

def generate_resolved_nwa():
    """NWA with push transitions resolved to normal transitions."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.3cm and 1.4cm,
    state/.style={circle, draw, thick, minimum size=0.55cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.55cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (3.5, 2.5) {Resolved NWA};
\node[font=\tiny] at (3.5, 2.15) {(push replaced with state edges)};
\node[state, fill=blue!15] (s0) at (0, 0) {0};
\node[accept] (s1) at (2, 1.5) {1};
\node[accept] (s2) at (4, 0) {2};
\node[accept] (s3) at (2, -1.5) {3};
\node[accept] (s4) at (6, 0) {4};

% 0 -> 1
\draw[->, thick, blue!70] (s0) to[bend left=10] node[pos=0.5, above, font=\tiny] {$\varepsilon|\{65\}$} (s1);

% 1 -> 2
\draw[->, thick, red!70] (s1) -- node[above,font=\tiny] {$\varepsilon|\{66\}$} (s2);

% 2 -> 3
\draw[->, thick, green!70] (s2) -- node[below,font=\tiny] {$\varepsilon|\{67\}$} (s3);

% 3 -> 1 (Cycle)
\draw[->, thick, blue!70] (s3) -- node[left,font=\tiny] {$\varepsilon|\{65\}$} (s1);

% 2 -> 4 (Exit)
\draw[->, thick, orange!70] (s2) -- node[above,font=\tiny] {$\varepsilon|\{68\}$} (s4);
\end{tikzpicture}
}"""

def generate_final_dwa():
    """Final DWA after determinization and simplification."""
    return r"""\resizebox{0.95\textw}{!}{
\begin{tikzpicture}[>=Stealth,
    node distance=1.8cm and 1.8cm,
    state/.style={circle, draw, thick, minimum size=0.7cm, font=\tiny},
    accept/.style={circle, draw, double, thick, minimum size=0.7cm, font=\tiny}]
\node[font=\bfseries\scriptsize] at (3.5, 2.2) {Final DWA};
\node[font=\tiny] at (3.5, 1.85) {(determinized \& simplified)};
\node[state, fill=blue!15] (q0) at (0, 0) {0};
\node[state] (q1) at (2, 1.5) {1};
\node[state] (q2) at (4, 0) {2};
\node[state] (q3) at (2, -1.5) {3};
\node[accept] (q4) at (6, 0) {4};

\draw[->, thick, blue!70] (q0) to[bend left=10] node[above,font=\tiny] {'a'} (q1);
\draw[->, thick, red!70] (q1) -- node[above,font=\tiny] {'b'} (q2);
\draw[->, thick, green!70] (q2) -- node[below,font=\tiny] {'c'} (q3);
\draw[->, thick, blue!70] (q3) -- node[left,font=\tiny] {'a'} (q1);
\draw[->, thick, orange!70] (q2) -- node[above,font=\tiny] {'d'} (q4);

\draw[->, thick, gray!70] (q4) to[loop right] node[right,font=\tiny] {$\star$} (q4);
\end{tikzpicture}
}"""

# ==============================================================================
# LATEX GENERATION - STRUCTURE WITH CONTENT
# ==============================================================================

def generate_latex_structure():
    """Generate the complete structure with content."""
    
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
        r"\newlength{\boxwidth}",
        r"\setlength{\boxwidth}{6cm}",
        r"\newlength{\boxheight}",
        r"\setlength{\boxheight}{4.5cm}",
        r"% Text area is slightly smaller than box to allow padding",
        r"\newlength{\textw}",
        r"\setlength{\textw}{5.6cm}",
        r"\newlength{\texth}",
        r"\setlength{\texth}{4.1cm}",
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
        f"  \\node[box] (tokenizer) {{{generate_tokenizer_dfa()}}};",
        f"  \\node[box, right=\\colgap of tokenizer] (parsetable) {{{generate_lalr_table()}}};",
        "",
        r"  % ====================================================================",
        r"  % ROW 1: Inputs (High above)",
        r"  % ====================================================================",
        r"  ",
        r"  % Calculate center axis of the graph",
        r"  \coordinate (center_axis) at ($(tokenizer.east)!0.5!(parsetable.west)$);",
        r"  ",
        r"  % Place Grammar HIGH above the center axis",
        f"  \\node[box, above=\\toprowgap of center_axis, anchor=south] (grammar) {{{generate_grammar_content()}}};",
        r"  ",
        r"  % Place Vocab to the left",
        f"  \\node[box, left=\\colgap of grammar] (vocab) {{{generate_vocab_content()}}};",
        "",
        r"  % ====================================================================",
        r"  % ROW 3 & 4: Processing (Split)",
        r"  % ====================================================================",
        r"  ",
        r"  % Characterizations below Parse Table",
        f"  \\node[box, below=\\midrowgap of parsetable] (characterizations) {{{generate_below_zero_chars()}}};",
        r"  ",
        r"  % Template DFAs below Characterizations",
        f"  \\node[box, below=2cm of characterizations] (template_dfas) {{{generate_template_dfas()}}};",
        r"  ",
        r"  % Terminal DWA below Tokenizer, vertically aligned with Template DFAs",
        f"  \\node[box] (terminal_dwa) at (tokenizer |- template_dfas) {{{generate_terminal_dwa()}}};",
        "",
        r"  % ====================================================================",
        r"  % CENTRAL COLUMN: Merged Flow",
        r"  % ====================================================================",
        r"  ",
        r"  % Determine the Y-level for the bracket merge",
        r"  % We look at where the bottom of the previous row is, and go halfway to the next node",
        r"  \path (template_dfas.south) -- ++(0, -2cm) coordinate (next_node_top);",
        r"  \coordinate (merge_bar_y) at ($(template_dfas.south)!0.5!(next_node_top)$);",
        r"  ",
        r"  % Center column nodes",
        f"  \\node[box] (dwa_with_templates) at (center_axis |- next_node_top) [anchor=north] {{{generate_dwa_with_templates()}}};",
        r"  ",
        f"  \\node[box, below=2cm of dwa_with_templates] (flattened_nwa) {{{generate_flattened_nwa()}}};",
        f"  \\node[box, below=2cm of flattened_nwa] (resolved_nwa) {{{generate_resolved_nwa()}}};",
        f"  \\node[box, below=2cm of resolved_nwa] (final_dwa) {{{generate_final_dwa()}}};",
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
        r"  % --- 3. Vertical Connectors ---",
        r"  \draw[flowarrow] (tokenizer) -- (terminal_dwa);",
        r"  \draw[flowarrow] (parsetable) -- (characterizations);",
        r"  \draw[flowarrow] (characterizations) -- (template_dfas);",
        r"  ",
        r"  % --- 4. The Bracket Merge (Row 4 -> Central) ---",
        r"  % Left Arm",
        r"  \draw[connector] (terminal_dwa.south) -- (terminal_dwa.south |- merge_bar_y);",
        r"  % Right Arm",
        r"  \draw[connector] (template_dfas.south) -- (template_dfas.south |- merge_bar_y);",
        r"  % Crossbar",
        r"  \draw[connector] (terminal_dwa.south |- merge_bar_y) -- (template_dfas.south |- merge_bar_y);",
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
    
    filename = "gcg-paper/paper/figures/pipeline_full.tex"
    with open(filename, "w") as f:
        f.write(latex_content)
    
    print(f"Generated {filename} with complete content in all boxes")

if __name__ == "__main__":
    main()