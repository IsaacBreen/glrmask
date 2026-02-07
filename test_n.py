#!/usr/bin/env python3
"""Test GLR parser timing at specific n."""
import _sep1, time, sys
sys.path.insert(0, 'scripts')
from test_diff import generate_diff_grammar

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
lines = open('testdata/finite_automata.rs').readlines()[:n]
path = f'/tmp/test_{n}lines.rs'
with open(path, 'w') as f:
    f.writelines(lines)

ebnf = generate_diff_grammar(path)
num_lines = ebnf.count('\n')
print(f"n={n}, grammar={num_lines} lines", flush=True)

t0 = time.time()
gd = _sep1.grammar_definition_from_ebnf(ebnf)
t1 = time.time()
print(f"Parse: {t1-t0:.3f}s", flush=True)

compiled = gd.compile()
t2 = time.time()
print(f"Compile: {t2-t1:.3f}s", flush=True)
print(f"Total: {t2-t0:.3f}s", flush=True)
