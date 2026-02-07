import time, sys, os
sys.path.insert(0, 'scripts')
from test_diff import generate_diff_grammar
import _sep1

source = 'testdata/finite_automata.rs'
with open(source) as f:
    all_lines = f.readlines()

total = len(all_lines)
print(f'Source file has {total} lines', flush=True)

for n in [10, 50, 100, 200, 500, 1000, 2000, 3482]:
    if n > total:
        break
    tmp = f'/tmp/test_{n}lines.rs'
    with open(tmp, 'w') as f:
        f.writelines(all_lines[:n])
    
    t0 = time.time()
    ebnf = generate_diff_grammar(tmp)
    gen_t = time.time() - t0
    nlines = len(ebnf.split('\n'))
    
    t0 = time.time()
    grammar_def = _sep1.grammar_definition_from_ebnf(ebnf)
    parse_t = time.time() - t0
    
    t0 = time.time()
    compiled = grammar_def.compile()
    compile_t = time.time() - t0
    
    print(f'n={n:5d} → grammar={nlines:6d}L  gen={gen_t:.3f}s  parse={parse_t:.3f}s  compile={compile_t:.3f}s', flush=True)

print('Done', flush=True)
