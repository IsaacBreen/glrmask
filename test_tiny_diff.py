import time, sys, os
sys.path.insert(0, 'scripts')
from test_diff import generate_diff_grammar
import _sep1

# Use a TINY source file to test timing
tiny_src = '/tmp/tiny_test.rs'
with open(tiny_src, 'w') as f:
    f.write('fn main() {\n    println!("hello");\n}\n')

print('=== TINY FILE (3 lines) ===', flush=True)
t0 = time.time()
ebnf = generate_diff_grammar(tiny_src)
lines = ebnf.split('\n')
print(f'Grammar: {len(lines)} lines, gen={time.time()-t0:.3f}s', flush=True)

t0 = time.time()
grammar_def = _sep1.grammar_definition_from_ebnf(ebnf)
print(f'EBNF parse: {time.time()-t0:.3f}s', flush=True)

# Skip optimize
t0 = time.time()
compiled = grammar_def.compile()
print(f'Compile (no opt): {time.time()-t0:.3f}s', flush=True)

print('Done', flush=True)
