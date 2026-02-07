import time, sys, os
sys.path.insert(0, 'scripts')
from test_diff import generate_diff_grammar
import _sep1

# Generate grammars of varying size and time JUST the EBNF text parsing (not GrammarDefinition construction)
source = 'testdata/finite_automata.rs'
with open(source) as f:
    all_lines = f.readlines()

total = len(all_lines)
print(f'Source file has {total} lines', flush=True)

for n in [10, 50, 100, 200, 500]:
    tmp = f'/tmp/test_{n}lines.rs'
    with open(tmp, 'w') as f:
        f.writelines(all_lines[:n])
    
    ebnf = generate_diff_grammar(tmp)
    nlines = len(ebnf.split('\n'))
    nchars = len(ebnf)
    
    # Time the EBNF -> GrammarDefinition conversion
    # This calls from_ebnf which does:
    # 1. EbnfParser::new(source).parse() - tokenize + parse
    # 2. from_parsed_rules() -> from_exprs_impl()
    t0 = time.time()
    grammar_def = _sep1.grammar_definition_from_ebnf(ebnf)
    parse_t = time.time() - t0
    
    print(f'n={n:5d} → grammar={nlines:6d}L ({nchars:8d} chars)  parse={parse_t:.3f}s', flush=True)

print('Done', flush=True)
