import time, sys, os
os.environ['MACRO_DEBUG_LEVEL'] = '3'
import _sep1
import tiktoken

sys.path.insert(0, 'scripts')
from test_diff import generate_diff_grammar

source_file = 'testdata/finite_automata.rs'
print('Generating grammar...', flush=True)
t0 = time.time()
ebnf = generate_diff_grammar(source_file)
print(f'Grammar gen: {time.time()-t0:.3f}s ({len(ebnf.splitlines())} lines)', flush=True)

enc = tiktoken.get_encoding('gpt2')
token_to_id = {}
for tid in range(enc.n_vocab):
    token_to_id[enc.decode_single_token_bytes(tid)] = tid
print(f'Vocab loaded: {len(token_to_id)} tokens', flush=True)

print('Parsing EBNF...', flush=True)
t0 = time.time()
grammar_def = _sep1.grammar_definition_from_ebnf(ebnf)
print(f'EBNF parse: {time.time()-t0:.3f}s', flush=True)

print('SKIPPING optimize() !!!', flush=True)

print('Compiling (no optimize)...', flush=True)
t0 = time.time()
compiled = grammar_def.compile()
print(f'Compile: {time.time()-t0:.3f}s', flush=True)

print('Creating constraint...', flush=True)
t0 = time.time()
constraint = _sep1.GrammarConstraint(compiled, token_to_id)
print(f'Constraint: {time.time()-t0:.3f}s', flush=True)

print('SUCCESS', flush=True)
