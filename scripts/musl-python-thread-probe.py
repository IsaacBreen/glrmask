"""Diagnostic only: replay the existing independent-template worker test."""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import glrmask

requested = int(sys.argv[1])
if requested:
    threading.stack_size(requested)
print('requested_stack_bytes', requested, flush=True)
v = glrmask.Vocab.from_id_to_bytes({0: b'a', 1: b'b', 2: b'ab', 17: b'', 29: b''})
template = glrmask.Grammar.from_glrm('extern token M start r nt r=M').prepare(v)

def worker(token):
    print('worker_enter', token, flush=True)
    request = template.clone()
    print('worker_clone_done', token, flush=True)
    request.bind('M', glrmask.Token(token))
    print('worker_bind_done', token, flush=True)
    compiled = request.link(optimization=glrmask.Optimization.FAST_BUILD)
    print('worker_link_done', token, flush=True)
    result = compiled.start().forced_tokens()
    print('worker_force_done', token, result, flush=True)
    return result, token

with ThreadPoolExecutor(max_workers=8) as workers:
    results = list(workers.map(worker, [17, 29] * 16))
assert all(result == [token] for result, token in results), results
assert [requirement.name for requirement in template.requirements] == ['M']
print('independent-template-workers PASS', flush=True)
