#!/usr/bin/env python3
"""Benchmark mask generation throughput for JSON schema constraint."""
import sys, time, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

schema_file = os.environ.get("SCHEMA_FILE", "gcg-paper/hard_schemas/schemas/SchemaStore_extra/apollo-router-2.9.0.json")
with open(schema_file) as f:
    schema = json.load(f)

import _sep1, tiktoken

enc = tiktoken.get_encoding("gpt2")
token_to_id = {}
for tid in range(enc.n_vocab):
    token_to_id[enc.decode_single_token_bytes(tid)] = tid

# Compile
t0 = time.time()
grammar = _sep1.grammar_definition_from_json_schema(json.dumps(schema))
compiled = grammar.compile()
constraint = _sep1.GrammarConstraint(compiled, token_to_id)
compile_time = time.time() - t0
print(f"Compile: {compile_time:.3f}s")

# Benchmark: step through a valid JSON input, measuring mask computation
test_json = b'{"experimental_chaos":{"enabled":true},"listen":{"host":"0.0.0.0","port":4000}}'

# Run multiple iterations for stable timing
NUM_ITERS = 20
total_time = 0
for _ in range(NUM_ITERS):
    state = _sep1.GrammarConstraintState(constraint)
    for byte in test_json:
        char_bytes = bytes([byte])
        if char_bytes in token_to_id:
            tid = token_to_id[char_bytes]
            t0 = time.perf_counter_ns()
            mask = state.get_mask()
            total_time += time.perf_counter_ns() - t0
            if mask.size > 0 and mask[tid]:
                state.commit(tid)

num_steps = 79 * NUM_ITERS  # 79 steps per iteration
print(f"Steps: {num_steps} ({NUM_ITERS} iterations)")
print(f"Total mask time: {total_time/1e6:.1f}ms ({total_time/1e3/num_steps:.1f}us/step)")

# First mask benchmark
first_times = []
for _ in range(100):
    s = _sep1.GrammarConstraintState(constraint)
    t0 = time.perf_counter_ns()
    m = s.get_mask()
    first_times.append(time.perf_counter_ns() - t0)
avg_first = sum(first_times) / len(first_times)
print(f"First mask: {avg_first/1e3:.1f}us avg ({min(first_times)/1e3:.1f}us min)")
