import re

with open('test_output.log', 'r') as f:
    text = f.read()

def get_time(pattern):
    m = re.search(pattern, text)
    if m:
        t_str = m.group(1)
        if t_str.endswith('ms'): return float(t_str[:-2])
        if t_str.endswith('s'): return float(t_str[:-1]) * 1000
        if t_str.endswith('µs'): return float(t_str[:-2]) / 1000
    return 0.0

total = get_time(r'TIMING: parser_dwa::total .*?([0-9.]+m?s|µ?s)')
setup = get_time(r'PHASE_TIMING: precompute1::setup = ([0-9.]+m?s|µ?s)')
dfs = get_time(r'PHASE_TIMING: precompute1::dfs = ([0-9.]+m?s|µ?s)')
finish = get_time(r'PHASE_TIMING: precompute1::finish = ([0-9.]+m?s|µ?s)')
parser_dwa_total = get_time(r'PHASE_TIMING: parser_dwa::total = ([0-9.]+m?s|µ?s)')
build_parser_dwa = get_time(r'TIMING: build_parser_dwa ([0-9.]+m?s|µ?s)')

terminal_nwa_states = re.search(r'Input terminal NWA: states=(\d+)', text)
terminal_nwa_trans = re.search(r'Input terminal NWA: .*transitions=(\d+)', text)
term_nwa_s = terminal_nwa_states.group(1) if terminal_nwa_states else '?'
term_nwa_t = terminal_nwa_trans.group(1) if terminal_nwa_trans else '?'

after_det = re.search(r'TIMING: DWA pre_minimize states=(\d+)', text)
after_det_s = after_det.group(1) if after_det else '?'

colpack = re.search(r'TIMING: DWA pass ColPackMinimize\+?.*?([0-9.]+m?s|µ?s)', text)
colpack_t = colpack.group(1) if colpack else '?'

post_min = re.search(r'TIMING: DWA post_minimize states=(\d+).*transitions=(\d+)', text)
post_min_s = post_min.group(1) if post_min else '?'
post_min_t = post_min.group(2) if post_min else '?'

mat_closures = get_time(r'TIMING: acyclic_det::materialize_weighted_closures ([0-9.]+m?s|µ?s)')
det_acyclic = get_time(r'TIMING: acyclic_det::determinize ([0-9.]+m?s|µ?s)')

print("📌 kb_143 Benchmark Results")
print(f"=== Total compile time: {build_parser_dwa + setup + dfs + finish}ms ===")
print("Phase breakdown:")
print("| Phase                                  | Time    |")
print("| -------------------------------------- | ------- |")
print(f"| precompute1::setup                     | {setup}ms  |")
print(f"| precompute1::dfs                       | {dfs}ms   |")
print(f"| precompute1::finish                    | {finish}ms |")
print(f"| parser_dwa::total                     | {parser_dwa_total}ms |")
print(f"| build_parser_dwa                     | {build_parser_dwa}ms |")
print("Terminal NWA/DWA:")
print(f"⦁ Terminal NWA: {term_nwa_s} states, {term_nwa_t} transitions")
print(f"⦁ After determinize: {after_det_s} DWA states → ColPackMinimize → {post_min_s} states, {post_min_t} transitions")
print(f"⦁ ColPackMinimize: {colpack_t}")
print(f"⦁ materialize_weighted_closures: {mat_closures}ms")
print("")
print("Key timings inside finish():")
print(f"⦁ determinize acyclic: {det_acyclic}ms")
print(f"⦁ determinize acyclic materialize: {mat_closures}ms")
print(f"⦁ ColPackMinimize: {colpack_t}")
print(f"⦁ Total finish: {finish}ms")
