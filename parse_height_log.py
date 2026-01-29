import re, math
from collections import defaultdict

LOG_PATH = "/tmp/test-js-dbg6.log"
MAX_HEIGHT = 99

heights = defaultdict(lambda: {
    "candidates": None,
    "color_graph_ms": None,
    "color_ms": None,
    "merge_ms": None,
    "comparisons": None,
})
current_height = None

re_height_start = re.compile(r"Height (\d+) start: (\d+) candidates")
re_height_candidates = re.compile(r"Height (\d+): (\d+) candidates")
re_coloring = re.compile(r"Coloring breakdown: signature=[^,]+, graph=([0-9.]+)(ms|µs|s), coloring=([0-9.]+)(ms|µs|s)")
re_incomp = re.compile(r"Incomp graph: (\d+) candidates, .*? (\d+) comparisons,")
re_merge = re.compile(r"Height (\d+): merge_state_into_builder ([0-9.]+)(ms|µs|s)")

unit_mult = {"s": 1000.0, "ms": 1.0, "µs": 0.001}

with open(LOG_PATH, "r", errors="replace") as f:
    for line in f:
        m = re_height_start.search(line)
        if m:
            h = int(m.group(1)); c = int(m.group(2))
            current_height = h
            if h <= MAX_HEIGHT:
                heights[h]["candidates"] = c
            continue
        m = re_height_candidates.search(line)
        if m:
            h = int(m.group(1)); c = int(m.group(2))
            current_height = h
            if h <= MAX_HEIGHT:
                heights[h]["candidates"] = c
            continue
        m = re_coloring.search(line)
        if m and current_height is not None and current_height <= MAX_HEIGHT:
            g_val = float(m.group(1)); g_unit = m.group(2)
            c_val = float(m.group(3)); c_unit = m.group(4)
            heights[current_height]["color_graph_ms"] = g_val * unit_mult[g_unit]
            heights[current_height]["color_ms"] = c_val * unit_mult[c_unit]
            continue
        m = re_incomp.search(line)
        if m and current_height is not None and current_height <= MAX_HEIGHT:
            heights[current_height]["comparisons"] = int(m.group(2))
            continue
        m = re_merge.search(line)
        if m:
            h = int(m.group(1))
            if h <= MAX_HEIGHT:
                val = float(m.group(2)); unit = m.group(3)
                heights[h]["merge_ms"] = val * unit_mult[unit]
            continue

rows = [(h, heights[h]) for h in range(0, MAX_HEIGHT + 1) if heights[h]["candidates"] is not None]

# correlation helper
def corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else None

cands = []
gtimes = []
comps = []
m_cands = []
mtimes = []
for h, data in rows:
    if data["candidates"] is not None and data["color_graph_ms"] is not None:
        cands.append(data["candidates"])
        gtimes.append(data["color_graph_ms"])
    if data["candidates"] is not None and data["comparisons"] is not None:
        comps.append((data["candidates"], data["comparisons"]))
    if data["candidates"] is not None and data["merge_ms"] is not None:
        m_cands.append(data["candidates"])
        mtimes.append(data["merge_ms"])

print("heights_captured", len(rows))
print("first_height", rows[0])
print("last_height", rows[-1])
if cands and gtimes:
    print("corr_candidates_graph_ms", corr(cands, gtimes))
if comps:
    c = [c for c, _ in comps]
    k = [k for _, k in comps]
    print("corr_candidates_comparisons", corr(c, k))
if m_cands and mtimes:
    print("corr_candidates_merge_ms", corr(m_cands, mtimes))

# bin averages
bins = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]
for lo, hi in bins:
    subset = [d for h, d in rows if lo <= h <= hi and d["candidates"] is not None]
    if not subset:
        continue
    avg_cand = sum(d["candidates"] for d in subset) / len(subset)
    avg_graph = sum(d["color_graph_ms"] or 0 for d in subset) / len(subset)
    avg_merge = sum(d["merge_ms"] or 0 for d in subset) / len(subset)
    avg_comp = sum(d["comparisons"] or 0 for d in subset) / len(subset)
    print(f"bin_{lo}_{hi}: avg_candidates={avg_cand:.2f}, avg_graph_ms={avg_graph:.2f}, avg_merge_ms={avg_merge:.2f}, avg_comparisons={avg_comp:.2f}")

for h, data in rows:
    if h == 99:
        print("height_99", data)
        break
