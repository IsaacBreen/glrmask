import argparse
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from copy import deepcopy

# Unique counters to avoid reusing filenames/env across runs, which can interfere with external caches.
_candidate_write_counter = 0
_benchmark_run_counter = 0


def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json_gz(path: str, data: Dict[str, Any]) -> None:
    """
    Atomic write: write to a temp file in the same directory and then replace.
    Prevents readers from observing partially-written files.
    """
    dir_name = os.path.dirname(path) or "."
    tmp_path = os.path.join(dir_name, f".tmp_{os.getpid()}_{int(time.time()*1_000_000)}.json.gz")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, path)


def values_list_to_dict(values_list: List[Tuple[int, Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
    """
    Convert trie3_god['values'] (list of [node_id, node]) to dict[node_id] = node.
    Node ids are normalized to int keys.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in values_list:
        out[int(k)] = v
    return out


def dict_to_values_list(values_dict: Dict[int, Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Convert dict[node_id] back to list format that loader expects: [[node_id, node], ...].
    Use sorted order for determinism.
    """
    items = sorted(values_dict.items(), key=lambda kv: int(kv[0]))
    return [[int(k), v] for k, v in items]


def collect_adjacency(values_dict: Dict[int, Dict[str, Any]]) -> Dict[int, Set[int]]:
    """
    Build adjacency ignoring pop/state filters: node -> set(dest_node_ids).
    """
    adj: Dict[int, Set[int]] = {nid: set() for nid in values_dict.keys()}
    for nid, node in values_dict.items():
        for edge in node.get("children") or []:
            _edge_key, dests = edge
            for dest_id, _state_bv_json in dests:
                adj.setdefault(int(nid), set()).add(int(dest_id))
    return adj


def compute_roots(precomputed3: List[Tuple[int, int]]) -> List[int]:
    """
    Extract all root node ids from precomputed3 mapping (tokenizer_state_id -> root_node_id).
    """
    roots = [int(r) for (_s, r) in precomputed3]
    return roots


def bfs_reachable(values_dict: Dict[int, Dict[str, Any]], roots: List[int], depth_limit: Optional[int]) -> Set[int]:
    """
    BFS over adjacency from all roots up to depth_limit (if provided).
    Returns the set of reachable node ids (including roots).
    """
    if not roots:
        return set()
    adj = collect_adjacency(values_dict)
    seen: Set[int] = set()
    q: List[Tuple[int, int]] = []
    for r in roots:
        if r in values_dict:  # Start only from existing nodes
            seen.add(int(r))
            q.append((int(r), 0))
    while q:
        u, d = q.pop(0)
        if depth_limit is not None and d >= depth_limit:
            continue
        for v in adj.get(u, set()):
            if v not in seen and v in values_dict:
                seen.add(v)
                q.append((v, d + 1))
    return seen


def prune_values_by_reachability(values_dict: Dict[int, Dict[str, Any]], keep: Set[int]) -> Dict[int, Dict[str, Any]]:
    """
    Keep only nodes in 'keep'. For remaining nodes, drop child destinations that point to
    removed nodes. Drop child entries with empty dest lists.
    """
    new_values: Dict[int, Dict[str, Any]] = {}
    for nid, node in values_dict.items():
        if nid not in keep:
            continue
        new_children = []
        for edge in node.get("children") or []:
            edge_key, dests = edge
            new_dests = []
            for dest_id, state_bv_json in dests:
                did = int(dest_id)
                if did in keep:
                    new_dests.append([did, state_bv_json])
            if new_dests:
                new_children.append([edge_key, new_dests])
        # Clone node shallowly and replace children
        new_node = dict(node)
        new_node["children"] = new_children
        new_values[nid] = new_node
    return new_values


def count_nodes_children_dests(values_dict: Dict[int, Dict[str, Any]]) -> Tuple[int, int, int]:
    nodes = len(values_dict)
    children = 0
    dests = 0
    for node in values_dict.values():
        ch = node.get("children") or []
        children += len(ch)
        for _, ds in ch:
            dests += len(ds)
    return nodes, children, dests


def ranges_to_token_set(ranges: List[List[int]]) -> Set[int]:
    """
    Convert a list of [start, end] ranges (inclusive) into a set of token ints.
    """
    out: Set[int] = set()
    for start, end in ranges:
        if start > end:
            continue
        out.update(range(int(start), int(end) + 1))
    return out


def token_set_to_ranges(tokens: Set[int]) -> List[List[int]]:
    """
    Convert a set of token ints to a minimal sorted list of [start, end] ranges (inclusive).
    """
    if not tokens:
        return []
    vals = sorted(int(t) for t in tokens)
    ranges: List[List[int]] = []
    rs = vals[0]
    re_ = vals[0]
    for v in vals[1:]:
        if v == re_ + 1:
            re_ = v
        else:
            ranges.append([rs, re_])
            rs = re_ = v
    ranges.append([rs, re_])
    return ranges


def write_candidate_constraint(tmp_dir: Path, original: Dict[str, Any], values_dict: Dict[int, Dict[str, Any]]) -> Path:
    """
    Build a candidate constraint JSON by replacing trie3_god['values'] and write to a temp .json.gz.
    Returns the file path.
    """
    global _candidate_write_counter
    data = dict(original)  # shallow copy top-level; nested parts reused
    trie = dict(data.get("trie3_god") or {})
    trie["values"] = dict_to_values_list(values_dict)
    data["trie3_god"] = trie
    # Use a unique filename on every write to avoid any external caching keyed by path.
    _candidate_write_counter += 1
    candidate_path = tmp_dir / f"candidate_constraint_{_candidate_write_counter:06d}.json.gz"
    save_json_gz(str(candidate_path), data)
    return candidate_path


def run_benchmarks_and_has_mismatch(
    repo_root: Path,
    constraint_path: Path,
    code_file: Path,
    baseline_model: Path,
    candidate_model: Path,
    env_extra: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Optional[int], str]:
    """
    Run run_benchmarks.sh for baseline and candidate against constraint_path+code_file.
    Returns (mismatch_found, mismatch_index_or_None, raw_stdout).
    """
    global _benchmark_run_counter
    _benchmark_run_counter += 1

    script = repo_root / "python" / "run_benchmarks.sh"
    if not script.exists():
        raise FileNotFoundError(f"run_benchmarks.sh not found at {script}")

    env = os.environ.copy()
    env["CONSTRAINT_FILE"] = str(constraint_path)
    env["CODE_FILE"] = str(code_file)
    env["SKIP_CPP_BUILD"] = "1"
    env["DISABLE_TQDM"] = "1"
    # Add a per-run cache buster to minimize any external caching keyed on env.
    env["BENCHMARK_RUN_ID"] = str(_benchmark_run_counter)
    env["CACHE_BUSTER"] = str(int(time.time() * 1_000_000))

    if env_extra:
        env.update(env_extra)

    cmd = ["bash", str(script), str(baseline_model), str(candidate_model)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    out_lines: List[str] = []
    try:
        for line in proc.stdout:
            out_lines.append(line)
    finally:
        proc.wait()
    stdout = "".join(out_lines)

    # Parse mismatch info from analyzer output
    # Example line:
    # Mask mismatch at token index 3 for model precompute3_model_pure_python
    mismatch_re = re.compile(r"Mask mismatch at token index\s+(\d+)\s+for model\s+(.+)")
    m = mismatch_re.search(stdout)
    if m:
        idx = int(m.group(1))
        return True, idx, stdout
    return False, None, stdout


def verify_initial_mismatch(
    repo_root: Path,
    constraint_in: Path,
    code_file: Path,
    baseline_model: Path,
    candidate_model: Path,
) -> None:
    ok, idx, out = run_benchmarks_and_has_mismatch(
        repo_root, constraint_in, code_file, baseline_model, candidate_model
    )
    if not ok:
        # Dump some lines for debugging
        snippet = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(
            "Initial constraint does not reproduce any mismatch; cannot reduce.\n"
            f"Last output lines:\n{snippet}"
        )


def bsearch_min_depth_with_mismatch(
    original: Dict[str, Any],
    values_dict: Dict[int, Dict[str, Any]],
    repo_root: Path,
    tmpdir: Path,
    code_file: Path,
    baseline_model: Path,
    candidate_model: Path,
    time_budget_deadline: float,
) -> Tuple[Dict[int, Dict[str, Any]], Optional[int]]:
    """
    Binary search the minimal BFS depth limit (from roots) that still produces a mismatch.
    Returns (best_values_dict, best_depth) or (original_values_dict, None) if depth pruning cannot keep mismatch.
    """
    precomp = original.get("precomputed3") or []
    roots = compute_roots(precomp)
    if not roots:
        # No roots to traverse; nothing we can do
        return values_dict, None

    # Compute upper bound for reachable max depth by BFS (no limit)
    full_reach = bfs_reachable(values_dict, roots, None)
    # If the graph is empty or trivial, skip this stage
    if not full_reach:
        return values_dict, None

    # Estimate "max depth" using naive layered BFS
    # We approximate by BFS layers from roots. This doesn't need to be exact for bsearch bounds.
    def estimate_max_depth(values: Dict[int, Dict[str, Any]]) -> int:
        adj = collect_adjacency(values)
        layer = set(int(r) for r in roots if r in values)
        seen = set(layer)
        depth = 0
        while layer:
            depth += 1
            nxt = set()
            for u in layer:
                for v in adj.get(u, set()):
                    if v not in seen and v in values:
                        seen.add(v)
                        nxt.add(v)
            layer = nxt
        return depth

    max_depth_est = estimate_max_depth(values_dict)
    if max_depth_est <= 1:
        return values_dict, None

    # Helper to test a depth
    def test_depth(d: int) -> Tuple[bool, Optional[int]]:
        kept_nodes = bfs_reachable(values_dict, roots, d)
        candidate_values = prune_values_by_reachability(values_dict, kept_nodes)
        cand_path = write_candidate_constraint(tmpdir, original, candidate_values)
        ok, idx, _ = run_benchmarks_and_has_mismatch(
            repo_root, cand_path, code_file, baseline_model, candidate_model
        )
        return ok, idx

    # Find a depth that mismatches: exponential search upwards if needed
    lo = 1
    hi = min(max_depth_est, 64)  # cap to avoid pathological
    found_any = False
    best_depth = None
    best_values = values_dict

    # First try small depths quickly
    d = 1
    while d <= hi and time.time() < time_budget_deadline:
        ok, _idx = test_depth(d)
        if ok:
            found_any = True
            best_depth = d
            # Narrow further with binary search between lo..d
            break
        d *= 2

    if not found_any:
        # Try hi explicitly if time remains
        if time.time() < time_budget_deadline:
            ok, _idx = test_depth(hi)
            if not ok:
                return values_dict, None
            best_depth = hi
        else:
            return values_dict, None

    # Binary search to minimize depth
    lo = 1
    hi = int(best_depth)
    while lo < hi and time.time() < time_budget_deadline:
        mid = (lo + hi) // 2
        ok, _idx = test_depth(mid)
        if ok:
            best_depth = mid
            hi = mid
        else:
            lo = mid + 1

    # Build final best candidate for best_depth
    kept_nodes = bfs_reachable(values_dict, roots, best_depth)
    best_values = prune_values_by_reachability(values_dict, kept_nodes)
    return best_values, best_depth


def ddmin_children(
    original: Dict[str, Any],
    values_dict: Dict[int, Dict[str, Any]],
    repo_root: Path,
    tmpdir: Path,
    code_file: Path,
    baseline_model: Path,
    candidate_model: Path,
    time_budget_deadline: float,
) -> Dict[int, Dict[str, Any]]:
    """
    Try to delete child entries per node (entire ((pop,llm_bv), dests) entries).
    Accept a deletion if mismatch persists. Prune unreachable nodes after each accepted deletion.
    """
    precomp = original.get("precomputed3") or []

    # Work through nodes in deterministic random order
    node_ids = list(values_dict.keys())
    random.shuffle(node_ids)

    improved = True
    pass_num = 0
    while improved and time.time() < time_budget_deadline:
        improved = False
        pass_num += 1
        n_before, c_before, _ = count_nodes_children_dests(values_dict)
        print(f"  [Pass {pass_num}] Starting child reduction. Nodes={n_before}, Children={c_before}")

        children_checked_in_pass = 0
        children_removed_in_pass = 0
        for nid in node_ids:
            if time.time() >= time_budget_deadline:
                break
            node = values_dict.get(nid)
            if not node:
                continue
            children = node.get("children") or []
            i = 0
            while i < len(children) and time.time() < time_budget_deadline:
                children_checked_in_pass += 1
                print(f"    - Checks: {children_checked_in_pass}, Reductions: {children_removed_in_pass}", end='\r')
                # Try removing the ith child
                trial_values = {k: dict(v) for k, v in values_dict.items()}
                trial_children = list((trial_values[nid].get("children") or []))
                trial_children.pop(i)
                trial_values[nid]["children"] = trial_children

                # Prune unreachable nodes
                roots = compute_roots(original.get("precomputed3") or [])
                kept = bfs_reachable(trial_values, roots, None)
                pruned_values = prune_values_by_reachability(trial_values, kept)

                cand_path = write_candidate_constraint(tmpdir, original, pruned_values)
                ok, _, _ = run_benchmarks_and_has_mismatch(
                    repo_root, cand_path, code_file, baseline_model, candidate_model
                )
                if ok:
                    # Accept deletion
                    values_dict = pruned_values
                    node = values_dict.get(nid)  # may be gone; refresh
                    children = (node.get("children") or []) if node else []
                    improved = True
                    children_removed_in_pass += 1
                    # No increment: keep same index i (new element at i after pop)
                    continue
                else:
                    # Revert: increment i
                    i += 1
        
        print() # Newline for the \r progress
        if children_removed_in_pass > 0:
            n_after, c_after, _ = count_nodes_children_dests(values_dict)
            print(f"  [Pass {pass_num}] Finished. Removed {children_removed_in_pass} children. New total: {c_after}")
        else:
            print(f"  [Pass {pass_num}] Finished. No reductions found.")

    return values_dict


def ddmin_dests(
    original: Dict[str, Any],
    values_dict: Dict[int, Dict[str, Any]],
    repo_root: Path,
    tmpdir: Path,
    code_file: Path,
    baseline_model: Path,
    candidate_model: Path,
    time_budget_deadline: float,
) -> Dict[int, Dict[str, Any]]:
    """
    For each remaining child entry, try to delete individual dest entries.
    Accept a deletion if mismatch persists. Prune unreachable nodes after accept.
    """
    precomp = original.get("precomputed3") or []

    node_ids = list(values_dict.keys())
    random.shuffle(node_ids)

    improved = True
    pass_num = 0
    while improved and time.time() < time_budget_deadline:
        improved = False
        pass_num += 1
        n_before, _, d_before = count_nodes_children_dests(values_dict)
        print(f"  [Pass {pass_num}] Starting dest reduction. Nodes={n_before}, Dests={d_before}")
        
        dests_checked_in_pass = 0
        dests_removed_in_pass = 0

        for nid in node_ids:
            if time.time() >= time_budget_deadline:
                break
            node = values_dict.get(nid)
            if not node:
                continue
            # Iterate over children using a dynamic while loop so we can handle mutations safely.
            ci = 0
            while time.time() < time_budget_deadline:
                node = values_dict.get(nid)
                if not node:
                    break
                children = node.get("children") or []
                if ci >= len(children):
                    break

                # We will repeatedly try to remove destinations from child at index ci.
                di = 0
                child_disappeared = False

                while time.time() < time_budget_deadline:
                    # Re-sync current node/children/dests on every attempt to avoid stale indices.
                    node_now = values_dict.get(nid)
                    if not node_now:
                        child_disappeared = True
                        break
                    children_now = node_now.get("children") or []
                    if ci >= len(children_now):
                        # The child at ci was removed by a prior accepted reduction.
                        child_disappeared = True
                        break

                    edge_key_now, dests_now = children_now[ci]
                    if di >= len(dests_now):
                        # Exhausted all dests for this child
                        break

                    dests_checked_in_pass += 1
                    print(f"    - Checks: {dests_checked_in_pass}, Reductions: {dests_removed_in_pass}", end='\r')

                    # Build a trial that removes dests_now[di]
                    trial_values = {k: dict(v) for k, v in values_dict.items()}
                    trial_children = list((trial_values[nid].get("children") or []))
                    ekey, dlist = trial_children[ci]
                    new_dlist = list(dlist)
                    new_dlist.pop(di)
                    if not new_dlist:
                        # If removing this dest empties the child, drop the whole child.
                        trial_children.pop(ci)
                    else:
                        trial_children[ci] = [ekey, new_dlist]
                    trial_values[nid]["children"] = trial_children

                    roots = compute_roots(original.get("precomputed3") or [])
                    kept = bfs_reachable(trial_values, roots, None)
                    pruned_values = prune_values_by_reachability(trial_values, kept)

                    cand_path = write_candidate_constraint(tmpdir, original, pruned_values)
                    ok, _, _ = run_benchmarks_and_has_mismatch(
                        repo_root, cand_path, code_file, baseline_model, candidate_model
                    )
                    if ok:
                        # Accept deletion and keep working on the same di position
                        values_dict = pruned_values
                        improved = True
                        dests_removed_in_pass += 1
                        # Don't increment di: the next element (if any) has shifted into di
                        continue
                    else:
                        # Try the next destination
                        di += 1

                if child_disappeared:
                    # The child at index ci was removed; do not advance ci so the next child shifts into this index.
                    continue
                else:
                    # Move to the next child
                    ci += 1
        
        print() # Newline for the \r progress
        if dests_removed_in_pass > 0:
            n_after, _, d_after = count_nodes_children_dests(values_dict)
            print(f"  [Pass {pass_num}] Finished. Removed {dests_removed_in_pass} dests. New total: {d_after}")
        else:
            print(f"  [Pass {pass_num}] Finished. No reductions found.")

    return values_dict


def remove_token_from_ranges(ranges: List[List[int]], token: int) -> List[List[int]]:
    """
    Given a list of [start, end] integer ranges, remove a single token.
    This may result in a range being split. Ranges are not merged.
    """
    new_ranges = []
    for start, end in ranges:
        if token < start or token > end:
            new_ranges.append([start, end])
        else:  # token is within [start, end]
            if start < token:
                new_ranges.append([start, token - 1])
            if token < end:
                new_ranges.append([token + 1, end])
    return new_ranges


def ddmin_llm_tokens(
    original: Dict[str, Any],
    values_dict: Dict[int, Dict[str, Any]],
    repo_root: Path,
    tmpdir: Path,
    code_file: Path,
    baseline_model: Path,
    candidate_model: Path,
    time_budget_deadline: float,
) -> Dict[int, Dict[str, Any]]:
    """
    For each child edge, try to remove chunks of tokens from the LLM token bitvector.
    Accept a deletion if mismatch persists. This is a faster, chunk-based approach.
    """
    # This reduction does not change graph reachability, so no pruning needed inside.

    node_ids = list(values_dict.keys())
    random.shuffle(node_ids)

    improved = True
    pass_num = 0
    while improved and time.time() < time_budget_deadline:
        improved = False
        pass_num += 1
        print(f"  [Pass {pass_num}] Starting LLM token reduction.")

        tokens_checked_in_pass = 0
        tokens_removed_in_pass = 0

        for nid in node_ids:
            if time.time() >= time_budget_deadline:
                break
            node = values_dict.get(nid)
            if not node:
                continue

            children = node.get("children") or []
            for ci in range(len(children)):
                if time.time() >= time_budget_deadline:
                    break

                MAX_ATTEMPTS_PER_EDGE = 10
                attempts_left = MAX_ATTEMPTS_PER_EDGE
                # Repeatedly try to reduce tokens for this edge
                while time.time() < time_budget_deadline and attempts_left > 0:
                    # Get current llm_bv_json from the potentially modified values_dict
                    current_node = values_dict.get(nid)
                    if not current_node:
                        break
                    current_children = current_node.get("children", [])
                    if ci >= len(current_children):
                        break

                    edge_key, dests = current_children[ci]
                    pop, llm_bv_json = edge_key

                    tokens_in_bv = []
                    for start, end in llm_bv_json:
                        tokens_in_bv.extend(range(start, end + 1))

                    if len(tokens_in_bv) < 2:
                        break  # Cannot reduce further

                    # Try removing roughly half the tokens
                    random.shuffle(tokens_in_bv)
                    tokens_to_remove = set(tokens_in_bv[:len(tokens_in_bv) // 2])

                    tokens_checked_in_pass += len(tokens_to_remove)
                    print(f"    - Checks: {tokens_checked_in_pass}, Reductions: {tokens_removed_in_pass}", end='\r')

                    # Build new llm_bv_json by removing tokens
                    remaining_tokens = sorted([t for t in tokens_in_bv if t not in tokens_to_remove])

                    new_llm_bv_json = []
                    if remaining_tokens:
                        range_start = remaining_tokens[0]
                        for i in range(1, len(remaining_tokens)):
                            if remaining_tokens[i] > remaining_tokens[i-1] + 1:
                                new_llm_bv_json.append([range_start, remaining_tokens[i-1]])
                                range_start = remaining_tokens[i]
                        new_llm_bv_json.append([range_start, remaining_tokens[-1]])

                    # Create a trial candidate
                    trial_values = {k: dict(v) for k, v in values_dict.items()}
                    trial_children = list((trial_values[nid].get("children") or []))

                    trial_children[ci] = [[pop, new_llm_bv_json], dests]
                    trial_values[nid]["children"] = trial_children

                    cand_path = write_candidate_constraint(tmpdir, original, trial_values)
                    ok, _, _ = run_benchmarks_and_has_mismatch(
                        repo_root, cand_path, code_file, baseline_model, candidate_model
                    )

                    if ok:
                        # Accept deletion
                        values_dict = trial_values
                        improved = True
                        tokens_removed_in_pass += len(tokens_to_remove)
                        attempts_left = MAX_ATTEMPTS_PER_EDGE  # Reset on success
                        # Continue the while loop to try and reduce this edge further
                    else:
                        # Failed to reduce with this chunk, try another random chunk
                        attempts_left -= 1

        print()  # Newline for the \r progress
        if tokens_removed_in_pass > 0:
            print(f"  [Pass {pass_num}] Finished. Removed {tokens_removed_in_pass} LLM tokens.")
        else:
            print(f"  [Pass {pass_num}] Finished. No reductions found.")

    return values_dict


def sweep_prune_llm_tokens(
    original: Dict[str, Any],
    values_dict: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    """

    Deterministic, single-sweep pruning of LLM tokens on edges based on:
      - forward token reachability from roots, and
      - backward token reachability to end nodes.

    For each edge (node -> dests) with llm_bv E:
      pruned_llm = E ∩ forward_tokens[node] ∩ (⋃_d backward_tokens[d])

    If pruned_llm is empty, the edge is removed.
    After pruning edges, prune unreachable nodes via BFS reachability.

    Returns (new_values_dict, stats).
    """
    # Helper: sum total tokens across all edges
    def total_llm_tokens(values: Dict[int, Dict[str, Any]]) -> int:
        tot = 0
        for node in values.values():
            for edge in node.get("children") or []:
                edge_key, _dests = edge
                _pop, llm_bv_json = edge_key
                tot += len(ranges_to_token_set(llm_bv_json))
        return tot

    # Build token universe from all edges
    universe: Set[int] = set()
    for node in values_dict.values():
        for edge in node.get("children") or []:
            edge_key, _dests = edge
            _pop, llm_bv_json = edge_key
            universe |= ranges_to_token_set(llm_bv_json)

    if not universe:
        # Nothing to prune
        return values_dict, {"tokens_removed": 0, "edges_removed": 0}

    # Precompute edges for faster fixed-point passes (avoid repeated conversions).
    # edges_map[node_id] = list of (edge_llm_tokens_set, [dest_ids...])
    edges_map: Dict[int, List[Tuple[Set[int], List[int]]]] = {}
    for nid, node in values_dict.items():
        edges_list: List[Tuple[Set[int], List[int]]] = []
        for edge in node.get("children") or []:
            edge_key, dests = edge
            _pop, llm_bv_json = edge_key
            e_tokens = ranges_to_token_set(llm_bv_json)
            dest_ids = [int(did) for did, _state_bv_json in dests]
            edges_list.append((e_tokens, dest_ids))
        edges_map[int(nid)] = edges_list

    # Forward token reachability: from roots, tokens that can reach each node.
    forward_tokens: Dict[int, Set[int]] = {}
    roots = compute_roots(original.get("precomputed3") or [])
    for r in roots:
        if r in values_dict:
            forward_tokens[int(r)] = set(universe)  # Initially, all tokens possible at roots

    changed = True
    while changed:
        changed = False
        for nid, edge_entries in edges_map.items():
            parent_tokens = forward_tokens.get(nid)
            if not parent_tokens:
                continue
            for e_tokens, dest_ids in edge_entries:
                through = parent_tokens & e_tokens
                if not through:
                    continue
                for did in dest_ids:
                    dest_set = forward_tokens.setdefault(did, set())
                    new = through - dest_set
                    if new:
                        dest_set.update(new)
                        changed = True

    # Backward token reachability: tokens that can reach an end node from each node.
    backward_tokens: Dict[int, Set[int]] = {}
    # Initialize end nodes to universe
    for nid, node in values_dict.items():
        value = node.get("value") or {}
        if bool(value.get("clean_end", False)):
            backward_tokens[int(nid)] = set(universe)

    changed = True
    while changed:
        changed = False
        for nid, edge_entries in edges_map.items():
            # Aggregate tokens across all outgoing edges considering their dests
            agg: Set[int] = set()
            for e_tokens, dest_ids in edge_entries:
                # Union of backward tokens over all dests for this edge
                union_dest_back = set()
                for did in dest_ids:
                    union_dest_back |= backward_tokens.get(did, set())
                if union_dest_back:
                    agg |= (e_tokens & union_dest_back)
            curr = backward_tokens.get(nid, set())
            new = agg - curr
            if new:
                if curr:
                    curr.update(new)
                else:
                    backward_tokens[nid] = agg
                changed = True

    # Prune edges based on forward/backward tokens
    tokens_before = total_llm_tokens(values_dict)
    edges_removed = 0
    for nid, node in list(values_dict.items()):
        parent_toks = forward_tokens.get(nid, set())
        children = list(node.get("children") or [])
        new_children = []
        for edge in children:
            edge_key, dests = edge
            pop, llm_bv_json = edge_key
            e_tokens = ranges_to_token_set(llm_bv_json)
            if not e_tokens:
                # Skip empty token edge (shouldn't happen at this stage)
                continue
            union_dest_back = set()
            for did, _state_bv_json in dests:
                union_dest_back |= backward_tokens.get(int(did), set())
            pruned = e_tokens & parent_toks & union_dest_back
            if pruned:
                new_llm_bv_json = token_set_to_ranges(pruned)
                new_children.append([[int(pop), new_llm_bv_json], dests])
            else:
                edges_removed += 1
                # Drop this edge entirely
        node["children"] = new_children

    # After edge pruning, prune unreachable nodes and empty dests
    roots = compute_roots(original.get("precomputed3") or [])
    kept = bfs_reachable(values_dict, roots, None)
    values_dict = prune_values_by_reachability(values_dict, kept)

    tokens_after = 0
    for node in values_dict.values():
        for edge in node.get("children") or []:
            edge_key, _dests = edge
            _pop, llm_bv_json = edge_key
            tokens_after += len(ranges_to_token_set(llm_bv_json))

    stats = {
        "tokens_removed": max(0, tokens_before - tokens_after),
        "edges_removed": edges_removed,
    }
    return values_dict, stats


def main():
    ap = argparse.ArgumentParser(description="Stochastic reducer for constraint trie while preserving a mask mismatch.")
    ap.add_argument("--constraint-in", required=True, help="Path to the source .json.gz constraint file.")
    ap.add_argument("--code", required=True, help="Path to the input code file used in benchmarks.")
    ap.add_argument("--baseline-model", default="python/aug25/models/precompute3_model_pure_python_standalone.py",
                    help="Path to baseline model module (default: standalone pure python).")
    ap.add_argument("--candidate-model", default="python/aug25/models/precompute3_model_pure_python.py",
                    help="Path to candidate model module (default: optimized pure python).")
    ap.add_argument("--output", required=True, help="Path to write the minimized .json.gz.")
    ap.add_argument("--time-budget-seconds", type=int, default=300, help="Default time budget per phase (seconds).")
    ap.add_argument("--phase1-time-budget", type=int, default=None, help="Time budget for Phase 1 (depth reduction). Overrides default.")
    ap.add_argument("--phase2-time-budget", type=int, default=None, help="Time budget for Phase 2 (child reduction). Overrides default.")
    ap.add_argument("--phase3-time-budget", type=int, default=None, help="Time budget for Phase 3 (dest reduction). Overrides default.")
    ap.add_argument("--phase4-time-budget", type=int, default=None, help="Time budget for Phase 4 (LLM token reduction). Overrides default.")
    ap.add_argument("--phase5-time-budget", type=int, default=None, help="Time budget for Phase 5 (token sweep). Overrides default.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for deterministic runs.")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    phase_budgets = {
        1: args.phase1_time_budget if args.phase1_time_budget is not None else args.time_budget_seconds,
        2: args.phase2_time_budget if args.phase2_time_budget is not None else args.time_budget_seconds,
        3: args.phase3_time_budget if args.phase3_time_budget is not None else args.time_budget_seconds,
        4: args.phase4_time_budget if args.phase4_time_budget is not None else args.time_budget_seconds,
        5: args.phase5_time_budget if args.phase5_time_budget is not None else args.time_budget_seconds,
    }

    repo_root = Path(__file__).resolve().parents[2]
    constraint_in = Path(args.constraint_in).resolve()
    code_file = Path(args.code).resolve()
    baseline_model = (repo_root / args.baseline_model).resolve() if not args.baseline_model.startswith("/") else Path(args.baseline_model).resolve()
    candidate_model = (repo_root / args.candidate_model).resolve() if not args.candidate_model.startswith("/") else Path(args.candidate_model).resolve()
    out_path = Path(args.output).resolve()

    # Sanity checks
    if not constraint_in.exists():
        print(f"Error: constraint input not found: {constraint_in}", file=sys.stderr)
        sys.exit(2)
    if not code_file.exists():
        print(f"Error: code file not found: {code_file}", file=sys.stderr)
        sys.exit(2)
    if not baseline_model.exists():
        print(f"Error: baseline model not found: {baseline_model}", file=sys.stderr)
        sys.exit(2)
    if not candidate_model.exists():
        print(f"Error: candidate model not found: {candidate_model}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(out_path.parent, exist_ok=True)

    print("Verifying initial mismatch on the original constraint...")
    verify_initial_mismatch(repo_root, constraint_in, code_file, baseline_model, candidate_model)
    print("Initial mismatch confirmed. Starting reduction...")

    original = load_json_gz(str(constraint_in))
    trie = original.get("trie3_god") or {}
    values_list = trie.get("values") or []
    values_dict = values_list_to_dict(values_list)
    accepted_values_dict = deepcopy(values_dict)

    # Initial sizes
    n0, c0, d0 = count_nodes_children_dests(values_dict)
    print(f"Initial trie size: {n0} nodes, {c0} children, {d0} dests")

    with tempfile.TemporaryDirectory(prefix="constraint_reduce_") as td:
        tmpdir = Path(td)

        # Phase 1: BFS depth minimization
        if phase_budgets[1] > 0:
            print(f"\n--- Phase 1: Reducing depth (budget: {phase_budgets[1]}s) ---")
            time_budget_deadline = time.time() + phase_budgets[1]
            start = time.time()
            pruned_values, best_depth = bsearch_min_depth_with_mismatch(
                original, values_dict, repo_root, tmpdir, code_file,
                baseline_model, candidate_model, time_budget_deadline
            )
            if best_depth is not None:
                values_dict = pruned_values
                accepted_values_dict = deepcopy(values_dict)
                n1, c1, d1 = count_nodes_children_dests(values_dict)
                print(f"--- Phase 1 finished. Minimal depth {best_depth} keeps mismatch. Size now: {n1} nodes, {c1} children, {d1} dests. Took {time.time()-start:.1f}s ---")
            else:
                print(f"--- Phase 1 finished. Depth pruning could not preserve mismatch; keeping full reachability. Took {time.time()-start:.1f}s ---")
        else:
            print("\n--- Phase 1: Skipped (budget is 0) ---")

        # Phase 2: Delete child edges per node (delta-debug)
        if phase_budgets[2] > 0:
            print(f"\n--- Phase 2: Reducing child edges (budget: {phase_budgets[2]}s) ---")
            time_budget_deadline = time.time() + phase_budgets[2]
            start = time.time()
            values_dict = ddmin_children(
                original, values_dict, repo_root, tmpdir, code_file,
                baseline_model, candidate_model, time_budget_deadline
            )
            accepted_values_dict = deepcopy(values_dict)
            n2, c2, d2 = count_nodes_children_dests(values_dict)
            print(f"--- Phase 2 finished. Size: {n2} nodes, {c2} children, {d2} dests. Took {time.time()-start:.1f}s ---")
        else:
            print("\n--- Phase 2: Skipped (budget is 0) ---")

        # Phase 3: Delete dests within child edges (delta-debug)
        if phase_budgets[3] > 0:
            print(f"\n--- Phase 3: Reducing destinations (budget: {phase_budgets[3]}s) ---")
            time_budget_deadline = time.time() + phase_budgets[3]
            start = time.time()
            values_dict = ddmin_dests(
                original, values_dict, repo_root, tmpdir, code_file,
                baseline_model, candidate_model, time_budget_deadline
            )
            accepted_values_dict = deepcopy(values_dict)
            n3, c3, d3 = count_nodes_children_dests(values_dict)
            print(f"--- Phase 3 finished. Size: {n3} nodes, {c3} children, {d3} dests. Took {time.time()-start:.1f}s ---")
        else:
            print("\n--- Phase 3: Skipped (budget is 0) ---")

        # Phase 4: Delete LLM tokens from edges (delta-debug)
        if phase_budgets[4] > 0:
            print(f"\n--- Phase 4: Reducing LLM tokens in edges (budget: {phase_budgets[4]}s) ---")
            time_budget_deadline = time.time() + phase_budgets[4]
            start = time.time()
            values_dict = ddmin_llm_tokens(
                original, values_dict, repo_root, tmpdir, code_file,
                baseline_model, candidate_model, time_budget_deadline
            )
            accepted_values_dict = deepcopy(values_dict)
            n4, c4, d4 = count_nodes_children_dests(values_dict)
            print(f"--- Phase 4 finished. Size: {n4} nodes, {c4} children, {d4} dests. Took {time.time()-start:.1f}s ---")
        else:
            print("\n--- Phase 4: Skipped (budget is 0) ---")

        # Phase 5: Deterministic single-sweep pruning by token reachability
        if phase_budgets[5] > 0:
            print(f"\n--- Phase 5: Sweeping unreachable tokens (forward/backward reachability) (budget: {phase_budgets[5]}s) ---")
            time_budget_deadline = time.time() + phase_budgets[5]
            if time.time() < time_budget_deadline:
                start = time.time()
                swept_values_dict, sweep_stats = sweep_prune_llm_tokens(original, values_dict)
                
                # Verify this aggressive, un-checked reduction
                sweep_cand_path = write_candidate_constraint(tmpdir, original, swept_values_dict)
                ok, _, _ = run_benchmarks_and_has_mismatch(
                    repo_root, sweep_cand_path, code_file, baseline_model, candidate_model
                )

                if ok:
                    print("  Sweep reduction is valid and preserves mismatch.")
                    values_dict = swept_values_dict
                    accepted_values_dict = deepcopy(values_dict)
                    n_after, c_after, d_after = count_nodes_children_dests(values_dict)
                    took = time.time() - start
                    print(f"--- Phase 5 finished. "
                          f"Tokens removed: {sweep_stats.get('tokens_removed', 0)}, "
                          f"Edges removed: {sweep_stats.get('edges_removed', 0)}. "
                          f"Size: {n_after} nodes, {c_after} children, {d_after} dests. "
                          f"Took {took:.1f}s ---")
                else:
                    print("  Sweep reduction was unsound (did not preserve mismatch). Discarding.")
                    n_after, c_after, d_after = count_nodes_children_dests(values_dict)
                    took = time.time() - start
                    print(f"--- Phase 5 finished. No changes applied. Size: {n_after} nodes, {c_after} children, {d_after} dests. Took {took:.1f}s ---")
            else:
                print("--- Phase 5: Skipped due to time budget exceeded before start ---")
        else:
            print("\n--- Phase 5: Skipped (budget is 0) ---")

        # Final write
        final_candidate_path = write_candidate_constraint(tmpdir, original, values_dict)

        # Verify mismatch (final check)
        print("Verifying final minimized constraint still mismatches...")
        ok, idx, _out = run_benchmarks_and_has_mismatch(
            repo_root, final_candidate_path, code_file, baseline_model, candidate_model
        )
        if not ok:
            print("Warning: final candidate unexpectedly does not mismatch. Reverting to last accepted state.", file=sys.stderr)
            final_candidate_path = write_candidate_constraint(tmpdir, original, accepted_values_dict)
            # Re-verify the last accepted state; if this also fails, abort to surface the issue early.
            print("Re-checking mismatch on last accepted candidate...")
            ok2, idx2, _out2 = run_benchmarks_and_has_mismatch(
                repo_root, final_candidate_path, code_file, baseline_model, candidate_model
            )
            if not ok2:
                print("Error: last accepted candidate also does not mismatch; aborting to avoid emitting an invalid artifact.", file=sys.stderr)
                # Optionally keep the last accepted file in tmpdir for post-mortem.
                # You can copy it manually from the printed tmp path if needed.
                sys.exit(3)

        # Copy to user-provided output path
        shutil.copyfile(final_candidate_path, out_path)
        print(f"Done. Minimized constraint written to: {out_path}")


if __name__ == "__main__":
    main()
