#!/usr/bin/env python3
"""Compare FastMinimize vs ColPackMinimize per height for Terminal DWA.

Runs `make test-js` twice (once per pass) with DWA_TRACE_HEIGHTS enabled and
prints a per-height table of color counts and gaps.
"""

import os
import re
import subprocess
import sys
from typing import Dict, Tuple

TRACE_RE = re.compile(r"TRACE: height (\d+) colors=(\d+)")
PRE_RE = re.compile(r"TIMING: DWA pre_minimize states=(\d+) transitions=(\d+)")
POST_RE = re.compile(r"TIMING: DWA post_minimize states=(\d+) transitions=(\d+)")


def run_pass(pass_name: str) -> Tuple[int, str]:
    env = os.environ.copy()
    env["TERMINAL_DWA_PASS"] = pass_name
    env["DWA_TRACE_HEIGHTS"] = "1"
    proc = subprocess.run(
        ["make", "test-js"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout


def parse_sections(output: str):
    sections = []
    current = None
    for line in output.splitlines():
        pre_match = PRE_RE.search(line)
        if pre_match:
            current = {
                "pre_states": int(pre_match.group(1)),
                "colors": {},
            }
            continue

        if current is None:
            continue

        trace_match = TRACE_RE.search(line)
        if trace_match:
            height = int(trace_match.group(1))
            count = int(trace_match.group(2))
            current["colors"][height] = count
            continue

        post_match = POST_RE.search(line)
        if post_match:
            current["post_states"] = int(post_match.group(1))
            sections.append(current)
            current = None

    if current is not None:
        sections.append(current)

    return sections


def select_section(sections):
    if not sections:
        return None
    return max(sections, key=lambda s: s.get("pre_states", 0))


def print_table(
    fast: Dict[int, int],
    colpack: Dict[int, int],
    fast_meta: dict,
    colpack_meta: dict,
) -> None:
    heights = sorted(set(fast.keys()) | set(colpack.keys()))
    if not heights:
        print("No TRACE height color counts found.")
        return

    fast_total = sum(fast.values())
    col_total = sum(colpack.values())
    print("FAST  : pre_states={} post_states={} sum_colors={}".format(
        fast_meta.get("pre_states"), fast_meta.get("post_states"), fast_total
    ))
    print("COLPK : pre_states={} post_states={} sum_colors={}".format(
        colpack_meta.get("pre_states"), colpack_meta.get("post_states"), col_total
    ))

    print("height\tfast\tcolpack\tgap")
    for h in heights:
        fast_val = fast.get(h)
        col_val = colpack.get(h)
        if fast_val is None or col_val is None:
            gap = "N/A"
        else:
            gap = str(fast_val - col_val)
        print(f"{h}\t{fast_val if fast_val is not None else 'N/A'}\t{col_val if col_val is not None else 'N/A'}\t{gap}")

    print("TOTAL\t{}\t{}\t{}".format(fast_total, col_total, fast_total - col_total))


def main() -> int:
    print("Running FastMinimize...")
    fast_code, fast_out = run_pass("fast")
    fast_sections = parse_sections(fast_out)
    fast_section = select_section(fast_sections)
    fast_colors = fast_section["colors"] if fast_section else {}

    print("Running ColPackMinimize...")
    col_code, col_out = run_pass("colpack")
    col_sections = parse_sections(col_out)
    col_section = select_section(col_sections)
    col_colors = col_section["colors"] if col_section else {}

    print_table(fast_colors, col_colors, fast_section or {}, col_section or {})

    if fast_code != 0:
        print(f"WARNING: FastMinimize run exited with code {fast_code}")
    if col_code != 0:
        print(f"WARNING: ColPackMinimize run exited with code {col_code}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
