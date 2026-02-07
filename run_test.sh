#!/bin/bash
cd /Users/isaacbreen/Projects2/grammars2024
echo "Starting test at $(date)"
NO_CACHE=1 MACRO_DEBUG_LEVEL=3 SOURCE_FILE="testdata/finite_automata.rs" python3 -u scripts/test_diff.py 2>&1 | tee /tmp/diff_test_fresh2.txt | grep -E "TIMING|build_tokenizer|num_tsids|suffix|precompute|Error|Traceback"
echo "Finished at $(date), exit code: $?"
