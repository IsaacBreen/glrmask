set -x
cd /Users/isaacbreen/Projects2/grammars2024
exec > profiling_output_dfs.txt 2>&1
export MACRO_DEBUG_LEVEL=5
export PROFILE_PRECOMPUTE1_DFS=1
export PROFILE_ENABLED=1
make test-js
