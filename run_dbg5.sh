set -x
cd /Users/isaacbreen/Projects2/grammars2024
exec > profiling_output_dbg5.txt 2>&1
MACRO_DEBUG_LEVEL=5 DEBUG_LEVEL=5 PROFILE_ENABLED=1 make test-js
