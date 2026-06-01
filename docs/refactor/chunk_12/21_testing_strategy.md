# Testing strategy after the split

The next compile/test phase should add or re-run tests in layers:

1. API smoke tests: `commit_token`, `commit_bytes`, and `commit_tokens` still mutate generation and state as expected.
2. Mask/Commit equivalence tests: accepting a token in the current mask should not be rejected by Commit.
3. Fast-path equivalence tests: force fast paths on/off and compare final states for the same byte fragments.
4. Longest-match tests: commit a prefix that can be extended and ensure delayed exclusions are respected.
5. Ignored-terminal tests: whitespace or ignored tokens should not require parser advance.
6. Profiling parity tests: profiled and unprofiled commits should produce identical parser frontiers.

This chunk deliberately did not compile or run those tests, but it made the test targets explicit.
