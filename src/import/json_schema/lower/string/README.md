# JSON Schema string lowering

This directory owns `string`-specific grammar emission.  The current chunk moves
the implementation into a phase-local directory so future work can split helper
strategies without changing the public module path `lower::string`.

Do not move raw schema loading, schema algebra, or compile/runtime code here.
This directory should only contain functions that lower already-loaded schema
assertions into grammar expressions/rules.
