# JSON Schema number lowering

This directory owns `number`-specific grammar emission.  The current chunk moves
the implementation into a phase-local directory so future work can split helper
strategies without changing the public module path `lower::number`.

Do not move raw schema loading, schema algebra, or compile/runtime code here.
This directory should only contain functions that lower already-loaded schema
assertions into grammar expressions/rules.
