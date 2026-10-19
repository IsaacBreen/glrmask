# Final Module/Expr qualification

The existing `core_batch_bench` and `core_batch_scaling` executables support
matched before/after qualification of the unchanged direct public APIs. Preserve
the baseline commit, compiler version, flags, executable hashes, exact masks
and raw measurements. Use separate owned canonical release targets. Do not use
development-profile timings as production acceptance data.

`forcing_bench.rs` is an external, dependency-free harness for the **current
canonical forced-output contract**. Link exactly the same source against each
revision's canonical release `glrmask` library. It covers all four public
optimization modes, cold and mask-warmed states, singleton and multi-spelling
text, exact token events, acceptance barriers, and passive markers. It compares
returned original IDs and verifies that queries do not alter source-state
acceptance, rejection or marker batches. JSON rows include sample counts and
nanosecond distributions.

The historical October 15 strict-singleton regression report is preserved, but
the public forcing contract changed on October 19. Its old 42.2/102.9 microsecond
numbers are **not a directly comparable current baseline**. Establish a fresh
matched pair rather than describing a semantic change as a performance fix.

Run several alternating AB/BA process pairs without competing builds. Compare
exact outputs before interpreting timings, preserve all cases including
regressions, and report hardware, warmup, samples, dispersion and capabilities.
Worker counts are not a claim about physical-core scalability. Reusing one
compiled definition is distinct from compiling one instance per occurrence.

The Module/Expr public test suites separately cover independently configured
owners, open/mixed persistence, vocabulary mismatch, failure atomicity, scoped
expression paths and crossing original tokens. Native C/C++, Node and installed
Python tests must exercise the artifacts actually built from the recorded source.
