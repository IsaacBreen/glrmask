# Follow-up backlog after Chunk 05

Chunk 05 creates the Parser-DWA subsystem boundary, but several cleanups should
wait until later chunks.

## High priority

1. **Move template construction out of historical `compiler::stages`.**
   Parser DWA still imports `crate::compiler::stages::templates::Templates`.
   That is acceptable for this chunk because templates are their own planned
   subsystem.  Later, move them to `src/compile/template_dfa/` or
   `src/compile/stack_effect/`.

2. **Replace compatibility wrapper at pipeline call sites.**
   The compile pipeline still calls the old positional wrapper.  Later, change it
   to construct `ParserDwaBuildInputs` directly and optionally retain the profile.

3. **Thread Parser-DWA profile into compile profiles.**
   `ParserDwaBuildOutput.profile` currently exists so this is possible.  The
   compatibility wrapper discards it.

4. **Convert profile environment variables to typed compile options.**
   `GLRMASK_PROFILE_PARSER_DWA_COMPOSE_DETAIL` still lives in `profiling.rs`.
   Later, all compile profile decisions should come from one typed options object.

5. **Audit the duplicate epsilon-closure implementations.**
   The first determinizer contains a local closure and also has a file-level
   `local_epsilon_closure`.  This is inherited from the monolith.  A later
   cleanup should reduce this to one implementation without changing behavior.

## Medium priority

6. **Rename `NWA` to a more paper-aware alias where appropriate.**
   In this subsystem it is really a weighted nondeterministic stack-prefix
   recognizer.  The generic automata type is fine, but local aliases could make
   docs clearer.

7. **Split `determinize.rs` if it grows again.**
   It is currently the largest file.  If later work changes it, split into
   `determinize/supports.rs`, `determinize/fallbacks.rs`, and
   `determinize/epsilon.rs`.

8. **Move `dwa_to_nwa` if still unused.**
   The helper remains from the monolith.  It should be removed or moved when the
   compile pass is allowed to surface dead code.

9. **Replace low-level `u32` state ids with newtypes in docs or code.**
   Do not do this until after structural refactors.  The current code uses raw
   automata ids everywhere.

## Low priority

10. **Rustdoc examples.**
    Internal modules can include toy automata examples after the API stabilizes.

11. **Profile-line naming cleanup.**
    The profile line names are preserved for compatibility.  Later, change them
    only if benchmark tooling is updated.

12. **More precise `ParserDwaOptions`.**
    Current options only carry minimization policy.  Later it can include profile
    mode, fallback strategy, or determinization tuning.
