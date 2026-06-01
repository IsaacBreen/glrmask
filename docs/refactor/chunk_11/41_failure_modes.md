# Runtime failure modes to guard against

Failure mode: stale mask cache.

- Symptom: token appears allowed after a commit that should disallow it.
- Likely cause: generation not incremented or cache generation check missing.

Failure mode: scratch leakage.

- Symptom: Commit result depends on previous token sequence even after frontier
  is equivalent.
- Likely cause: `CommitBuffers::clear_all` misses a map or vector.

Failure mode: internal/original token confusion.

- Symptom: mask bits are shifted, tokens outside vocabulary appear allowed, or
  Python binding sees wrong ids.
- Likely cause: dense internal token id used directly as original token id.

Failure mode: template-DFA mismatch.

- Symptom: optimized Commit accepts/rejects differently from reference GLR
  advance.
- Likely cause: template recognizer compiled for wrong stack-effect language or
  terminal id space.

Failure mode: EOS inconsistency.

- Symptom: completion masks differ from commit success near accepting states.
- Likely cause: EOS handled in materialization but not in commit equivalence
  assertions.

Failure mode: diagnostic code changes semantics.

- Symptom: enabling profiling changes result.
- Likely cause: profile path has a separate implementation not checked against
  the unprofiled path.
