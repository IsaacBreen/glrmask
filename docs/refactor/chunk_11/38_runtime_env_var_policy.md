# Runtime environment-variable policy

Runtime environment variables should be localized by subsystem.

Commit-local variables now live in `runtime/commit/options.rs`:

- `GLRMASK_DISABLE_TEMPLATE_DFA_ADVANCE`
- `GLRMASK_VALIDATE_TEMPLATE_DFA_ADVANCE`

Mask-local profile variables should live in `runtime/mask/profile.rs` or a later
`runtime/mask/options.rs`:

- queue debug flags;
- inner profile flags;
- delta-profile flags;
- single-path fallback flags.

General rule:

```text
algorithm.rs should not call std::env directly;
options.rs/profile.rs may call std::env directly;
public RuntimeOptions should eventually override env vars explicitly.
```

This is not only style.  Direct environment reads make reproducibility harder.
For a publication artifact, benchmark configuration must be visible and
replayable.
