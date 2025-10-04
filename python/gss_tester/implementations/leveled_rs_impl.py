try:
    # The native module is built and placed in the python/ directory or installed in the venv
    from leveled_gss_rs import LeveledGSS as LeveledRSGSS
except ImportError as e:
    raise ImportError(
        "Could not import the Rust-based LeveledGSS implementation. "
        "Please build the native module by running `maturin develop` in `python/leveled_rs/`"
    ) from e

# Alias for test runner discovery
Leveled_rsGSS = LeveledRSGSS
