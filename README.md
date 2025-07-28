```bash
RUSTFLAGS=-Awarnings CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --color=always --package sep1 --lib test_js_constraint_integration --profile test -- --nocapture 2>&1 | tee .temp2; python analyze_timings.py .temp2
```

```bash
RUSTFLAGS=-Awarnings CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --color=always --package sep1 --lib test_js_constraint_isolated_and_minimized --profile test -- --nocapture 2>&1 | tee .temp2; python analyze_timings.py .temp2
```