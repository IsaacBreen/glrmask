```bash
RUSTFLAGS=-Awarnings CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --color=always --package sep1 --lib test_js_parser_isolated_object_literal --profile test -- --nocapture | tee .temp2
```