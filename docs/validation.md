# Validation order after structural cleanup

Repair in this order: module paths, visibility, rustfmt, clippy, pure unit tests, public integration tests, serialization compatibility, benchmark parity. Do not tune performance before semantic tests pass.
