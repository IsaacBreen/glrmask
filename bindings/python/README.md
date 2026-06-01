# Python bindings

Split by concept: `vocab.rs`, `constraint.rs`, `state.rs`, `conversion.rs`, `state_lifetime.rs`, and `module.rs`. The split uses `include!` to change the file tree without introducing additional PyO3 module-visibility risk before compile repair.
