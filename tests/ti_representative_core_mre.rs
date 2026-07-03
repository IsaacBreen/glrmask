//! Minimal witness that a representative-only terminal-DWA core is incomplete.
//!
//! This is deliberately a diagnostic test of the superseded design. It first
//! enables `GLRMASK_TI_REPRESENTATIVE_CORE_MRE`, which emits only one terminal
//! representative during the trie walk, and verifies that the exact artifact
//! comparator finds the expected mismatch. It then disables the diagnostic
//! switch and verifies that the normal direct-transport construction compiles
//! the identical grammar.
//!
//! The full scanner/TSID/DWA trace is documented in
//! `docs/terminal-interchangeability-representative-core-mre.md`.

use std::{env, ffi::OsString};

use glrmask::{Constraint, Vocab};

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = env::var_os(key);
        unsafe { env::set_var(key, value) };
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => unsafe { env::set_var(self.key, value) },
            None => unsafe { env::remove_var(self.key) },
        }
    }
}

fn compile_mre() -> glrmask::Result<Constraint> {
    let vocab = Vocab::new(vec![(0u32, b"c".to_vec())], None);
    Constraint::from_lark(
        r#"
            start: A B | B A
            A: "ca"
            B: "cb"
        "#,
        &vocab,
    )
}

#[test]
fn representative_only_core_loses_b_prefix_but_direct_transport_recovers_it() {
    let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
    let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
    let _feature = EnvVarGuard::set("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY", "1");
    let representative_core = EnvVarGuard::set("GLRMASK_TI_REPRESENTATIVE_CORE_MRE", "1");

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compile_mre().expect("the diagnostic build must reach the exact comparator");
    }))
    .expect_err("the representative-only core must lose B's prefix edge");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic payload");
    assert!(
        message.contains("state=0 token=0 word=[1] baseline_accepts=true candidate_accepts=false"),
        "unexpected diagnostic panic: {message}",
    );

    // With the diagnostic core disabled, direct transport preserves the exact
    // baseline terminal language for the same grammar and vocabulary.
    drop(representative_core);
    compile_mre().expect("direct transport must agree with the baseline");
}
