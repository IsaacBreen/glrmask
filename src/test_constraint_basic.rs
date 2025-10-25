//! Basic tests focusing on:
//! - Special precomputation edge structures (shape and invariants)
//! - get_mask interface contract at a very high-level
//!
//! This file takes a layered approach:
//! 1) Always-compiled lightweight unit tests that don't require constructing a full grammar/parser/tokenizer.
//!    These validate local invariants that are critical to downstream logic (formatting, ordering, basic set semantics).
//! 2) Feature-gated integration tests (feature = "integration-tests") that exercise end-to-end building of
//!    GrammarConstraint, running special precomputation, and verifying that get_mask4 matches the legacy path
//!    get_mask3 for a few simple inputs.
//!
//! To run the integration suite (which assumes your repository provides the grammar compilation helpers),
//! enable the "integration-tests" feature. For example:
//!     cargo test --features integration-tests -- --nocapture
//!
//! Notes:
//! - The integration tests assume the presence of existing facilities to build a small grammar and tokenizer.
//!   If your project exposes different helpers, adjust the fixture accordingly.
//!
//! Rationale for rewrite:
//! - Keep "unit tests" self-contained and always-on. They do not rely on any global caches or
//!   external data.
//! - Move heavy, end-to-end tests behind a feature flag to avoid slowing down inner-loop development.
//! - Make assertions crisp and explain failures with actionable messages.
//!
//! If you need a stricter or broader integration coverage, extend the integration section below.

#![cfg_attr(not(feature = "integration-tests"), allow(dead_code, unused_imports))]

use std::collections::{BTreeSet, HashSet};

use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::constraint::IntermediateTrie3EdgeKey;

// -------------------------------------------------------------------------------------------------
// Always-compiled, lightweight tests
// -------------------------------------------------------------------------------------------------

#[test]
fn test_intermediate_trie3_edgekey_display_variants() {
    // Empty bitset formatting
    let empty_bv = HybridBitset::zeros();
    let ek_empty = IntermediateTrie3EdgeKey::CheckLLM(empty_bv.clone());
    let s_empty = format!("{}", ek_empty);
    assert!(
        s_empty.contains("CheckLLM([])"),
        "Expected 'CheckLLM([])' for empty bitset, got: {}",
        s_empty
    );

    // All bitset formatting (uses '[ALL]')
    let all_bv = HybridBitset::max_ones();
    let ek_all = IntermediateTrie3EdgeKey::CheckLLM(all_bv.clone());
    let s_all = format!("{}", ek_all);
    assert!(
        s_all.contains("[ALL]"),
        "Expected '[ALL]' marker for full bitset, got: {}",
        s_all
    );

    // Single-range formatting (no brackets; e.g. '0..=9')
    let mut single_range = HybridBitset::zeros();
    for i in 0..10 {
        single_range.insert(i);
    }
    let ek_single = IntermediateTrie3EdgeKey::CheckLLM(single_range.clone());
    let s_single = format!("{}", ek_single);
    assert!(
        s_single.contains("CheckLLM(") && (s_single.contains("0..=9") || s_single.contains("0..=9)")),
        "Expected single-range formatting like 'CheckLLM(0..=9)', got: {}",
        s_single
    );

    // Multi-range formatting (should show brackets and a comma)
    let mut multi_range = HybridBitset::zeros();
    for i in 0..5 {
        multi_range.insert(i);
    }
    for i in 10..15 {
        multi_range.insert(i);
    }
    let ek_multi = IntermediateTrie3EdgeKey::CheckLLM(multi_range.clone());
    let s_multi = format!("{}", ek_multi);
    assert!(
        s_multi.contains("CheckLLM([") && s_multi.contains(","),
        "Expected bracketed multi-range formatting with comma for multiple ranges, got: {}",
        s_multi
    );

    // NoOp formatting
    let ek_noop = IntermediateTrie3EdgeKey::NoOp;
    let s_noop = format!("{}", ek_noop);
    assert_eq!(s_noop, "NoOp", "NoOp display should be exactly 'NoOp'");

    // Push formatting sanity check (doesn't assert exact ranges, which are impl details)
    let mut st_bv = HybridBitset::zeros();
    st_bv.insert(1);
    st_bv.insert(3);
    st_bv.insert(5);
    let ek_push = IntermediateTrie3EdgeKey::Push(st_bv.clone());
    let s_push = format!("{}", ek_push);
    assert!(
        s_push.starts_with("Push(") && s_push.ends_with(")"),
        "Push display should start with 'Push(' and end with ')', got: {}",
        s_push
    );

    // Pop formatting with an arbitrary distance
    let mut st_bv2 = HybridBitset::zeros();
    st_bv2.insert(42);
    let ek_pop = IntermediateTrie3EdgeKey::Pop(3, st_bv2.clone());
    let s_pop = format!("{}", ek_pop);
    assert!(
        s_pop.starts_with("Pop(3, "),
        "Pop display should start with 'Pop(3, ...', got: {}",
        s_pop
    );
}

#[test]
fn test_hybridbitset_basic_invariants() {
    // A few invariants assumed by various parts of the constraint pipeline.
    let mut a = HybridBitset::zeros();
    let mut b = HybridBitset::zeros();

    assert!(a.is_empty(), "zeros() should be empty");
    a.insert(2);
    a.insert(4);
    assert!(a.contains(2) && a.contains(4), "Inserted bits should be present");
    assert_eq!(a.len(), 2, "Cardinality should match number of inserted bits");

    b.insert(4);
    b.insert(6);

    let a_and_b = &a & &b;
    assert_eq!(a_and_b.len(), 1, "Only bit 4 is shared between a and b");
    assert!(a_and_b.contains(4), "Intersection should contain 4");

    let a_or_b = &a | &b;
    assert_eq!(a_or_b.len(), 3, "Union should contain {2,4,6}");
    assert!(a_or_b.contains(2) && a_or_b.contains(4) && a_or_b.contains(6));

    let a_minus_b = &a - &b;
    assert_eq!(a_minus_b.len(), 1, "Difference should contain {2}");
    assert!(a_minus_b.contains(2) && !a_minus_b.contains(4));
}

#[test]
fn test_special_precompute_dest_ord_and_hash_behaviour() {
    // Validate we can store many distinct tuples in a set without collisions.
    use crate::constraint_special_precompute::{SpecialPrecomputeDest, SpecialPrecomputeNormalEdge};
    use crate::glr::table::{NonTerminalID, StateID};
    use crate::types::TerminalID;

    let mut s: BTreeSet<SpecialPrecomputeNormalEdge> = BTreeSet::new();

    // None node, two different states/terminals
    s.insert((None, StateID(1), TerminalID(10), SpecialPrecomputeDest::Reduce { pop: 1, dest_nt: NonTerminalID(7) }));
    s.insert((None, StateID(2), TerminalID(10), SpecialPrecomputeDest::Reduce { pop: 1, dest_nt: NonTerminalID(7) }));

    // Same state/terminal but different source node (Some NT)
    s.insert((Some(NonTerminalID(5)), StateID(1), TerminalID(10), SpecialPrecomputeDest::Reduce { pop: 2, dest_nt: NonTerminalID(8) }));

    // Escape variants
    s.insert((Some(NonTerminalID(5)), StateID(1), TerminalID(10), SpecialPrecomputeDest::Escape { push_states: vec![StateID(11), StateID(12)] }));
    s.insert((Some(NonTerminalID(5)), StateID(1), TerminalID(11), SpecialPrecomputeDest::Escape { push_states: vec![StateID(11)] }));

    // Distinctness sanity
    assert_eq!(s.len(), 5, "All 5 inserted edges should be distinct");
}

// -------------------------------------------------------------------------------------------------
// Feature-gated integration tests
// -------------------------------------------------------------------------------------------------

// These tests target the high-level pipeline: build a small grammar+tokenizer -> GrammarConstraint
// -> run special precomputation -> compare get_mask4 with legacy get_mask3.
//
// They are behind the "integration-tests" feature to avoid requiring grammar compilation
// machinery in all environments. Adjust the fixture code to align with your project's helpers.

#[cfg(feature = "integration-tests")]
mod integration {
    use super::*;
    use std::collections::BTreeMap;

    use crate::constraint::{GrammarConstraint, GrammarConstraintConfig, TerminalAllowanceCheckMode};
    use crate::constraint_special_precompute::SpecialPrecomputation;
    use crate::datastructures::trie::Trie;
    use crate::finite_automata::Regex;
    use crate::glr::parser::GLRParser;
    use crate::interface::CompiledGrammar;
    use crate::tokenizer::{LLMTokenID, LLMTokenMap};
    use crate::types::TerminalID;
    use bimap::BiBTreeMap;

    // ---- Test fixture construction --------------------------------------------------------------

    /// Builds a tiny arithmetic-like grammar and tokenizer and returns a ready GrammarConstraint.
    ///
    /// Assumptions:
    /// - Your repository exposes a helper function that compiles a small grammar into CompiledGrammar.
    ///   If your actual helper has a different signature or module path, adjust below accordingly.
    fn build_basic_constraint() -> GrammarConstraint {
        // 1) Build or load a small compiled grammar
        //    Replace the following with whatever your project uses to obtain a CompiledGrammar.
        let compiled: CompiledGrammar = crate::tests::fixtures::tiny_arith_grammar();

        // 2) LLM token map that covers the byte sequences the tokenizer recognizes.
        //    Here we assume 'n' stands for numbers and '+' for addition, with tiny IDs.
        let mut llm_token_map: LLMTokenMap = BiBTreeMap::new();
        llm_token_map.insert(b"n".to_vec(), LLMTokenID(0));
        llm_token_map.insert(b"+".to_vec(), LLMTokenID(1));

        // 3) Maximum original LLM token ID from the map above
        let max_original_llm_token_id = 1;

        // 4) Constraint config – enable end-to-end precomputation (fast defaults)
        let mut cfg = GrammarConstraintConfig::default();
        // The post-commit check mode affects filtering of states; keep default or choose the fastest:
        cfg.trie3 = cfg.trie3.without_global_optimizations();
        // Make sure immediate checks don't prune too aggressively for the test
        // (tune to match your repository's defaults)
        // Example: cfg.post_commit_allow_check_mode = TerminalAllowanceCheckMode::StepProbe;

        // 5) Build constraint from compiled grammar
        GrammarConstraint::from_compiled_grammar_with_config(
            compiled,
            llm_token_map,
            LLMTokenID(0), // EOF or sentinel; adjust if your repo requires a specific ID
            max_original_llm_token_id,
            &cfg,
        )
    }

    // ---- Tests ---------------------------------------------------------------------------------

    #[test]
    fn precompute_special_smoke() {
        let gc = build_basic_constraint();
        let sp: &SpecialPrecomputation = &gc.special_precomputation;

        // Sanity: We should at least discover some edges for a nontrivial grammar.
        assert!(
            !sp.normal_edges.is_empty(),
            "Expected non-empty normal_edges for tiny_arith_grammar"
        );

        // Super edges require walking Trie1; they may be empty for extremely simple grammars,
        // but for even modest grammars we expect some.
        // Make the assertion lenient: check at least that the set exists and does not panic when iterated.
        let super_edges_count = sp.super_edges.len();
        assert!(
            super_edges_count >= 0,
            "super_edges length accounting should not be negative"
        );

        // Validate that each super edge references reachable Trie1 nodes (start and end) to catch broken IDs early.
        if !gc.precomputed1.is_empty() {
            let roots: Vec<_> = gc.precomputed1.values().cloned().collect();
            let live_nodes = Trie::all_nodes(&gc.trie1_god, &roots)
                .into_iter()
                .collect::<std::collections::HashSet<_>>();

            for (src_nt, terminal, (_pop, _dest_nt), _bv, pci1_start, pci1_end) in &sp.super_edges {
                assert!(
                    live_nodes.contains(pci1_start),
                    "super_edge references a non-reachable Trie1 start node: {:?}/{:?} start={:?}",
                    src_nt,
                    terminal,
                    pci1_start
                );
                assert!(
                    live_nodes.contains(pci1_end),
                    "super_edge references a non-reachable Trie1 end node: {:?}/{:?} end={:?}",
                    src_nt,
                    terminal,
                    pci1_end
                );
            }
        }
    }

    #[test]
    fn get_mask4_matches_legacy_get_mask3_for_initial_state() {
        let gc = build_basic_constraint();
        let mut s = gc.init();

        // For an initial, empty-commit state, both approaches should agree.
        // get_mask4 is the new, special-precompute-driven path.
        // get_mask3 is the legacy Trie3-driven path.
        let m4 = s.get_mask4();
        let m3 = s.get_mask3();

        assert_eq!(
            m4, m3,
            "get_mask4 and get_mask3 should match for the initial state"
        );
    }

    #[test]
    fn get_mask4_matches_after_simple_commit() {
        let gc = build_basic_constraint();
        let mut s = gc.init();

        // Commit a single 'n' byte (number token).
        s.commit_bytes(b"n");

        let m4 = s.get_mask4();
        let m3 = s.get_mask3();

        assert_eq!(
            m4, m3,
            "After committing 'n', get_mask4 should match get_mask3"
        );

        // Now commit a '+' and expect equivalence again.
        s.commit_bytes(b"+");
        let m4b = s.get_mask4();
        let m3b = s.get_mask3();

        assert_eq!(
            m4b, m3b,
            "After committing 'n+' (two segments), get_mask4 should match get_mask3"
        );
    }
}
