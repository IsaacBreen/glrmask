//! Production boundary-shard terminal-DWA construction (static-link redesign).
//!
//! For a composition with a merged tokenizer and a spliced (control-eliminated)
//! LR table, the boundary shard for start component `i` is built by running the
//! STANDARD terminal-DWA trie walk on the merged tokenizer from `Commit_i`
//! (component `i`'s token-start states), keeping only crossing paths at the NWA
//! level, then standard templates + parser-DWA construction (see
//! `build_boundary_shard` in step 4). Exact by construction: the walk is what a
//! monolithic compile would run on the same merged DFA, restricted to `i`
//! starts; non-crossing paths are already covered at runtime by `A_i`.
//!
//! Plan note: `prepared-static-linker-architecture-2026-09-17.md` §2 (rev 4).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::constraint_compose::accepted_original_tokens;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::ComposedTable;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa as tdwa;
use crate::ds::bitset::BitSet;
use crate::Vocab;

/// Inputs for one boundary-shard terminal-DWA build.
pub(crate) struct BoundaryWalkInputs<'a> {
    /// Merged (disjoint-union) tokenizer of the composition being linked.
    pub merged_tokenizer: &'a Tokenizer,
    /// Full model vocabulary (no `T_i` restriction: Phase 1 showed trie
    /// restriction loses via the minimize pathology).
    pub vocab: &'a Vocab,
    /// Analyzed composed grammar (for always-allowed follows + the walk's
    /// terminal observation scope).
    pub grammar: &'a AnalyzedGrammar,
    /// Pairwise disallowed terminal follows of the composed table.
    pub disallowed_follows: &'a BTreeMap<u32, BitSet>,
    /// Canonical ignore terminal of the merged tokenizer, if any.
    pub ignore_terminal: Option<u32>,
    /// Composed table terminal offsets (terminal ownership per component).
    pub terminal_offsets: &'a [u32],
    /// Start component `i` (parent = 0, children in order).
    pub component_index: usize,
    /// `Commit_i`: token-start states of component `i` as merged-tokenizer
    /// raw state ids (`tokenizer.num_states()` entries). The production
    /// convention is all states of component `i` (see
    /// `commit_states_for_component`): a sound superset of the true runtime
    /// commit states.
    pub commit_states: &'a [bool],
    /// Link-once shared equivalence for the merged tokenizer (step 1).
    pub shared_equivalence: &'a tdwa::l2p::SharedL2pEquivalence,
    /// Prebuilt flat transition table for the merged tokenizer, shared across
    /// shards. Built on demand when `None`.
    pub flat_trans: Option<&'a Arc<[u32]>>,
}

/// Per-stage timings for one shard build (milliseconds wall).
#[derive(Debug, Clone, Default)]
pub(crate) struct BoundaryWalkProfile {
    pub setup_ms: f64,
    pub walk_ms: f64,
    pub id_map_ms: f64,
    pub terminal_dwa_ms: f64,
    pub compact_ms: f64,
    pub determinize_ms: f64,
    pub minimize_ms: f64,
}

/// Output of one boundary-shard terminal-DWA build.
pub(crate) struct BoundaryWalkOutput {
    /// Minimized crossing-only terminal DWA (empty language iff `X_i` is empty).
    pub dwa: DWA,
    /// Shard-local id_map (compacted; TSIDs are shard-private per plan §2.4.4).
    pub id_map: InternalIdMap,
    pub profile: BoundaryWalkProfile,
}

/// Token-start states of one component: all merged-tokenizer raw states owned
/// by component `i` (`tokenizer_offsets[i] .. + num_states_of_i`).
///
/// This is a sound superset of the runtime commit states (every token the
/// runtime starts in `i` starts in one of these states), hence sound for
/// crossing detection; it is pessimistic for cost (narrowing to the true
/// commit set is future work).
pub(crate) fn commit_states_for_component(
    tokenizer_offsets: &[u32],
    num_states_of_component: u32,
    index: usize,
    total_states: usize,
) -> Vec<bool> {
    let mut keep = vec![false; total_states];
    let start = tokenizer_offsets[index] as usize;
    let end = start + num_states_of_component as usize;
    keep[start..end].fill(true);
    keep
}

/// Build the crossing terminal DWA for one start component.
///
/// Runs the standard L2P walk (`build_l2p_id_map_and_terminal_dwa_mode`) on
/// the merged tokenizer with `seed_state_filter = Commit_i`, the link-shared
/// equivalence (TI discovery skipped), and the NWA-level crossing filter for
/// `component_index`. Returns `None` only when the vocab is empty.
pub(crate) fn build_boundary_terminal_dwa(
    inputs: &BoundaryWalkInputs,
) -> Option<BoundaryWalkOutput> {
    let setup_started = Instant::now();
    let tokenizer = inputs.merged_tokenizer;
    let num_terms = inputs.grammar.num_terminals as usize;
    let coloring = tdwa::types::TerminalColoring::identity(num_terms);
    let always_allowed =
        tdwa::grammar_helpers::compute_always_allowed_follows(inputs.grammar);
    let active = vec![true; num_terms];
    let owned_flat;
    let flat: &Arc<[u32]> = match inputs.flat_trans {
        Some(flat) => flat,
        None => {
            owned_flat = Arc::from(tdwa::l1::build_flat_transition_table(tokenizer));
            &owned_flat
        }
    };
    let crossing = tdwa::l2p::L2pCrossingFilter {
        terminal_offsets: inputs.terminal_offsets,
        start_component: inputs.component_index,
    };
    let shard_options = tdwa::l2p::L2pShardBuildOptions {
        shared_equivalence: Some(inputs.shared_equivalence),
        skip_ti_discovery: true,
        crossing_filter: Some(crossing),
    };
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1000.0;
    let walk_started = Instant::now();
    let result = tdwa::l2p::build_l2p_id_map_and_terminal_dwa_mode(
        "boundary_shard",
        tokenizer,
        inputs.vocab,
        &coloring,
        false,
        inputs.ignore_terminal,
        inputs.grammar,
        &always_allowed,
        &active,
        inputs.disallowed_follows,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(flat),
        None,
        None,
        false,
        Some(inputs.commit_states),
        Some(&shard_options),
    )?;
    let walk_ms = walk_started.elapsed().as_secs_f64() * 1000.0;
    let profile = BoundaryWalkProfile {
        setup_ms,
        walk_ms,
        id_map_ms: result.profile.id_map_ms,
        terminal_dwa_ms: result.profile.terminal_dwa_ms,
        compact_ms: result.profile.compact_ms,
        determinize_ms: result.profile.determinize_ms,
        minimize_ms: result.profile.minimize_ms,
    };
    Some(BoundaryWalkOutput {
        dwa: result.dwa,
        id_map: result.id_map,
        profile,
    })
}

/// Original model tokens accepted by a shard terminal DWA (candidate-token
/// trigger + gate helper). The shard DWAs are acyclic (asserted).
pub(crate) fn boundary_accepted_tokens(dwa: &DWA, id_map: &InternalIdMap) -> BTreeSet<u32> {
    accepted_original_tokens(dwa, id_map)
}

/// Distinct terminal labels emitted by a terminal DWA (template selection).
pub(crate) fn boundary_emitted_terminals(dwa: &DWA, num_terminals: usize) -> Vec<bool> {
    let mut selected = vec![false; num_terminals];
    for state in dwa.states() {
        for &label in state.transitions.keys() {
            if label >= 0
                && let Some(slot) = selected.get_mut(label as usize)
            {
                *slot = true;
            }
        }
    }
    selected
}

/// Terminal offsets of a composed table (re-exported shape for shard builds).
pub(crate) fn composed_terminal_offsets(composed_table: &ComposedTable) -> &[u32] {
    composed_table.terminal_offsets.as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::Lexer;
    use crate::compiler::constraint_compose::{
        CompiledSubgrammarInput, component_ignores_are_globally_erasable,
        eliminate_composed_runtime_controls, load_vocab, merged_ignore_terminals,
        merged_retained_terminal_exprs, merged_terminal_display_names,
    };
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::table::{
        GLRTable, SubgrammarTableInput, compose_subgrammar_tables,
    };
    use crate::compiler::pipeline::compute_disallowed_follows;
    use crate::grammar::flat::TerminalID;
    use crate::runtime::Constraint;

    struct LowLevelComposed {
        table: ComposedTable,
        tokenizer: Tokenizer,
        tokenizer_offsets: Vec<u32>,
        terminal_names: Vec<String>,
        ignore_canonical: Option<u32>,
    }

    fn terminal_id(constraint: &Constraint, name: &str) -> TerminalID {
        constraint
            .terminal_display_names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap() as u32
    }

    // Existing composition code only: table splice + control elimination +
    // disjoint-union tokenizer. Mirrors the phase1 probe helper.
    fn low_level_compose(
        parent: &Constraint,
        children: &[CompiledSubgrammarInput<'_>],
    ) -> LowLevelComposed {
        let global_ignores = component_ignores_are_globally_erasable(parent, children);
        let table_inputs: Vec<SubgrammarTableInput> = children
            .iter()
            .map(|child| SubgrammarTableInput {
                placeholder_terminal: child.placeholder_terminal,
                additional_placeholder_terminals: &[],
                table: &child.constraint.table,
                ignore_terminal: (!global_ignores)
                    .then_some(child.constraint.ignore_terminal)
                    .flatten(),
                start_nullable: child.constraint.table.embedded_start_nullable(),
            })
            .collect();
        let mut composed = compose_subgrammar_tables(
            &parent.table,
            (!global_ignores).then_some(parent.ignore_terminal).flatten(),
            &table_inputs,
        )
        .expect("compose tables");
        eliminate_composed_runtime_controls(&mut composed).expect("eliminate controls");
        let terminal_names = merged_terminal_display_names(parent, children);
        let mut tokenizer_inputs: Vec<(&Tokenizer, u32)> =
            Vec::with_capacity(children.len() + 1);
        tokenizer_inputs.push((&parent.tokenizer, composed.terminal_offsets[0]));
        for (index, child) in children.iter().enumerate() {
            tokenizer_inputs
                .push((&child.constraint.tokenizer, composed.terminal_offsets[index + 1]));
        }
        let (mut merged, tokenizer_offsets) =
            Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
        if merged.terminal_exprs().is_none() {
            let all: Vec<&Constraint> = std::iter::once(parent)
                .chain(children.iter().map(|child| child.constraint))
                .collect();
            if let Some(exprs) = merged_retained_terminal_exprs(
                &all,
                &composed.terminal_offsets,
                composed.table.num_terminals,
            ) {
                merged.restore_terminal_exprs(Some(exprs)).expect("restore merged exprs");
            }
        }
        let ignores = merged_ignore_terminals(
            parent,
            children,
            &composed.terminal_offsets,
            global_ignores,
        );
        assert_eq!(
            tokenizer_offsets[0], 1,
            "merged state 0 must be the fresh reset fan-out"
        );
        LowLevelComposed {
            table: composed,
            tokenizer: merged,
            tokenizer_offsets,
            terminal_names,
            ignore_canonical: ignores.canonical,
        }
    }

    fn analyzed_grammar(table: &GLRTable, names: &[String]) -> AnalyzedGrammar {
        let augmented_start =
            table.rules.first().expect("composed table has augmented start").lhs;
        AnalyzedGrammar::from_composed_rules(
            table.rules.clone(),
            table.num_terminals,
            names.to_vec(),
            table.nonterminal_display_names.clone(),
            augmented_start,
        )
    }

    fn shared_equivalence_for(
        tokenizer: &Tokenizer,
        vocab: &Vocab,
        ignore_terminal: Option<u32>,
        grammar: &AnalyzedGrammar,
        active_terminals: &[bool],
        disallowed: &BTreeMap<u32, BitSet>,
        flat: &Arc<[u32]>,
    ) -> tdwa::l2p::SharedL2pEquivalence {
        tdwa::l2p::compute_shared_l2p_equivalence(
            "boundary_shard_test",
            tokenizer,
            vocab,
            ignore_terminal,
            grammar,
            active_terminals,
            disallowed,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(flat),
            None,
            None,
        )
        .expect("shared equivalence must compute")
    }

    #[test]
    fn toy_two_component_crossing_sets() {
        // Parent `doc ::= PA SUB PB` with child `item ::= CC CD` bound at SUB.
        // Hand-verified crossing sets (trace the merged lexer from each
        // component's states; the composed table's pairwise follows prune the
        // rest):
        // - "ac": PA[parent] then CC[child], pair (PA,CC) allowed -> X_parent.
        // - "db": CD[child] then PB[parent], pair (CD,PB) allowed -> X_child.
        // - "ca": pair (CC,PA) never adjacent -> pruned everywhere.
        // - "bd": pair (PB,CD) never adjacent (PB is last) -> pruned.
        // - "ab", "cd": single-component paths -> internal, not crossing.
        // - single bytes: no reset strictly inside -> never crossing.
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"d".to_vec()),
            (4, b"ac".to_vec()),
            (5, b"db".to_vec()),
            (6, b"ca".to_vec()),
            (7, b"bd".to_vec()),
            (8, b"ab".to_vec()),
            (9, b"cd".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start doc;
                t PA ::= "a";
                t PB ::= "b";
                t SUB ::= @token(999);
                nt doc ::= PA SUB PB;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start item;
                t CC ::= "c";
                t CD ::= "d";
                nt item ::= CC CD;
            "#,
            &vocab,
        )
        .unwrap();
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let composed = low_level_compose(&parent, &children);
        let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = compute_disallowed_follows(&grammar);
        let active = vec![true; grammar.num_terminals as usize];
        let flat: Arc<[u32]> =
            Arc::from(tdwa::l1::build_flat_transition_table(&composed.tokenizer));
        let shared = shared_equivalence_for(
            &composed.tokenizer,
            &vocab,
            composed.ignore_canonical,
            &grammar,
            &active,
            &disallowed,
            &flat,
        );
        for (index, (num_states, expected)) in
            [(parent.tokenizer.num_states(), vec![4u32]), (child.tokenizer.num_states(), vec![5u32])]
                .into_iter()
                .enumerate()
        {
            let commit = commit_states_for_component(
                &composed.tokenizer_offsets,
                num_states,
                index,
                composed.tokenizer.num_states() as usize,
            );
            let output = build_boundary_terminal_dwa(&BoundaryWalkInputs {
                merged_tokenizer: &composed.tokenizer,
                vocab: &vocab,
                grammar: &grammar,
                disallowed_follows: &disallowed,
                ignore_terminal: composed.ignore_canonical,
                terminal_offsets: &composed.table.terminal_offsets,
                component_index: index,
                commit_states: &commit,
                shared_equivalence: &shared,
                flat_trans: Some(&flat),
            })
            .expect("toy shard walk must produce a DWA");
            assert!(output.dwa.is_acyclic(), "toy shard {index} DWA must be acyclic");
            let tokens = boundary_accepted_tokens(&output.dwa, &output.id_map);
            assert_eq!(
                tokens.into_iter().collect::<Vec<_>>(),
                expected,
                "toy shard {index} crossing set",
            );
        }
    }

    fn restore_component(component: &mut Constraint, label: &str) {
        if component.tokenizer.terminal_exprs().is_none()
            && let Some(exprs) = component.retained_terminal_exprs().map(|exprs| exprs.to_vec())
        {
            component
                .tokenizer
                .restore_terminal_exprs(Some(exprs))
                .expect("restore component terminal exprs");
        }
        let inline_rules = component.table.rules.len();
        let retained = component.retained_table_rules().expect("decode retained rules").len();
        if inline_rules != retained {
            component.table.rules =
                component.retained_table_rules().expect("decode retained rules").to_vec();
        }
        component
            .materialize_composition_metadata_for_compilation()
            .expect("materialize composition metadata");
        eprintln!("BOUNDARY_WALK restore {label} inline_rules={inline_rules} retained_rules={retained}");
    }

    /// Production-path selected10 crossing gate: one shared equivalence, one
    /// standard walk per component, NWA-level crossing filter. Asserts the
    /// 143-token dispatch crossing set (MINBOUND oracle dump) and the
    /// 26-state true-minimal crossing DWA, plus the empty core shard.
    #[test]
    #[ignore]
    fn selected10_boundary_walk_crossing() {
        use std::path::Path;

        let root = std::env::var("PHASE1_DIR").unwrap_or_else(|_| {
            "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29".to_string()
        });
        let root = Path::new(&root).to_path_buf();
        let vocab_path = std::env::var("PHASE1_VOCAB")
            .unwrap_or_else(|_| root.join("vocab_dump.bin").to_string_lossy().into_owned());
        let dump_dir = std::env::var("PHASE1_DUMP_DIR")
            .unwrap_or_else(|_| "/tmp/grammars25-redesign".to_string());
        let vocab = load_vocab(&vocab_path);
        let mut core =
            Constraint::load(&std::fs::read(root.join("core.bin")).expect("read core.bin"))
                .expect("load core");
        let dispatch_name = std::env::var("PHASE1_DISPATCH")
            .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
        let mut dispatch =
            Constraint::load(&std::fs::read(root.join(&dispatch_name)).expect("read dispatch"))
                .expect("load dispatch");
        restore_component(&mut core, "core");
        restore_component(&mut dispatch, "dispatch");
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &dispatch,
        }];
        let composed = low_level_compose(&core, &children);
        let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = compute_disallowed_follows(&grammar);
        let active = vec![true; grammar.num_terminals as usize];
        let flat: Arc<[u32]> =
            Arc::from(tdwa::l1::build_flat_transition_table(&composed.tokenizer));
        let shared_started = Instant::now();
        let shared = shared_equivalence_for(
            &composed.tokenizer,
            &vocab,
            composed.ignore_canonical,
            &grammar,
            &active,
            &disallowed,
            &flat,
        );
        let shared_wall_ms = shared_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "BOUNDARY_WALK shared id_map_ms={:.3} wall_ms={:.3} tsids={} itokens={}",
            shared.id_map_ms,
            shared_wall_ms,
            shared.id_map.num_tsids(),
            shared.id_map.num_internal_tokens(),
        );

        let oracle_text = std::fs::read_to_string(format!("{dump_dir}/minbound-tokens.txt"))
            .expect(
                "oracle dump missing; regenerate with the MINBOUND driver \
                 (see E-phase1-walk.md §2.1) or run without PHASE1_SKIP_ORACLE via the probe",
            );
        let oracle: BTreeSet<u32> =
            oracle_text.split_whitespace().map(|value| value.parse::<u32>().unwrap()).collect();
        assert_eq!(oracle.len(), 143, "oracle dump must hold the 143 MINBOUND tokens");

        let component_states = [core.tokenizer.num_states(), dispatch.tokenizer.num_states()];
        for (index, num_states) in component_states.into_iter().enumerate() {
            let commit = commit_states_for_component(
                &composed.tokenizer_offsets,
                num_states,
                index,
                composed.tokenizer.num_states() as usize,
            );
            let output = build_boundary_terminal_dwa(&BoundaryWalkInputs {
                merged_tokenizer: &composed.tokenizer,
                vocab: &vocab,
                grammar: &grammar,
                disallowed_follows: &disallowed,
                ignore_terminal: composed.ignore_canonical,
                terminal_offsets: &composed.table.terminal_offsets,
                component_index: index,
                commit_states: &commit,
                shared_equivalence: &shared,
                flat_trans: Some(&flat),
            })
            .expect("selected10 shard walk must produce a DWA");
            assert!(output.dwa.is_acyclic(), "shard {index} DWA must be acyclic");
            let tokens = boundary_accepted_tokens(&output.dwa, &output.id_map);
            let emitted =
                boundary_emitted_terminals(&output.dwa, grammar.num_terminals as usize);
            let emitted_count = emitted.iter().filter(|slot| **slot).count();
            let profile = &output.profile;
            eprintln!(
                "BOUNDARY_WALK shard={index} states={} trans={} tokens={} emitted_terms={} \
                 setup_ms={:.3} walk_ms={:.3} id_map_ms={:.3} dwa_ms={:.3} det_ms={:.3} min_ms={:.3} compact_ms={:.3}",
                output.dwa.num_states(),
                output.dwa.num_transitions(),
                tokens.len(),
                emitted_count,
                profile.setup_ms,
                profile.walk_ms,
                profile.id_map_ms,
                profile.terminal_dwa_ms,
                profile.determinize_ms,
                profile.minimize_ms,
                profile.compact_ms,
            );
            if index == 0 {
                assert!(tokens.is_empty(), "core shard must be empty, got {}", tokens.len());
                assert_eq!(output.dwa.num_states(), 1, "empty core shard DWA shape");
            } else {
                assert_eq!(tokens.len(), 143, "dispatch shard token count");
                assert_eq!(tokens, oracle, "dispatch shard must match the oracle set");
                assert_eq!(
                    output.dwa.num_states(),
                    26,
                    "dispatch crossing DWA must be the true-minimal 26-state form"
                );
            }
        }
    }
}
