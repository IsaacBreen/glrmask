//! Production boundary-shard terminal-DWA construction (static-link redesign).
//!
//! For a composition with a merged tokenizer and a provider-materialized
//! exact boundary table (live leaf coordinate), the boundary shard for start
//! component `i` is built by running the STANDARD terminal-DWA trie walk on
//! the merged tokenizer from `Commit_i` (component `i`'s token-start states),
//! keeping only crossing paths at the NWA level, then standard templates +
//! parser-DWA construction (see `build_boundary_shard` in step 4). Exact by
//! construction: the walk is what a monolithic compile would run on the same
//! merged DFA, restricted to `i` starts; non-crossing paths are already
//! covered at runtime by `A_i`.
//!
//! Plan note: `prepared-static-linker-architecture-2026-09-17.md` §2 (rev 4).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::{Lexer, Tokenizer};
use crate::automata::weighted_u32::dwa::DWA;
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::compiler::constraint_compose::{
    CompiledSubgrammarInput, PublishedStaticBoundaryShard, WalkBoundaryShardWork,
    accepted_original_tokens, build_segmented_parser_links,
    component_ignores_are_globally_erasable, eliminate_composed_runtime_controls,
    leaf_ignores_are_globally_erasable, merged_ignore_terminals,
    merged_leaf_ignore_terminals, merged_leaf_terminal_display_names,
    merged_retained_terminal_exprs, merged_terminal_display_names,
};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::parser::{
    DisjointComponentActionProvider, ParserComponentTableSource, ScopedParserSymbol,
    ScopedSubgrammarLink, materialize_control_eliminated_scoped_provider_table,
};
use crate::compiler::glr::table::{
    ComposedTable, GLRTable, SubgrammarTableInput, compose_subgrammar_tables_with_rules,
};
use crate::compiler::pipeline::compute_disallowed_follows;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa as tdwa;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::runtime::Constraint;
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
    /// Retain non-crossing paths (bypass the NWA-level crossing filter).
    /// The link sets this only for the parent shard when a child can be
    /// traversed zero-width (nullable start): the composed parser then
    /// accepts parent-internal paths the parent alone rejects, and only the
    /// unfiltered walk covers them. Child shards always filter (their
    /// component mask covers every dropped path); see
    /// `BoundaryShardLinkInputs::retain_parent_non_crossing_paths`.
    pub retain_non_crossing_paths: bool,
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
    /// Minimized terminal DWA: crossing-only by default (empty language iff
    /// `X_i` is empty), full walk output when non-crossing paths are retained.
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
/// equivalence (TI discovery skipped), the NWA-level crossing filter for
/// `component_index` (bypassed when `retain_non_crossing_paths` is set), and
/// core compaction skipped (the shard parser indexes (parser-stack x tsid),
/// which needs the exact Step-1 coordinate).
/// Returns `None` only when the vocab is empty.
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
        crossing_filter: if inputs.retain_non_crossing_paths {
            None
        } else {
            Some(crossing)
        },
        // The shard parser indexes (parser-stack x tsid); compaction would
        // merge stack-distinguishable states and over-admit (inner-6).
        skip_core_compact: true,
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

/// Link-level inputs for building every boundary shard.
pub(crate) struct BoundaryShardLinkInputs<'a> {
    pub merged_tokenizer: &'a Tokenizer,
    pub vocab: &'a Vocab,
    pub grammar: &'a AnalyzedGrammar,
    pub disallowed_follows: &'a BTreeMap<u32, BitSet>,
    pub ignore_terminal: Option<u32>,
    pub terminal_offsets: &'a [u32],
    pub tokenizer_offsets: &'a [u32],
    pub component_state_counts: &'a [u32],
    /// Retain non-crossing paths in the PARENT shard (component 0). Child
    /// shards always filter. The filter drops exactly the single-component
    /// paths, and a dropped path gaps the mask only if the composed table
    /// accepts its terminal sequence while no component mask admits the
    /// token. For a child-C-internal sequence at a C-topped stack, the
    /// boundary table's C rows are C's own rows, so composed acceptance
    /// implies C acceptance and C's mask covers it. For a parent-internal
    /// sequence, the parent mask covers it unless a zero-width empty-child
    /// derivation enables a parent-alone rejection — possible only when a
    /// linked child is start-nullable. Multi-component paths are crossing
    /// from every seed's view and are always kept. Nullable bound children
    /// decline loudly at signed-context certification, so retention only
    /// fires on embedded-nullable-but-effectively-nonnullable links.
    pub retain_parent_non_crossing_paths: bool,
    /// Explicit per-start-component walk plans (nested links). When `None`,
    /// one plan per component is derived from `component_state_counts` /
    /// `tokenizer_offsets` exactly as before (flat behavior unchanged).
    pub walk_plans: Option<Vec<BoundaryShardWalkPlan>>,
}

/// One built shard walk with a nonempty crossing set.
pub(crate) struct BuiltBoundaryShardWalk {
    pub start_component: usize,
    pub output: BoundaryWalkOutput,
    pub candidate_tokens: BTreeSet<u32>,
}

/// Walk plan for one start component: which leaf index the crossing filter
/// views from, the full merged-tokenizer commit set, and whether non-crossing
/// paths are retained. Flat links use one plan per direct component with the
/// component's own commit states; nested links use one plan per top-level
/// component where a block plan views from the block-root leaf with the union
/// of the block leaves' commit states (every block-internal crossing touches
/// a leaf outside the block root, so the filter keeps exactly the
/// block-crossing gaps, mirroring the proven nested fixture).
pub(crate) struct BoundaryShardWalkPlan {
    pub start_component: usize,
    pub filter_component: usize,
    pub commit_states: Vec<bool>,
    pub retain_non_crossing_paths: bool,
}

/// Link-level shard profile: shared-once cost + per-shard profiles.
pub(crate) struct BoundaryShardLinkProfile {
    pub shared_id_map_ms: f64,
    pub shared_wall_ms: f64,
    pub shared_tsids: usize,
    pub shared_itokens: usize,
    pub flat_ms: f64,
    pub per_shard: Vec<(usize, BoundaryWalkProfile)>,
}

/// Build every boundary shard walk for a link: one shared equivalence, then
/// one standard walk per component (in parallel unless macro parallelism is
/// disabled). Components with empty crossing sets are skipped (the runtime
/// skips missing shards). Returns `None` only when the vocab is empty.
pub(crate) fn build_boundary_shard_walks(
    inputs: &BoundaryShardLinkInputs,
) -> Option<(Vec<BuiltBoundaryShardWalk>, BoundaryShardLinkProfile)> {
    use rayon::prelude::*;

    let flat_started = Instant::now();
    let flat: Arc<[u32]> =
        Arc::from(tdwa::l1::build_flat_transition_table(inputs.merged_tokenizer));
    let flat_ms = flat_started.elapsed().as_secs_f64() * 1000.0;
    let num_terms = inputs.grammar.num_terminals as usize;
    let active = vec![true; num_terms];
    let shared_started = Instant::now();
    let shared = tdwa::l2p::compute_shared_l2p_equivalence(
        "boundary_shard",
        inputs.merged_tokenizer,
        inputs.vocab,
        inputs.ignore_terminal,
        inputs.grammar,
        &active,
        inputs.disallowed_follows,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(&flat),
        None,
        None,
    )?;
    let shared_wall_ms = shared_started.elapsed().as_secs_f64() * 1000.0;
    let link_profile = BoundaryShardLinkProfile {
        shared_id_map_ms: shared.id_map_ms,
        shared_wall_ms,
        shared_tsids: shared.id_map.num_tsids() as usize,
        shared_itokens: shared.id_map.num_internal_tokens() as usize,
        flat_ms,
        per_shard: Vec::new(),
    };
    let num_components = inputs.component_state_counts.len();
    // Walk plans: explicit (nested) or derived per component (flat, unchanged).
    let plans: Vec<BoundaryShardWalkPlan> = match &inputs.walk_plans {
        Some(plans) => plans
            .iter()
            .map(|plan| BoundaryShardWalkPlan {
                start_component: plan.start_component,
                filter_component: plan.filter_component,
                commit_states: plan.commit_states.clone(),
                retain_non_crossing_paths: plan.retain_non_crossing_paths,
            })
            .collect(),
        None => (0..num_components)
            .map(|index| BoundaryShardWalkPlan {
                start_component: index,
                filter_component: index,
                commit_states: commit_states_for_component(
                    inputs.tokenizer_offsets,
                    inputs.component_state_counts[index],
                    index,
                    inputs.merged_tokenizer.num_states() as usize,
                ),
                retain_non_crossing_paths: inputs.retain_parent_non_crossing_paths
                    && index == 0,
            })
            .collect(),
    };
    // `None` means "empty crossing set, skip"; walk failure is impossible
    // (the shared equivalence above proves the vocab is nonempty).
    let build_one = |plan: &BoundaryShardWalkPlan| -> Option<BuiltBoundaryShardWalk> {
        let output = build_boundary_terminal_dwa(&BoundaryWalkInputs {
            merged_tokenizer: inputs.merged_tokenizer,
            vocab: inputs.vocab,
            grammar: inputs.grammar,
            disallowed_follows: inputs.disallowed_follows,
            ignore_terminal: inputs.ignore_terminal,
            terminal_offsets: inputs.terminal_offsets,
            component_index: plan.filter_component,
            commit_states: &plan.commit_states,
            shared_equivalence: &shared,
            flat_trans: Some(&flat),
            retain_non_crossing_paths: plan.retain_non_crossing_paths,
        })
        .expect("nonempty-vocab shard walks must produce a DWA");
        let candidate_tokens = boundary_accepted_tokens(&output.dwa, &output.id_map);
        if candidate_tokens.is_empty() {
            return None;
        }
        Some(BuiltBoundaryShardWalk {
            start_component: plan.start_component,
            output,
            candidate_tokens,
        })
    };
    let mut built: Vec<BuiltBoundaryShardWalk> = if crate::compiler::macro_parallelism_disabled()
    {
        let mut timings = Vec::with_capacity(plans.len());
        let built: Vec<Option<BuiltBoundaryShardWalk>> = plans
            .iter()
            .map(|plan| {
                let started = Instant::now();
                let result = build_one(plan);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect();
        crate::compiler::report_macro_item_timings("boundary_walk_shard_walks", &timings);
        built.into_iter().flatten().collect()
    } else {
        plans.par_iter().map(build_one).collect::<Vec<_>>().into_iter().flatten().collect()
    };
    built.sort_by_key(|shard| shard.start_component);
    let mut link_profile = link_profile;
    for shard in &built {
        link_profile.per_shard.push((shard.start_component, shard.output.profile.clone()));
    }
    Some((built, link_profile))
}

/// Inputs for a production static boundary link over one composition.
pub(crate) struct WalkStaticLinkInputs<'a> {
    pub parent: &'a Constraint,
    pub children: &'a [CompiledSubgrammarInput<'a>],
    pub vocab: &'a Vocab,
    /// Hybrid selection (parent first, then supplied children): a set bit
    /// requests a static shard. `None` requests static shards everywhere.
    pub static_components: Option<&'a BitSet>,
    /// Terminal offsets of the coordinator's own composed table. The walk link
    /// builds its own control-free splice; the offsets must agree (both lay
    /// out parent-then-children) and are pinned here so a layout change fails
    /// loudly instead of misrouting terminal ownership.
    pub expected_terminal_offsets: &'a [u32],
}

/// Output of a production static boundary link: published static shards plus
/// the per-component candidate metadata the install site needs.
pub(crate) struct WalkStaticLinkOutput {
    pub published_shards: Vec<PublishedStaticBoundaryShard>,
    pub boundary_tokens_by_start_component: Vec<Vec<u32>>,
    /// Effective static selection after virtual-residual exclusions (indexed
    /// in component order). Components not selected need exact dynamic shards.
    pub effective_static_components: BitSet,
    /// True when no component takes a static shard: the caller installs exact
    /// dynamic shards for every component instead.
    pub all_dynamic: bool,
    /// Expected runtime scoped leaf packing (link-component order) and total.
    /// The caller pins the installing runtime's layout against these after
    /// install; a mismatch (nesting the up-front check missed, a layout
    /// change) is a loud decline, never a silent dynamic swap.
    pub expected_leaf_tokenizer_offsets: Vec<u32>,
    pub expected_total_tokenizer_states: u32,
    /// True when the link expanded nested children to leaves. The install
    /// site recursively clears inner-overlay shards in this case (the outer
    /// block shards cover block-inner crossings).
    pub has_nested_components: bool,
}

/// Whether the link's parent is itself a segmented composition. The walk
/// expands nested *children* into intact leaves (nested static links over
/// acyclic, effectively nonnullable children are supported), but an
/// already-composed parent has no intact local table (its table is a composed
/// table with linker controls), so a static link with a composed parent
/// declines loudly (callers needing that shape use Dynamic).
pub(crate) fn walk_static_link_parent_needs_dynamic_fallback(parent: &Constraint) -> bool {
    parent.has_recursive_segmented_parser_tree()
}

/// A link expanded to intact leaves: the parent (leaf 0) plus every direct
/// child, where nested children (themselves segmented compositions) recurse
/// into their overlays' intact component constraints. Links are remapped to
/// leaf coordinates: outer links target block-root leaves, inner-overlay
/// links are translated through each subtree's wrapper-to-leaf map. The
/// runtime recursive layout visits leaves in this same pre-order, so the
/// walk's leaf packing matches the installing runtime's.
pub(crate) struct NestedLinkExpansion<'a> {
    /// Intact leaf constraints in link-component order (leaf 0 is the parent).
    pub leaves: Vec<&'a Constraint>,
    /// Full link set in leaf coordinates (outer + remapped inner links).
    pub links: Vec<ScopedSubgrammarLink>,
    /// Back-to-back leaf terminal offsets plus total.
    pub leaf_terminal_offsets: Vec<u32>,
    pub num_terminals: u32,
    /// Leaf index of each top-level component's root (top 0 -> leaf 0).
    pub top_root_leaves: Vec<u32>,
    /// Leaf indices per top-level component (for block commit unions and
    /// virtual-residual checks).
    pub top_leaf_ranges: Vec<Vec<usize>>,
    /// Top-level terminal offsets/sizes (coordinator layout pin).
    pub top_terminal_offsets: Vec<u32>,
    pub top_terminal_sizes: Vec<u32>,
    /// True when any direct child expanded to more than one leaf.
    pub nested: bool,
}

fn expand_subtree_leaves<'a>(
    constraint: &'a Constraint,
    leaves: &mut Vec<&'a Constraint>,
    links: &mut Vec<ScopedSubgrammarLink>,
) -> Result<u32, String> {
    if !constraint.has_recursive_segmented_parser_tree() {
        leaves.push(constraint);
        return u32::try_from(leaves.len() - 1)
            .map_err(|_| "nested link leaf index overflow".to_string());
    }
    let overlay = constraint.static_dynamic_overlay.as_ref().ok_or_else(|| {
        "nested link subtree reports a segmented tree but has no overlay".to_string()
    })?;
    if overlay.segmented_parser_components.is_empty() {
        return Err("nested link subtree has no segmented components".to_string());
    }
    let mut roots = Vec::with_capacity(overlay.segmented_parser_components.len());
    for component in &overlay.segmented_parser_components {
        roots.push(expand_subtree_leaves(&component.constraint, leaves, links)?);
    }
    for link in &overlay.segmented_parser_links {
        let parent = *roots.get(link.parent_component as usize).ok_or_else(|| {
            format!(
                "nested link references unknown inner parent {}",
                link.parent_component
            )
        })?;
        let child = *roots.get(link.child_component as usize).ok_or_else(|| {
            format!(
                "nested link references unknown inner child {}",
                link.child_component
            )
        })?;
        links.push(ScopedSubgrammarLink {
            parent_component: parent,
            slot_terminal: link.slot_terminal,
            child_component: child,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        });
    }
    Ok(roots[0])
}

/// Expand one production link to intact leaves with leaf-coordinate links.
pub(crate) fn expand_nested_link_leaves<'a>(
    parent: &'a Constraint,
    children: &'a [CompiledSubgrammarInput<'a>],
) -> Result<NestedLinkExpansion<'a>, String> {
    if walk_static_link_parent_needs_dynamic_fallback(parent) {
        return Err(
            "walk static link does not support an already-composed parent; use the Dynamic boundary backend"
                .to_string(),
        );
    }
    let mut leaves: Vec<&'a Constraint> = Vec::new();
    let mut links: Vec<ScopedSubgrammarLink> = Vec::new();
    leaves.push(parent);
    let mut top_root_leaves = vec![0u32];
    let mut top_leaf_ranges = vec![vec![0usize]];
    // Outer links over direct children (block-root remap happens below).
    let outer_links = build_segmented_parser_links(children)?;
    let mut nested = false;
    for child in children.iter() {
        let first_leaf = leaves.len();
        let root = expand_subtree_leaves(child.constraint, &mut leaves, &mut links)?;
        let range: Vec<usize> = (first_leaf..leaves.len()).collect();
        if range.len() != 1 {
            nested = true;
        }
        let root_leaf = u32::try_from(root)
            .map_err(|_| "nested link root leaf index overflow".to_string())?;
        top_root_leaves.push(root_leaf);
        top_leaf_ranges.push(range);
    }
    // Remap outer links (parent 0, child i+1) to leaf coordinates.
    for link in outer_links {
        let child_top = link.child_component as usize;
        if child_top == 0 || child_top > children.len() {
            return Err(format!(
                "nested link outer reference targets unknown top component {child_top}"
            ));
        }
        links.push(ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: link.slot_terminal,
            child_component: top_root_leaves[child_top],
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        });
    }
    let mut leaf_terminal_offsets = Vec::with_capacity(leaves.len());
    let mut total = 0u32;
    for leaf in &leaves {
        leaf_terminal_offsets.push(total);
        total = total
            .checked_add(leaf.table.num_terminals)
            .ok_or_else(|| "nested link terminal domain overflow".to_string())?;
    }
    let mut top_terminal_offsets = Vec::with_capacity(children.len() + 1);
    let mut top_terminal_sizes = Vec::with_capacity(children.len() + 1);
    for range in &top_leaf_ranges {
        let start = leaf_terminal_offsets[range[0]];
        let size: u32 = range
            .iter()
            .map(|&leaf| leaves[leaf].table.num_terminals)
            .try_fold(0u32, |acc, size| {
                acc.checked_add(size)
                    .ok_or_else(|| "nested link block size overflow".to_string())
            })?;
        top_terminal_offsets.push(start);
        top_terminal_sizes.push(size);
    }
    Ok(NestedLinkExpansion {
        leaves,
        links,
        leaf_terminal_offsets,
        num_terminals: total,
        top_root_leaves,
        top_leaf_ranges,
        top_terminal_offsets,
        top_terminal_sizes,
        nested,
    })
}

/// Whole-link exact-dynamic output: no static shards, every component dynamic.
pub(crate) fn dynamic_fallback_walk_link_output(num_components: usize) -> WalkStaticLinkOutput {
    WalkStaticLinkOutput {
        published_shards: Vec::new(),
        boundary_tokens_by_start_component: vec![Vec::new(); num_components],
        effective_static_components: BitSet::new(num_components),
        all_dynamic: true,
        expected_leaf_tokenizer_offsets: Vec::new(),
        expected_total_tokenizer_states: 0,
        has_nested_components: false,
    }
}

/// Effective static selection: the hybrid request minus components whose
/// tokenizer has a virtual residual runtime. Those can query lazily-allocated
/// tokenizer states at runtime that the link-time private TSID map cannot
/// cover, so their shards stay on the exact dynamic walker.
fn effective_static_selection(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    static_components: Option<&BitSet>,
) -> BitSet {
    let num_components = children.len() + 1;
    let mut effective = BitSet::new(num_components);
    for index in 0..num_components {
        let component =
            if index == 0 { parent } else { children[index - 1].constraint };
        let selected = static_components.is_none_or(|bits| bits.contains(index));
        if selected && !component.tokenizer.has_virtual_residual_runtime() {
            effective.set(index);
        }
    }
    effective
}

/// Unbound subgrammar slots of a link as composed-terminal IDs: special-token
/// terminals that are recorded late-grammar slots or whose backing token has
/// no bytes (out-of-vocab linker sentinels), minus the placeholders bound by
/// this link. Bound slots are spliced (gone from actions and rules); unbound
/// slots keep live placeholder shifts that would admit phantom terminal paths
/// through grammars that contribute nothing, so the caller empties them in
/// its private boundary-table copy.
pub(crate) fn unbound_link_slot_terminals(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
) -> Result<Vec<TerminalID>, String> {
    let bound: BTreeSet<u32> = children
        .iter()
        .flat_map(|child| {
            std::iter::once(child.placeholder_terminal)
                .chain(child.additional_placeholder_terminals.iter().copied())
        })
        .collect();
    let mut unbound = Vec::new();
    for special in &parent.special_token_terminals {
        if bound.contains(&special.terminal_id) {
            continue;
        }
        if parent.is_late_grammar_placeholder_terminal(special.terminal_id)
            || parent.token_bytes_for_id(special.token_id).is_none()
        {
            unbound.push(terminal_offsets[0].checked_add(special.terminal_id).ok_or_else(
                || "unbound parent slot terminal offset overflow".to_string(),
            )?);
        }
    }
    for (child_index, child) in children.iter().enumerate() {
        let constraint = child.constraint;
        for special in &constraint.special_token_terminals {
            if constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                || constraint.token_bytes_for_id(special.token_id).is_none()
            {
                unbound.push(
                    terminal_offsets[child_index + 1]
                        .checked_add(special.terminal_id)
                        .ok_or_else(|| {
                            format!(
                                "unbound slot terminal offset overflow for child {child_index}"
                            )
                        })?,
                );
            }
        }
    }
    unbound.sort_unstable();
    unbound.dedup();
    Ok(unbound)
}

/// Intact component tables for one flat link, with per-component standalone
/// ignore terminals (mirroring the splice inputs).
struct LinkComponentTables<'a> {
    tables: Vec<&'a GLRTable>,
    ignores: Vec<Option<TerminalID>>,
}

impl ParserComponentTableSource for LinkComponentTables<'_> {
    fn component_count(&self) -> usize {
        self.tables.len()
    }

    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.tables.get(component as usize).copied()
    }

    fn component_ignore_terminal(&self, component: u32) -> Option<TerminalID> {
        self.ignores.get(component as usize).copied().flatten()
    }
}

/// Exact boundary table for one flat link (AUDIT/PIN ROUTE ONLY): the
/// disjoint-component provider over the intact component tables (per-link
/// Call/Return with call-specific parent targets), materialized and
/// control-eliminated. Production static linking no longer uses this object
/// (see `build_walk_static_boundary_link`); it remains for the called-frame
/// regression pins that compare provider-exact against splice-inexact
/// publication. Its state alphabet is the live leaf coordinate (components
/// back-to-back, local ids intact), and call-site provenance survives
/// elimination by construction (Call pushes the call-specific parent target
/// beneath the shared child start; Finish exposes it), so divergent
/// multi-site placeholders are exact. A single flattened shared start row
/// cannot express that provenance and must not be used here.
fn link_provider_boundary_table(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    num_terminals: u32,
    global_ignores: bool,
) -> Result<GLRTable, String> {
    let mut tables = Vec::with_capacity(children.len() + 1);
    let mut ignores = Vec::with_capacity(children.len() + 1);
    tables.push(&parent.table);
    ignores.push((!global_ignores).then_some(parent.ignore_terminal).flatten());
    for child in children {
        tables.push(&child.constraint.table);
        ignores.push(
            (!global_ignores)
                .then_some(child.constraint.ignore_terminal)
                .flatten(),
        );
    }
    for (index, table) in tables.iter().enumerate() {
        if !table.control_terminals.is_empty() {
            return Err(format!(
                "walk static link component {index} carries linker controls; control-bearing components need dynamic shards",
            ));
        }
    }
    let source = LinkComponentTables { tables, ignores };
    let links = build_segmented_parser_links(children)?;
    let provider = DisjointComponentActionProvider::new(&source, &links)?;
    let mut terminal_symbols = Vec::with_capacity(num_terminals as usize);
    for global in 0..num_terminals {
        let owner = terminal_offsets
            .iter()
            .rposition(|&offset| offset <= global)
            .ok_or_else(|| {
                format!("composed terminal {global} has no owning component")
            })?;
        let local = global.checked_sub(terminal_offsets[owner]).ok_or_else(|| {
            format!("composed terminal {global} underflows its component offset")
        })?;
        let mut symbols = SmallVec::<[ScopedParserSymbol; 4]>::new();
        symbols.push(ScopedParserSymbol::Terminal {
            component: owner as u32,
            terminal: local,
        });
        terminal_symbols.push(symbols);
    }
    let table =
        materialize_control_eliminated_scoped_provider_table(&provider, &terminal_symbols)?;
    if table.num_terminals != num_terminals {
        return Err(format!(
            "provider boundary table terminal domain {} differs from composed {num_terminals}",
            table.num_terminals,
        ));
    }
    if !table.control_terminals.is_empty() {
        return Err("provider boundary table unexpectedly retains controls".to_string());
    }
    Ok(table)
}

/// Build every static boundary shard for one production link: a packed splice
/// (rules + terminal layout only), the signed-transfer link context over
/// intact local tables, a merged tokenizer, one shared equivalence, one
/// standard crossing-filtered walk per component, then per-shard signed-NWA
/// compilation (scoped ordinary transfers + Entry/Finish exports, exact
/// negative resolution, table-free normalization) and publication. This
/// mirrors the gate's `link_and_time` walks; no provider-materialized table
/// and no exact_control_elimination exists on this route.
pub(crate) fn build_walk_static_boundary_link(
    inputs: &WalkStaticLinkInputs,
) -> Result<WalkStaticLinkOutput, String> {
    let parent = inputs.parent;
    let children = inputs.children;
    // Expand nested children to intact leaves (flat links: leaves are exactly
    // the direct components, so the flat path below is unchanged).
    let expansion = expand_nested_link_leaves(parent, children)?;
    if expansion.nested {
        return build_walk_static_boundary_link_nested(inputs, &expansion);
    }
    let vocab = inputs.vocab;
    let num_components = children.len() + 1;
    let effective =
        effective_static_selection(parent, children, inputs.static_components);

    // The signed-transfer route links intact local tables and never invokes
    // control elimination. Reject control-bearing components first, before any
    // elimination-adjacent call, so even declined inputs cannot trigger it.
    for component in std::iter::once(parent).chain(children.iter().map(|child| child.constraint)) {
        if !component.table.control_terminals.is_empty() {
            return Err(
                "walk static link requires control-free component tables".to_string(),
            );
        }
    }

    // Packed splice: rules + terminal layout only. The packed shape is the
    // exact production construction (per-caller overlays); it feeds the
    // grammar/follows analysis and the coordinator-layout pin, never the
    // boundary parser. The boundary parser is compiled by the signed-transfer
    // compiler below from intact local tables; no provider-materialized table
    // and no exact_control_elimination exists on this route.
    let global_ignores = component_ignores_are_globally_erasable(parent, children);
    let parent_rules = parent.retained_table_rules()?;
    let child_rules = children
        .iter()
        .map(|child| child.constraint.retained_table_rules())
        .collect::<Result<Vec<_>, String>>()?;
    let table_inputs: Vec<SubgrammarTableInput> = children
        .iter()
        .map(|child| SubgrammarTableInput {
            placeholder_terminal: child.placeholder_terminal,
            additional_placeholder_terminals: child.additional_placeholder_terminals,
            table: &child.constraint.table,
            ignore_terminal: (!global_ignores)
                .then_some(child.constraint.ignore_terminal)
                .flatten(),
            start_nullable: child.constraint.table.embedded_start_nullable(),
        })
        .collect();
    let mut composed = compose_subgrammar_tables_with_rules(
        &parent.table,
        parent_rules,
        (!global_ignores).then_some(parent.ignore_terminal).flatten(),
        &table_inputs,
        child_rules.as_slice(),
    )?;
    eliminate_composed_runtime_controls(&mut composed)?;
    if !composed.table.control_terminals.is_empty() || !composed.control_terminals.is_empty()
    {
        return Err(
            "walk static link requires a control-free spliced table".to_string(),
        );
    }
    if composed.terminal_offsets.as_slice() != inputs.expected_terminal_offsets {
        return Err(
            "walk static link terminal layout differs from coordinator table".to_string(),
        );
    }
    // Signed-transfer link context: intact local tables, provider-layout
    // injections, validated Entry/Finish contracts.
    let unbound = unbound_link_slot_terminals(parent, children, &composed.terminal_offsets)?;
    for &terminal in &unbound {
        if terminal as usize >= composed.table.num_terminals as usize {
            return Err(format!(
                "unbound slot terminal {terminal} lies outside the composed terminal domain",
            ));
        }
    }
    let signed_context = crate::compiler::boundary_transfer::build_signed_link_context(
        parent,
        children,
        &composed.terminal_offsets,
        composed.table.num_terminals,
        global_ignores,
        unbound.into_iter().collect(),
    )?;

    // Merged tokenizer + ignores (mirror the gate's low_level_compose).
    let terminal_names = merged_terminal_display_names(parent, children);
    let tokenizer_inputs: Vec<(&Tokenizer, u32)> = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
        .map(|(index, constraint)| {
            (constraint.composition_tokenizer(), composed.terminal_offsets[index])
        })
        .collect();
    let (mut merged, tokenizer_offsets) =
        Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    if tokenizer_offsets.first() != Some(&1) {
        return Err(
            "merged tokenizer must place the fresh reset fan-out at state 0".to_string(),
        );
    }
    if merged.terminal_exprs().is_none() {
        let all: Vec<&Constraint> = std::iter::once(parent)
            .chain(children.iter().map(|child| child.constraint))
            .collect();
        if let Some(exprs) = merged_retained_terminal_exprs(
            &all,
            &composed.terminal_offsets,
            composed.table.num_terminals,
        ) {
            merged.restore_terminal_exprs(Some(exprs))?;
        }
    }
    let ignores =
        merged_ignore_terminals(parent, children, &composed.terminal_offsets, global_ignores);

    // Grammar + pairwise follows over the non-emptied composed rules.
    let augmented_start = composed
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
    let grammar = AnalyzedGrammar::from_composed_rules(
        composed.table.rules.clone(),
        composed.table.num_terminals,
        terminal_names,
        composed.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    let disallowed = compute_disallowed_follows(&grammar);
    let component_state_counts: Vec<u32> = tokenizer_inputs
        .iter()
        .map(|(tokenizer, _)| tokenizer.num_states())
        .collect();

    // Expected runtime scoped leaf packing (link-component order).
    let mut expected_leaf_tokenizer_offsets = Vec::with_capacity(num_components);
    let mut expected_total = 0u32;
    for &count in &component_state_counts {
        expected_leaf_tokenizer_offsets.push(expected_total);
        expected_total = expected_total
            .checked_add(count)
            .ok_or_else(|| "scoped tokenizer state count overflow".to_string())?;
    }

    let empty_output = || WalkStaticLinkOutput {
        published_shards: Vec::new(),
        boundary_tokens_by_start_component: vec![Vec::new(); num_components],
        effective_static_components: effective.clone(),
        all_dynamic: false,
        expected_leaf_tokenizer_offsets: expected_leaf_tokenizer_offsets.clone(),
        expected_total_tokenizer_states: expected_total,
        has_nested_components: false,
    };
    let retain_parent_non_crossing_paths = children
        .iter()
        .any(|child| child.constraint.table.embedded_start_nullable());
    let Some((built, _link_profile)) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
        merged_tokenizer: &merged,
        vocab,
        grammar: &grammar,
        disallowed_follows: &disallowed,
        ignore_terminal: ignores.canonical,
        terminal_offsets: &composed.terminal_offsets,
        tokenizer_offsets: &tokenizer_offsets,
        component_state_counts: &component_state_counts,
        retain_parent_non_crossing_paths,
        walk_plans: None,
    }) else {
        // Empty vocab: nothing can cross; pure component masks, no shards.
        return Ok(empty_output());
    };

    // Signed-transfer shard publication: scoped ordinary transfers +
    // Entry/Finish exports composed in the control-aware signed NWA, exact
    // negative resolution, table-free normalization. No provider table, no
    // elimination; unbound slots carry empty-language semantics inside the
    // fragment library.
    let mut published = Vec::new();
    let mut tokens_by_component: Vec<Vec<u32>> = vec![Vec::new(); num_components];
    for shard in built {
        tokens_by_component[shard.start_component] =
            shard.candidate_tokens.iter().copied().collect();
        if !effective.contains(shard.start_component) {
            continue;
        }
        let emitted = boundary_emitted_terminals(&shard.output.dwa, composed.table.num_terminals as usize);
        let library = crate::compiler::boundary_transfer::build_fragment_library(
            &signed_context,
            &emitted,
            shard.start_component as u32,
        )?;
        let compiled = crate::compiler::boundary_transfer::compile_signed_shard_parser(
            &signed_context,
            &library,
            &shard.output.dwa,
            &shard.output.id_map,
            shard.start_component as u32,
        )?;
        let work = WalkBoundaryShardWork {
            start_component: shard.start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
            id_map: shard.output.id_map,
            candidate_tokens: shard
                .candidate_tokens
                .into_iter()
                .collect::<Vec<_>>()
                .into(),
        };
        let (one, _publish_profile) = crate::compiler::boundary_transfer::publish_signed_shard(
            work,
            compiled,
            &tokenizer_offsets,
            &component_state_counts,
        )?;
        published.push(one);
    }
    published.sort_by_key(|shard| shard.start_component);
    Ok(WalkStaticLinkOutput {
        published_shards: published,
        boundary_tokens_by_start_component: tokens_by_component,
        effective_static_components: effective,
        all_dynamic: false,
        expected_leaf_tokenizer_offsets,
        expected_total_tokenizer_states: expected_total,
        has_nested_components: false,
    })
}

/// Recursive packed splice for one nested block: inner blocks first, then
/// this level (rules + terminal layout only, mirroring the proven nested
/// fixture). Only the returned rules/offsets feed grammar analysis; no parser
/// behavior is ever derived from the packed shape.
fn splice_nested_block(
    root_leaf: usize,
    expansion: &NestedLinkExpansion<'_>,
    global_ignores: bool,
) -> Result<ComposedTable, String> {
    let root = expansion.leaves[root_leaf];
    let mut child_roots: Vec<u32> = expansion
        .links
        .iter()
        .filter(|link| link.parent_component as usize == root_leaf)
        .map(|link| link.child_component)
        .collect();
    child_roots.sort_unstable();
    child_roots.dedup();
    let mut child_tables = Vec::with_capacity(child_roots.len());
    let mut slot_lists: Vec<Vec<u32>> = Vec::with_capacity(child_roots.len());
    let mut nullables = Vec::with_capacity(child_roots.len());
    for &child_root in &child_roots {
        child_tables.push(splice_nested_block(
            child_root as usize,
            expansion,
            global_ignores,
        )?);
        let mut slots: Vec<u32> = expansion
            .links
            .iter()
            .filter(|link| {
                link.parent_component as usize == root_leaf
                    && link.child_component == child_root
            })
            .map(|link| link.slot_terminal)
            .collect();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return Err(format!(
                "nested link block {root_leaf} has a child block with no slot terminal"
            ));
        }
        let nullable = expansion
            .links
            .iter()
            .find(|link| {
                link.parent_component as usize == root_leaf
                    && link.child_component == child_root
            })
            .map(|link| link.child_start_nullable)
            .unwrap_or(false);
        slot_lists.push(slots);
        nullables.push(nullable);
    }
    let child_rules: Vec<&[crate::grammar::flat::Rule]> = child_tables
        .iter()
        .map(|tabled: &ComposedTable| tabled.table.rules.as_slice())
        .collect();
    let table_inputs: Vec<SubgrammarTableInput> = child_roots
        .iter()
        .zip(child_tables.iter())
        .zip(slot_lists.iter())
        .zip(nullables.iter())
        .map(|(((&child_root, table), slots), &nullable)| {
            let (placeholder, additionals) = slots
                .split_first()
                .expect("nested slot list is nonempty");
            let child_leaf = expansion.leaves[child_root as usize];
            SubgrammarTableInput {
                placeholder_terminal: *placeholder,
                additional_placeholder_terminals: additionals,
                table: &table.table,
                ignore_terminal: (!global_ignores)
                    .then_some(child_leaf.ignore_terminal)
                    .flatten(),
                start_nullable: nullable,
            }
        })
        .collect();
    let mut composed = compose_subgrammar_tables_with_rules(
        &root.table,
        root.retained_table_rules()?,
        (!global_ignores).then_some(root.ignore_terminal).flatten(),
        &table_inputs,
        child_rules.as_slice(),
    )?;
    eliminate_composed_runtime_controls(&mut composed)?;
    if !composed.table.control_terminals.is_empty() || !composed.control_terminals.is_empty() {
        return Err(
            "walk static link requires a control-free spliced table".to_string(),
        );
    }
    Ok(composed)
}

/// Unbound subgrammar slots in leaf coordinates: a slot is bound iff some
/// link fills it; the rest follows the flat helper's rule per leaf
/// (late-grammar placeholders or out-of-vocab linker sentinels).
fn nested_unbound_slot_terminals(
    expansion: &NestedLinkExpansion<'_>,
) -> Result<Vec<TerminalID>, String> {
    let bound: BTreeSet<u32> = expansion
        .links
        .iter()
        .map(|link| {
            expansion.leaf_terminal_offsets[link.parent_component as usize]
                .checked_add(link.slot_terminal)
                .ok_or_else(|| "bound slot terminal offset overflow".to_string())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut unbound = Vec::new();
    for (leaf_index, leaf) in expansion.leaves.iter().enumerate() {
        for special in &leaf.special_token_terminals {
            let global = expansion.leaf_terminal_offsets[leaf_index]
                .checked_add(special.terminal_id)
                .ok_or_else(|| {
                    format!("slot terminal offset overflow for leaf {leaf_index}")
                })?;
            if bound.contains(&global) {
                continue;
            }
            if leaf.is_late_grammar_placeholder_terminal(special.terminal_id)
                || leaf.token_bytes_for_id(special.token_id).is_none()
            {
                unbound.push(global);
            }
        }
    }
    unbound.sort_unstable();
    unbound.dedup();
    Ok(unbound)
}

/// Nested production static link: the flat pipeline over recursively expanded
/// intact leaves with the full multi-level link set (see the flat function
/// for the pipeline stages). One walk per top-level component; publication
/// per top-level component with leaf tokenizer coordinates.
fn build_walk_static_boundary_link_nested(
    inputs: &WalkStaticLinkInputs,
    expansion: &NestedLinkExpansion<'_>,
) -> Result<WalkStaticLinkOutput, String> {
    let vocab = inputs.vocab;
    let num_components = inputs.children.len() + 1;
    let leaves = &expansion.leaves;

    // The signed-transfer route links intact local tables and never invokes
    // control elimination. Reject control-bearing leaves first.
    for leaf in leaves.iter() {
        if !leaf.table.control_terminals.is_empty() {
            return Err(
                "walk static link requires control-free component tables".to_string(),
            );
        }
    }

    // Effective static selection over top-level components: a requested block
    // needs every leaf in its subtree free of virtual residual runtimes, whose
    // lazily-allocated tokenizer states the link-time coordinate cannot cover.
    // A requested-but-blocked block is a loud error, never a silent dynamic
    // swap; unrequested blocks stay on the exact dynamic walker (hybrid).
    let mut effective = BitSet::new(num_components);
    for top in 0..num_components {
        let requested = inputs
            .static_components
            .is_none_or(|bits| bits.contains(top));
        if !requested {
            continue;
        }
        let mut blocked = None;
        for &leaf_index in &expansion.top_leaf_ranges[top] {
            if leaves[leaf_index].tokenizer.has_virtual_residual_runtime() {
                blocked = Some(leaf_index);
                break;
            }
        }
        if let Some(leaf_index) = blocked {
            return Err(format!(
                "walk static link component {top} requested a static shard but leaf {leaf_index} has a virtual residual runtime; leave it unselected (hybrid) or use the Dynamic boundary backend",
            ));
        }
        effective.set(top);
    }

    let global_ignores = leaf_ignores_are_globally_erasable(leaves);

    // Signed-transfer context over leaves first: the bounded-closure
    // certificate declines nullable bound children and link-level cycles
    // loudly here, before any walks run.
    let unbound = nested_unbound_slot_terminals(expansion)?;
    for &terminal in &unbound {
        if terminal >= expansion.num_terminals {
            return Err(format!(
                "unbound slot terminal {terminal} lies outside the composed terminal domain",
            ));
        }
    }
    let signed_context = crate::compiler::boundary_transfer::build_signed_link_context_from_parts(
        leaves.iter().map(|leaf| &leaf.table).collect(),
        leaves.iter().map(|leaf| leaf.ignore_terminal).collect(),
        expansion.links.clone(),
        &expansion.leaf_terminal_offsets,
        expansion.num_terminals,
        global_ignores,
        unbound.into_iter().collect(),
    )?;

    // Packed recursive splice (rules + layout only) for grammar analysis and
    // the coordinator-layout pin.
    let composed = splice_nested_block(0, expansion, global_ignores)?;
    if composed.table.num_terminals != expansion.num_terminals {
        return Err(format!(
            "nested walk terminal domain {} differs from leaf domain {}",
            composed.table.num_terminals, expansion.num_terminals,
        ));
    }
    if composed.terminal_offsets.as_slice() != inputs.expected_terminal_offsets {
        return Err(
            "walk static link terminal layout differs from coordinator table".to_string(),
        );
    }
    if composed.terminal_offsets.as_slice() != expansion.top_terminal_offsets.as_slice() {
        return Err(
            "nested walk top-level layout differs from leaf expansion".to_string(),
        );
    }

    // Merged tokenizer + names + ignores over leaves.
    let terminal_names = merged_leaf_terminal_display_names(leaves);
    let tokenizer_inputs: Vec<(&Tokenizer, u32)> = leaves
        .iter()
        .enumerate()
        .map(|(index, leaf)| {
            (
                leaf.composition_tokenizer(),
                expansion.leaf_terminal_offsets[index],
            )
        })
        .collect();
    let (mut merged, tokenizer_offsets) =
        Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    if tokenizer_offsets.first() != Some(&1) {
        return Err(
            "merged tokenizer must place the fresh reset fan-out at state 0".to_string(),
        );
    }
    if merged.terminal_exprs().is_none() {
        if let Some(exprs) = merged_retained_terminal_exprs(
            leaves,
            &expansion.leaf_terminal_offsets,
            expansion.num_terminals,
        ) {
            merged.restore_terminal_exprs(Some(exprs))?;
        }
    }
    let ignores =
        merged_leaf_ignore_terminals(leaves, &expansion.leaf_terminal_offsets, global_ignores);

    // Grammar + pairwise follows over the spliced rules.
    let augmented_start = composed
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
    let grammar = AnalyzedGrammar::from_composed_rules(
        composed.table.rules.clone(),
        composed.table.num_terminals,
        terminal_names,
        composed.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    let disallowed = compute_disallowed_follows(&grammar);
    let leaf_state_counts: Vec<u32> = tokenizer_inputs
        .iter()
        .map(|(tokenizer, _)| tokenizer.num_states())
        .collect();

    // Expected runtime scoped leaf packing (leaf order).
    let mut expected_leaf_tokenizer_offsets = Vec::with_capacity(leaves.len());
    let mut expected_total = 0u32;
    for &count in &leaf_state_counts {
        expected_leaf_tokenizer_offsets.push(expected_total);
        expected_total = expected_total
            .checked_add(count)
            .ok_or_else(|| "scoped tokenizer state count overflow".to_string())?;
    }

    let empty_output = || WalkStaticLinkOutput {
        published_shards: Vec::new(),
        boundary_tokens_by_start_component: vec![Vec::new(); num_components],
        effective_static_components: effective.clone(),
        all_dynamic: false,
        expected_leaf_tokenizer_offsets: expected_leaf_tokenizer_offsets.clone(),
        expected_total_tokenizer_states: expected_total,
        has_nested_components: true,
    };
    let retain_parent_non_crossing_paths = leaves
        .iter()
        .skip(1)
        .any(|leaf| leaf.table.embedded_start_nullable());
    // One walk plan per top-level component: blocks walk from their root leaf
    // with the union of the block leaves' commit states.
    let total_states = merged.num_states() as usize;
    let mut walk_plans = Vec::with_capacity(num_components);
    for top in 0..num_components {
        let mut commit = vec![false; total_states];
        for &leaf_index in &expansion.top_leaf_ranges[top] {
            let start = tokenizer_offsets[leaf_index] as usize;
            let end = start + leaf_state_counts[leaf_index] as usize;
            if end > commit.len() {
                return Err(
                    "nested block commit range lies outside the merged tokenizer".to_string(),
                );
            }
            commit[start..end].fill(true);
        }
        walk_plans.push(BoundaryShardWalkPlan {
            start_component: top,
            filter_component: expansion.top_root_leaves[top] as usize,
            commit_states: commit,
            retain_non_crossing_paths: retain_parent_non_crossing_paths && top == 0,
        });
    }
    let Some((built, _link_profile)) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
        merged_tokenizer: &merged,
        vocab,
        grammar: &grammar,
        disallowed_follows: &disallowed,
        ignore_terminal: ignores.canonical,
        terminal_offsets: &expansion.leaf_terminal_offsets,
        tokenizer_offsets: &tokenizer_offsets,
        component_state_counts: &leaf_state_counts,
        retain_parent_non_crossing_paths,
        walk_plans: Some(walk_plans),
    }) else {
        // Empty vocab: nothing can cross; pure component masks, no shards.
        return Ok(empty_output());
    };

    // Signed-transfer shard publication over the leaf context, one shard per
    // top-level component.
    let mut published = Vec::new();
    let mut tokens_by_component: Vec<Vec<u32>> = vec![Vec::new(); num_components];
    for shard in built {
        tokens_by_component[shard.start_component] =
            shard.candidate_tokens.iter().copied().collect();
        if !effective.contains(shard.start_component) {
            continue;
        }
        let emitted = boundary_emitted_terminals(&shard.output.dwa, composed.table.num_terminals as usize);
        let library = crate::compiler::boundary_transfer::build_fragment_library(
            &signed_context,
            &emitted,
            shard.start_component as u32,
        )?;
        let compiled = crate::compiler::boundary_transfer::compile_signed_shard_parser(
            &signed_context,
            &library,
            &shard.output.dwa,
            &shard.output.id_map,
            shard.start_component as u32,
        )?;
        let work = WalkBoundaryShardWork {
            start_component: shard.start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
            id_map: shard.output.id_map,
            candidate_tokens: shard
                .candidate_tokens
                .into_iter()
                .collect::<Vec<_>>()
                .into(),
        };
        let (one, _publish_profile) = crate::compiler::boundary_transfer::publish_signed_shard(
            work,
            compiled,
            &tokenizer_offsets,
            &leaf_state_counts,
        )?;
        published.push(one);
    }
    published.sort_by_key(|shard| shard.start_component);
    Ok(WalkStaticLinkOutput {
        published_shards: published,
        boundary_tokens_by_start_component: tokens_by_component,
        effective_static_components: effective,
        all_dynamic: false,
        expected_leaf_tokenizer_offsets,
        expected_total_tokenizer_states: expected_total,
        has_nested_components: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
    use crate::compiler::constraint_compose::{
        CompiledSubgrammarInput, SegmentedBoundaryBackend,
        component_ignores_are_globally_erasable, compose_constraints_owned_parent_segmented,
        eliminate_composed_runtime_controls, install_published_static_boundary_shards,
        load_vocab, merged_ignore_terminals, merged_retained_terminal_exprs,
        merged_terminal_display_names, publish_walk_boundary_shard_work, WalkBoundaryShardWork,
    };
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::table::{
        GLRTable, SubgrammarTableInput, compose_subgrammar_tables, control_elimination_budget_exhausted,
        control_elimination_run_count, empty_terminals_in_composed_table,
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

    /// Step-4c boundary parser table: the control-free spliced composed table
    /// (the Phase-1 object — never the dynamic path's control-bearing
    /// recursive table) with every unbound slot terminal emptied. Controls
    /// are a runtime device of the dynamic path; they must not appear in the
    /// static boundary parser's table at all, so a nonzero control count is
    /// a hard error, not a fallback. `unbound_slots` holds
    /// (component_index, component-local terminal) pairs; they are resolved
    /// to composed IDs through the composed terminal offsets.
    fn prepare_spliced_boundary_table(
        name: &str,
        composed: &LowLevelComposed,
        unbound_slots: &[(usize, TerminalID)],
    ) -> (Arc<GLRTable>, f64) {
        assert!(
            composed.table.table.control_terminals.is_empty(),
            "{name}: spliced composed table must be control-free, has {:?}",
            composed.table.table.control_terminals,
        );
        assert!(
            composed.table.control_terminals.is_empty(),
            "{name}: composed control set must be empty",
        );
        let started = Instant::now();
        let mut table = composed.table.table.clone();
        let emptied: Vec<TerminalID> = unbound_slots
            .iter()
            .map(|(component, local)| {
                composed.table.terminal_offsets[*component]
                    .checked_add(*local)
                    .expect("slot terminal offset overflow")
            })
            .collect();
        for &terminal in &emptied {
            assert!(
                (terminal as usize) < table.num_terminals as usize,
                "{name}: unbound slot terminal {terminal} out of range",
            );
        }
        empty_terminals_in_composed_table(&mut table, &emptied);
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "BOUNDARY_TABLE name={name} states={} terms={} controls={} emptied={} prep_ms={ms:.3}",
            table.num_states,
            table.num_terminals,
            table.control_terminals.len(),
            emptied.len(),
        );
        (Arc::new(table), ms)
    }

    /// Pin the installing runtime's scoped tokenizer packing: leaves packed
    /// back-to-back from 0 in link-component order (parent first). The walk
    /// shard's private TSID map is indexed by exactly this coordinate; a
    /// layout change here must update `publish_walk_boundary_shard_work`.
    fn assert_scoped_tokenizer_packing(
        name: &str,
        installed: &Constraint,
        component_state_counts: &[u32],
    ) {
        let layout = installed
            .recursive_parser_layout()
            .expect("recursive layout")
            .expect("recursive layout present");
        let mut expected = Vec::with_capacity(component_state_counts.len());
        let mut next = 0u32;
        for &count in component_state_counts {
            expected.push(next);
            next = next.checked_add(count).expect("scoped total overflow");
        }
        assert_eq!(
            layout.leaf_tokenizer_state_offsets, expected,
            "{name}: runtime leaf packing must match link-component order",
        );
        assert_eq!(
            layout.total_tokenizer_states, next,
            "{name}: runtime scoped total must match link components",
        );
    }

    /// Component-local IDs of `TOOL_ARGS_SLOT_*` terminals in `constraint`
    /// except the bound ones. The sweep parents/children follow the
    /// `TOOL_ARGS_SLOT_{index}` slot convention; the shared `slot`-named
    /// JSON-string terminals (`subgrammar2::"..."`) never match the prefix.
    fn unbound_slot_terminals(
        constraint: &Constraint,
        bound_names: &[&str],
    ) -> Vec<TerminalID> {
        constraint
            .terminal_display_names
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                name.starts_with("TOOL_ARGS_SLOT_") && !bound_names.contains(&name.as_str())
            })
            .map(|(id, _)| id as TerminalID)
            .collect()
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
                retain_non_crossing_paths: false,
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

    struct Selected10Outer {
        vocab: Vocab,
        core: Constraint,
        dispatch: Constraint,
        composed: LowLevelComposed,
        grammar: AnalyzedGrammar,
        disallowed: BTreeMap<u32, BitSet>,
    }

    fn load_selected10_outer() -> Selected10Outer {
        use std::path::Path;

        let root = std::env::var("PHASE1_DIR").unwrap_or_else(|_| {
            "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29".to_string()
        });
        let root = Path::new(&root).to_path_buf();
        let vocab_path = std::env::var("PHASE1_VOCAB")
            .unwrap_or_else(|_| root.join("vocab_dump.bin").to_string_lossy().into_owned());
        let vocab = load_vocab(&vocab_path);
        assert_eq!(
            vocab.entries_map().len(),
            128_256,
            "selected10 expects the CFA Llama-3.1 vocabulary",
        );
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
        Selected10Outer { vocab, core, dispatch, composed, grammar, disallowed }
    }

    fn require_release_selected10_gate(name: &str) {
        assert!(
            !cfg!(debug_assertions),
            "{name} is a selected10 integration/performance gate and must be run with an optimized test binary; rerun with `cargo test --release ...`"
        );
    }

    /// Production-path selected10 crossing gate: one shared equivalence, one
    /// standard walk per component, NWA-level crossing filter. Asserts the
    /// 143-token dispatch crossing set (MINBOUND oracle dump) and the
    /// 26-state true-minimal crossing DWA, plus the empty core shard.
    #[test]
    #[ignore]
    fn selected10_boundary_walk_crossing() {
        require_release_selected10_gate("selected10_boundary_walk_crossing");
        let fixture = load_selected10_outer();
        let vocab = &fixture.vocab;
        let core = &fixture.core;
        let dispatch = &fixture.dispatch;
        let composed = &fixture.composed;
        let grammar = &fixture.grammar;
        let disallowed = &fixture.disallowed;
        let active = vec![true; grammar.num_terminals as usize];
        let flat: Arc<[u32]> =
            Arc::from(tdwa::l1::build_flat_transition_table(&composed.tokenizer));
        let shared_started = Instant::now();
        let shared = shared_equivalence_for(
            &composed.tokenizer,
            vocab,
            composed.ignore_canonical,
            grammar,
            &active,
            disallowed,
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

        // Frozen output of the independent Phase-1 MINBOUND B-A oracle for
        // this exact selected10 fixture. The original artifact is preserved
        // in the 2026-09-18 linker compaction with SHA-256
        // 689820493b69f3a603c1c38235a74b4b91aa2a6449f22cddbdbc5a5e3d60bf6d.
        // Keep the oracle in-repo so this regression never depends on
        // historical /tmp state or a machine-local PHASE1_DUMP_DIR.
        let oracle_text = include_str!("testdata/selected10_minbound_tokens.txt");
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
                vocab,
                grammar,
                disallowed_follows: disallowed,
                ignore_terminal: composed.ignore_canonical,
                terminal_offsets: &composed.table.terminal_offsets,
                component_index: index,
                commit_states: &commit,
                shared_equivalence: &shared,
                flat_trans: Some(&flat),
                retain_non_crossing_paths: false,
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

    /// Convergence-bound regression gate (Phase 2 step 4): the selected10
    /// outer recursive table's control elimination provably diverges (cyclic
    /// reduce graph defeats the DFS visiting set), so it must decline via
    /// the convergence budget quickly instead of stalling the link.
    #[test]
    #[ignore]
    fn recursive_table_convergence_bound_fails_fast() {
        let fixture = load_selected10_outer();
        let dyn_children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let dynamic = compose_constraints_owned_parent_segmented(
            fixture.core.clone(),
            &dyn_children,
            &fixture.vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        let started = Instant::now();
        let error = dynamic
            .recursive_control_eliminated_parser_table()
            .expect_err("outer recursive table must decline via the convergence budget");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("BOUNDARY_INSTALL convergence_decline_ms={elapsed_ms:.3} error={error}");
        assert!(
            control_elimination_budget_exhausted(&error),
            "decline must be budget exhaustion, got: {error}"
        );
        assert!(elapsed_ms < 15_000.0, "decline must fail fast, took {elapsed_ms:.0} ms");
    }

    /// Step-4 install gate: build walk shards through the link orchestration,
    /// publish against the spliced control-free boundary table (unbound
    /// slots emptied), install by replacing the dynamic composition's
    /// dynamic shards, and require mask-for-mask equality with the dynamic
    /// backend over a byte-prefix corpus. No fallback: a nonzero control
    /// count or a publish failure is a hard gate failure.
    #[test]
    #[ignore]
    fn selected10_walk_shard_install_matches_dynamic() {
        require_release_selected10_gate("selected10_walk_shard_install_matches_dynamic");
        let fixture = load_selected10_outer();
        let dyn_children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let compose_started = Instant::now();
        let dynamic = compose_constraints_owned_parent_segmented(
            fixture.core.clone(),
            &dyn_children,
            &fixture.vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        eprintln!(
            "BOUNDARY_INSTALL dynamic_compose_ms={:.3}",
            compose_started.elapsed().as_secs_f64() * 1000.0,
        );

        let component_state_counts =
            [fixture.core.tokenizer.num_states(), fixture.dispatch.tokenizer.num_states()];
        let link_started = Instant::now();
        let (built, link_profile) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
            merged_tokenizer: &fixture.composed.tokenizer,
            vocab: &fixture.vocab,
            grammar: &fixture.grammar,
            disallowed_follows: &fixture.disallowed,
            ignore_terminal: fixture.composed.ignore_canonical,
            terminal_offsets: &fixture.composed.table.terminal_offsets,
            tokenizer_offsets: &fixture.composed.tokenizer_offsets,
            component_state_counts: &component_state_counts,
            // Core and dispatch schemas are never start-nullable.
            retain_parent_non_crossing_paths: false,
            walk_plans: None,
        })
        .expect("walk shards");
        eprintln!(
            "BOUNDARY_INSTALL walks shared_ms={:.3} flat_ms={:.3} shards={:?} link_ms={:.3}",
            link_profile.shared_wall_ms,
            link_profile.flat_ms,
            built
                .iter()
                .map(|shard| (
                    shard.start_component,
                    shard.candidate_tokens.len(),
                    shard.output.profile.walk_ms,
                ))
                .collect::<Vec<_>>(),
            link_started.elapsed().as_secs_f64() * 1000.0,
        );
        assert_eq!(built.len(), 1, "only the dispatch shard is nonempty");
        assert_eq!(built[0].start_component, 1);
        assert_eq!(built[0].candidate_tokens.len(), 143);

        // Outer unbound slots: dispatch's TOOL_ARGS_SLOT_* (component 1, none
        // bound — the 10-way segmented dispatch link is unavailable) plus any
        // unbound parent slots (core binds its only slot; expect none).
        let mut unbound: Vec<(usize, TerminalID)> = unbound_slot_terminals(&fixture.core, &[])
            .into_iter()
            .map(|local| (0usize, local))
            .collect();
        unbound.extend(
            unbound_slot_terminals(&fixture.dispatch, &[]).into_iter().map(|local| (1usize, local)),
        );
        let (boundary_table, table_ms) =
            prepare_spliced_boundary_table("outer", &fixture.composed, &unbound);
        eprintln!("BOUNDARY_INSTALL table_ms={table_ms:.3}");
        let mut published = Vec::with_capacity(built.len());
        for shard in built {
            let work = WalkBoundaryShardWork {
                start_component: shard.start_component as u32,
                terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
                id_map: shard.output.id_map,
                candidate_tokens: shard.candidate_tokens.into_iter().collect::<Vec<_>>().into(),
            };
            let (shard, profile) = publish_walk_boundary_shard_work(
                work,
                &boundary_table,
                &fixture.composed.tokenizer_offsets,
                &component_state_counts,
            )
            .expect("publish walk shard");
            eprintln!(
                "BOUNDARY_INSTALL shard={} parser_states={} parser_trans={} candidates={} uses_composed={} terms={} templates_ms={:.3} materialize_ms={:.3} normalize_ms={:.3}",
                shard.start_component,
                profile.parser_states,
                profile.parser_trans,
                shard.candidate_tokens.len(),
                shard.boundary.uses_composed_tsid_coordinate,
                profile.terms,
                profile.templates_ms,
                profile.materialize_ms,
                profile.normalize_ms,
            );
            published.push(shard);
        }

        let mut static_comp = dynamic.clone();
        install_published_static_boundary_shards(
            static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
            published,
        )
        .expect("install");
        assert!(
            static_comp.uses_compact_segmented_parser_runtime(),
            "installed composition must stay on the compact runtime",
        );
        assert_scoped_tokenizer_packing("outer", &static_comp, &component_state_counts);
        run_install_differential(&dynamic, &mut static_comp, false);
    }

    /// Byte-prefix mask differential between a dynamic composition and its
    /// walk-shard-installed twin. `fallback` labels runs where convergence
    /// decline left the dynamic shards in place (vacuous by construction).
    fn run_install_differential(
        dynamic: &Constraint,
        static_comp: &mut Constraint,
        fallback: bool,
    ) {

        let prefixes: Vec<&[u8]> = vec![
            b"",
            b"const x = tools",
            b"const x = tools.tool_0(",
            b"const x = tools.tool_3({",
            b"const x = tools.tool_3({\"p\": ",
            b"const x = tools.tool_9({\"outer\": {\"inner\": [1, ",
            b"function f(a, b) { return ",
            b"for (let i = 0; i < ",
            b"const s = \"hello",
            b"const x = { a: ",
            b"const x = tools.",
            b"tools",
        ];
        let mut positions = 0usize;
        let mut checksum: u64 = 0xcbf29ce484222325;
        for (scenario, prefix) in prefixes.iter().enumerate() {
            let scenario_started = Instant::now();
            let mut st_dyn = dynamic.start();
            let mut st_static = static_comp.start();
            let mut check = |st_dyn: &mut crate::runtime::ConstraintState<'_>,
                             st_static: &mut crate::runtime::ConstraintState<'_>,
                             scenario: usize,
                             consumed: usize| {
                let mask_dyn = st_dyn.mask();
                let mask_static = st_static.mask();
                assert_eq!(
                    mask_static,
                    mask_dyn,
                    "mask mismatch scenario={scenario} consumed={consumed}",
                );
                for (index, &word) in mask_dyn.iter().enumerate() {
                    checksum ^= (word as u64).wrapping_add(index as u64);
                    checksum = checksum.wrapping_mul(0x100000001b3);
                }
                positions += 1;
            };
            check(&mut st_dyn, &mut st_static, scenario, 0);
            for (consumed, &byte) in prefix.iter().enumerate() {
                let r_dyn = st_dyn.commit_bytes(&[byte]);
                let r_static = st_static.commit_bytes(&[byte]);
                assert_eq!(
                    r_dyn.is_ok(),
                    r_static.is_ok(),
                    "commit divergence scenario={scenario} consumed={consumed}",
                );
                if r_dyn.is_err() {
                    break;
                }
                check(&mut st_dyn, &mut st_static, scenario, consumed + 1);
            }
            eprintln!(
                "BOUNDARY_INSTALL scenario={scenario} len={} ms={:.3}",
                prefix.len(),
                scenario_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        eprintln!(
            "BOUNDARY_INSTALL fallback={fallback} positions={positions} checksum={checksum:016x}"
        );
        assert!(positions > 100, "corpus must cover real positions");
    }

    /// Dispatch-parent grammar source for the inner-link sweep. Mirrors
    /// `dispatcher_literal_names_parent_source` in
    /// `examples/static_link_measure.rs` (10 `.tool_i(` slots over the same
    /// placeholder token IDs).
    fn dispatch_parent_source() -> String {
        let mut source = String::from("start suffix;\n");
        for index in 0..10 {
            source.push_str(&format!(
                "t TOOL_ARGS_SLOT_{index} ::= @token({});\n",
                128_320 + 10 + index
            ));
        }
        source.push_str("nt suffix ::=\n    ");
        for index in 0..10 {
            if index != 0 {
                source.push_str("\n  | ");
            }
            source.push_str(&format!(r#"".tool_{index}(" TOOL_ARGS_SLOT_{index} ")""#));
        }
        source.push_str(";\n");
        source
    }

    /// Per-link stage timing record for the sweep table.
    struct LinkTiming {
        name: String,
        dyn_compose_ms: f64,
        low_level_ms: f64,
        shared_ms: f64,
        flat_ms: f64,
        walk_ms: f64,
        det_ms: f64,
        min_ms: f64,
        shards_built: usize,
        table_ms: f64,
        templates_ms: f64,
        parser_ms: f64,
        install_ms: f64,
        installed: usize,
        diff_positions: usize,
        diff_mismatches: usize,
        diff_checksum: u64,
        diff_ms: f64,
    }

    impl LinkTiming {
        fn total_link_ms(&self) -> f64 {
            self.shared_ms
                + self.walk_ms
                + self.table_ms
                + self.templates_ms
                + self.parser_ms
                + self.install_ms
        }
    }

    /// Seeded RNG token-walk differential between two compositions (dynamic
    /// reference vs installed/fallback twin). Language-agnostic: drives on
    /// the dynamic masks. Returns (positions, mismatches, checksum).
    fn rng_choose_token(mask: &[u32], rng: &mut u64) -> Option<u32> {
        let allowed: usize = mask.iter().map(|word| word.count_ones() as usize).sum();
        if allowed == 0 {
            return None;
        }
        *rng ^= *rng >> 12;
        *rng ^= *rng << 25;
        *rng ^= *rng >> 27;
        let draw = rng.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mut rank = draw as usize % allowed;
        for (word_index, &word) in mask.iter().enumerate() {
            let count = word.count_ones() as usize;
            if rank >= count {
                rank -= count;
                continue;
            }
            let mut live = word;
            for _ in 0..rank {
                live &= live - 1;
            }
            let bit = live.trailing_zeros() as usize;
            return Some((word_index * 32 + bit) as u32);
        }
        None
    }

    fn rng_differential(
        dynamic: &Constraint,
        other: &Constraint,
        steps: usize,
        seed: u64,
    ) -> (usize, usize, u64, Option<String>) {
        fn mask_bits(mask: &[u32]) -> Vec<u32> {
            let mut out = Vec::new();
            for (word_index, &word) in mask.iter().enumerate() {
                let mut live = word;
                while live != 0 {
                    let bit = live.trailing_zeros() as usize;
                    out.push((word_index * 32 + bit) as u32);
                    live &= live - 1;
                }
            }
            out
        }
        let mut rng = seed;
        let mut checksum: u64 = 0xcbf29ce484222325;
        let mut positions = 0usize;
        let mut mismatches = 0usize;
        let mut first: Option<String> = None;
        let mut st_dyn = dynamic.start();
        let mut st_other = other.start();
        let mut check = |st_dyn: &mut crate::runtime::ConstraintState<'_>,
                         st_other: &mut crate::runtime::ConstraintState<'_>,
                         step_index: usize,
                         first: &mut Option<String>| {
            let mask_dyn = st_dyn.mask();
            let mask_other = st_other.mask();
            for (index, &word) in mask_dyn.iter().enumerate() {
                checksum ^= (word as u64).wrapping_add(index as u64);
                checksum = checksum.wrapping_mul(0x100000001b3);
            }
            positions += 1;
            if mask_dyn != mask_other {
                mismatches += 1;
                if first.is_none() {
                    let dyn_bits = mask_bits(&mask_dyn);
                    let other_bits = mask_bits(&mask_other);
                    let dyn_set: BTreeSet<u32> = dyn_bits.into_iter().collect();
                    let other_set: BTreeSet<u32> = other_bits.into_iter().collect();
                    let dyn_only: Vec<u32> =
                        dyn_set.difference(&other_set).copied().take(12).collect();
                    let other_only: Vec<u32> =
                        other_set.difference(&dyn_set).copied().take(12).collect();
                    *first = Some(format!(
                        "mask step={step_index} dyn_admits={} other_admits={} dyn_only={dyn_only:?} other_only={other_only:?}",
                        dyn_set.len(),
                        other_set.len(),
                    ));
                }
            }
        };
        check(&mut st_dyn, &mut st_other, 0, &mut first);
        let mut path: Vec<u32> = Vec::new();
        for step in 0..steps {
            let mask_dyn = st_dyn.mask();
            let Some(token) = rng_choose_token(&mask_dyn, &mut rng) else {
                break;
            };
            path.push(token);
            let r_dyn = st_dyn.commit_token(token);
            let r_other = st_other.commit_token(token);
            if r_dyn.is_ok() != r_other.is_ok() {
                mismatches += 1;
                if first.is_none() {
                    first = Some(format!(
                        "commit step={} token={token} dyn_ok={} other_ok={}",
                        step + 1,
                        r_dyn.is_ok(),
                        r_other.is_ok(),
                    ));
                }
                break;
            }
            if r_dyn.is_err() {
                break;
            }
            check(&mut st_dyn, &mut st_other, step + 1, &mut first);
        }
        if mismatches > 0 {
            eprintln!("BOUNDARY_LINK_PATH positions={positions} path={path:?}");
        }
        (positions, mismatches, checksum, first)
    }

    /// Run the full walk-shard link for one composition and time every stage.
    /// Publication uses the production signed-transfer route (same fragment
    /// library, composer, and publisher as `build_walk_static_boundary_link`),
    /// so this sweep validates what production installs. There is no table
    /// stage on this route (`table_ms` is exactly 0).
    fn link_and_time(
        name: &str,
        parent: &Constraint,
        slots: &[(String, &Constraint)],
        vocab: &Vocab,
        diff_steps: usize,
    ) -> LinkTiming {
        let inputs: Vec<CompiledSubgrammarInput> = slots
            .iter()
            .map(|(slot, child)| CompiledSubgrammarInput {
                placeholder_terminal: terminal_id(parent, slot),
                additional_placeholder_terminals: &[],
                constraint: child,
            })
            .collect();
        let compose_started = Instant::now();
        let dynamic = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        let dyn_compose_ms = compose_started.elapsed().as_secs_f64() * 1000.0;
        let low_level_started = Instant::now();
        let composed = low_level_compose(parent, &inputs);
        let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = compute_disallowed_follows(&grammar);
        let low_level_ms = low_level_started.elapsed().as_secs_f64() * 1000.0;
        let counts: Vec<u32> = std::iter::once(parent.tokenizer.num_states())
            .chain(inputs.iter().map(|child| child.constraint.tokenizer.num_states()))
            .collect();
        let retain_parent_non_crossing_paths = inputs
            .iter()
            .any(|child| child.constraint.table.embedded_start_nullable());
        let (built, profile) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
            merged_tokenizer: &composed.tokenizer,
            vocab,
            grammar: &grammar,
            disallowed_follows: &disallowed,
            ignore_terminal: composed.ignore_canonical,
            terminal_offsets: &composed.table.terminal_offsets,
            tokenizer_offsets: &composed.tokenizer_offsets,
            component_state_counts: &counts,
            retain_parent_non_crossing_paths,
            walk_plans: None,
        })
        .expect("walk shards");
        let shards_built = built.len();
        let (mut walk_ms, mut det_ms, mut min_ms) = (0.0, 0.0, 0.0);
        for (_, shard_profile) in &profile.per_shard {
            walk_ms += shard_profile.walk_ms;
            det_ms += shard_profile.determinize_ms;
            min_ms += shard_profile.minimize_ms;
        }
        // Signed-transfer shard publication (production route): scoped ordinary
        // transfers + Entry/Finish exports, exact resolution, table-free
        // normalization. No composed/provider table exists here; unbound slots
        // use the same empty-language definition as production
        // (`unbound_link_slot_terminals`). There is no table stage anymore, so
        // `table_ms` below is exactly 0.
        use crate::compiler::boundary_transfer::{
            build_fragment_library, build_signed_link_context, compile_signed_shard_parser,
            publish_signed_shard,
        };
        let global_ignores = component_ignores_are_globally_erasable(parent, &inputs);
        let unbound_set: std::collections::BTreeSet<TerminalID> =
            unbound_link_slot_terminals(parent, &inputs, &composed.table.terminal_offsets)
                .expect("unbound slots")
                .into_iter()
                .collect();
        let signed_context = build_signed_link_context(
            parent,
            &inputs,
            &composed.table.terminal_offsets,
            composed.table.table.num_terminals,
            global_ignores,
            unbound_set,
        )
        .expect("signed link context");
        let table_ms = 0.0;
        let (mut templates_ms, mut parser_ms, mut install_ms) = (0.0, 0.0, 0.0);
        let mut static_comp = dynamic.clone();
        let mut published = Vec::with_capacity(built.len());
        for shard in built {
            let emitted = boundary_emitted_terminals(
                &shard.output.dwa,
                composed.table.table.num_terminals as usize,
            );
            let library = build_fragment_library(
                &signed_context,
                &emitted,
                shard.start_component as u32,
            )
            .expect("fragment library");
            templates_ms += library.templates_ms;
            let compiled = compile_signed_shard_parser(
                &signed_context,
                &library,
                &shard.output.dwa,
                &shard.output.id_map,
                shard.start_component as u32,
            )
            .expect("signed shard compile");
            parser_ms += compiled.compose_ms + compiled.resolve_ms + compiled.normalize_ms;
            let work = WalkBoundaryShardWork {
                start_component: shard.start_component as u32,
                terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
                id_map: shard.output.id_map,
                candidate_tokens: shard
                    .candidate_tokens
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into(),
            };
            let (one, _publish_profile) = publish_signed_shard(
                work,
                compiled,
                &composed.tokenizer_offsets,
                &counts,
            )
            .expect("publish signed shard");
            published.push(one);
        }
        let installed = published.len();
        if !published.is_empty() {
            let install_started = Instant::now();
            install_published_static_boundary_shards(
                static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
                published,
            )
            .expect("install");
            install_ms = install_started.elapsed().as_secs_f64() * 1000.0;
        }
        assert!(
            static_comp.uses_compact_segmented_parser_runtime(),
            "{name}: installed composition must stay on the compact runtime",
        );
        assert_scoped_tokenizer_packing(name, &static_comp, &counts);
        let diff_started = Instant::now();
        let (diff_positions, diff_mismatches, diff_checksum, diff_first) =
            rng_differential(&dynamic, &static_comp, diff_steps, 0x9e37_79b9_7f4a_7c15);
        let diff_ms = diff_started.elapsed().as_secs_f64() * 1000.0;
        let timing = LinkTiming {
            name: name.to_string(),
            dyn_compose_ms,
            low_level_ms,
            shared_ms: profile.shared_wall_ms,
            flat_ms: profile.flat_ms,
            walk_ms,
            det_ms,
            min_ms,
            shards_built,
            table_ms,
            templates_ms,
            parser_ms,
            install_ms,
            installed,
            diff_positions,
            diff_mismatches,
            diff_checksum,
            diff_ms,
        };
        eprintln!(
            "BOUNDARY_LINK name={} dyn={:.1} low={:.1} shared={:.1} flat={:.1} walk={:.1} det={:.1} min={:.1} shards={} table={:.1} templates={:.1} parser={:.1} install={:.1} installed={} diff_pos={} diff_mm={} diffck={:016x} diff_ms={:.0} total_link={:.1}",
            timing.name,
            timing.dyn_compose_ms,
            timing.low_level_ms,
            timing.shared_ms,
            timing.flat_ms,
            timing.walk_ms,
            timing.det_ms,
            timing.min_ms,
            timing.shards_built,
            timing.table_ms,
            timing.templates_ms,
            timing.parser_ms,
            timing.install_ms,
            timing.installed,
            timing.diff_positions,
            timing.diff_mismatches,
            timing.diff_checksum,
            timing.diff_ms,
            timing.total_link_ms(),
        );
        if let Some(detail) = &diff_first {
            eprintln!("BOUNDARY_LINK_FIRST_MISMATCH name={name} {detail}");
        }
        timing
    }

    /// Step-4 link timing sweep: full walk-shard link over every available
    /// composition (10× dispatch-parent+schema + 1× core+dispatch outer)
    /// with per-stage max/median and >3×-median outlier flags.
    #[test]
    #[ignore]
    fn link_timing_all_compositions() {
        use std::path::Path;

        let root = std::env::var("PHASE1_DIR").unwrap_or_else(|_| {
            "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29".to_string()
        });
        let root = Path::new(&root).to_path_buf();
        let vocab_path = std::env::var("PHASE1_VOCAB")
            .unwrap_or_else(|_| root.join("vocab_dump.bin").to_string_lossy().into_owned());
        let vocab = load_vocab(&vocab_path);
        let parent =
            Constraint::from_glrm_grammar(&dispatch_parent_source(), &vocab).expect("parent");
        let mut schema_paths: Vec<_> = std::fs::read_dir(&root)
            .expect("read cache dir")
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().to_string_lossy().into_owned();
                name.starts_with("schema-").then_some(name)
            })
            .collect();
        schema_paths.sort();
        assert_eq!(schema_paths.len(), 10, "cache must hold the 10 schema constraints");
        let mut schemas = Vec::with_capacity(10);
        for name in &schema_paths {
            let mut schema =
                Constraint::load(&std::fs::read(root.join(name)).expect("read schema"))
                    .expect("load schema");
            restore_component(&mut schema, name);
            schemas.push(schema);
        }
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

        let mut timings = Vec::new();
        for (index, schema) in schemas.iter().enumerate() {
            timings.push(link_and_time(
                &format!("inner-{index}"),
                &parent,
                &[(format!("TOOL_ARGS_SLOT_{index}"), schema)],
                &vocab,
                64,
            ));
        }
        // NOTE: no 10-way segmented dispatch link — N-way segmented compose
        // rejects it ("component 2 has a non-functional LR-state relation").
        // The architecture composes dispatch flat and segments only the outer
        // link, so the sweep covers the 10 synthetic inner links + outer.
        timings.push(link_and_time(
            "outer",
            &core,
            &[("PROGRAMMATIC_TOOL_SUFFIX".to_string(), &dispatch)],
            &vocab,
            64,
        ));

        let stages: &[(&str, fn(&LinkTiming) -> f64)] = &[
            ("shared", |t| t.shared_ms),
            ("walk", |t| t.walk_ms),
            ("det", |t| t.det_ms),
            ("min", |t| t.min_ms),
            ("templates", |t| t.templates_ms),
            ("parser", |t| t.parser_ms),
            ("table", |t| t.table_ms),
            ("install", |t| t.install_ms),
            ("total_link", |t| t.total_link_ms()),
        ];
        for (stage, get) in stages {
            let mut values: Vec<f64> = timings.iter().map(get).collect();
            values.sort_by(f64::total_cmp);
            let median = values[values.len() / 2];
            let max = values[values.len() - 1];
            let argmax = timings.iter().map(get).enumerate().max_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let flagged = max > 3.0 * median;
            eprintln!(
                "BOUNDARY_STAGE stage={stage} n={} median_ms={median:.1} max_ms={max:.1} max_link={} flagged_3x={flagged}",
                values.len(),
                argmax.map(|(i, _)| timings[i].name.as_str()).unwrap_or("?"),
            );
        }
        let total_mismatches: usize = timings.iter().map(|timing| timing.diff_mismatches).sum();
        let failing: Vec<&str> = timings
            .iter()
            .filter(|timing| timing.diff_mismatches > 0)
            .map(|timing| timing.name.as_str())
            .collect();
        eprintln!("BOUNDARY_SWEEP links={} total_mismatches={total_mismatches} failing={failing:?}", timings.len());
        for timing in &timings {
            assert_eq!(
                timing.installed,
                timing.shards_built,
                "link {} must install every built shard",
                timing.name,
            );
        }
        assert_eq!(total_mismatches, 0, "walk-shard sweep must match DynamicDirect on every link");
    }

    /// Called-frame non-equivalence pin (Phase 2b reconciliation audit).
    ///
    /// The compact segmented runtime evaluates boundary shards against live
    /// GSS stacks in the leaf/provider coordinate, where an in-progress call
    /// appears as Call-pushed frames `[.., parent_target, child_start, ..]`.
    /// The spliced composed table speaks a different calling convention (child
    /// start overlaid on callers, returns via reduce+goto through caller
    /// states, fresh appended child-state identities), so its template labels
    /// never match called frames: a splice-published shard systematically
    /// under-admits wherever the live stack holds an active call. The
    /// provider-materialized table speaks the live coordinate with call-site
    /// guards and is exact here. The Phase 2 sweep stayed green only because
    /// no corpus position needed a child-shard admission (the full sweep
    /// passes with every child shard removed, outer with zero shards).
    ///
    /// Fixture: one placeholder called from two states with divergent
    /// continuations (`L SUB x | R SUB y`, child `a`), fused exit tokens
    /// `ax`/`ay` discriminated by call-site guards. Asserts DynamicDirect
    /// admits each fused token at its own call site, the provider-published
    /// shard matches DynamicDirect exactly, and the splice-published shard
    /// misses both fused tokens. If the splice side ever starts admitting
    /// them, the coordinate story changed: revisit this pin and the audit.
    #[test]
    fn splice_boundary_shard_underadmits_called_frames_divergent() {
        fn admits(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }

        let vocab = Vocab::new(vec![
            (0, b"ax".to_vec()),
            (1, b"ay".to_vec()),
            (2, b"L".to_vec()),
            (3, b"R".to_vec()),
            (4, b"a".to_vec()),
            (5, b"x".to_vec()),
            (6, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let dynamic = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;

        let composed = low_level_compose(&parent, &inputs);
        let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = compute_disallowed_follows(&grammar);
        let counts: Vec<u32> = vec![
            parent.tokenizer.num_states(),
            child.tokenizer.num_states(),
        ];
        let (built, _) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
            merged_tokenizer: &composed.tokenizer,
            vocab: &vocab,
            grammar: &grammar,
            disallowed_follows: &disallowed,
            ignore_terminal: composed.ignore_canonical,
            terminal_offsets: &composed.table.terminal_offsets,
            tokenizer_offsets: &composed.tokenizer_offsets,
            component_state_counts: &counts,
            retain_parent_non_crossing_paths: inputs
                .iter()
                .any(|child| child.constraint.table.embedded_start_nullable()),
            walk_plans: None,
        })
        .expect("walk shards");
        assert_eq!(built.len(), 1, "only the child shard crosses here");
        assert_eq!(built[0].start_component, 1);
        assert!(
            unbound_slot_terminals(&parent, &["SUB"]).is_empty()
                && unbound_slot_terminals(&child, &[]).is_empty(),
            "fixture has no unbound slots; emptying must not be involved",
        );
        let (splice_table, _) =
            prepare_spliced_boundary_table("divergent", &composed, &[]);
        let global_ignores = component_ignores_are_globally_erasable(&parent, &inputs);
        let provider_table = Arc::new(
            link_provider_boundary_table(
                &parent,
                &inputs,
                &composed.table.terminal_offsets,
                composed.table.table.num_terminals,
                global_ignores,
            )
            .expect("provider boundary table"),
        );

        let publish_against = |table: &Arc<GLRTable>| {
            let shard = &built[0];
            let work = WalkBoundaryShardWork {
                start_component: shard.start_component as u32,
                terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa.clone()),
                id_map: shard.output.id_map.clone(),
                candidate_tokens: shard
                    .candidate_tokens
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
                    .into(),
            };
            let (one, _) =
                publish_walk_boundary_shard_work(work, table, &composed.tokenizer_offsets, &counts)
                    .expect("publish walk shard");
            let mut installed = dynamic.clone();
            install_published_static_boundary_shards(
                installed.static_dynamic_overlay.as_mut().expect("overlay"),
                vec![one],
            )
            .expect("install");
            assert!(
                installed.uses_compact_segmented_parser_runtime(),
                "installed composition must stay on the compact runtime",
            );
            installed
        };
        let splice_static = publish_against(&splice_table);
        let provider_static = publish_against(&provider_table);

        // After L: fused `ax` (token 0) must be admitted; after R: `ay` (1).
        for (commit, fused, sibling, site) in [(2u32, 0u32, 1u32, "L"), (3u32, 1u32, 0u32, "R")] {
            let mut st_dyn = dynamic.start();
            let mut st_splice = splice_static.start();
            let mut st_provider = provider_static.start();
            st_dyn.commit_token(commit).unwrap();
            st_splice.commit_token(commit).unwrap();
            st_provider.commit_token(commit).unwrap();
            let mask_dyn = st_dyn.mask();
            let mask_splice = st_splice.mask();
            let mask_provider = st_provider.mask();
            assert!(
                admits(&mask_dyn, fused),
                "dynamic must admit fused token {fused} after {site}",
            );
            assert!(
                !admits(&mask_dyn, sibling),
                "dynamic must reject sibling fused token {sibling} after {site}",
            );
            assert_eq!(
                mask_provider, mask_dyn,
                "provider shard must match DynamicDirect after {site}",
            );
            assert!(
                !admits(&mask_splice, fused),
                "splice shard under-admits fused token {fused} after {site} \
                 (called-frame labels unmatched); if this now admits, revisit the audit pin",
            );
        }
    }

    /// Production-outer decline pin (Phase 2b reconciliation audit).
    ///
    /// The provider-materialized boundary table's `exact_control_elimination`
    /// diverges on the selected10 outer (core+dispatch) link at the known
    /// row-918 / terminal-3 cell (`state=918 terminal=3`, guard-differentiated
    /// cyclic fan-out over a ~10-state reduce cycle — the identical cell the
    /// old segmented-static path hung on), so current production static
    /// composition DECLINES the outer link loudly via the convergence cap
    /// instead of linking it. Phase 2 linked outer exactly via the splice in
    /// ~7.6 s; that object is inexact on called frames (see the divergent pin
    /// above), so neither table object currently serves outer statically.
    ///
    /// If this test observes convergence (Ok), the elimination pathology is
    /// fixed and the production contract can be revisited; the panic message
    /// says so. Fast decline (well under the 10 s wall cap) is part of the
    /// pin: the budget must fail fast, never hang.
    #[test]
    #[ignore]
    fn provider_boundary_table_diverges_on_selected10_outer() {
        let fixture = load_selected10_outer();
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let global_ignores = component_ignores_are_globally_erasable(&fixture.core, &children);
        let started = Instant::now();
        let result = link_provider_boundary_table(
            &fixture.core,
            &children,
            &fixture.composed.table.terminal_offsets,
            fixture.composed.table.table.num_terminals,
            global_ignores,
        );
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(table) => panic!(
                "provider boundary table unexpectedly converged on outer: states={} terms={} \
                 controls={} ms={ms:.1}; the elimination pathology is fixed — update this pin \
                 and revisit the production static-link contract",
                table.num_states,
                table.num_terminals,
                table.control_terminals.len(),
            ),
            Err(error) => {
                eprintln!("BOUNDARY_PROVIDER_DIVERGENCE ms={ms:.1} error={error}");
                assert!(
                    error.contains("state=918")
                        && error.contains("terminal=3")
                        && error.contains("stack_effect_visits"),
                    "outer decline must be the row-918 convergence cap, got: {error}",
                );
                assert!(
                    ms < 60_000.0,
                    "convergence cap must fail fast, took {ms:.1} ms",
                );
            }
        }
    }

    /// Prepared-transfer outer link through the NEW signed-transfer production
    /// route (advisor v3 Prototype 1, milestones F + 9).
    ///
    /// Links only LOCAL tables: the ordinary core terminal-3 transfer plus
    /// the dispatch Finish export under this link's child-start/return-pop
    /// policy, with the slot Entry shape validated. Records local
    /// characterization cycle status / read bound / push bound / sizes for
    /// both transfers, then compiles, installs, and differentially validates
    /// the full outer link. This path never calls `exact_control_elimination`
    /// by construction (proven by the run counter); there is no global table
    /// object here at all.
    #[test]
    #[ignore]
    fn prepared_transfer_row918_no_global_elimination() {
        use crate::compiler::boundary_transfer::{
            StateInjection, diagnose_transfer, instantiate_entry, instantiate_finish,
            scope_characterization, validate_shared_child_links, validate_slot_entry_shape,
        };
        use crate::compiler::constraint_compose::build_segmented_parser_links;
        use crate::compiler::stages::templates::characterize::characterize_selected_terminals_for_terminal_count;

        let fixture = load_selected10_outer();
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let links = build_segmented_parser_links(&children).expect("segmented links");
        validate_shared_child_links(&links).expect("incoming links agree");
        assert_eq!(links.len(), 1, "outer link has one child link");
        let link = &links[0];
        let parent_injection = StateInjection { offset: 0 };
        let child_injection = StateInjection {
            offset: fixture.core.table.num_states,
        };
        // Ordinary local transfer for core terminal 3 (the row-918 terminal).
        let mut selected = vec![false; fixture.core.table.num_terminals as usize];
        selected[3] = true;
        let characterized = characterize_selected_terminals_for_terminal_count(
            &fixture.core.table,
            fixture.core.table.num_terminals,
            &selected,
        );
        let local_t3 = characterized.get(&3).expect("core terminal 3 characterizes");
        let scoped_t3 =
            scope_characterization(local_t3, &parent_injection).expect("scope core terminal 3");
        let diag_t3 = diagnose_transfer(&scoped_t3);
        eprintln!(
            "BOUNDARY_TRANSFER t3 escapes={} reduces={} nt_escapes={} nt_rereduces={} read={} push={} cycle={:?} kinds={:?}",
            diag_t3.num_escapes,
            diag_t3.num_reduces,
            diag_t3.num_nt_escapes,
            diag_t3.num_nt_rereduces,
            diag_t3.max_input_read,
            diag_t3.max_push_len,
            diag_t3.cycle_status,
            diag_t3.endpoint_kinds,
        );
        assert!(
            diag_t3.cycle_status.is_none(),
            "core terminal-3 local transfer must be finite",
        );
        // Slot Entry: validate the provider-supported shape, then instantiate.
        validate_slot_entry_shape(&fixture.core.table, link.slot_terminal)
            .expect("outer slot supports static Entry");
        let mut slot_selected = vec![false; fixture.core.table.num_terminals as usize];
        slot_selected[link.slot_terminal as usize] = true;
        let slot_characterized = characterize_selected_terminals_for_terminal_count(
            &fixture.core.table,
            fixture.core.table.num_terminals,
            &slot_selected,
        );
        let local_slot = slot_characterized
            .get(&link.slot_terminal)
            .expect("slot terminal characterizes");
        let scoped_slot =
            scope_characterization(local_slot, &parent_injection).expect("scope slot");
        let scoped_child_start = child_injection
            .scope_state(link.child_start)
            .expect("scope child start");
        let entry = instantiate_entry(scoped_slot, scoped_child_start, 0);
        eprintln!(
            "BOUNDARY_TRANSFER entry escapes={} reduces={} read={} push={} cycle={:?} kinds={:?}",
            entry.diagnostics.num_escapes,
            entry.diagnostics.num_reduces,
            entry.diagnostics.max_input_read,
            entry.diagnostics.max_push_len,
            entry.diagnostics.cycle_status,
            entry.diagnostics.endpoint_kinds,
        );
        assert!(
            entry.diagnostics.cycle_status.is_none(),
            "slot Entry transfer must be finite",
        );
        assert!(
            entry
                .characterization
                .escapes
                .iter()
                .all(|escape| escape.pushes.last() == Some(&scoped_child_start)),
            "every Entry escape must carry the scoped child-start push",
        );
        // Finish export for the dispatch child under this link's policy.
        let (finish, has_local_eof_effects) =
            instantiate_finish(&fixture.dispatch.table, link, &child_injection)
                .expect("dispatch finish transfer");
        eprintln!(
            "BOUNDARY_TRANSFER finish return_pop={} nullable={} escapes={} reduces={} nt_escapes={} nt_rereduces={} read={} push={} cycle={:?} kinds={:?} local_eof={has_local_eof_effects}",
            link.return_pop,
            link.child_start_nullable,
            finish.diagnostics.num_escapes,
            finish.diagnostics.num_reduces,
            finish.diagnostics.num_nt_escapes,
            finish.diagnostics.num_nt_rereduces,
            finish.diagnostics.max_input_read,
            finish.diagnostics.max_push_len,
            finish.diagnostics.cycle_status,
            finish.diagnostics.endpoint_kinds,
        );
        assert!(
            finish.diagnostics.cycle_status.is_none(),
            "dispatch finish transfer must be finite",
        );
        assert!(
            !has_local_eof_effects,
            "bounded flat prototype requires canonical EOF completion (reductions + pure Accept); local EOF work needs outer control-choice points",
        );
        // Full outer link through the NEW signed-transfer production route
        // (milestone 9): local transfers, Ready-port composition, exact
        // resolution, table-free normalization — and no control elimination.
        let elim_before = control_elimination_run_count();
        let link_output = build_walk_static_boundary_link(&WalkStaticLinkInputs {
            parent: &fixture.core,
            children: &children,
            vocab: &fixture.vocab,
            static_components: None,
            expected_terminal_offsets: &fixture.composed.table.terminal_offsets,
        })
        .expect("outer link compiles through the signed-transfer route");
        assert_eq!(
            control_elimination_run_count(),
            elim_before,
            "signed-transfer outer link must not invoke control elimination",
        );
        assert!(
            !link_output.all_dynamic,
            "outer link must take the static route",
        );
        for shard in &link_output.published_shards {
            assert!(
                link_output
                    .effective_static_components
                    .contains(shard.start_component as usize),
                "published shard {} must be effectively static",
                shard.start_component,
            );
        }
        eprintln!(
            "BOUNDARY_TRANSFER outer published={:?} tokens={:?}",
            link_output
                .published_shards
                .iter()
                .map(|shard| shard.start_component)
                .collect::<Vec<_>>(),
            link_output.boundary_tokens_by_start_component,
        );
        // Install the new-route shards over a dynamic composition and require
        // mask-for-mask equality: the outer shard must WORK, not just compile.
        let dyn_children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let dynamic = compose_constraints_owned_parent_segmented(
            fixture.core.clone(),
            &dyn_children,
            &fixture.vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        let mut static_comp = dynamic.clone();
        install_published_static_boundary_shards(
            static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
            link_output.published_shards,
        )
        .expect("install");
        assert!(
            static_comp.uses_compact_segmented_parser_runtime(),
            "installed composition must stay on the compact runtime",
        );
        for (index, component) in static_comp
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .enumerate()
        {
            if let Some(shard) = component.boundary.as_ref() {
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed outer component {index} must be a StaticParser shard",
                );
            }
        }
        let component_state_counts =
            [fixture.core.tokenizer.num_states(), fixture.dispatch.tokenizer.num_states()];
        assert_scoped_tokenizer_packing("outer-signed", &static_comp, &component_state_counts);
        run_install_differential(&dynamic, &mut static_comp, false);
    }

    /// Divergent called-frame fixture through the PRODUCTION static link
    /// (milestones E/G/H at fixture scale).
    ///
    /// Nullable bound children are OUTSIDE the signed-transfer static
    /// linker's supported class (bounded flat closure certificate): silent
    /// Entry/Return episodes have no uniform bound, so the link declines
    /// loudly here instead of under-admitting. This pins that decline.
    #[test]
    fn nullable_static_link_declines_loudly() {
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"a".to_vec()),
            (2, b"x".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        assert!(
            child.table.embedded_start_nullable(),
            "pin fixture must stay effectively nullable",
        );
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let composed = low_level_compose(&parent, &inputs);
        let error = match build_walk_static_boundary_link(&WalkStaticLinkInputs {
            parent: &parent,
            children: &inputs,
            vocab: &vocab,
            static_components: None,
            expected_terminal_offsets: &composed.table.terminal_offsets,
        }) {
            Err(error) => error,
            Ok(_) => panic!("nullable static links must decline loudly (general C* is future work)"),
        };
        assert!(
            error.contains("nullable"),
            "decline must name nullability, got: {error}",
        );
    }

    /// Divergent called-frame fixture through the PRODUCTION static link
    /// (milestones E/G/H at fixture scale).
    ///
    /// `build_walk_static_boundary_link` must install a real static shard that
    /// matches DynamicDirect at both call sites (caller-sensitive fused
    /// tokens), with no `DynamicDirect` backend anywhere in the installed
    /// overlay. Removing the child shard (ablation) must lose the fused
    /// tokens — a nonempty shard that never fires is not evidence. With
    /// `GLRMASK_STRICT_STATIC_TRAP_DYNAMIC=1`, masking the installed static
    /// composition must not trip the DynamicDirect trap.
    #[test]
    #[ignore]
    fn walk_static_link_divergent_fixture_matches_dynamic_and_ablates() {
        fn admits(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }

        // Panic-safe env restore: Drop removes the trap var even if an assert
        // below fails, so the trap cannot leak into other tests. The unsafe
        // blocks are sound because the caller holds TEST_ENV_LOCK, serializing
        // all process-env mutation in the test build.
        struct TrapGuard;
        impl TrapGuard {
            fn set() -> Self {
                unsafe {
                    std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1");
                }
                Self
            }
        }
        impl Drop for TrapGuard {
            fn drop(&mut self) {
                unsafe {
                    std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
                }
            }
        }

        let vocab = Vocab::new(vec![
            (0, b"ax".to_vec()),
            (1, b"ay".to_vec()),
            (2, b"L".to_vec()),
            (3, b"R".to_vec()),
            (4, b"a".to_vec()),
            (5, b"x".to_vec()),
            (6, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let composed = low_level_compose(&parent, &inputs);
        // No control elimination anywhere on the new parser-side path: the
        // signed-transfer compiler links local tables only.
        let elim_before = control_elimination_run_count();
        let link_output = build_walk_static_boundary_link(&WalkStaticLinkInputs {
            parent: &parent,
            children: &inputs,
            vocab: &vocab,
            static_components: None,
            expected_terminal_offsets: &composed.table.terminal_offsets,
        })
        .expect("production static link on the divergent fixture");
        assert_eq!(
            control_elimination_run_count(),
            elim_before,
            "signed-transfer link must not invoke control elimination",
        );
        assert!(
            !link_output.all_dynamic,
            "fixture link must be static, not all-dynamic",
        );
        assert_eq!(
            link_output.published_shards.len(),
            1,
            "only the child shard crosses here",
        );
        assert_eq!(link_output.published_shards[0].start_component, 1);

        let dynamic = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        let mut installed = dynamic.clone();
        install_published_static_boundary_shards(
            installed.static_dynamic_overlay.as_mut().expect("overlay"),
            link_output
                .published_shards
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        )
        .expect("install");
        // No DynamicDirect backend anywhere in the installed overlay.
        for (index, component) in installed
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .enumerate()
        {
            if let Some(shard) = component.boundary.as_ref() {
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
        }
        // Call-site-exact admission at both sites.
        for (commit, fused, sibling, site) in [(2u32, 0u32, 1u32, "L"), (3u32, 1u32, 0u32, "R")] {
            let mut st_dyn = dynamic.start();
            let mut st_static = installed.start();
            st_dyn.commit_token(commit).unwrap();
            st_static.commit_token(commit).unwrap();
            let mask_dyn = st_dyn.mask();
            let mask_static = st_static.mask();
            assert!(admits(&mask_dyn, fused), "dynamic admits fused {fused} after {site}");
            assert!(!admits(&mask_dyn, sibling), "dynamic rejects sibling {sibling} after {site}");
            assert_eq!(mask_static, mask_dyn, "static matches dynamic after {site}");
        }
        // Strict-static trap: no DynamicDirect evaluation on this path.
        // Masks are collected under the env lock, then the guard is dropped
        // (removing the var) before asserting, so a failure cannot leak the
        // trap into other tests.
        let trapped: Vec<(u32, Vec<u32>)> = {
            let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
            let _trap = TrapGuard::set();
            let mut trapped = Vec::new();
            for (commit, fused, _) in [(2u32, 0u32, "L"), (3u32, 1u32, "R")] {
                let mut st = installed.start();
                st.commit_token(commit).unwrap();
                trapped.push((fused, st.mask()));
            }
            trapped
        };
        for (fused, mask) in &trapped {
            assert!(admits(mask, *fused), "static admits fused {fused} under the trap");
        }
        // Ablation: with the child shard removed, the fused tokens must go
        // missing (the shard is load-bearing, not redundant).
        let mut ablated = dynamic.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("install empty");
        for (commit, fused, site) in [(2u32, 0u32, "L"), (3u32, 1u32, "R")] {
            let mut st = ablated.start();
            st.commit_token(commit).unwrap();
            assert!(
                !admits(&st.mask(), fused),
                "ablated composition must lose fused token {fused} after {site}",
            );
        }
    }

    /// Nested depth-2 end-to-end fixture through the nested signed-transfer
    /// link (production walk/compile/publish/install, strict trap, curated
    /// differential, light ablation).
    ///
    /// Shape: P ::= "L" SUB SUB "x" | "R" SUB SUB "y" (two adjacent call
    /// sites into the same mid child M); M ::= "m" SUB2 (leading visible
    /// terminal, trailing slot); G ::= "g". All links effectively
    /// nonnullable, acyclic, canonical. Fused vocabulary tokens span every
    /// adjacent cross-leaf terminal pair ("Lm", "Rm", "mg", "gm", "gx",
    /// "gy"); without them no walk path can cross components (single bytes
    /// never cross) and the shards would be vacuous.
    ///
    /// This fixture is end-to-end integration only: the 2H certificate
    /// (depth 2, bound 4) is asserted, but the R,R,E,E load-bearing proof
    /// lives in `boundary_transfer::nested_ready_depth_four_is_load_bearing`.
    /// Here the arbiter is a curated reachable-prefix differential of the
    /// installed StaticParser composition against DynamicDirect, run under
    /// the strict-static trap; a light empty-shard ablation confirms the
    /// installed shards are genuinely used.
    ///
    /// Link architecture under test: two top-level walks (parent commit with
    /// the parent-leaf crossing filter; whole-block commit with the mid-leaf
    /// crossing filter — a path survives iff it touches a terminal outside
    /// the filter leaf, so the block walk keeps every leaf-crossing gap
    /// including g→m), each compiled with the full 3-leaf signed context and
    /// installed on the dynamic nested composition (mid overlay shards
    /// cleared: the outer block shard covers every block-start crossing, so
    /// the inner dynamic shards are redundant and must not fire under the
    /// strict trap).
    #[test]
    fn nested_static_link_depth_two_chain_matches_dynamic_and_ablates() {
        use crate::compiler::glr::parser::ScopedSubgrammarLink;

        fn admits(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }

        // Panic-safe trap guard (mirrors the flat divergent fixture): Drop
        // removes the var even on failure. The whole test holds TEST_ENV_LOCK
        // so no foreign trap can leak into the dynamic reference phase.
        struct TrapGuard(bool);
        impl TrapGuard {
            fn set() -> Self {
                unsafe {
                    std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1");
                }
                Self(true)
            }
            fn clear(&mut self) {
                if self.0 {
                    unsafe {
                        std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
                    }
                    self.0 = false;
                }
            }
        }
        impl Drop for TrapGuard {
            fn drop(&mut self) {
                self.clear();
            }
        }

        fn compile_publish_shard(
            ctx: &crate::compiler::boundary_transfer::SignedLinkContext<'_>,
            dwa: &DWA,
            id_map: &InternalIdMap,
            start: u32,
            num_terminals: u32,
            tok_offsets: &[u32],
            leaf_counts: &[u32],
        ) -> crate::compiler::constraint_compose::PublishedStaticBoundaryShard {
            let emitted = boundary_emitted_terminals(dwa, num_terminals as usize);
            let library =
                crate::compiler::boundary_transfer::build_fragment_library(ctx, &emitted, start)
                    .expect("nested fragment library");
            let compiled = crate::compiler::boundary_transfer::compile_signed_shard_parser(
                ctx, &library, dwa, id_map, start,
            )
            .expect("nested shard compile");
            let candidates: Arc<[u32]> = boundary_accepted_tokens(dwa, id_map)
                .into_iter()
                .collect::<Vec<_>>()
                .into();
            let work = WalkBoundaryShardWork {
                start_component: start,
                terminal_automaton: TerminalAutomaton::Dwa(dwa.clone()),
                id_map: id_map.clone(),
                candidate_tokens: candidates,
            };
            crate::compiler::boundary_transfer::publish_signed_shard(
                work,
                compiled,
                tok_offsets,
                leaf_counts,
            )
            .expect("nested shard publish")
            .0
        }

        fn install_nested(
            base: &Constraint,
            top_shards: Vec<crate::compiler::constraint_compose::PublishedStaticBoundaryShard>,
        ) -> Constraint {
            let mut installed = base.clone();
            {
                let overlay = installed.static_dynamic_overlay.as_mut().expect("overlay");
                assert_eq!(
                    overlay.segmented_parser_components.len(),
                    2,
                    "nested install needs the dynamic 2-component top overlay",
                );
                let mid_arc = &mut overlay.segmented_parser_components[1].constraint;
                let mid_mut = std::sync::Arc::make_mut(mid_arc);
                install_published_static_boundary_shards(
                    mid_mut.static_dynamic_overlay.as_mut().expect("mid overlay"),
                    Vec::new(),
                )
                .expect("clear mid shards");
            }
            install_published_static_boundary_shards(
                installed.static_dynamic_overlay.as_mut().expect("overlay"),
                top_shards,
            )
            .expect("install nested shards");
            installed
        }

        // Token ids: 0 L, 1 R, 2 x, 3 y, 4 m, 5 g, 6 Lm, 7 Rm, 8 mg, 9 gm,
        // 10 gx, 11 gy.
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const M: u32 = 4;
        const G: u32 = 5;
        const LM: u32 = 6;
        const RM: u32 = 7;
        const MG: u32 = 8;
        const GM: u32 = 9;
        const GX: u32 = 10;
        const GY: u32 = 11;

        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"m".to_vec()),
            (5, b"g".to_vec()),
            (6, b"Lm".to_vec()),
            (7, b"Rm".to_vec()),
            (8, b"mg".to_vec()),
            (9, b"gm".to_vec()),
            (10, b"gx".to_vec()),
            (11, b"gy".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB SUB "x" | "R" SUB SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let mid = Constraint::from_glrm_grammar(
            r#"
                start m;
                t SUB2 ::= @token(998);
                nt m ::= "m" SUB2;
            "#,
            &vocab,
        )
        .unwrap();
        let grandchild = Constraint::from_glrm_grammar(
            r#"
                start g;
                nt g ::= "g";
            "#,
            &vocab,
        )
        .unwrap();
        assert!(
            !mid.table.embedded_start_nullable(),
            "nested fixture mid must stay effectively nonnullable",
        );
        assert!(
            !grandchild.table.embedded_start_nullable(),
            "nested fixture grandchild must stay effectively nonnullable",
        );
        assert!(parent.ignore_terminal.is_none());
        assert!(mid.ignore_terminal.is_none());
        assert!(grandchild.ignore_terminal.is_none());
        let sub_p = terminal_id(&parent, "SUB");
        let sub2_m = terminal_id(&mid, "SUB2");

        // Reference nesting through the production dynamic composer (nested
        // dynamic masking is proven by existing tests; it is the arbiter).
        let mid_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: sub2_m,
            additional_placeholder_terminals: &[],
            constraint: &grandchild,
        }];
        let mid_dyn = compose_constraints_owned_parent_segmented(
            mid.clone(),
            &mid_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("inner dynamic compose")
        .constraint;
        let outer_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: sub_p,
            additional_placeholder_terminals: &[],
            constraint: &mid_dyn,
        }];
        let outer_dyn = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("outer dynamic compose")
        .constraint;
        assert!(
            outer_dyn.uses_compact_segmented_parser_runtime(),
            "nested dynamic composition must stay on the compact runtime",
        );
        assert!(
            mid_dyn.uses_compact_segmented_parser_runtime(),
            "nested inner composition must stay on the compact runtime",
        );

        // Leaf layout pins from the real compose machinery: leaves
        // [[0],[1,0],[1,1]] with back-to-back terminal ranges.
        let layout = outer_dyn
            .recursive_parser_layout()
            .expect("layout")
            .expect("compact layout");
        let paths: Vec<Vec<u32>> = layout
            .leaves
            .iter()
            .map(|leaf| leaf.component_path.clone())
            .collect();
        assert_eq!(
            paths,
            vec![vec![0], vec![1, 0], vec![1, 1]],
            "nested runtime leaf order must be parent, mid, grandchild",
        );
        let n_p = parent.table.num_terminals;
        let n_m = mid.table.num_terminals;
        let n_g = grandchild.table.num_terminals;
        let outer_overlay = outer_dyn.static_dynamic_overlay.as_ref().expect("overlay");
        assert_eq!(outer_overlay.segmented_parser_components.len(), 2);
        let mid_overlay = mid_dyn.static_dynamic_overlay.as_ref().expect("inner overlay");
        assert_eq!(mid_overlay.segmented_parser_components.len(), 2);
        let t0 = outer_overlay.segmented_parser_components[0].terminal_offset;
        let t1 = outer_overlay.segmented_parser_components[1].terminal_offset;
        assert_eq!((t0, t1), (0, n_p), "outer terminal layout must pack parent first");
        let leaf_offsets = vec![0, n_p, n_p + n_m];
        let num_terminals = n_p + n_m + n_g;

        // Expanded links with live return pops and nullability.
        let rp_mid = mid_dyn.composition_child_return_pop().expect("mid return pop");
        let rp_g = grandchild
            .composition_child_return_pop()
            .expect("grandchild return pop");
        assert!(!mid_dyn.composition_start_nullable().expect("mid null"));
        assert!(!grandchild.composition_start_nullable().expect("g null"));
        let links = vec![
            ScopedSubgrammarLink {
                parent_component: 0,
                slot_terminal: sub_p,
                child_component: 1,
                child_start: 0,
                return_pop: rp_mid,
                child_start_nullable: false,
            },
            ScopedSubgrammarLink {
                parent_component: 1,
                slot_terminal: sub2_m,
                child_component: 2,
                child_start: 0,
                return_pop: rp_g,
                child_start_nullable: false,
            },
        ];

        // Recursive machine splice (layout-only, never parser behavior):
        // inner splice first, then the inner composed table as one child.
        let mut splice1 = crate::compiler::glr::table::compose_subgrammar_tables_with_rules(
            &mid.table,
            mid.retained_table_rules().expect("mid rules"),
            None,
            &[SubgrammarTableInput {
                placeholder_terminal: sub2_m,
                additional_placeholder_terminals: &[],
                table: &grandchild.table,
                ignore_terminal: None,
                start_nullable: false,
            }],
            &[grandchild.retained_table_rules().expect("g rules")],
        )
        .expect("inner splice");
        eliminate_composed_runtime_controls(&mut splice1).expect("inner eliminate");
        let mut splice2 = crate::compiler::glr::table::compose_subgrammar_tables_with_rules(
            &parent.table,
            parent.retained_table_rules().expect("parent rules"),
            None,
            &[SubgrammarTableInput {
                placeholder_terminal: sub_p,
                additional_placeholder_terminals: &[],
                table: &splice1.table,
                ignore_terminal: None,
                start_nullable: false,
            }],
            &[splice1.table.rules.as_slice()],
        )
        .expect("outer splice");
        eliminate_composed_runtime_controls(&mut splice2).expect("outer eliminate");
        assert!(
            splice2.table.control_terminals.is_empty(),
            "nested splice must be control-free",
        );
        assert_eq!(
            splice1.terminal_offsets.as_slice(),
            &[0, n_m],
            "inner splice layout must pack mid then grandchild",
        );
        assert_eq!(
            splice2.terminal_offsets.as_slice(),
            &[0, n_p],
            "outer splice layout must pack parent then the mid block",
        );
        assert_eq!(
            splice2.table.num_terminals, num_terminals,
            "nested terminal domain must be exactly the three leaf ranges",
        );

        // Merged tokenizer over intact leaf tokenizers (production union).
        let tokenizer_inputs = vec![
            (parent.composition_tokenizer(), leaf_offsets[0]),
            (mid.composition_tokenizer(), leaf_offsets[1]),
            (grandchild.composition_tokenizer(), leaf_offsets[2]),
        ];
        let (mut merged, tok_offsets) =
            Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
        assert_eq!(
            tok_offsets.first(),
            Some(&1),
            "merged tokenizer must place the fresh reset fan-out at state 0",
        );
        let c_p = parent.composition_tokenizer().num_states();
        let c_m = mid.composition_tokenizer().num_states();
        let c_g = grandchild.composition_tokenizer().num_states();
        let leaf_counts = vec![c_p, c_m, c_g];
        if merged.terminal_exprs().is_none() {
            let all = vec![&parent, &mid, &grandchild];
            if let Some(exprs) = merged_retained_terminal_exprs(
                &all,
                &leaf_offsets,
                num_terminals,
            ) {
                merged.restore_terminal_exprs(Some(exprs)).expect("restore exprs");
            }
        }

        // Grammar + follows over the spliced rules (production analysis).
        let mut terminal_names = parent.terminal_display_names.clone();
        terminal_names.extend(mid.terminal_display_names.iter().cloned());
        terminal_names.extend(grandchild.terminal_display_names.iter().cloned());
        assert_eq!(terminal_names.len(), num_terminals as usize);
        let augmented_start = splice2
            .table
            .rules
            .first()
            .map(|rule| rule.lhs)
            .expect("spliced rules");
        let grammar = AnalyzedGrammar::from_composed_rules(
            splice2.table.rules.clone(),
            num_terminals,
            terminal_names,
            splice2.table.nonterminal_display_names.clone(),
            augmented_start,
        );
        let disallowed = compute_disallowed_follows(&grammar);

        // Two top-level walks sharing one link equivalence: the parent walk
        // (parent commit, parent-leaf filter) and the block walk (whole-block
        // commit, mid-leaf filter). The filter keeps a path iff it touches a
        // terminal outside the filter leaf, so the block walk keeps every
        // leaf-crossing gap (m→g, g→m, g→x) while the parent walk keeps the
        // outer entries (L→m, R→m).
        let active = vec![true; num_terminals as usize];
        let flat: Arc<[u32]> = Arc::from(tdwa::l1::build_flat_transition_table(&merged));
        let shared =
            shared_equivalence_for(&merged, &vocab, None, &grammar, &active, &disallowed, &flat);
        let total_states = merged.num_states() as usize;
        let run_walk = |component_index: usize, commit: &[bool]| {
            build_boundary_terminal_dwa(&BoundaryWalkInputs {
                merged_tokenizer: &merged,
                vocab: &vocab,
                grammar: &grammar,
                disallowed_follows: &disallowed,
                ignore_terminal: None,
                terminal_offsets: &leaf_offsets,
                component_index,
                commit_states: commit,
                retain_non_crossing_paths: false,
                shared_equivalence: &shared,
                flat_trans: Some(&flat),
            })
            .expect("nonempty-vocab nested walks must produce a DWA")
        };
        let commit_p = commit_states_for_component(&tok_offsets, c_p, 0, total_states);
        let mut commit_block = commit_states_for_component(&tok_offsets, c_m, 1, total_states);
        let commit_g = commit_states_for_component(&tok_offsets, c_g, 2, total_states);
        for (slot, extra) in commit_block.iter_mut().zip(commit_g.iter()) {
            *slot |= *extra;
        }
        let walk_p = run_walk(0, &commit_p);
        let walk_b = run_walk(1, &commit_block);
        let cand_p = boundary_accepted_tokens(&walk_p.dwa, &walk_p.id_map);
        let cand_b = boundary_accepted_tokens(&walk_b.dwa, &walk_b.id_map);
        assert!(
            !cand_p.is_empty(),
            "parent walk must cross (fused Lm/Rm expected)",
        );
        assert!(
            !cand_b.is_empty(),
            "block walk must cross (fused mg/gm/gx expected)",
        );
        assert!(
            cand_b.contains(&GM) || cand_b.contains(&MG),
            "block walk must cover an inner-boundary fused token, got {cand_b:?}",
        );
        assert!(
            cand_b.contains(&GX) || cand_b.contains(&GY),
            "block walk must cover an outer-boundary fused token, got {cand_b:?}",
        );

        // Signed-transfer compilation over the full 3-leaf context. The 2H
        // certificate is asserted: depth-2 nonnullable links allow exactly 4
        // control events per gap.
        // Nested unbound-slot scan over intact leaves: a slot is bound iff
        // some link fills it, located by (parent leaf offset + local slot).
        // The flat helper assumes children are leaves (no bound check for
        // child specials), so the nested link generalizes it here.
        let bound_global: std::collections::BTreeSet<u32> = links
            .iter()
            .map(|link| {
                leaf_offsets[link.parent_component as usize]
                    .checked_add(link.slot_terminal)
                    .expect("bound slot offset overflow")
            })
            .collect();
        let mut unbound = std::collections::BTreeSet::new();
        for (constraint, base) in
            [&parent, &mid, &grandchild].into_iter().zip(leaf_offsets.iter())
        {
            for special in &constraint.special_token_terminals {
                let global = base
                    .checked_add(special.terminal_id)
                    .expect("slot terminal offset overflow");
                if bound_global.contains(&global) {
                    continue;
                }
                if constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                    || constraint.token_bytes_for_id(special.token_id).is_none()
                {
                    unbound.insert(global);
                }
            }
        }
        assert!(
            unbound.is_empty(),
            "nested fixture must bind every slot, got {unbound:?}",
        );
        let example_children = [CompiledSubgrammarInput {
            placeholder_terminal: sub_p,
            additional_placeholder_terminals: &[],
            constraint: &mid_dyn,
        }];
        let global_ignores = component_ignores_are_globally_erasable(&parent, &example_children);
        let signed_context =
            crate::compiler::boundary_transfer::build_signed_link_context_from_parts(
                vec![&parent.table, &mid.table, &grandchild.table],
                vec![None, None, None],
                links,
                &leaf_offsets,
                num_terminals,
                global_ignores,
                unbound,
            )
            .expect("nested signed context");
        assert_eq!(signed_context.closure.max_nesting_depth, 2);
        assert_eq!(signed_context.closure.max_controls_per_gap, 4);
        let shard_p = compile_publish_shard(
            &signed_context,
            &walk_p.dwa,
            &walk_p.id_map,
            0,
            num_terminals,
            &tok_offsets,
            &leaf_counts,
        );
        let shard_b = compile_publish_shard(
            &signed_context,
            &walk_b.dwa,
            &walk_b.id_map,
            1,
            num_terminals,
            &tok_offsets,
            &leaf_counts,
        );
        assert_eq!(
            (shard_p.start_component, shard_b.start_component),
            (0, 1),
            "nested link must publish exactly the parent and block shards",
        );

        let installed = install_nested(&outer_dyn, vec![shard_p, shard_b]);
        for (index, component) in installed
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .enumerate()
        {
            let shard = component.boundary.as_ref().unwrap_or_else(|| {
                panic!("installed top component {index} must carry a static shard")
            });
            assert!(
                matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                ),
                "installed component {index} must be a StaticParser shard",
            );
        }
        let mid_installed = &installed
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components[1]
            .constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("mid overlay");
        assert!(
            mid_installed
                .segmented_parser_components
                .iter()
                .all(|component| component.boundary.is_none()),
            "mid overlay must carry no shards (outer block shard covers block-start crossings)",
        );
        assert!(
            mid_installed.segmented_boundary_shards.is_empty(),
            "mid overlay shard list must be cleared",
        );

        // Curated reachable-prefix corpus (fast): prefixes of the two
        // complete L/R strings plus fused-spelling entry points. Only fully
        // mask-admitted paths are kept. This covers root entry, mid entry,
        // grandchild entry, both returns, and both call sites without an
        // exhaustive state search.
        struct Node {
            path: Vec<u32>,
            mask: Vec<u32>,
        }
        let candidates: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![L, M],
            vec![R, M],
            vec![LM],
            vec![RM],
            vec![L, M, G],
            vec![R, M, G],
            vec![LM, G],
            vec![RM, G],
            vec![L, M, G, M],
            vec![R, M, G, M],
            vec![L, M, G, M, G],
            vec![R, M, G, M, G],
            vec![L, M, G, M, G, X],
            vec![R, M, G, M, G, Y],
        ];
        let mut nodes = Vec::new();
        for path in &candidates {
            let mut st = outer_dyn.start();
            let mut ok = true;
            for &token in path {
                // Mask-gated stepping: the runtime treats committing a
                // non-admitted token whose bytes still advance the lexer as
                // a loud mismatch, so only mask-admitted steps extend a
                // corpus prefix (mirrors the production differential walk).
                if !admits(&st.mask(), token) {
                    ok = false;
                    break;
                }
                if st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let mask = st.mask();
            nodes.push(Node { path: path.clone(), mask });
        }
        assert!(
            nodes.len() >= 10,
            "nested corpus must cover real positions, got {}",
            nodes.len()
        );
        // The corpus must contain the deep gap sources on both sites.
        for key_path in [
            vec![L, M, G],
            vec![R, M, G],
            vec![L, M, G, M, G],
            vec![R, M, G, M, G],
        ] {
            assert!(
                nodes.iter().any(|node| node.path == key_path),
                "nested corpus must visit gap source {key_path:?}",
            );
        }

        // Phase 2 (trap armed): every recorded path replays identically on
        // the installed static composition. Any hidden dynamic fallback on
        // the static side panics here instead of contributing exact masks.
        let mut trap = TrapGuard::set();
        let mut mismatches = 0usize;
        for node in &nodes {
            let mut st = installed.start();
            for &token in &node.path {
                st.commit_token(token).expect("static replay");
            }
            if st.mask() != node.mask {
                mismatches += 1;
            }
        }
        trap.clear();
        assert_eq!(mismatches, 0, "nested static must match dynamic on every path");

        // Readable call-site discrimination through two nesting levels. Note
        // the fused spellings overlap single-byte prefixes: gx (bytes g,x)
        // is admissible where g is still unconsumed, i.e. after Lmgm, not
        // after Lmgmg; gm is admissible after Lm; mg is admissible after L.
        for (prefix, fused, sibling, site) in [
            (vec![L, M, G, M], GX, GY, "L"),
            (vec![R, M, G, M], GY, GX, "R"),
        ] {
            let mut st = installed.start();
            for &token in &prefix {
                st.commit_token(token).unwrap();
            }
            let mask = st.mask();
            assert!(admits(&mask, fused), "nested admits {fused} after {site}mgm");
            assert!(!admits(&mask, sibling), "nested rejects {sibling} after {site}mgm");
        }
        {
            let mut st = installed.start();
            st.commit_token(L).unwrap();
            assert!(admits(&st.mask(), M), "nested admits m after L");
            assert!(admits(&st.mask(), MG), "nested admits mg after L");
            assert!(!admits(&st.mask(), RM), "nested rejects Rm after L");
        }
        {
            let mut st = installed.start();
            st.commit_token(LM).unwrap();
            assert!(admits(&st.mask(), GM), "nested admits gm after Lm");
            assert!(!admits(&st.mask(), GX), "nested rejects gx after Lm");
        }

        // Light ablation: empty shards must lose admissions (confirms the
        // installed shards are genuinely used, not redundant). Removing the
        // boundary can only remove admissions (component masks are
        // unchanged), so every ablated mask is a subset; at least one must
        // be strict or the fixture is vacuous. (The R,R,E,E load-bearing
        // control proof lives in
        // `boundary_transfer::nested_ready_depth_four_is_load_bearing`.)
        let ablated = install_nested(&outer_dyn, Vec::new());
        // Lockstep walk: at every shared prefix the shard-less mask must be
        // a subset of the dynamic mask; a prefix where the reference admits
        // a token the ablated mask lacks (or a final-mask difference) proves
        // the installed shards are genuinely used. Mask-gated stepping keeps
        // the runtime commit oracle satisfied on the shard-less side.
        fn is_subset(a: &[u32], b: &[u32]) -> bool {
            a.iter().zip(b.iter()).all(|(x, y)| x & !y == 0) && a.len() <= b.len()
        }
        let mut strict = 0usize;
        for node in &nodes {
            let mut st_dyn = outer_dyn.start();
            let mut st_abl = ablated.start();
            assert!(
                is_subset(&st_abl.mask(), &st_dyn.mask()),
                "ablated masks must be subsets at path {:?}",
                Vec::<u32>::new(),
            );
            let mut diverged = false;
            for &token in &node.path {
                assert!(
                    admits(&st_dyn.mask(), token),
                    "corpus path must stay dynamic-admitted at {:?}",
                    node.path,
                );
                if !admits(&st_abl.mask(), token) {
                    diverged = true;
                    break;
                }
                st_dyn.commit_token(token).expect("dynamic replay");
                st_abl.commit_token(token).expect("ablated replay");
                assert!(
                    is_subset(&st_abl.mask(), &st_dyn.mask()),
                    "ablated masks must be subsets at path {:?}",
                    node.path,
                );
            }
            if !diverged {
                assert_eq!(
                    st_dyn.mask(),
                    node.mask,
                    "dynamic replay must reproduce the recorded mask",
                );
                if st_abl.mask() != node.mask {
                    diverged = true;
                }
            }
            if diverged {
                strict += 1;
            }
        }
        assert!(
            strict > 0,
            "nested ablation must lose admissions somewhere (fixture vacuous otherwise)",
        );
    }
}
