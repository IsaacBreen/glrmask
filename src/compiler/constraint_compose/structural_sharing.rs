use super::*;
use crate::compiler::glr::table::row::{ActionRow, GotoRow};
use crate::grammar::flat::{NonterminalID, Symbol};

// -------------------------------------------------------------------------
// Exact structural sharing for independently compiled subgrammars
// -------------------------------------------------------------------------
//
// Composition historically kept every child parser state disjoint.  That is
// correct but unnecessarily preserves copies of the same parser machine.  The
// quotient below is deliberately *not* a grammar-language-equivalence oracle.
// It uses only sufficient, mechanically checkable congruences:
//
//  * terminals may be related only when their retained lexer Expr values are
//    exactly equal, or when they are the same local terminal of the exact same
//    compiled artifact reused at multiple call sites, and neither terminal has
//    control/skip/ignore/special-token semantics;
//  * nonterminals are related by greatest structural bisimulation of their
//    production equations modulo the terminal relation, with the augmented
//    start and boundary/non-boundary role kept distinct. This relation is a
//    *candidate-isomorphism certificate* for independently compiled children;
//    nonterminal IDs are not physically identified by the ordinary quotient,
//    because equal generated languages do not imply equal caller goto columns;
//  * LR states are related by greatest row bisimulation modulo terminal
//    aliases, concrete nonterminal IDs, and the state relation itself. Every
//    observable guard set is
//    included in the initial state colour, so no quotient can make a guarded
//    stack predicate less discriminating.
//
// The materialized table keeps every original terminal and nonterminal ID. A
// merged state is simply given the union of the terminal aliases from its
// members, and the refinement proves that all aliases in one terminal-language
// class have the same quotient action under concrete nonterminal semantics.
// Existing parser DWAs then need no recompilation: the already supported
// local-state -> composed-state relation is just post-composed with the LR
// quotient map.
//
// A full proof is kept in docs/subgrammar-structural-sharing-proof.md.  Keep
// the implementation conservative when adding new table metadata: if a field
// can observe LR-state identity, it must either be transported through the
// quotient or included as an observation in the refinement.

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct StructuralSharingReport {
    pub(super) terminal_aliases: usize,
    pub(super) nonterminals_before: usize,
    pub(super) nonterminal_classes: usize,
    pub(super) contextual_candidate_groups: usize,
    pub(super) contextual_states_saved: usize,
    pub(super) states_before: usize,
    pub(super) states_after: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum StructuralSymbolSignature {
    Terminal(u32),
    Nonterminal(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NonterminalRefinementSignature {
    previous_class: u32,
    productions: Vec<Vec<StructuralSymbolSignature>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StructuralStateSignature {
    previous_class: u32,
    /// A state ID here intentionally makes the row a singleton.  This is used
    /// only when one physical row assigns observably different behavior to two
    /// terminal/nonterminal aliases that our symbol proof otherwise relates.
    alias_conflict_nonce: Option<u32>,
    guard_memberships: Vec<u32>,
    actions: Vec<(u32, Action, bool)>,
    gotos: Vec<(u32, u32, bool)>,
    advance: Vec<u32>,
    wide_frontiers: Vec<(u32, Vec<u32>)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ComponentActionSignature {
    Shift(u32, bool),
    StackShifts(Vec<(u32, Vec<u32>)>),
    GuardedStackShifts(Vec<(Vec<(u32, Vec<u32>)>, u32, Vec<u32>)>),
    Reduce(u32, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(u32, u32)>,
        accept: bool,
    },
    Accept,
    ReplaceShifts(Vec<u32>),
    Skip,
}

fn component_action_signature(action: &Action) -> ComponentActionSignature {
    match action {
        Action::Shift(target, replace) => ComponentActionSignature::Shift(*target, *replace),
        Action::StackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| (shift.pop, shift.pushes.clone()))
                .collect::<Vec<_>>();
            shifts.sort();
            shifts.dedup();
            ComponentActionSignature::StackShifts(shifts)
        }
        Action::GuardedStackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| {
                    let mut guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states = guard.states.clone();
                            states.sort_unstable();
                            states.dedup();
                            (guard.pop, states)
                        })
                        .collect::<Vec<_>>();
                    guards.sort();
                    guards.dedup();
                    (
                        guards,
                        shift.pop,
                        shift.pushes.clone(),
                    )
                })
                .collect::<Vec<_>>();
            shifts.sort();
            shifts.dedup();
            ComponentActionSignature::GuardedStackShifts(shifts)
        }
        Action::Reduce(nonterminal, len) => {
            ComponentActionSignature::Reduce(*nonterminal, *len)
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => ComponentActionSignature::Split {
            shift: *shift,
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => ComponentActionSignature::Accept,
        Action::ReplaceShifts(targets) => {
            ComponentActionSignature::ReplaceShifts(targets.iter().copied().collect())
        }
        Action::Skip => ComponentActionSignature::Skip,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ComponentStateSignature {
    previous_class: u32,
    actions: Vec<(u32, ComponentActionSignature, bool)>,
    gotos: Vec<(u32, u32, bool)>,
    advance: Vec<u32>,
}

pub(super) fn structural_sharing_enabled() -> bool {
    std::env::var_os("GLRMASK_COMPOSE_DISABLE_STRUCTURAL_SHARING").is_none()
}

fn composition_terminal_classes(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    composed: &ComposedTable,
) -> Vec<u32> {
    let num_terminals = composed.table.num_terminals as usize;
    let mut classes = (0..num_terminals as u32).collect::<Vec<_>>();
    let mut ineligible = BitSet::new(num_terminals);
    for &terminal in &composed.table.control_terminals {
        ineligible.set(terminal as usize);
    }
    for &terminal in &composed.table.skip_terminals {
        ineligible.set(terminal as usize);
    }
    for child in children {
        // Placeholder terminals are linker controls even on the legacy splice
        // path where they are removed rather than retained in control_terminals.
        ineligible.set(child.placeholder_terminal as usize);
    }

    let components = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    for (component_index, component) in components.iter().enumerate() {
        let offset = composed.terminal_offsets[component_index];
        if let Some(ignore) = component.ignore_terminal {
            ineligible.set((offset + ignore) as usize);
        }
        for special in &component.special_token_terminals {
            ineligible.set((offset + special.terminal_id) as usize);
        }
    }

    // Expr equality is a deliberately sufficient condition.  It proves equal
    // byte languages without invoking a potentially expensive regex-language
    // equivalence check, and it remains exact for Exclude/Intersect because
    // those operators are retained inside Expr.
    let mut representative_by_expr = FxHashMap::<Expr, u32>::default();
    let mut representative_by_artifact_terminal = FxHashMap::<(usize, u32), u32>::default();
    for (component_index, component) in components.iter().enumerate() {
        let offset = composed.terminal_offsets[component_index];
        for local_terminal in 0..component.tokenizer.num_terminals() {
            let global_terminal = offset + local_terminal;
            if global_terminal as usize >= num_terminals
                || ineligible.contains(global_terminal as usize)
            {
                continue;
            }
            // Reusing one compiled child artifact at multiple call sites is an
            // exact identity proof even after save/load, where compile-time
            // lexer Exprs are intentionally not serialized.
            let artifact_key = (*component as *const Constraint as usize, local_terminal);
            let artifact_representative = representative_by_artifact_terminal
                .get(&artifact_key)
                .copied();
            let expr = component.tokenizer.terminal_expr(local_terminal);
            let expr_representative = expr
                .and_then(|expr| representative_by_expr.get(expr).copied());
            let representative = artifact_representative
                .or(expr_representative)
                .unwrap_or(global_terminal);
            classes[global_terminal as usize] = representative;
            representative_by_artifact_terminal
                .entry(artifact_key)
                .or_insert(representative);
            if let Some(expr) = expr {
                representative_by_expr
                    .entry(expr.clone())
                    .or_insert(representative);
            }
        }
    }
    classes
}

fn dense_initial_nonterminal_classes(
    table: &crate::compiler::glr::table::GLRTable,
    boundary_nonterminals: &BTreeSet<NonterminalID>,
) -> Vec<u32> {
    let num_nonterminals = table.nonterminal_display_names.len();
    let augmented_start = table.rules.first().map(|rule| rule.lhs);
    let mut class_by_anchor = BTreeMap::<(bool, bool), u32>::new();
    let mut classes = Vec::with_capacity(num_nonterminals);
    for nonterminal in 0..num_nonterminals as u32 {
        let anchor = (
            Some(nonterminal) == augmented_start,
            boundary_nonterminals.contains(&nonterminal),
        );
        let next = class_by_anchor.len() as u32;
        let class = *class_by_anchor.entry(anchor).or_insert(next);
        classes.push(class);
    }
    classes
}

fn structural_nonterminal_classes(
    table: &crate::compiler::glr::table::GLRTable,
    terminal_classes: &[u32],
    boundary_nonterminals: &BTreeSet<NonterminalID>,
) -> Vec<u32> {
    let num_nonterminals = table.nonterminal_display_names.len();
    if num_nonterminals == 0 {
        return Vec::new();
    }
    let mut productions = vec![Vec::<Vec<Symbol>>::new(); num_nonterminals];
    for rule in &table.rules {
        if let Some(slot) = productions.get_mut(rule.lhs as usize) {
            slot.push(rule.rhs.clone());
        }
    }

    let mut classes = dense_initial_nonterminal_classes(table, boundary_nonterminals);
    loop {
        let mut class_by_signature = FxHashMap::<NonterminalRefinementSignature, u32>::default();
        let mut next_classes = Vec::with_capacity(num_nonterminals);
        for nonterminal in 0..num_nonterminals {
            let mut normalized_productions = productions[nonterminal]
                .iter()
                .map(|rhs| {
                    rhs.iter()
                        .map(|symbol| match *symbol {
                            Symbol::Terminal(terminal) => StructuralSymbolSignature::Terminal(
                                terminal_classes
                                    .get(terminal as usize)
                                    .copied()
                                    .unwrap_or(terminal),
                            ),
                            Symbol::Nonterminal(child) => {
                                StructuralSymbolSignature::Nonterminal(
                                    classes.get(child as usize).copied().unwrap_or(child),
                                )
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            // Rule order is not part of CFG semantics.  Preserve duplicate
            // productions (rather than deduplicating) so this criterion is at
            // least as strict as structural equality of the derivation graph.
            normalized_productions.sort();
            let signature = NonterminalRefinementSignature {
                previous_class: classes[nonterminal],
                productions: normalized_productions,
            };
            let next = class_by_signature.len() as u32;
            let class = *class_by_signature.entry(signature).or_insert(next);
            next_classes.push(class);
        }
        if next_classes == classes {
            return classes;
        }
        classes = next_classes;
    }
}

fn remap_action_for_structural_quotient(
    action: &Action,
    state_map: &[u32],
    nonterminal_map: &[u32],
) -> Action {
    let map_state = |state: u32| state_map.get(state as usize).copied().unwrap_or(state);
    let map_nonterminal = |nonterminal: u32| {
        nonterminal_map
            .get(nonterminal as usize)
            .copied()
            .unwrap_or(nonterminal)
    };
    match action {
        Action::Shift(target, replace) => Action::Shift(map_state(*target), *replace),
        Action::ReplaceShifts(targets) => {
            let mut targets = targets.iter().map(|&target| map_state(target)).collect::<Vec<_>>();
            targets.sort_unstable();
            targets.dedup();
            Action::ReplaceShifts(targets.into())
        }
        Action::StackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| crate::compiler::glr::table::StackShift {
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| map_state(state)).collect(),
                })
                .collect::<Vec<_>>();
            shifts.sort_by(|left, right| {
                left.pop
                    .cmp(&right.pop)
                    .then_with(|| left.pushes.cmp(&right.pushes))
            });
            shifts.dedup();
            Action::StackShifts(shifts)
        }
        Action::GuardedStackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| {
                    let mut guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states = guard
                                .states
                                .iter()
                                .map(|&state| map_state(state))
                                .collect::<Vec<_>>();
                            states.sort_unstable();
                            states.dedup();
                            crate::compiler::glr::table::StackShiftGuard {
                                pop: guard.pop,
                                states,
                            }
                        })
                        .collect::<Vec<_>>();
                    guards.sort();
                    guards.dedup();
                    crate::compiler::glr::table::GuardedStackShift {
                        guards,
                        pop: shift.pop,
                        pushes: shift.pushes.iter().map(|&state| map_state(state)).collect(),
                    }
                })
                .collect::<Vec<_>>();
            shifts.sort();
            shifts.dedup();
            Action::GuardedStackShifts(shifts)
        }
        Action::Reduce(nonterminal, len) => {
            Action::Reduce(map_nonterminal(*nonterminal), *len)
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            let mut reduces = reduces
                .iter()
                .map(|&(nonterminal, len)| (map_nonterminal(nonterminal), len))
                .collect::<Vec<_>>();
            reduces.sort_unstable();
            reduces.dedup();
            Action::Split {
                shift: shift.map(|(target, replace)| (map_state(target), replace)),
                reduces,
                accept: *accept,
            }
        }
        Action::Accept => Action::Accept,
        Action::Skip => Action::Skip,
    }
}

fn guarded_state_memberships(
    table: &crate::compiler::glr::table::GLRTable,
) -> Vec<Vec<u32>> {
    let mut memberships = vec![Vec::<u32>::new(); table.num_states as usize];
    let mut guard_index = 0u32;
    for row in &table.action {
        for (_, action) in row.iter() {
            if let Action::GuardedStackShifts(shifts) = action {
                for shift in shifts {
                    for guard in &shift.guards {
                        for &state in &guard.states {
                            if let Some(slot) = memberships.get_mut(state as usize) {
                                slot.push(guard_index);
                            }
                        }
                        guard_index += 1;
                    }
                }
            }
        }
    }
    memberships
}

fn structural_state_signature(
    table: &crate::compiler::glr::table::GLRTable,
    state: u32,
    previous_classes: &[u32],
    terminal_classes: &[u32],
    nonterminal_classes: &[u32],
    guard_memberships: &[Vec<u32>],
) -> StructuralStateSignature {
    let state_index = state as usize;
    let mut alias_conflict = false;
    let mut action_by_terminal_class = BTreeMap::<u32, (Action, bool)>::new();
    for (terminal, action) in table.action[state_index].iter() {
        let terminal_class = terminal_classes
            .get(terminal as usize)
            .copied()
            .unwrap_or(terminal);
        let normalized = remap_action_for_structural_quotient(
            action,
            previous_classes,
            nonterminal_classes,
        );
        let forwarded = table.forwarded_shifts.contains(&(state, terminal));
        match action_by_terminal_class.entry(terminal_class) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((normalized, forwarded));
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get() != &(normalized, forwarded) {
                    alias_conflict = true;
                }
            }
        }
    }

    let mut goto_by_nonterminal_class = BTreeMap::<u32, (u32, bool)>::new();
    for (&nonterminal, &(target, replace)) in table.goto[state_index].iter() {
        let nonterminal_class = nonterminal_classes
            .get(nonterminal as usize)
            .copied()
            .unwrap_or(nonterminal);
        let normalized = (
            previous_classes
                .get(target as usize)
                .copied()
                .unwrap_or(target),
            replace,
        );
        match goto_by_nonterminal_class.entry(nonterminal_class) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(normalized);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get() != &normalized {
                    alias_conflict = true;
                }
            }
        }
    }

    let mut advance = if table.advance.len() == table.num_states as usize {
        table.advance[state_index]
            .iter()
            .map(|terminal| {
                terminal_classes
                    .get(terminal)
                    .copied()
                    .unwrap_or(terminal as u32)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    advance.sort_unstable();
    advance.dedup();

    let mut wide_by_terminal_class = BTreeMap::<u32, Vec<u32>>::new();
    for descriptor in table
        .direct_regular_wide_frontiers
        .iter()
        .filter(|descriptor| descriptor.source_state == state)
    {
        let terminal_class = terminal_classes
            .get(descriptor.terminal as usize)
            .copied()
            .unwrap_or(descriptor.terminal);
        let mut targets = descriptor
            .target_states
            .iter()
            .map(|&target| {
                previous_classes
                    .get(target as usize)
                    .copied()
                    .unwrap_or(target)
            })
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        match wide_by_terminal_class.entry(terminal_class) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(targets);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get() != &targets {
                    alias_conflict = true;
                }
            }
        }
    }

    StructuralStateSignature {
        previous_class: previous_classes[state_index],
        alias_conflict_nonce: alias_conflict.then_some(state),
        guard_memberships: guard_memberships
            .get(state_index)
            .cloned()
            .unwrap_or_default(),
        actions: action_by_terminal_class
            .into_iter()
            .map(|(terminal, (action, forwarded))| (terminal, action, forwarded))
            .collect(),
        gotos: goto_by_nonterminal_class
            .into_iter()
            .map(|(nonterminal, (target, replace))| (nonterminal, target, replace))
            .collect(),
        advance,
        wide_frontiers: wide_by_terminal_class.into_iter().collect(),
    }
}

fn structural_state_classes(
    table: &crate::compiler::glr::table::GLRTable,
    terminal_classes: &[u32],
    nonterminal_map: &[u32],
) -> Vec<u32> {
    let num_states = table.num_states as usize;
    if num_states == 0 {
        return Vec::new();
    }
    // Guard membership is a direct observation of a stack-state identity.  By
    // colouring states with exact membership in every guard set before any
    // refinement, every final guard set is a union of quotient classes.
    let guard_memberships = guarded_state_memberships(table);
    let mut classes = vec![0u32; num_states];
    loop {
        let mut class_by_signature = FxHashMap::<StructuralStateSignature, u32>::default();
        let mut next_classes = Vec::with_capacity(num_states);
        for state in 0..num_states as u32 {
            let signature = structural_state_signature(
                table,
                state,
                &classes,
                terminal_classes,
                nonterminal_map,
                &guard_memberships,
            );
            let next = class_by_signature.len() as u32;
            let class = *class_by_signature.entry(signature).or_insert(next);
            next_classes.push(class);
        }
        if next_classes == classes {
            return classes;
        }
        classes = next_classes;
    }
}

/// Find state correspondences in the independently compiled child tables,
/// before caller-specific continuation terminals are spliced into their rows.
///
/// This is the crucial distinction from quotienting the already-composed LR
/// table. Two copies of the same submachine can have different return
/// lookaheads after linking even though their standalone machines are
/// structurally identical. The resulting groups are only *candidates*; the
/// table-level contextual quotient subsequently proves that the caller can be
/// recovered exactly from the predecessor state before accepting a merge.
fn component_structural_state_groups(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    composed: &ComposedTable,
    terminal_classes: &[u32],
    nonterminal_classes: &[u32],
) -> Vec<Vec<u32>> {
    if children.len() < 2 {
        return Vec::new();
    }

    let parent_nonterminals = parent.table.nonterminal_display_names.len() as u32;
    let mut child_nonterminal_offsets = Vec::with_capacity(children.len());
    let mut next_nonterminal = parent_nonterminals;
    for child in children {
        child_nonterminal_offsets.push(next_nonterminal);
        next_nonterminal += child.constraint.table.nonterminal_display_names.len() as u32;
    }

    let mut virtual_state_offsets = Vec::with_capacity(children.len());
    let mut next_virtual_state = 0u32;
    for child in children {
        virtual_state_offsets.push(next_virtual_state);
        next_virtual_state += child.constraint.table.num_states;
    }
    let virtual_state_count = next_virtual_state as usize;
    if virtual_state_count == 0 {
        return Vec::new();
    }

    // Distinguish each component's start role from internal rows, but allow
    // starts from different components to be compared with one another.
    let mut classes = vec![0u32; virtual_state_count];
    for (component_index, child) in children.iter().enumerate() {
        if child.constraint.table.num_states != 0 {
            classes[virtual_state_offsets[component_index] as usize] = 1;
        }
    }

    loop {
        let mut class_by_signature = FxHashMap::<ComponentStateSignature, u32>::default();
        let mut next_classes = vec![0u32; virtual_state_count];
        for (component_index, child) in children.iter().enumerate() {
            let table = &child.constraint.table;
            let state_offset = virtual_state_offsets[component_index];
            let terminal_offset = composed.terminal_offsets[component_index + 1];
            let nonterminal_offset = child_nonterminal_offsets[component_index];
            let local_state_classes = (0..table.num_states as usize)
                .map(|state| classes[state_offset as usize + state])
                .collect::<Vec<_>>();
            let local_nonterminal_classes = (0..table.nonterminal_display_names.len())
                .map(|nonterminal| {
                    nonterminal_classes
                        .get(nonterminal_offset as usize + nonterminal)
                        .copied()
                        .unwrap_or(nonterminal_offset + nonterminal as u32)
                })
                .collect::<Vec<_>>();

            for local_state in 0..table.num_states {
                let mut actions = table.action[local_state as usize]
                    .iter()
                    .map(|(terminal, action)| {
                        let terminal_class = if terminal < table.num_terminals {
                            terminal_classes
                                .get((terminal_offset + terminal) as usize)
                                .copied()
                                .unwrap_or(terminal_offset + terminal)
                        } else {
                            terminal
                        };
                        let normalized = remap_action_for_structural_quotient(
                            action,
                            &local_state_classes,
                            &local_nonterminal_classes,
                        );
                        (
                            terminal_class,
                            component_action_signature(&normalized),
                            table.forwarded_shifts.contains(&(local_state, terminal)),
                        )
                    })
                    .collect::<Vec<_>>();
                actions.sort();

                let mut gotos = table.goto[local_state as usize]
                    .iter()
                    .map(|(&nonterminal, &(target, replace))| {
                        (
                            local_nonterminal_classes
                                .get(nonterminal as usize)
                                .copied()
                                .unwrap_or(nonterminal_offset + nonterminal),
                            local_state_classes[target as usize],
                            replace,
                        )
                    })
                    .collect::<Vec<_>>();
                gotos.sort_unstable();

                let mut advance = if table.advance.len() == table.num_states as usize {
                    table.advance[local_state as usize]
                        .iter()
                        .map(|terminal| {
                            if terminal < table.num_terminals as usize {
                                terminal_classes
                                    .get(terminal_offset as usize + terminal)
                                    .copied()
                                    .unwrap_or(terminal_offset + terminal as u32)
                            } else {
                                terminal as u32
                            }
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                advance.sort_unstable();
                advance.dedup();

                let virtual_state = state_offset + local_state;
                let signature = ComponentStateSignature {
                    previous_class: classes[virtual_state as usize],
                    actions,
                    gotos,
                    advance,
                };
                let next = class_by_signature.len() as u32;
                next_classes[virtual_state as usize] =
                    *class_by_signature.entry(signature).or_insert(next);
            }
        }
        if next_classes == classes {
            break;
        }
        classes = next_classes;
    }

    let class_count = classes.iter().copied().max().map_or(0, |class| class + 1) as usize;
    let mut members = vec![Vec::<(usize, u32)>::new(); class_count];
    for (component_index, child) in children.iter().enumerate() {
        let state_offset = virtual_state_offsets[component_index];
        for local_state in 0..child.constraint.table.num_states {
            members[classes[(state_offset + local_state) as usize] as usize]
                .push((component_index, local_state));
        }
    }

    let mut out = Vec::new();
    for class_members in members {
        let distinct_components = class_members
            .iter()
            .map(|&(component, _)| component)
            .collect::<BTreeSet<_>>();
        if distinct_components.len() < 2 {
            continue;
        }
        let mut composed_states = Vec::<u32>::new();
        for (component_index, local_state) in class_members {
            if let Some(targets) = composed.state_relations[component_index + 1]
                .get(local_state as usize)
            {
                composed_states.extend_from_slice(targets);
            }
        }
        composed_states.sort_unstable();
        composed_states.dedup();
        if composed_states.len() >= 2 {
            out.push(composed_states);
        }
    }
    out
}

fn remap_composed_state_relations(composed: &mut ComposedTable, state_map: &[u32]) {
    for relation in &mut composed.state_relations {
        for targets in relation {
            for target in targets.iter_mut() {
                *target = state_map
                    .get(*target as usize)
                    .copied()
                    .unwrap_or(*target);
            }
            targets.sort_unstable();
            targets.dedup();
        }
    }
}

pub(super) fn contextually_share_composed_states(
    composed: &mut ComposedTable,
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> (usize, usize) {
    let terminal_classes = composition_terminal_classes(parent, children, composed);
    if terminal_classes
        .iter()
        .enumerate()
        .all(|(terminal, &class)| terminal as u32 == class)
    {
        return (0, 0);
    }
    let nonterminal_classes = structural_nonterminal_classes(
        &composed.table,
        &terminal_classes,
        &composed.boundary_nonterminals,
    );
    let groups = component_structural_state_groups(
        parent,
        children,
        composed,
        &terminal_classes,
        &nonterminal_classes,
    );
    if groups.is_empty() {
        return (0, 0);
    }
    let before = composed.table.num_states as usize;
    let state_map = composed
        .table
        .share_context_distinguishable_states_exact(&groups);
    remap_composed_state_relations(composed, &state_map);
    let after = composed.table.num_states as usize;
    (groups.len(), before.saturating_sub(after))
}

pub(super) fn quotient_composed_table_structurally(
    composed: &mut ComposedTable,
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> Result<StructuralSharingReport, String> {
    let states_before = composed.table.num_states as usize;
    let nonterminals_before = composed.table.nonterminal_display_names.len();
    if states_before <= 1 {
        return Ok(StructuralSharingReport {
            states_before,
            states_after: states_before,
            nonterminals_before,
            nonterminal_classes: nonterminals_before,
            ..StructuralSharingReport::default()
        });
    }

    let terminal_classes = composition_terminal_classes(parent, children, composed);
    let terminal_aliases = terminal_classes
        .iter()
        .enumerate()
        .filter(|&(terminal, &class)| terminal as u32 != class)
        .count();
    if terminal_aliases == 0 {
        return Ok(StructuralSharingReport {
            terminal_aliases,
            nonterminals_before,
            nonterminal_classes: nonterminals_before,
            states_before,
            states_after: states_before,
            ..StructuralSharingReport::default()
        });
    }

    let nonterminal_classes = structural_nonterminal_classes(
        &composed.table,
        &terminal_classes,
        &composed.boundary_nonterminals,
    );
    let nonterminal_class_count = nonterminal_classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class as usize + 1);
    // Structural nonterminal equivalence is intentionally *not* used by this
    // ordinary LR quotient. Equal nonterminal languages do not imply equal
    // goto behavior in one caller state. Keep concrete nonterminal identity
    // here; the stronger cross-child sharing path uses the structural relation
    // only to propose candidates and then preserves source behavior with
    // stack-context guards.
    let nonterminal_identity = (0..nonterminals_before as u32).collect::<Vec<_>>();
    let state_classes =
        structural_state_classes(&composed.table, &terminal_classes, &nonterminal_identity);
    let state_class_count = state_classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class as usize + 1);
    if state_class_count == states_before {
        return Ok(StructuralSharingReport {
            terminal_aliases,
            nonterminals_before,
            nonterminal_classes: nonterminal_class_count,
            states_before,
            states_after: states_before,
            ..StructuralSharingReport::default()
        });
    }

    let mut members = vec![Vec::<u32>::new(); state_class_count];
    for (state, &class) in state_classes.iter().enumerate() {
        members[class as usize].push(state as u32);
    }

    let old_action = std::mem::take(&mut composed.table.action);
    let old_goto = std::mem::take(&mut composed.table.goto);
    let old_advance = std::mem::take(&mut composed.table.advance);
    let has_advance = old_advance.len() == states_before;
    let mut new_action = Vec::<ActionRow>::with_capacity(state_class_count);
    let mut new_goto = Vec::<GotoRow>::with_capacity(state_class_count);
    let mut new_advance = has_advance
        .then(|| Vec::<BitSet>::with_capacity(state_class_count))
        .unwrap_or_default();

    for class_members in &members {
        let mut action_row = ActionRow::default();
        let mut goto_row = GotoRow::default();
        let mut advance_row = BitSet::new(composed.table.num_terminals as usize);
        for &old_state in class_members {
            for (terminal, action) in old_action[old_state as usize].iter() {
                let remapped = remap_action_for_structural_quotient(
                    action,
                    &state_classes,
                    &nonterminal_identity,
                );
                if let Some(previous) = action_row.insert(terminal, remapped.clone())
                    && previous != remapped
                {
                    return Err(format!(
                        "structural-sharing proof violation: quotient state has incompatible actions for terminal {terminal}",
                    ));
                }
            }
            for (&nonterminal, &(target, replace)) in old_goto[old_state as usize].iter() {
                let canonical_nonterminal = nonterminal;
                let remapped = (state_classes[target as usize], replace);
                if let Some(previous) = goto_row.insert(canonical_nonterminal, remapped)
                    && previous != remapped
                {
                    return Err(format!(
                        "structural-sharing proof violation: quotient state members {class_members:?} have incompatible gotos for nonterminal {canonical_nonterminal}; old_state={old_state} original_nonterminal={nonterminal} previous={previous:?} remapped={remapped:?}",
                    ));
                }
            }
            if has_advance {
                for terminal in old_advance[old_state as usize].iter() {
                    advance_row.set(terminal);
                }
            }
        }
        new_action.push(action_row);
        new_goto.push(goto_row);
        if has_advance {
            new_advance.push(advance_row);
        }
    }

    composed.table.action = new_action;
    composed.table.goto = new_goto;
    composed.table.advance = new_advance;
    composed.table.num_states = state_class_count as u32;

    composed.table.forwarded_shifts = composed
        .table
        .forwarded_shifts
        .iter()
        .map(|&(state, terminal)| (state_classes[state as usize], terminal))
        .collect();
    for descriptor in &mut composed.table.direct_regular_wide_frontiers {
        descriptor.source_state = state_classes[descriptor.source_state as usize];
        for target in &mut descriptor.target_states {
            *target = state_classes[*target as usize];
        }
        descriptor.target_states.sort_unstable();
        descriptor.target_states.dedup();
    }
    for relation in &mut composed.state_relations {
        for targets in relation {
            for target in targets.iter_mut() {
                *target = state_classes[*target as usize];
            }
            targets.sort_unstable();
            targets.dedup();
        }
    }
    composed.table.rebuild_guarded_shift_index();
    composed.table.compress_default_action_rows();

    Ok(StructuralSharingReport {
        terminal_aliases,
        nonterminals_before,
        nonterminal_classes: nonterminal_class_count,
        states_before,
        states_after: state_class_count,
        ..StructuralSharingReport::default()
    })
}
