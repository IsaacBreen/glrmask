use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rayon::prelude::*;

use super::row::{ActionRow, GotoRow};
use super::{
    Action, AdmissionPolicy, GLRTable, GlrTableConstruction, GuardedStackShift, StackShift,
    StackShiftGuard,
};
use crate::glr::analysis::EOF;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};

#[derive(Debug, Clone, Copy)]
pub struct SubgrammarTableInput<'a> {
    /// Terminal in the parent table whose shift is replaced by this child's
    /// start row. The terminal remains in the merged ID domain but becomes
    /// unreachable from the merged lexer/table.
    pub placeholder_terminal: TerminalID,
    pub table: &'a GLRTable,
    /// Ignore terminal local to this component, when one exists.  The legacy
    /// splice path does not inspect it; the explicit-control reference path
    /// installs an identity action for it on every child-owned state.
    pub ignore_terminal: Option<TerminalID>,
    /// Source-language nullability of the child's start nonterminal. The
    /// standalone compiled table intentionally omits a root-only epsilon path,
    /// so composition must carry this fact separately.
    pub start_nullable: bool,
}

#[derive(Debug)]
pub struct ComposedTable {
    pub table: GLRTable,
    pub terminal_offsets: Vec<TerminalID>,
    /// Parent placeholder terminal replaced by each child, in child order.
    /// These labels remain in the merged terminal ID domain but have no lexer
    /// language after composition. Keeping the IDs lets higher layers compose
    /// cached terminal-adjacency summaries algebraically instead of rebuilding
    /// FIRST/FOLLOW over the fully merged rule graph.
    pub placeholder_terminals: Vec<TerminalID>,
    /// One relation per input table, parent first. A local parser state may map
    /// to several merged states: a standalone child start state maps to every
    /// parent call-site state, and its accept state maps to the corresponding
    /// placeholder-shift destinations. Internal states map one-to-one.
    pub state_relations: Vec<Vec<Vec<u32>>>,
    /// Remapped root nonterminal of every child linked by this composition.
    /// Boundary repair derives its correctness-critical lexical frontier from
    /// FIRST(root) ∪ FOLLOW(root) over the fully composed grammar.  Unlike the
    /// legacy direct-row seed set, this follows nullable, adjacent, and nested
    /// subgrammars transitively.
    pub boundary_nonterminals: BTreeSet<NonterminalID>,
    /// Internal terminal labels consumed only by linker control closure.  They
    /// have table actions but no lexer language and must never be exposed as
    /// model special tokens.
    pub control_terminals: BTreeSet<TerminalID>,
}

fn remap_rule(rule: &Rule, terminal_offset: TerminalID, nonterminal_offset: NonterminalID) -> Rule {
    Rule {
        lhs: rule.lhs + nonterminal_offset,
        rhs: rule
            .rhs
            .iter()
            .map(|symbol| match *symbol {
                Symbol::Terminal(terminal) => Symbol::Terminal(terminal + terminal_offset),
                Symbol::Nonterminal(nonterminal) => {
                    Symbol::Nonterminal(nonterminal + nonterminal_offset)
                }
            })
            .collect(),
    }
}

fn child_root_nonterminal(table: &GLRTable) -> Result<NonterminalID, String> {
    child_root_nonterminal_from_rules(&table.rules)
}

fn child_root_nonterminal_from_rules(rules: &[Rule]) -> Result<NonterminalID, String> {
    let Some(augmented) = rules.first() else {
        return Err("standalone child table contains no augmented-start rule".to_string());
    };
    match augmented.rhs.as_slice() {
        [Symbol::Nonterminal(root)] => Ok(*root),
        rhs => Err(format!(
            "standalone child augmented-start rule must contain exactly one nonterminal, found {rhs:?}"
        )),
    }
}

fn ensure_epsilon_rule(rules: &mut Vec<Rule>, nonterminal: NonterminalID) {
    if !rules
        .iter()
        .any(|rule| rule.lhs == nonterminal && rule.rhs.is_empty())
    {
        rules.push(Rule {
            lhs: nonterminal,
            rhs: Vec::new(),
        });
    }
}

fn rules_make_start_nullable(rules: &[Rule], start: NonterminalID) -> bool {
    let mut nullable = BTreeSet::<NonterminalID>::new();
    loop {
        let before = nullable.len();
        for rule in rules {
            if rule.rhs.iter().all(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
                Symbol::Terminal(_) => false,
            }) {
                nullable.insert(rule.lhs);
            }
        }
        if nullable.len() == before {
            break;
        }
    }
    nullable.contains(&start)
}

fn remap_state(
    state: u32,
    state_map: &[u32],
    start_replacement: Option<u32>,
    accept_replacement: Option<u32>,
    start_state: u32,
    accept_state: u32,
) -> Result<u32, String> {
    if state == start_state {
        return start_replacement.ok_or_else(|| {
            "child internal table action unexpectedly targets the child start state".to_string()
        });
    }
    if state == accept_state {
        return accept_replacement.ok_or_else(|| {
            "child internal table action unexpectedly targets the child accept state".to_string()
        });
    }
    state_map
        .get(state as usize)
        .copied()
        .filter(|&mapped| mapped != u32::MAX)
        .ok_or_else(|| format!("missing merged state mapping for child state {state}"))
}

fn remap_action(
    action: &Action,
    state_map: &[u32],
    terminal_offset: TerminalID,
    nonterminal_offset: NonterminalID,
    start_replacement: Option<u32>,
    accept_replacement: Option<u32>,
    start_state: u32,
    accept_state: u32,
) -> Result<Action, String> {
    let map_state = |state| {
        remap_state(
            state,
            state_map,
            start_replacement,
            accept_replacement,
            start_state,
            accept_state,
        )
    };
    let _ = terminal_offset;
    Ok(match action {
        Action::Shift(target, replace) => Action::Shift(map_state(*target)?, *replace),
        Action::StackShifts(shifts) => Action::StackShifts(
            shifts
                .iter()
                .map(|shift| {
                    Ok(StackShift {
                        pop: shift.pop,
                        pushes: shift
                            .pushes
                            .iter()
                            .map(|&state| map_state(state))
                            .collect::<Result<Vec<_>, String>>()?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        Action::GuardedStackShifts(shifts) => Action::GuardedStackShifts(
            shifts
                .iter()
                .map(|shift| {
                    Ok(GuardedStackShift {
                        guards: shift
                            .guards
                            .iter()
                            .map(|guard| {
                                Ok(StackShiftGuard {
                                    pop: guard.pop,
                                    states: guard
                                        .states
                                        .iter()
                                        .map(|&state| map_state(state))
                                        .collect::<Result<Vec<_>, String>>()?,
                                })
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                        pop: shift.pop,
                        pushes: shift
                            .pushes
                            .iter()
                            .map(|&state| map_state(state))
                            .collect::<Result<Vec<_>, String>>()?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        Action::Reduce(nonterminal, len) => {
            Action::Reduce(nonterminal + nonterminal_offset, *len)
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => Action::Split {
            shift: shift
                .map(|(target, replace)| -> Result<(u32, bool), String> {
                    Ok((map_state(target)?, replace))
                })
                .transpose()?,
            reduces: reduces
                .iter()
                .map(|&(nonterminal, len)| (nonterminal + nonterminal_offset, len))
                .collect(),
            accept: *accept,
        },
        Action::Accept => Action::Accept,
        Action::ReplaceShifts(targets) => Action::ReplaceShifts(
            targets
                .iter()
                .map(|&target| map_state(target))
                .collect::<Result<Vec<_>, String>>()?
                .into(),
        ),
        Action::Skip => Action::Skip,
    })
}

#[derive(Default)]
struct ActionAlternatives {
    shifts: Vec<StackShift>,
    reduces: Vec<(NonterminalID, u32)>,
    accept: bool,
}

impl ActionAlternatives {
    fn add(&mut self, action: Action) -> Result<(), String> {
        match action {
            Action::Shift(target, replace) => self.shifts.push(StackShift {
                pop: u32::from(replace),
                pushes: vec![target],
            }),
            Action::StackShifts(shifts) => self.shifts.extend(shifts),
            Action::ReplaceShifts(targets) => {
                self.shifts.extend(targets.iter().map(|&target| StackShift {
                    pop: 1,
                    pushes: vec![target],
                }));
            }
            Action::Reduce(nonterminal, len) => self.reduces.push((nonterminal, len)),
            Action::Split {
                shift,
                reduces,
                accept,
            } => {
                if let Some((target, replace)) = shift {
                    self.shifts.push(StackShift {
                        pop: u32::from(replace),
                        pushes: vec![target],
                    });
                }
                self.reduces.extend(reduces);
                self.accept |= accept;
            }
            Action::Accept => self.accept = true,
            Action::Skip => self.shifts.push(StackShift {
                pop: 0,
                pushes: Vec::new(),
            }),
            Action::GuardedStackShifts(_) => {
                return Err(
                    "cannot yet merge a guarded stack-shift cell at a subgrammar entry boundary"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Action, String> {
        self.shifts.sort_by(|left, right| {
            left.pop
                .cmp(&right.pop)
                .then_with(|| left.pushes.cmp(&right.pushes))
        });
        self.shifts.dedup();
        self.reduces.sort_unstable();
        self.reduces.dedup();

        if self.shifts.is_empty() {
            if !self.accept && self.reduces.len() == 1 {
                let (nonterminal, len) = self.reduces[0];
                return Ok(Action::Reduce(nonterminal, len));
            }
            return Ok(Action::Split {
                shift: None,
                reduces: self.reduces,
                accept: self.accept,
            });
        }

        if self.shifts.len() == 1
            && self.shifts[0].pushes.len() == 1
            && self.shifts[0].pop <= 1
        {
            let shift = &self.shifts[0];
            let simple = (shift.pushes[0], shift.pop == 1);
            if self.reduces.is_empty() && !self.accept {
                return Ok(Action::Shift(simple.0, simple.1));
            }
            return Ok(Action::Split {
                shift: Some(simple),
                reduces: self.reduces,
                accept: self.accept,
            });
        }

        if self.reduces.is_empty() && !self.accept {
            return Ok(Action::StackShifts(self.shifts));
        }

        Err(
            "subgrammar entry action collision contains multiple shifts plus reductions/accept"
                .to_string(),
        )
    }
}

fn merge_action_cell(row: &mut ActionRow, terminal: TerminalID, action: Action) -> Result<(), String> {
    let Some(existing) = row.remove(&terminal) else {
        row.insert(terminal, action);
        return Ok(());
    };
    if existing == action {
        row.insert(terminal, existing);
        return Ok(());
    }
    let mut alternatives = ActionAlternatives::default();
    alternatives.add(existing)?;
    alternatives.add(action)?;
    row.insert(terminal, alternatives.finish()?);
    Ok(())
}

fn simple_shift(action: &Action) -> Option<(u32, bool)> {
    match action {
        Action::Shift(target, replace) => Some((*target, *replace)),
        Action::Split {
            shift: Some((target, replace)),
            reduces,
            accept: false,
        } if reduces.is_empty() => Some((*target, *replace)),
        _ => None,
    }
}

fn reduction_only(action: &Action) -> bool {
    match action {
        Action::Reduce(_, _) => true,
        Action::Split {
            shift: None,
            reduces,
            accept: false,
        } => !reduces.is_empty(),
        _ => false,
    }
}

fn table_has_guarded_stack_shifts(table: &GLRTable) -> bool {
    if table.guarded_shift_index.len() == table.num_states as usize {
        return table.guarded_shift_index.iter().any(|row| !row.is_empty());
    }
    table.action.iter().any(|row| {
        row.values()
            .any(|action| matches!(action, Action::GuardedStackShifts(_)))
    })
}

fn accept_state(table: &GLRTable) -> Result<u32, String> {
    let states = table
        .action
        .iter()
        .enumerate()
        .filter_map(|(state, row)| {
            matches!(row.get(&EOF), Some(Action::Accept)).then_some(state as u32)
        })
        .collect::<Vec<_>>();
    match states.as_slice() {
        [state] => Ok(*state),
        _ => Err(format!(
            "standalone child table must have exactly one EOF accept state, found {}",
            states.len()
        )),
    }
}

/// Compose already-built parse tables by replacing each parent placeholder
/// shift with the corresponding child's start row.
///
/// Child internal states are copied once and shared by every call site. The
/// child start state and augmented accept state are not copied:
///
/// * the start row is overlaid onto every parent caller state;
/// * the start-row goto of the child's root nonterminal is redirected to that
///   caller's original placeholder-shift target, preserving the ordinary
///   parent wrapper reduction/continuation;
/// * child EOF reductions are copied onto the union of the parent continuation
///   terminals for that placeholder.
pub fn compose_subgrammar_tables(
    parent: &GLRTable,
    parent_scoped_ignore_terminal: Option<TerminalID>,
    children: &[SubgrammarTableInput<'_>],
) -> Result<ComposedTable, String> {
    let child_rules = children
        .iter()
        .map(|child| child.table.rules.as_slice())
        .collect::<Vec<_>>();
    compose_subgrammar_tables_with_rules(
        parent,
        &parent.rules,
        parent_scoped_ignore_terminal,
        children,
        &child_rules,
    )
}

pub fn compose_subgrammar_tables_with_rules(
    parent: &GLRTable,
    parent_rules: &[Rule],
    parent_scoped_ignore_terminal: Option<TerminalID>,
    children: &[SubgrammarTableInput<'_>],
    child_rules: &[&[Rule]],
) -> Result<ComposedTable, String> {
    if children.len() != child_rules.len() {
        return Err("child table/rule override count mismatch".to_owned());
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    let total_started_at = profile.then(Instant::now);
    let source_has_guarded_stack_shifts = table_has_guarded_stack_shifts(parent)
        || children
            .iter()
            .any(|child| table_has_guarded_stack_shifts(child.table));
    let mut terminal_offsets = Vec::with_capacity(children.len() + 1);
    terminal_offsets.push(0);
    let mut next_terminal = parent.num_terminals;
    for child in children {
        terminal_offsets.push(next_terminal);
        next_terminal = next_terminal
            .checked_add(child.table.num_terminals)
            .ok_or_else(|| "merged terminal ID overflow".to_string())?;
    }

    let parent_nonterminals = parent.nonterminal_display_names.len() as u32;
    let mut nonterminal_offsets = Vec::with_capacity(children.len());
    let mut next_nonterminal = parent_nonterminals;
    for child in children {
        nonterminal_offsets.push(next_nonterminal);
        next_nonterminal = next_nonterminal
            .checked_add(child.table.nonterminal_display_names.len() as u32)
            .ok_or_else(|| "merged nonterminal ID overflow".to_string())?;
    }

    let mut action = parent.action.clone();
    let mut goto = parent.goto.clone();
    if let Some(ignore) = parent_scoped_ignore_terminal {
        for row in &mut action {
            merge_action_cell(row, ignore, identity_skip_action())?;
        }
    }
    let mut state_relations = Vec::with_capacity(children.len() + 1);
    state_relations.push(
        (0..parent.num_states)
            .map(|state| vec![state])
            .collect::<Vec<_>>(),
    );
    let mut next_state = parent.num_states;

    let parent_root = child_root_nonterminal_from_rules(parent_rules)?;
    let mut rules = parent_rules.to_vec();
    if parent.embedded_start_nullable() {
        ensure_epsilon_rule(&mut rules, parent_root);
    }
    let parent_rule_count = rules.len();
    let mut nonterminal_display_names = parent.nonterminal_display_names.clone();
    let mut terminal_display_suffixes = Vec::<String>::new();
    let mut forwarded_shifts = parent.forwarded_shifts.clone();
    let mut direct_regular_wide_frontiers = parent.direct_regular_wide_frontiers.clone();
    let mut boundary_nonterminals = BTreeSet::<NonterminalID>::new();
    let mut rows_needing_compress = BTreeSet::<usize>::new();
    let mut skip_terminals = parent.skip_terminals.clone();
    if let Some(ignore) = parent_scoped_ignore_terminal {
        skip_terminals.insert(ignore);
    }

    // Observable entry terminals for non-nullable child placeholders.  These
    // are used when one child is followed immediately by another: the first
    // child's EOF reduction must be keyed by the next child's first *real*
    // terminal, never by the impossible placeholder symbol itself.
    let placeholder_entry_terminals = children
        .iter()
        .enumerate()
        .map(|(child_index, input)| {
            let terminal_offset = terminal_offsets[child_index + 1];
            let mut entries = input
                .table
                .action[0]
                .keys()
                .filter(|&terminal| terminal != EOF)
                .map(|terminal| terminal + terminal_offset)
                .collect::<BTreeSet<_>>();
            if let Some(ignore) = input.ignore_terminal {
                entries.insert(ignore + terminal_offset);
            }
            (input.placeholder_terminal, entries)
        })
        .collect::<BTreeMap<_, _>>();

    for (child_index, child_input) in children.iter().enumerate() {
        let child = child_input.table;
        let child_rules = child_rules[child_index];
        let terminal_offset = terminal_offsets[child_index + 1];
        // A child may itself already be a composed/scoped table. Its Skip
        // actions are copied below like ordinary LR actions, so their metadata
        // must be transported as well. `child_input.ignore_terminal` only
        // describes a standalone component-level ignore and is intentionally
        // None for such precomposed children.
        skip_terminals.extend(
            child
                .skip_terminals
                .iter()
                .map(|terminal| terminal + terminal_offset),
        );
        let nonterminal_offset = nonterminal_offsets[child_index];
        let child_root_local = child_root_nonterminal_from_rules(child_rules)?;
        let child_root = child_root_local + nonterminal_offset;
        boundary_nonterminals.insert(child_root);
        let child_start = 0u32;
        let child_accept = accept_state(child)?;

        // Keep the rule metadata semantically aligned with the manually
        // spliced table. Replacing a placeholder terminal by the child's root
        // nonterminal preserves RHS length, so existing reduce actions and
        // parser states remain valid, while FOLLOW/template analysis can now
        // see nullable children and true parent↔child adjacency.
        for rule in &mut rules[..parent_rule_count] {
            for symbol in &mut rule.rhs {
                if *symbol == Symbol::Terminal(child_input.placeholder_terminal) {
                    *symbol = Symbol::Nonterminal(child_root);
                }
            }
        }

        let mut call_sites = Vec::<(u32, u32, bool)>::new();
        let mut precursor_actions = Vec::<(u32, Action)>::new();
        for state in 0..parent.num_states {
            let Some(placeholder_action) = parent.action(state, child_input.placeholder_terminal)
            else {
                continue;
            };
            let Some((target, replace)) = simple_shift(placeholder_action) else {
                if reduction_only(placeholder_action) {
                    precursor_actions.push((state, placeholder_action.clone()));
                    continue;
                }
                return Err(format!(
                    "placeholder terminal {} has unsupported action {:?} in parent state {state}",
                    child_input.placeholder_terminal, placeholder_action
                ));
            };
            call_sites.push((state, target, replace));
        }
        if call_sites.is_empty() {
            return Err(format!(
                "placeholder terminal {} has no shift call sites in the parent table",
                child_input.placeholder_terminal
            ));
        }
        let mut continuation_terminals = BTreeSet::<TerminalID>::new();
        for &(_, target, _) in &call_sites {
            continuation_terminals.extend(parent.action[target as usize].keys());
        }
        if let Some(ignore) = parent_scoped_ignore_terminal {
            continuation_terminals.insert(ignore);
        }
        let raw_continuations = std::mem::take(&mut continuation_terminals);
        for terminal in raw_continuations {
            if let Some(entries) = placeholder_entry_terminals.get(&terminal) {
                continuation_terminals.extend(entries.iter().copied());
            } else {
                continuation_terminals.insert(terminal);
            }
        }
        // A placeholder lookahead may first trigger one or more parent
        // reductions before reaching the actual shift call site. Once the
        // placeholder is replaced, those precursor reductions must be keyed
        // by every real child-entry lookahead instead. Otherwise later
        // sequential subgrammar calls fail before reaching their caller row.
        let mut child_entry_terminals = placeholder_entry_terminals
            .get(&child_input.placeholder_terminal)
            .cloned()
            .unwrap_or_default();
        if child_input.start_nullable {
            child_entry_terminals.extend(continuation_terminals.iter().copied());
        }
        for (state, precursor_action) in precursor_actions {
            rows_needing_compress.insert(state as usize);
            action[state as usize].remove(&child_input.placeholder_terminal);
            for &terminal in &child_entry_terminals {
                merge_action_cell(
                    &mut action[state as usize],
                    terminal,
                    precursor_action.clone(),
                )?;
            }
        }

        let mut child_state_map = vec![u32::MAX; child.num_states as usize];
        for child_state in 0..child.num_states {
            if child_state == child_start || child_state == child_accept {
                continue;
            }
            child_state_map[child_state as usize] = next_state;
            next_state += 1;
            action.push(ActionRow::default());
            goto.push(GotoRow::default());
        }

        // If the child has a scope-local ignore distinct from the parent's
        // scope, consuming that ignore is the first observable proof that the
        // zero-width call boundary has been crossed.  Refine the caller state
        // with one same-depth phase state per call site.  The phase carries
        // exactly the child's start-row behavior but not parent-scope ignore
        // actions.  This is ordinary LR state splitting, not a control/epsilon
        // transition: the first child-ignore token replaces caller -> phase.
        let mut child_scope_phase = BTreeMap::<u32, u32>::new();
        if child_input.ignore_terminal.is_some() {
            for &(caller, _, _) in &call_sites {
                let phase = next_state;
                next_state += 1;
                action.push(ActionRow::default());
                goto.push(GotoRow::default());
                child_scope_phase.insert(caller, phase);
            }
        }

        let mut child_relation = vec![Vec::<u32>::new(); child.num_states as usize];
        child_relation[child_start as usize] =
            call_sites.iter().map(|&(caller, _, _)| caller).collect();
        child_relation[child_start as usize]
            .extend(child_scope_phase.values().copied());
        child_relation[child_accept as usize] =
            call_sites.iter().map(|&(_, target, _)| target).collect();
        for child_state in 0..child.num_states {
            let mapped = child_state_map[child_state as usize];
            if mapped != u32::MAX {
                child_relation[child_state as usize].push(mapped);
            }
        }
        for targets in &mut child_relation {
            targets.sort_unstable();
            targets.dedup();
        }
        state_relations.push(child_relation);

        if let Some(local_ignore) = child_input.ignore_terminal {
            skip_terminals.insert(local_ignore + terminal_offset);
        }
        let row_transport_started_at = profile.then(Instant::now);
        let mapped_child_rows = (0..child.num_states)
            .into_par_iter()
            .filter(|&child_state| child_state != child_start && child_state != child_accept)
            .map(|child_state| {
            let merged_state = child_state_map[child_state as usize] as usize;
            let mut mapped_action_row = ActionRow::default();
            let mut mapped_goto_row = GotoRow::default();
            for (terminal, child_action) in child.action[child_state as usize].iter() {
                let mapped_action = remap_action(
                    child_action,
                    &child_state_map,
                    terminal_offset,
                    nonterminal_offset,
                    None,
                    None,
                    child_start,
                    child_accept,
                )?;
                if terminal == EOF {
                    for &continuation in &continuation_terminals {
                        merge_action_cell(
                            &mut mapped_action_row,
                            continuation,
                            mapped_action.clone(),
                        )?;
                    }
                } else {
                    merge_action_cell(
                        &mut mapped_action_row,
                        terminal + terminal_offset,
                        mapped_action,
                    )?;
                }
            }
            if let Some(local_ignore) = child_input.ignore_terminal {
                merge_action_cell(
                    &mut mapped_action_row,
                    local_ignore + terminal_offset,
                    identity_skip_action(),
                )?;
            }
            mapped_action_row.compress_default(next_terminal);
            for (nonterminal, &(target, replace)) in child.goto[child_state as usize].iter() {
                let target = remap_state(
                    target,
                    &child_state_map,
                    None,
                    None,
                    child_start,
                    child_accept,
                )?;
                mapped_goto_row.insert(nonterminal + nonterminal_offset, (target, replace));
            }
            Ok::<_, String>((merged_state, mapped_action_row, mapped_goto_row))
        })
            .collect::<Result<Vec<_>, String>>()?;
        for (merged_state, mapped_action_row, mapped_goto_row) in mapped_child_rows {
            action[merged_state] = mapped_action_row;
            goto[merged_state] = mapped_goto_row;
        }
        if let Some(started_at) = row_transport_started_at {
            eprintln!(
                "[glrmask/profile][subgrammar_table_row_transport] child={} states={} ms={:.3}",
                child_index,
                child.num_states,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        for &(caller_state, placeholder_target, placeholder_replace) in &call_sites {
            rows_needing_compress.insert(caller_state as usize);
            action[caller_state as usize].remove(&child_input.placeholder_terminal);
            for (terminal, child_action) in child.action[child_start as usize].iter() {
                let mapped_action = remap_action(
                    child_action,
                    &child_state_map,
                    terminal_offset,
                    nonterminal_offset,
                    Some(caller_state),
                    Some(placeholder_target),
                    child_start,
                    child_accept,
                )?;
                if terminal == EOF {
                    for &continuation in &continuation_terminals {
                        merge_action_cell(
                            &mut action[caller_state as usize],
                            continuation,
                            mapped_action.clone(),
                        )?;
                    }
                } else {
                    merge_action_cell(
                        &mut action[caller_state as usize],
                        terminal + terminal_offset,
                        mapped_action,
                    )?;
                }
            }
            for (nonterminal, &(target, replace)) in child.goto[child_start as usize].iter() {
                let is_child_root = *nonterminal == child_root_local;
                let target = remap_state(
                    target,
                    &child_state_map,
                    Some(caller_state),
                    Some(placeholder_target),
                    child_start,
                    child_accept,
                )?;
                goto[caller_state as usize].insert(
                    nonterminal + nonterminal_offset,
                    (
                        target,
                        if is_child_root {
                            placeholder_replace
                        } else {
                            replace
                        },
                    ),
                );
            }
            if child_input.start_nullable {
                for &continuation in &continuation_terminals {
                    merge_action_cell(
                        &mut action[caller_state as usize],
                        continuation,
                        Action::Reduce(child_root, 0),
                    )?;
                }
            }

            if let (Some(local_ignore), Some(&phase_state)) = (
                child_input.ignore_terminal,
                child_scope_phase.get(&caller_state),
            ) {
                rows_needing_compress.insert(phase_state as usize);
                let scoped_ignore = local_ignore + terminal_offset;
                // Consuming the first child-scope ignore refines only the state
                // identity; it must not change LR stack depth.
                merge_action_cell(
                    &mut action[caller_state as usize],
                    scoped_ignore,
                    Action::Shift(phase_state, true),
                )?;
                merge_action_cell(
                    &mut action[phase_state as usize],
                    scoped_ignore,
                    identity_skip_action(),
                )?;
                skip_terminals.insert(scoped_ignore);

                // Phase state is the child-entry refinement of this caller.
                // Rebuild the same child-start actions/gotos with the phase as
                // the start-state representative.  Parent-scope actions are
                // deliberately absent.
                for (terminal, child_action) in child.action[child_start as usize].iter() {
                    let mapped_action = remap_action(
                        child_action,
                        &child_state_map,
                        terminal_offset,
                        nonterminal_offset,
                        Some(phase_state),
                        Some(placeholder_target),
                        child_start,
                        child_accept,
                    )?;
                    if terminal == EOF {
                        for &continuation in &continuation_terminals {
                            merge_action_cell(
                                &mut action[phase_state as usize],
                                continuation,
                                mapped_action.clone(),
                            )?;
                        }
                    } else {
                        merge_action_cell(
                            &mut action[phase_state as usize],
                            terminal + terminal_offset,
                            mapped_action,
                        )?;
                    }
                }
                for (nonterminal, &(target, replace)) in
                    child.goto[child_start as usize].iter()
                {
                    let is_child_root = *nonterminal == child_root_local;
                    let target = remap_state(
                        target,
                        &child_state_map,
                        Some(phase_state),
                        Some(placeholder_target),
                        child_start,
                        child_accept,
                    )?;
                    goto[phase_state as usize].insert(
                        nonterminal + nonterminal_offset,
                        (
                            target,
                            if is_child_root {
                                placeholder_replace
                            } else {
                                replace
                            },
                        ),
                    );
                }
            }
        }

        for rule in child_rules {
            rules.push(remap_rule(rule, terminal_offset, nonterminal_offset));
        }
        if child_input.start_nullable {
            ensure_epsilon_rule(&mut rules, child_root);
        }
        nonterminal_display_names.extend(
            child
                .nonterminal_display_names
                .iter()
                .map(|name| format!("child{child_index}::{name}")),
        );
        terminal_display_suffixes.extend(
            (0..child.num_terminals).map(|terminal| format!("child{child_index}::T{terminal}")),
        );

        for &(state, terminal) in &child.forwarded_shifts {
            if state == child_start {
                for &(caller, _, _) in &call_sites {
                    forwarded_shifts.insert((caller, terminal + terminal_offset));
                }
            } else if state != child_accept {
                forwarded_shifts.insert((
                    child_state_map[state as usize],
                    terminal + terminal_offset,
                ));
            }
        }
        for frontier in &child.direct_regular_wide_frontiers {
            if frontier.source_state == child_accept {
                continue;
            }
            let source_states = if frontier.source_state == child_start {
                call_sites
                    .iter()
                    .map(|&(caller, _, _)| caller)
                    .collect::<Vec<_>>()
            } else {
                vec![child_state_map[frontier.source_state as usize]]
            };
            for source_state in source_states {
                let mut target_states = Vec::with_capacity(frontier.target_states.len());
                let mut valid = true;
                for &target in &frontier.target_states {
                    match remap_state(
                        target,
                        &child_state_map,
                        Some(source_state),
                        None,
                        child_start,
                        child_accept,
                    ) {
                        Ok(target) => target_states.push(target),
                        Err(_) => {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid {
                    direct_regular_wide_frontiers.push(
                        super::DirectRegularWideFrontierDescriptor {
                            source_state,
                            terminal: frontier.terminal + terminal_offset,
                            target_states,
                        },
                    );
                }
            }
        }
    }

    let result_start_nullable = rules_make_start_nullable(&rules, parent_root);
    let mut table = GLRTable {
        action,
        goto,
        num_states: next_state,
        num_terminals: next_terminal,
        num_rules: rules.len() as u32,
        rules,
        nonterminal_display_names,
        construction: GlrTableConstruction::Lalr,
        admission_policy: AdmissionPolicy::ExactSimulation,
        advance: Vec::<BitSet>::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts,
        control_terminals: Default::default(),
        skip_terminals,
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers,
    };
    let _ = terminal_display_suffixes;
    let advance_started_at = profile.then(Instant::now);
    table.rebuild_advance_rows_from_actions();
    let advance_ms = advance_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let unconditional_started_at = profile.then(Instant::now);
    table.rebuild_unconditional_advance_rows();
    let unconditional_ms = unconditional_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let guarded_started_at = profile.then(Instant::now);
    if source_has_guarded_stack_shifts {
        table.rebuild_guarded_shift_index();
    } else {
        table.guarded_shift_index = vec![Default::default(); table.num_states as usize];
    }
    let guarded_ms = guarded_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let compress_started_at = profile.then(Instant::now);
    for state in rows_needing_compress {
        table.action[state].compress_default(table.num_terminals);
    }
    let compress_ms = compress_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    table.set_embedded_start_nullable(result_start_nullable);
    if let Some(started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][subgrammar_table_compose] states={} advance_ms={advance_ms:.3} unconditional_ms={unconditional_ms:.3} guarded_ms={guarded_ms:.3} compress_ms={compress_ms:.3} total_ms={:.3}",
            table.num_states,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Ok(ComposedTable {
        table,
        terminal_offsets,
        placeholder_terminals: children
            .iter()
            .map(|child| child.placeholder_terminal)
            .collect(),
        state_relations,
        boundary_nonterminals,
        control_terminals: BTreeSet::new(),
    })
}

fn identity_skip_action() -> Action {
    Action::Skip
}

/// Reference subgrammar linker which preserves call and return boundaries as
/// internal zero-width control terminals.
///
/// Unlike [`compose_subgrammar_tables`], this path does not identify the child
/// start with the parent caller or the child accept with the parent
/// continuation.  Each call site gets a complete remapped child table.  The
/// parent's placeholder terminal becomes an internal control label:
///
/// * the original placeholder shift enters the copied child start state;
/// * child-local EOF reductions are executed under the same control label;
/// * child acceptance becomes a stack shift back to the original parent
///   continuation.
///
/// The control labels have no lexer language.  A caller compiling terminal
/// paths must make them available only as zero-width transitions at lexeme
/// boundaries.  This deliberately simple representation is the semantic
/// reference against which future state merging/epsilon elimination can be
/// validated.
pub fn compose_subgrammar_tables_explicit(
    parent: &GLRTable,
    parent_scoped_ignore_terminal: Option<TerminalID>,
    children: &[SubgrammarTableInput<'_>],
) -> Result<ComposedTable, String> {
    let child_rules = children
        .iter()
        .map(|child| child.table.rules.as_slice())
        .collect::<Vec<_>>();
    compose_subgrammar_tables_explicit_with_rules(
        parent,
        &parent.rules,
        parent_scoped_ignore_terminal,
        children,
        &child_rules,
    )
}

pub fn compose_subgrammar_tables_explicit_with_rules(
    parent: &GLRTable,
    parent_rules: &[Rule],
    parent_scoped_ignore_terminal: Option<TerminalID>,
    children: &[SubgrammarTableInput<'_>],
    child_rules: &[&[Rule]],
) -> Result<ComposedTable, String> {
    if children.len() != child_rules.len() {
        return Err("child table/rule override count mismatch".to_owned());
    }
    let mut terminal_offsets = Vec::with_capacity(children.len() + 1);
    terminal_offsets.push(0);
    let mut next_terminal = parent.num_terminals;
    for child in children {
        terminal_offsets.push(next_terminal);
        next_terminal = next_terminal
            .checked_add(child.table.num_terminals)
            .ok_or_else(|| "merged terminal ID overflow".to_string())?;
    }

    let parent_nonterminals = parent.nonterminal_display_names.len() as u32;
    let mut nonterminal_offsets = Vec::with_capacity(children.len());
    let mut next_nonterminal = parent_nonterminals;
    for child in children {
        nonterminal_offsets.push(next_nonterminal);
        next_nonterminal = next_nonterminal
            .checked_add(child.table.nonterminal_display_names.len() as u32)
            .ok_or_else(|| "merged nonterminal ID overflow".to_string())?;
    }

    let mut action = parent.action.clone();
    let mut goto = parent.goto.clone();
    if let Some(ignore) = parent_scoped_ignore_terminal {
        for row in &mut action {
            merge_action_cell(row, ignore, identity_skip_action())?;
        }
    }

    let mut state_relations = Vec::with_capacity(children.len() + 1);
    state_relations.push(
        (0..parent.num_states)
            .map(|state| vec![state])
            .collect::<Vec<_>>(),
    );
    let mut next_state = parent.num_states;

    let parent_root = child_root_nonterminal_from_rules(parent_rules)?;
    let mut rules = parent_rules.to_vec();
    if parent.embedded_start_nullable() {
        ensure_epsilon_rule(&mut rules, parent_root);
    }
    let parent_rule_count = rules.len();
    let mut nonterminal_display_names = parent.nonterminal_display_names.clone();
    let mut forwarded_shifts = parent.forwarded_shifts.clone();
    let mut direct_regular_wide_frontiers = parent.direct_regular_wide_frontiers.clone();
    let mut boundary_nonterminals = BTreeSet::<NonterminalID>::new();
    let mut control_terminals = parent.control_terminals.clone();
    let mut skip_terminals = parent.skip_terminals.clone();
    if let Some(ignore) = parent_scoped_ignore_terminal {
        skip_terminals.insert(ignore);
    }

    for (child_index, child_input) in children.iter().enumerate() {
        let child = child_input.table;
        let child_rules = child_rules[child_index];
        let terminal_offset = terminal_offsets[child_index + 1];
        let nonterminal_offset = nonterminal_offsets[child_index];
        let child_root_local = child_root_nonterminal_from_rules(child_rules)?;
        let child_root = child_root_local + nonterminal_offset;
        boundary_nonterminals.insert(child_root);
        let child_start = 0u32;
        let child_accept = accept_state(child)?;
        let control = child_input.placeholder_terminal;
        control_terminals.insert(control);
        control_terminals.extend(
            child
                .control_terminals
                .iter()
                .map(|terminal| terminal + terminal_offset),
        );
        skip_terminals.extend(
            child
                .skip_terminals
                .iter()
                .map(|terminal| terminal + terminal_offset),
        );
        if let Some(ignore) = child_input.ignore_terminal {
            skip_terminals.insert(ignore + terminal_offset);
        }

        for rule in &mut rules[..parent_rule_count] {
            for symbol in &mut rule.rhs {
                if *symbol == Symbol::Terminal(control) {
                    *symbol = Symbol::Nonterminal(child_root);
                }
            }
        }

        let mut call_sites = Vec::<(u32, u32, bool)>::new();
        for state in 0..parent.num_states {
            let Some(placeholder_action) = parent.action(state, control) else {
                continue;
            };
            if let Some((target, replace)) = simple_shift(placeholder_action) {
                call_sites.push((state, target, replace));
            } else if !reduction_only(placeholder_action) {
                return Err(format!(
                    "placeholder terminal {control} has unsupported action {placeholder_action:?} in parent state {state}",
                ));
            }
        }
        if call_sites.is_empty() {
            return Err(format!(
                "placeholder terminal {control} has no shift call sites in the parent table",
            ));
        }

        let Some(&(child_root_target, child_root_replace)) =
            child.goto[child_start as usize].get(&child_root_local)
        else {
            return Err("child start row has no goto for its root nonterminal".to_string());
        };
        if child_root_target != child_accept {
            return Err(format!(
                "child root goto targets state {child_root_target}, expected accept state {child_accept}",
            ));
        }
        let return_pop = if child_root_replace { 1 } else { 2 };

        let mut child_relation = vec![Vec::<u32>::new(); child.num_states as usize];

        // Deliberately copy the whole child once per call site.  This makes the
        // continuation and active ignore scope explicit in the parser-state
        // identity.  Sharing internal states is a later optimization which can
        // be proved against this representation.
        for &(caller_state, placeholder_target, placeholder_replace) in &call_sites {
            let mut state_map = vec![u32::MAX; child.num_states as usize];
            for local_state in 0..child.num_states {
                state_map[local_state as usize] = next_state;
                child_relation[local_state as usize].push(next_state);
                next_state += 1;
                action.push(ActionRow::default());
                goto.push(GotoRow::default());
            }

            let mapped_start = state_map[child_start as usize];
            let mapped_accept = state_map[child_accept as usize];
            action[caller_state as usize].insert(
                control,
                Action::Shift(mapped_start, placeholder_replace),
            );

            // A compiled child can retain exact root-nullability metadata even
            // after its internal epsilon reduction has been normalized out of
            // the LR action rows.  The explicit-control linker must therefore
            // make the empty child derivation explicit too.  From the mapped
            // child start, reducing an empty root and then taking the ordinary
            // child-return control has net stack effect `pop start; push parent
            // continuation`, independent of whether the child's root goto
            // itself uses replace semantics.
            if child_input.start_nullable {
                merge_action_cell(
                    &mut action[mapped_start as usize],
                    control,
                    Action::StackShifts(vec![StackShift {
                        pop: 1,
                        pushes: vec![placeholder_target],
                    }]),
                )?;
            }

            for local_state in 0..child.num_states {
                let merged_state = state_map[local_state as usize] as usize;
                for (terminal, child_action) in child.action[local_state as usize].iter() {
                    if terminal == EOF && local_state == child_accept {
                        if !matches!(child_action, Action::Accept) {
                            return Err(format!(
                                "child accept state {child_accept} has unsupported EOF action {child_action:?}",
                            ));
                        }
                        merge_action_cell(
                            &mut action[merged_state],
                            control,
                            Action::StackShifts(vec![StackShift {
                                pop: return_pop,
                                pushes: vec![placeholder_target],
                            }]),
                        )?;
                        continue;
                    }

                    let mapped_action = remap_action(
                        child_action,
                        &state_map,
                        terminal_offset,
                        nonterminal_offset,
                        Some(mapped_start),
                        Some(mapped_accept),
                        child_start,
                        child_accept,
                    )?;
                    let mapped_terminal = if terminal == EOF {
                        control
                    } else {
                        terminal + terminal_offset
                    };
                    merge_action_cell(
                        &mut action[merged_state],
                        mapped_terminal,
                        mapped_action,
                    )?;
                }

                if let Some(local_ignore) = child_input.ignore_terminal {
                    merge_action_cell(
                        &mut action[merged_state],
                        local_ignore + terminal_offset,
                        identity_skip_action(),
                    )?;
                }

                for (nonterminal, &(target, replace)) in child.goto[local_state as usize].iter() {
                    goto[merged_state].insert(
                        nonterminal + nonterminal_offset,
                        (state_map[target as usize], replace),
                    );
                }
            }

            for &(state, terminal) in &child.forwarded_shifts {
                forwarded_shifts.insert((
                    state_map[state as usize],
                    terminal + terminal_offset,
                ));
            }
            for frontier in &child.direct_regular_wide_frontiers {
                direct_regular_wide_frontiers.push(
                    super::DirectRegularWideFrontierDescriptor {
                        source_state: state_map[frontier.source_state as usize],
                        terminal: frontier.terminal + terminal_offset,
                        target_states: frontier
                            .target_states
                            .iter()
                            .map(|&target| state_map[target as usize])
                            .collect(),
                    },
                );
            }
        }

        for targets in &mut child_relation {
            targets.sort_unstable();
            targets.dedup();
        }
        state_relations.push(child_relation);

        for rule in child_rules {
            rules.push(remap_rule(rule, terminal_offset, nonterminal_offset));
        }
        if child_input.start_nullable {
            ensure_epsilon_rule(&mut rules, child_root);
        }
        nonterminal_display_names.extend(
            child
                .nonterminal_display_names
                .iter()
                .map(|name| format!("child{child_index}::{name}")),
        );
    }

    let result_start_nullable = rules_make_start_nullable(&rules, parent_root);
    let mut table = GLRTable {
        action,
        goto,
        num_states: next_state,
        num_terminals: next_terminal,
        num_rules: rules.len() as u32,
        rules,
        nonterminal_display_names,
        construction: GlrTableConstruction::Lalr,
        admission_policy: AdmissionPolicy::ExactSimulation,
        advance: Vec::<BitSet>::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts,
        control_terminals: control_terminals.clone(),
        skip_terminals,
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers,
    };
    table.rebuild_advance_rows_from_actions();
    table.rebuild_guarded_shift_index();
    table.compress_default_action_rows();
    table.set_embedded_start_nullable(result_start_nullable);

    Ok(ComposedTable {
        table,
        terminal_offsets,
        placeholder_terminals: children
            .iter()
            .map(|child| child.placeholder_terminal)
            .collect(),
        state_relations,
        boundary_nonterminals,
        control_terminals,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};

    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::grammar::ast::lower;
    use crate::grammar::glrm::from_glrm;

    fn table(source: &str) -> (GLRTable, AnalyzedGrammar) {
        let named = from_glrm(source).unwrap();
        let grammar = lower(&named).unwrap();
        let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
        (GLRTable::build(&analyzed), analyzed)
    }

    fn apply_stack_shift(stack: &[u32], pop: u32, pushes: &[u32]) -> Option<Vec<u32>> {
        if pop as usize >= stack.len() {
            return None;
        }
        let mut next = stack[..stack.len() - pop as usize].to_vec();
        next.extend_from_slice(pushes);
        Some(next)
    }

    fn guard_matches(stack: &[u32], guard: &StackShiftGuard) -> bool {
        let Some(index) = stack.len().checked_sub(guard.pop as usize + 1) else {
            return false;
        };
        guard.states.binary_search(&stack[index]).is_ok()
            || guard.states.contains(&stack[index])
    }

    fn shift_results(table: &GLRTable, initial: Vec<u32>, terminal: TerminalID) -> Vec<Vec<u32>> {
        let mut pending = VecDeque::from([initial]);
        let mut visited = BTreeSet::<Vec<u32>>::new();
        let mut shifted = BTreeSet::<Vec<u32>>::new();
        while let Some(stack) = pending.pop_front() {
            if !visited.insert(stack.clone()) {
                continue;
            }
            let Some(&top) = stack.last() else {
                continue;
            };
            let Some(action) = table.action(top, terminal) else {
                continue;
            };
            let mut add_reduce = |nonterminal: NonterminalID, len: u32| {
                let Some(mut reduced) = apply_stack_shift(&stack, len, &[]) else {
                    return;
                };
                let Some(&source) = reduced.last() else {
                    return;
                };
                let Some(&(target, replace)) = table.goto[source as usize].get(&nonterminal)
                else {
                    return;
                };
                if replace {
                    reduced.pop();
                }
                reduced.push(target);
                pending.push_back(reduced);
            };
            match action {
                Action::Skip => {
                    shifted.insert(stack.clone());
                }
                Action::Shift(target, replace) => {
                    if let Some(next) = apply_stack_shift(
                        &stack,
                        u32::from(*replace),
                        std::slice::from_ref(target),
                    ) {
                        shifted.insert(next);
                    }
                }
                Action::StackShifts(shifts) => {
                    for shift in shifts {
                        if let Some(next) = apply_stack_shift(&stack, shift.pop, &shift.pushes) {
                            shifted.insert(next);
                        }
                    }
                }
                Action::GuardedStackShifts(shifts) => {
                    for shift in shifts {
                        if shift.guards.iter().all(|guard| guard_matches(&stack, guard))
                            && let Some(next) =
                                apply_stack_shift(&stack, shift.pop, &shift.pushes)
                        {
                            shifted.insert(next);
                        }
                    }
                }
                Action::Reduce(nonterminal, len) => add_reduce(*nonterminal, *len),
                Action::Split {
                    shift,
                    reduces,
                    accept: _,
                } => {
                    if let Some((target, replace)) = shift
                        && let Some(next) = apply_stack_shift(
                            &stack,
                            u32::from(*replace),
                            std::slice::from_ref(target),
                        )
                    {
                        shifted.insert(next);
                    }
                    for &(nonterminal, len) in reduces {
                        add_reduce(nonterminal, len);
                    }
                }
                Action::Accept => {}
                Action::ReplaceShifts(targets) => {
                    for target in targets.iter() {
                        if let Some(next) = apply_stack_shift(
                            &stack,
                            1,
                            std::slice::from_ref(target),
                        ) {
                            shifted.insert(next);
                        }
                    }
                }
            }
        }
        shifted.into_iter().collect()
    }

    fn accepts(table: &GLRTable, word: &[TerminalID]) -> bool {
        let mut stacks = BTreeSet::from([vec![0u32]]);
        let close_controls = |mut stacks: BTreeSet<Vec<u32>>| {
            if table.control_terminals.is_empty() {
                return stacks;
            }
            loop {
                let mut next = stacks.clone();
                for stack in &stacks {
                    for &control in &table.control_terminals {
                        next.extend(shift_results(table, stack.clone(), control));
                    }
                }
                if next == stacks {
                    return stacks;
                }
                stacks = next;
            }
        };
        stacks = close_controls(stacks);
        for &terminal in word {
            let mut next = BTreeSet::new();
            for stack in stacks {
                next.extend(shift_results(table, stack, terminal));
            }
            if next.is_empty() {
                return false;
            }
            stacks = close_controls(next);
        }

        let mut pending = VecDeque::from_iter(close_controls(stacks));
        let mut visited = BTreeSet::new();
        while let Some(stack) = pending.pop_front() {
            if !visited.insert(stack.clone()) {
                continue;
            }
            let Some(&top) = stack.last() else {
                continue;
            };
            match table.action(top, EOF) {
                Some(Action::Accept)
                | Some(Action::Split {
                    accept: true, ..
                }) => return true,
                Some(Action::Skip) => {
                    pending.push_back(stack);
                }
                Some(Action::Shift(target, replace)) => {
                    if let Some(next) = apply_stack_shift(
                        &stack,
                        u32::from(*replace),
                        std::slice::from_ref(target),
                    ) {
                        pending.push_back(next);
                    }
                }
                Some(Action::ReplaceShifts(targets)) => {
                    for target in targets.iter() {
                        if let Some(next) = apply_stack_shift(
                            &stack,
                            1,
                            std::slice::from_ref(target),
                        ) {
                            pending.push_back(next);
                        }
                    }
                }
                Some(Action::StackShifts(shifts)) => {
                    for shift in shifts {
                        if let Some(next) = apply_stack_shift(&stack, shift.pop, &shift.pushes) {
                            pending.push_back(next);
                        }
                    }
                }
                Some(Action::GuardedStackShifts(shifts)) => {
                    for shift in shifts {
                        if shift.guards.iter().all(|guard| guard_matches(&stack, guard))
                            && let Some(next) =
                                apply_stack_shift(&stack, shift.pop, &shift.pushes)
                        {
                            pending.push_back(next);
                        }
                    }
                }
                Some(Action::Reduce(nonterminal, len)) => {
                    let Some(mut reduced) = apply_stack_shift(&stack, *len, &[]) else {
                        continue;
                    };
                    let Some(&source) = reduced.last() else {
                        continue;
                    };
                    let Some(&(target, replace)) = table.goto[source as usize].get(nonterminal)
                    else {
                        continue;
                    };
                    if replace {
                        reduced.pop();
                    }
                    reduced.push(target);
                    pending.push_back(reduced);
                }
                Some(Action::Split { shift, reduces, .. }) => {
                    if let Some((target, replace)) = shift
                        && let Some(next) = apply_stack_shift(
                            &stack,
                            u32::from(*replace),
                            std::slice::from_ref(target),
                        )
                    {
                        pending.push_back(next);
                    }
                    for &(nonterminal, len) in reduces {
                        let Some(mut reduced) = apply_stack_shift(&stack, len, &[]) else {
                            continue;
                        };
                        let Some(&source) = reduced.last() else {
                            continue;
                        };
                        let Some(&(target, replace)) =
                            table.goto[source as usize].get(&nonterminal)
                        else {
                            continue;
                        };
                        if replace {
                            reduced.pop();
                        }
                        reduced.push(target);
                        pending.push_back(reduced);
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn terminal(analyzed: &AnalyzedGrammar, name: &str) -> TerminalID {
        analyzed
            .terminal_display_names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap() as TerminalID
    }

    fn enumerate_words(
        alphabet: &[TerminalID],
        max_len: usize,
        mut visit: impl FnMut(&[TerminalID]),
    ) {
        fn rec(
            alphabet: &[TerminalID],
            remaining: usize,
            word: &mut Vec<TerminalID>,
            visit: &mut impl FnMut(&[TerminalID]),
        ) {
            visit(word);
            if remaining == 0 {
                return;
            }
            for &terminal in alphabet {
                word.push(terminal);
                rec(alphabet, remaining - 1, word, visit);
                word.pop();
            }
        }
        rec(alphabet, max_len, &mut Vec::new(), &mut visit);
    }

    #[test]
    fn composed_table_matches_monolithic_language_with_two_call_sites() {
        let (child, child_analysis) = table(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" SUB "!";
            "#,
        );
        let (monolithic, monolithic_analysis) = table(
            r#"
                start document;
                g inner ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "<" inner ">" inner "!";
            "#,
        );

        let placeholder = terminal(&parent_analysis, "SUB");
        let composed = compose_subgrammar_tables(
            &parent,
            None,
            &[SubgrammarTableInput {
                placeholder_terminal: placeholder,
                table: &child,
                ignore_terminal: None,
                start_nullable: false,
            }],
        )
        .unwrap();
        assert_eq!(composed.state_relations.len(), 2);
        assert_eq!(composed.state_relations[1][0].len(), 2);

        let composed_alphabet = [
            terminal(&parent_analysis, "<"),
            composed.terminal_offsets[1] + terminal(&child_analysis, "a"),
            composed.terminal_offsets[1] + terminal(&child_analysis, "b"),
            terminal(&parent_analysis, ">"),
            terminal(&parent_analysis, "!"),
        ];
        let monolithic_alphabet = [
            terminal(&monolithic_analysis, "<"),
            terminal(&monolithic_analysis, "a"),
            terminal(&monolithic_analysis, "b"),
            terminal(&monolithic_analysis, ">"),
            terminal(&monolithic_analysis, "!"),
        ];
        enumerate_words(&composed_alphabet, 7, |composed_word| {
            let monolithic_word = composed_word
                .iter()
                .map(|terminal| {
                    let index = composed_alphabet
                        .iter()
                        .position(|candidate| candidate == terminal)
                        .unwrap();
                    monolithic_alphabet[index]
                })
                .collect::<Vec<_>>();
            assert_eq!(
                accepts(&composed.table, composed_word),
                accepts(&monolithic, &monolithic_word),
                "language mismatch for alphabet indexes {:?}",
                composed_word
                    .iter()
                    .map(|terminal| composed_alphabet.iter().position(|x| x == terminal).unwrap())
                    .collect::<Vec<_>>(),
            );
        });
    }

    #[test]
    fn explicit_control_table_matches_optimized_splice_without_scoped_ignore() {
        let (child, child_analysis) = table(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" SUB "!";
            "#,
        );
        let input = SubgrammarTableInput {
            placeholder_terminal: terminal(&parent_analysis, "SUB"),
            table: &child,
            ignore_terminal: None,
            start_nullable: false,
        };
        let optimized = compose_subgrammar_tables(&parent, None, std::slice::from_ref(&input)).unwrap();
        let explicit =
            compose_subgrammar_tables_explicit(&parent, None, std::slice::from_ref(&input))
                .unwrap();
        assert!(!explicit.control_terminals.is_empty());
        assert!(explicit.table.num_states > optimized.table.num_states);

        let child_offset = explicit.terminal_offsets[1];
        let alphabet = [
            terminal(&parent_analysis, "<"),
            terminal(&parent_analysis, ">"),
            terminal(&parent_analysis, "!"),
            child_offset + terminal(&child_analysis, "a"),
            child_offset + terminal(&child_analysis, "b"),
        ];
        enumerate_words(&alphabet, 6, |word| {
            assert_eq!(
                accepts(&explicit.table, word),
                accepts(&optimized.table, word),
                "explicit control table differs from optimized splice for {word:?}",
            );
        });
    }

    #[test]

    fn explicit_control_table_completes_nullable_child_before_continuation() {
        let (child, child_analysis) = table(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                t DONE ::= @token(1000);
                nt document ::= SUB DONE;
            "#,
        );
        let input = SubgrammarTableInput {
            placeholder_terminal: terminal(&parent_analysis, "SUB"),
            table: &child,
            ignore_terminal: None,
            start_nullable: true,
        };
        let optimized = compose_subgrammar_tables(&parent, None, std::slice::from_ref(&input)).unwrap();
        let explicit =
            compose_subgrammar_tables_explicit(&parent, None, std::slice::from_ref(&input))
                .unwrap();

        let done = terminal(&parent_analysis, "DONE");
        let child_a = explicit.terminal_offsets[1] + terminal(&child_analysis, "a");
        for word in [vec![done], vec![child_a, done]] {
            assert!(
                accepts(&optimized.table, &word),
                "optimized nullable composition rejected {word:?}",
            );
            assert!(
                accepts(&explicit.table, &word),
                "explicit nullable composition rejected {word:?}",
            );
        }
    }

    #[test]
    fn compiled_control_actions_match_explicit_adjacent_child_reference() {
        let (child, child_analysis) = table(
            r#"
                start child;
                nt child ::= "a";
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB SUB "!";
            "#,
        );
        let input = SubgrammarTableInput {
            placeholder_terminal: terminal(&parent_analysis, "SUB"),
            table: &child,
            ignore_terminal: None,
            start_nullable: false,
        };
        let explicit =
            compose_subgrammar_tables_explicit(&parent, None, std::slice::from_ref(&input))
                .unwrap()
                .table;
        let mut compiled = explicit.clone();
        let report = compiled.eliminate_control_terminals_exact().unwrap();
        assert!(compiled.control_terminals.is_empty());
        assert!(report.compiled_cells > 0);

        let alphabet = [
            terminal(&parent_analysis, "X"),
            explicit.num_terminals - child.num_terminals + terminal(&child_analysis, "a"),
            terminal(&parent_analysis, "!"),
        ];
        enumerate_words(&alphabet, 5, |word| {
            assert_eq!(
                accepts(&compiled, word),
                accepts(&explicit, word),
                "control elimination changed adjacent-child language for {word:?}",
            );
        });
    }

    #[test]
    fn compiled_control_actions_match_nullable_special_continuation_reference() {
        let (child, child_analysis) = table(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                t DONE ::= @token(1000);
                nt document ::= SUB DONE;
            "#,
        );
        let input = SubgrammarTableInput {
            placeholder_terminal: terminal(&parent_analysis, "SUB"),
            table: &child,
            ignore_terminal: None,
            start_nullable: true,
        };
        let explicit =
            compose_subgrammar_tables_explicit(&parent, None, std::slice::from_ref(&input))
                .unwrap()
                .table;
        let mut compiled = explicit.clone();
        compiled.eliminate_control_terminals_exact().unwrap();
        assert!(compiled.control_terminals.is_empty());

        let alphabet = [
            explicit.num_terminals - child.num_terminals + terminal(&child_analysis, "a"),
            terminal(&parent_analysis, "DONE"),
        ];
        enumerate_words(&alphabet, 3, |word| {
            assert_eq!(
                accepts(&compiled, word),
                accepts(&explicit, word),
                "control elimination changed nullable-special language for {word:?}",
            );
        });
    }

    #[test]
    fn compiled_control_actions_preserve_scoped_ignore_ownership() {
        let (child, child_analysis) = table(
            r#"
                start child;
                t C_WS ::= "\t"+;
                t A ::= "a";
                nt child ::= A;
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                t P_WS ::= " "+;
                t L ::= "<";
                t R ::= ">";
                nt document ::= L SUB R;
            "#,
        );
        let parent_ws = terminal(&parent_analysis, "P_WS");
        let child_ws = terminal(&child_analysis, "C_WS");
        let input = SubgrammarTableInput {
            placeholder_terminal: terminal(&parent_analysis, "SUB"),
            table: &child,
            ignore_terminal: Some(child_ws),
            start_nullable: false,
        };
        let composed = compose_subgrammar_tables_explicit(
            &parent,
            Some(parent_ws),
            std::slice::from_ref(&input),
        )
        .unwrap();
        let explicit = composed.table;
        let child_offset = composed.terminal_offsets[1];
        let mut compiled = explicit.clone();
        compiled.eliminate_control_terminals_exact().unwrap();

        let alphabet = [
            terminal(&parent_analysis, "L"),
            terminal(&parent_analysis, "R"),
            parent_ws,
            child_offset + child_ws,
            child_offset + terminal(&child_analysis, "A"),
        ];
        enumerate_words(&alphabet, 5, |word| {
            assert_eq!(
                accepts(&compiled, word),
                accepts(&explicit, word),
                "control elimination changed scoped-ignore language for {word:?}",
            );
        });
    }

    #[test]
    fn compiled_control_actions_preserve_nested_control_chains() {
        let (leaf, leaf_analysis) = table(
            r#"
                start leaf;
                nt leaf ::= "a";
            "#,
        );
        let (middle_parent, middle_analysis) = table(
            r#"
                start child;
                t LEAF ::= @token(998);
                nt child ::= LEAF LEAF;
            "#,
        );
        let middle_input = SubgrammarTableInput {
            placeholder_terminal: terminal(&middle_analysis, "LEAF"),
            table: &leaf,
            ignore_terminal: None,
            start_nullable: false,
        };
        let middle = compose_subgrammar_tables_explicit(
            &middle_parent,
            None,
            std::slice::from_ref(&middle_input),
        )
        .unwrap();
        let middle_leaf_a = middle.terminal_offsets[1] + terminal(&leaf_analysis, "a");

        let (outer_parent, outer_analysis) = table(
            r#"
                start document;
                t CHILD ::= @token(999);
                t DONE ::= @token(1000);
                nt document ::= CHILD DONE;
            "#,
        );
        let outer_input = SubgrammarTableInput {
            placeholder_terminal: terminal(&outer_analysis, "CHILD"),
            table: &middle.table,
            ignore_terminal: None,
            start_nullable: false,
        };
        let outer = compose_subgrammar_tables_explicit(
            &outer_parent,
            None,
            std::slice::from_ref(&outer_input),
        )
        .unwrap();
        let outer_middle_offset = outer.terminal_offsets[1];
        let explicit = outer.table;
        assert!(explicit.control_terminals.len() >= 2);
        let mut compiled = explicit.clone();
        compiled.eliminate_control_terminals_exact().unwrap();
        assert_eq!(compiled.unconditional_advance.len(), compiled.num_states as usize);
        let mut rebuilt = compiled.clone();
        rebuilt.rebuild_unconditional_advance_rows();
        assert_eq!(compiled.unconditional_advance, rebuilt.unconditional_advance);

        let alphabet = [
            outer_middle_offset + middle_leaf_a,
            terminal(&outer_analysis, "DONE"),
        ];
        enumerate_words(&alphabet, 4, |word| {
            assert_eq!(
                accepts(&compiled, word),
                accepts(&explicit, word),
                "control elimination changed nested-control language for {word:?}",
            );
        });
    }

    #[test]

    fn optimized_splice_matches_explicit_for_adjacent_calls_to_same_child() {
        let (child, child_analysis) = table(
            r#"
                start child;
                nt child ::= "a";
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB SUB "!";
            "#,
        );
        let input = SubgrammarTableInput {
            placeholder_terminal: terminal(&parent_analysis, "SUB"),
            table: &child,
            ignore_terminal: None,
            start_nullable: false,
        };
        let optimized = compose_subgrammar_tables(&parent, None, std::slice::from_ref(&input)).unwrap();
        let explicit =
            compose_subgrammar_tables_explicit(&parent, None, std::slice::from_ref(&input))
                .unwrap();
        let child_offset = explicit.terminal_offsets[1];
        let alphabet = [
            terminal(&parent_analysis, "X"),
            child_offset + terminal(&child_analysis, "a"),
            terminal(&parent_analysis, "!"),
        ];
        enumerate_words(&alphabet, 5, |word| {
            assert_eq!(
                accepts(&optimized.table, word),
                accepts(&explicit.table, word),
                "direct splice differs from explicit control semantics for adjacent calls: {word:?}",
            );
        });
    }

    #[test]
    fn explicit_control_table_scopes_parent_and_child_ignore_at_boundaries() {
        let (child, child_analysis) = table(
            r#"
                start child;
                t C_WS ::= "\t"+;
                t A ::= "a";
                nt child ::= A;
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                t P_WS ::= " "+;
                t L ::= "<";
                t R ::= ">";
                nt document ::= L SUB R;
            "#,
        );
        let parent_ws = terminal(&parent_analysis, "P_WS");
        let child_ws = terminal(&child_analysis, "C_WS");
        let input = SubgrammarTableInput {
            placeholder_terminal: terminal(&parent_analysis, "SUB"),
            table: &child,
            ignore_terminal: Some(child_ws),
            start_nullable: false,
        };
        let composed = compose_subgrammar_tables_explicit(
            &parent,
            Some(parent_ws),
            std::slice::from_ref(&input),
        )
        .unwrap();
        let direct = compose_subgrammar_tables(
            &parent,
            Some(parent_ws),
            std::slice::from_ref(&input),
        )
        .unwrap();
        let child_offset = composed.terminal_offsets[1];
        let child_ws = child_offset + child_ws;
        let a = child_offset + terminal(&child_analysis, "A");
        let l = terminal(&parent_analysis, "L");
        let r = terminal(&parent_analysis, "R");

        assert_eq!(composed.table.skip_terminals, BTreeSet::from([parent_ws, child_ws]));
        assert!(composed.table.action.iter().all(|row| {
            row.iter().all(|(terminal, action)| {
                !composed.table.skip_terminals.contains(&terminal)
                    || matches!(action, Action::Skip)
            })
        }));

        assert!(accepts(
            &composed.table,
            &[parent_ws, l, parent_ws, child_ws, a, child_ws, parent_ws, r, parent_ws],
        ));
        assert!(accepts(&composed.table, &[l, child_ws, a, child_ws, r]));

        assert!(!accepts(&composed.table, &[child_ws, l, a, r]));
        assert!(!accepts(&composed.table, &[l, child_ws, parent_ws, a, r]));
        assert!(!accepts(&composed.table, &[l, a, parent_ws, child_ws, r]));

        let alphabet = [l, r, parent_ws, child_ws, a];
        enumerate_words(&alphabet, 5, |word| {
            assert_eq!(
                accepts(&direct.table, word),
                accepts(&composed.table, word),
                "direct scoped-ignore splice differs from explicit semantics for {word:?}",
            );
        });
    }

    #[test]
    fn composed_table_matches_monolithic_with_nullable_child() {
        let (child, child_analysis) = table(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
        );
        let (monolithic, monolithic_analysis) = table(
            r#"
                start document;
                nt item ::= "a";
                nt document ::= "X" item? "!";
            "#,
        );
        let composed = compose_subgrammar_tables(
            &parent,
            None,
            &[SubgrammarTableInput {
                placeholder_terminal: terminal(&parent_analysis, "SUB"),
                table: &child,
                ignore_terminal: None,
                start_nullable: true,
            }],
        )
        .unwrap();

        let composed_x = terminal(&parent_analysis, "X");
        let composed_a = composed.terminal_offsets[1] + terminal(&child_analysis, "a");
        let composed_bang = terminal(&parent_analysis, "!");
        let monolithic_x = terminal(&monolithic_analysis, "X");
        let monolithic_a = terminal(&monolithic_analysis, "a");
        let monolithic_bang = terminal(&monolithic_analysis, "!");
        for (composed_word, monolithic_word) in [
            (vec![composed_x, composed_bang], vec![monolithic_x, monolithic_bang]),
            (
                vec![composed_x, composed_a, composed_bang],
                vec![monolithic_x, monolithic_a, monolithic_bang],
            ),
        ] {
            assert_eq!(
                accepts(&composed.table, &composed_word),
                accepts(&monolithic, &monolithic_word),
                "nullable language mismatch for {composed_word:?}",
            );
            assert!(accepts(&monolithic, &monolithic_word));
        }
    }

    #[test]
    fn composed_table_matches_monolithic_with_two_distinct_children() {
        let (left, left_analysis) = table(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
        );
        let (right, right_analysis) = table(
            r#"
                start child;
                nt child ::= "c" "d";
            "#,
        );
        let (parent, parent_analysis) = table(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "<" LEFT ">,<" RIGHT ">" "!";
            "#,
        );
        let (monolithic, monolithic_analysis) = table(
            r#"
                start document;
                g left ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                g right ::= {
                    start child;
                    nt child ::= "c" "d";
                };
                nt document ::= "<" left ">,<" right ">" "!";
            "#,
        );
        let composed = compose_subgrammar_tables(
            &parent,
            None,
            &[
                SubgrammarTableInput {
                    placeholder_terminal: terminal(&parent_analysis, "LEFT"),
                    table: &left,
                    ignore_terminal: None,
                    start_nullable: false,
                },
                SubgrammarTableInput {
                    placeholder_terminal: terminal(&parent_analysis, "RIGHT"),
                    table: &right,
                    ignore_terminal: None,
                    start_nullable: false,
                },
            ],
        )
        .unwrap();
        let composed_word = [
            terminal(&parent_analysis, "<"),
            composed.terminal_offsets[1] + terminal(&left_analysis, "a"),
            composed.terminal_offsets[1] + terminal(&left_analysis, "b"),
            terminal(&parent_analysis, ">,<"),
            composed.terminal_offsets[2] + terminal(&right_analysis, "c"),
            composed.terminal_offsets[2] + terminal(&right_analysis, "d"),
            terminal(&parent_analysis, ">"),
            terminal(&parent_analysis, "!"),
        ];
        let monolithic_word = [
            terminal(&monolithic_analysis, "<"),
            terminal(&monolithic_analysis, "a"),
            terminal(&monolithic_analysis, "b"),
            terminal(&monolithic_analysis, ">,<"),
            terminal(&monolithic_analysis, "c"),
            terminal(&monolithic_analysis, "d"),
            terminal(&monolithic_analysis, ">"),
            terminal(&monolithic_analysis, "!"),
        ];
        assert!(accepts(&monolithic, &monolithic_word));
        assert!(
            accepts(&composed.table, &composed_word),
            "composed table rejected the valid two-child word",
        );
        let mut invalid = composed_word;
        invalid[5] = composed.terminal_offsets[1] + terminal(&left_analysis, "b");
        assert!(!accepts(&composed.table, &invalid));
    }
}
