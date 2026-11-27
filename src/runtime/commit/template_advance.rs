use crate::compiler::glr::labels::{
    DEFAULT_LABEL,
    is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::glr::parser::ParserGSS;
use crate::grammar::flat::TerminalID;
use crate::runtime::{CommitTemplateDfas, FastCommitTemplateDfas};
use crate::runtime::constraint::Constraint;
use crate::runtime::parser_backend::PreparedRelation;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

pub(crate) fn advance_stacks_template_dfa(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: TerminalID,
) -> Option<ParserGSS> {
    if let Some(provider) = constraint.template_composition_provider() {
        return Some(crate::compiler::glr::parser::advance_provider_control_closed_stacks(
            &provider, stack, terminal));
    }
    let relation = constraint.template_parser()?.local_relation(terminal as usize)?;
    let output = advance_prepared_relation(relation, stack);
    debug_validate_template_output(&relation.source, stack, &output, terminal);
    Some(output)
}

/// Shared borrowed relation dispatch for ordinary and scoped composition
/// advances. The prepared shift is an exact specialization of the same finite
/// program, not an LR shortcut. Try it before cloning the interpreter input.
#[inline]
pub(crate) fn advance_prepared_relation(
    relation: &PreparedRelation,
    stack: &ParserGSS,
) -> ParserGSS {
    let prepared = relation.fast.as_ref();
    if let Some(output) = prepared.and_then(|fast|
        super::simple_read_shift::apply(fast, stack))
    {
        #[cfg(test)]
        PREPARED_RELATION_SHORTCUTS.with(|count| count.set(count.get() + 1));
        output
    } else {
        advance_with_prepared_template(&relation.source, stack.clone(), prepared)
    }
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static PREPARED_RELATION_SHORTCUTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn advance_stacks_template_dfa_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: TerminalID,
) -> Option<ParserGSS> {
    if let Some(provider) = constraint.template_composition_provider() {
        return Some(crate::compiler::glr::parser::advance_provider_control_closed_stacks(
            &provider, &stack, terminal));
    }
    let relation = constraint.template_parser()?.local_relation(terminal as usize)?;
    let input = debug_template_literal_enabled().then(|| stack.clone());
    let fast = relation.fast.as_ref();
    let output = match fast.and_then(|fast| super::simple_read_shift::apply(fast, &stack)) {
        Some(output) => output,
        None => advance_with_prepared_template(&relation.source, stack, fast),
    };
    if let Some(input) = input { debug_validate_template_output(&relation.source, &input, &output, terminal); }
    Some(output)
}

// Opt-in independent interpreter used only to diagnose relation/GSS errors.
// It never references an LR table and is excluded from timed runs.
fn debug_template_literal_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| crate::env::var_os("GLRMASK_DIAG_TEMPLATE_LITERAL").is_some())
}

fn debug_validate_template_output(
    template: &CommitTemplateDfas, input: &ParserGSS, output: &ParserGSS, terminal: u32,
) {
    if !debug_template_literal_enabled() { return; }
    let Some(inputs) = input.to_stacks(4_096) else { return; };
    let mut outputs = Vec::new();
    let mut operations = 0usize;
    for (stack, accumulator) in &inputs {
        let mut work = vec![(Phase::Pop, template.pop.start_state, stack.clone())];
        let mut visited = rustc_hash::FxHashSet::default();
        while let Some((phase, id, stack)) = work.pop() {
            operations += 1;
            if operations > 1_000_000 { return; }
            if !visited.insert((phase, id, stack.clone())) { continue; }
            let state = match phase {
                Phase::Pop => &template.pop.states[id as usize],
                Phase::Read => &template.read.states[id as usize],
                Phase::Push => &template.push.states[id as usize],
            };
            if state.is_accepting { outputs.push((stack.clone(), accumulator.clone())); }
            match phase {
                Phase::Pop => {
                    if let Some(&top) = stack.last()
                        && let Some(&next) = state.transitions.get(&(top as i32))
                            .or_else(|| state.transitions.get(&DEFAULT_LABEL)) {
                            let mut popped = stack.clone(); popped.pop();
                            work.push((Phase::Pop, next, popped));
                        }
                    if let Some(next) = template.pop_to_read.get(id as usize).copied().flatten() {
                        work.push((Phase::Read, next, stack.clone()));
                    }
                    if let Some(next) = template.pop_to_push.get(id as usize).copied().flatten() {
                        work.push((Phase::Push, next, stack));
                    }
                }
                Phase::Read => {
                    if let Some(top) = stack.last()
                        && let Some(&next) = state.transitions.get(&(*top as i32)) {
                            work.push((Phase::Read, next, stack.clone()));
                        }
                    if let Some(next) = template.read_to_push.get(id as usize).copied().flatten() {
                        work.push((Phase::Push, next, stack));
                    }
                }
                Phase::Push => {
                    for (&label, &next) in &state.transitions {
                        let mut pushed = stack.clone();
                        pushed.push(negative_to_positive_label(label) as u32);
                        work.push((Phase::Push, next, pushed));
                    }
                }
            }
        }
    }
    if terminal == 1 && crate::env::var_os("GLRMASK_DIAG_TEMPLATE_TRACE").is_some() {
        use std::io::Write;
        let path = crate::env::var("GLRMASK_DIAG_TEMPLATE_TRACE").unwrap();
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        let record = serde_json::json!({
            "terminal":terminal,
            "input":inputs.iter().map(|(stack,_)|stack).collect::<Vec<_>>(),
            "output":outputs.iter().map(|(stack,_)|stack).collect::<Vec<_>>(),
            "accumulators":format!("{inputs:?}"),
        });
        writeln!(file, "{record}").unwrap();
    }
    let expected = ParserGSS::from_stacks(&outputs);
    if output.semantically_eq(&expected, 65_536) != Some(true) {
        let fixture = serde_json::json!({
            "terminal":terminal,"template":template,
            "input_stacks":inputs.iter().map(|(stack,_)|stack).collect::<Vec<_>>(),
            "literal_stacks":outputs.iter().map(|(stack,_)|stack).collect::<Vec<_>>(),
            "actual_stacks":output.to_stacks(4096).map(|stacks|stacks.into_iter().map(|(stack,_)|stack).collect::<Vec<_>>()),
            "input_debug":format!("{input:?}"),
        });
        let path = crate::env::var("GLRMASK_DIAG_TEMPLATE_LITERAL_PATH")
            .unwrap_or_else(|_| "template-literal-difference.json".into());
        std::fs::write(path, serde_json::to_vec(&fixture).unwrap()).unwrap();
        panic!("template GSS output differs from independent literal interpretation for terminal {terminal}");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Phase {
    Pop,
    Read,
    Push,
}

#[cfg(test)]
fn advance_with_template(template: &CommitTemplateDfas, stack: ParserGSS) -> ParserGSS {
    advance_with_prepared_template(template, stack, None)
}

pub(crate) fn advance_with_prepared_template(template: &CommitTemplateDfas, stack: ParserGSS,
    prepared: Option<&FastCommitTemplateDfas>) -> ParserGSS {
    advance_relation(template, stack, prepared, None)
}

/// A scope changes only the concrete stack alphabet. Stored stack frames
/// outside the local alphabet remain untouched; DEFAULT never crosses it.
pub(crate) fn advance_with_template_coordinate(
    template: &CommitTemplateDfas, stack: ParserGSS, scope: Option<(u32, u32)>,
) -> ParserGSS {
    advance_relation(template, stack, None, scope)
}

#[inline]
pub(super) fn local_symbol(scope: Option<(u32, u32)>, top: u32) -> Option<u32> {
    match scope {
        Some((offset, count)) => top.checked_sub(offset).filter(|&symbol| symbol < count),
        None => Some(top),
    }
}

#[inline]
pub(super) fn global_symbol(scope: Option<(u32, u32)>, local: u32) -> u32 {
    scope.map_or(local, |(offset, _)| offset + local)
}

pub(crate) fn advance_with_prepared_coordinate(
    template: &CommitTemplateDfas, stack: ParserGSS,
    prepared: Option<&FastCommitTemplateDfas>, scope: (u32, u32),
) -> ParserGSS {
    advance_relation(template, stack, prepared, Some(scope))
}

fn advance_relation(
    template: &CommitTemplateDfas, stack: ParserGSS,
    prepared: Option<&FastCommitTemplateDfas>, scope: Option<(u32, u32)>,
) -> ParserGSS {
    let global = |symbol| global_symbol(scope, symbol);
    let local = |top| local_symbol(scope, top);
    // Uniform inputs permit exact union reassociation in either coordinate.
    // The unprepared scoped reference retains its ordered traversal.
    let sparse_input = (scope.is_none() || prepared.is_some())
        && stack.single_interface_lower_id().is_some();
    let seeded = prepared.and_then(|index| index.phase_dag.as_ref())
        .and_then(|cursor| cursor.apply_input_cursor(template, &stack, scope));
    let (mut output, mut worklist) = match seeded {
        Some(result) => {
            let mut worklist = SmallVec::<[(Phase, u32, ParserGSS); 8]>::new();
            for (phase, state, input) in result.residual.into_iter().rev() {
                let phase = match phase { 0 => Phase::Pop, 1 => Phase::Read, _ => Phase::Push };
                worklist.push((phase, state, input));
            }
            (result.output, worklist)
        }
        None => {
            let mut worklist = SmallVec::<[(Phase, u32, ParserGSS); 8]>::new();
            worklist.push((Phase::Pop, template.pop.start_state, stack));
            (ParserGSS::empty(), worklist)
        }
    };
    // Retain every visited source GSS. Raw pointer keys alone are not safe:
    // temporary isolate/push results can be dropped and their addresses reused
    // later in the same evaluation.
    let mut visited = FxHashMap::<(Phase, u32, usize), ParserGSS>::default();

    while let Some((phase, state_id, gss)) = worklist.pop() {
        if gss.is_empty() {
            continue;
        }
        // A LIFO path walker is cheapest for ordinary sparse actions, but
        // repeated POPs through a shared ambiguous stack must not expand its
        // concrete paths. Switch the exact *remaining work*, not the original
        // input, to state-level dataflow once the inline frontier grows. This
        // adds no retained input clone or second GSS representation to the
        // normal path. Uniform annotations permit union reassociation.
        if sparse_input && worklist.len() >= 8
            && let Some(plan) = prepared.and_then(|p| p.phase_dag.as_ref())
        {
            worklist.push((phase, state_id, gss));
            return plan.apply(template, worklist.drain(..), output, scope);
        }
        let visit_key = (phase, state_id, gss.ptr_key());
        if let Some(source) = visited.get(&visit_key) {
            debug_assert!(source.ptr_eq(&gss));
            continue;
        }
        visited.insert(visit_key, gss.clone());

        match phase {
            Phase::Pop => {
                let Some(dfa_state) = template.pop.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                if sparse_input {
                    // Intersect the automaton row with the actual GSS frontier,
                    // not with every stack symbol in the grammar. A JSON/JS
                    // template row can contain hundreds of labels while the live
                    // GSS typically exposes only one or two. DEFAULT still has
                    // lower priority than *every* explicit edge, including a
                    // rejecting edge retained as a wildcard exception.
                    if let Some(top) = gss.single_exclusive_top_value() {
                        if let Some(symbol) = local(top)
                            && let Some(&target) = dfa_state.transitions.get(&(symbol as i32))
                            .or_else(|| dfa_state.transitions.get(&DEFAULT_LABEL))
                        {
                            worklist.push((Phase::Pop, target, gss.popn(1)));
                        }
                    } else {
                        for top in gss.peek_values() {
                            if let Some(symbol) = local(top)
                            && let Some(&target) = dfa_state.transitions.get(&(symbol as i32))
                                .or_else(|| dfa_state.transitions.get(&DEFAULT_LABEL))
                            {
                                let branch = gss.pop_top_value(&top);
                                if !branch.is_empty() {
                                    worklist.push((Phase::Pop, target, branch));
                                }
                            }
                        }
                    }
                } else {
                    // Preserve the original BTreeMap row order for correlated
                    // annotations. Only live top labels can produce a branch;
                    // sorting them preserves the exact old merge order while
                    // avoiding scans of all absent grammar-state labels.
                    let tops = gss.peek_values();
                    let mut explicit_tops = tops.clone();
                    explicit_tops.sort_unstable();
                    for &top in &explicit_tops {
                        let Some(symbol) = local(top) else { continue; };
                        let label = symbol as i32;
                        if label == DEFAULT_LABEL || is_negative_label(label) { continue; }
                        if let Some(&target) = dfa_state.transitions.get(&label) {
                            let isolated = gss.isolate(Some(top));
                            if !isolated.is_empty() {
                                worklist.push((Phase::Pop, target, isolated.popn(1)));
                            }
                        }
                    }
                    if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                        // The old fallback visits remaining GSS top values in
                        // their native order, separately from explicit edges.
                        for top in tops {
                            let Some(symbol) = local(top) else { continue; };
                            if dfa_state.transitions.contains_key(&(symbol as i32)) { continue; }
                            let branch = gss.isolate(Some(top)).popn(1);
                            if !branch.is_empty() {
                                worklist.push((Phase::Pop, target, branch));
                            }
                        }
                    }
                }

                if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize) {
                    worklist.push((Phase::Read, *read_state, gss.clone()));
                }
                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Read => {
                let Some(dfa_state) = template.read.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                if sparse_input {
                    if let Some(top) = gss.single_exclusive_top_value() {
                        if let Some(symbol) = local(top)
                            && let Some(&target) = dfa_state.transitions.get(&(symbol as i32)) {
                            worklist.push((Phase::Read, target, gss.clone()));
                        }
                    } else {
                        for top in gss.peek_values() {
                            if let Some(symbol) = local(top)
                            && let Some(&target) = dfa_state.transitions.get(&(symbol as i32)) {
                                let branch = gss.isolate(Some(top));
                                if !branch.is_empty() {
                                    worklist.push((Phase::Read, target, branch));
                                }
                            }
                        }
                    }
                } else {
                    let mut tops = gss.peek_values();
                    tops.sort_unstable();
                    for top in tops {
                        let Some(symbol) = local(top) else { continue; };
                        let label = symbol as i32;
                        if is_negative_label(label) { continue; }
                        if let Some(&target) = dfa_state.transitions.get(&label) {
                            let isolated = gss.isolate(Some(top));
                            if !isolated.is_empty() {
                                worklist.push((Phase::Read, target, isolated));
                            }
                        }
                    }
                }

                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Push => {
                if sparse_input
                    && let Some(plan) = prepared.and_then(|p| p.push_suffixes.get(state_id as usize))
                    .and_then(Option::as_ref)
                    && let Some(outputs) = plan.apply(&gss, scope.map_or(0, |(offset, _)| offset))
                {
                    output = output.merge(&outputs);
                    continue;
                }
                if sparse_input
                    && let Some(plan) = prepared.and_then(|p| p.phase_dag.as_ref())
                    && let Some(outputs) = plan.apply_large_push(template, state_id, &gss, scope)
                {
                    output = output.merge(&outputs);
                    continue;
                }
                let Some(dfa_state) = template.push.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        panic!(
                            "commit template push DFA contains non-push label {label} at state {state_id}"
                        );
                    }
                    worklist.push((
                        Phase::Push,
                        target,
                        gss.push(global(negative_to_positive_label(label) as u32)),
                    ));
                }
            }
        }
    }

    output
}

#[cfg(test)]
fn advance_with_template_reference(template: &CommitTemplateDfas, stack: ParserGSS) -> ParserGSS {
    advance_with_template_coordinate_reference(template, stack, None)
}

/// The exact phase interpreter over a borrowed immutable program. A scoped
/// coordinate translates only inspected/written symbols; caller frames retain
/// their original values and annotations. DEFAULT is local at every POP.
#[cfg(test)]
fn advance_with_template_coordinate_reference(
    template: &CommitTemplateDfas, stack: ParserGSS, scope: Option<(u32, u32)>,
) -> ParserGSS {
    let global = |local: u32| scope.map_or(local, |(offset, _)| offset + local);
    let local = |top: u32| match scope {
        Some((offset, count)) => top.checked_sub(offset).filter(|&symbol| symbol < count),
        None => Some(top),
    };

    let mut output = ParserGSS::empty();
    let mut worklist = vec![(Phase::Pop, template.pop.start_state, stack)];
    // Retain every visited source GSS. Raw pointer keys alone are not safe:
    // temporary isolate/push results can be dropped and their addresses reused
    // later in the same evaluation.
    let mut visited = FxHashMap::<(Phase, u32, usize), ParserGSS>::default();

    while let Some((phase, state_id, gss)) = worklist.pop() {
        if gss.is_empty() {
            continue;
        }
        let visit_key = (phase, state_id, gss.ptr_key());
        if let Some(source) = visited.get(&visit_key) {
            debug_assert!(source.ptr_eq(&gss));
            continue;
        }
        visited.insert(visit_key, gss.clone());

        match phase {
            Phase::Pop => {
                let Some(dfa_state) = template.pop.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                // Match the ordinary interpreter's live-frontier lookup.
                // Translating sorted concrete tops preserves the original
                // explicit BTreeMap row order and correlated merge order.
                let tops = gss.peek_values();
                let mut explicit_tops = tops.clone();
                explicit_tops.sort_unstable();
                for top in explicit_tops {
                    let Some(symbol) = local(top) else { continue; };
                    if let Some(&target) = dfa_state.transitions.get(&(symbol as i32)) {
                        let branch = gss.isolate(Some(top)).popn(1);
                        if !branch.is_empty() { worklist.push((Phase::Pop, target, branch)); }
                    }
                }
                if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                    for top in tops {
                        let Some(symbol) = local(top) else { continue; };
                        if dfa_state.transitions.contains_key(&(symbol as i32)) {
                            continue;
                        }
                        let branch = gss.isolate(Some(top)).popn(1);
                        if !branch.is_empty() {
                            worklist.push((Phase::Pop, target, branch));
                        }
                    }
                }

                if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize) {
                    worklist.push((Phase::Read, *read_state, gss.clone()));
                }
                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Read => {
                let Some(dfa_state) = template.read.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                let mut tops = gss.peek_values();
                tops.sort_unstable();
                for top in tops {
                    let Some(symbol) = local(top) else { continue; };
                    if let Some(&target) = dfa_state.transitions.get(&(symbol as i32)) {
                        let branch = gss.isolate(Some(top));
                        if !branch.is_empty() { worklist.push((Phase::Read, target, branch)); }
                    }
                }

                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Push => {
                let Some(dfa_state) = template.push.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        panic!(
                            "commit template push DFA contains non-push label {label} at state {state_id}"
                        );
                    }
                    worklist.push((
                        Phase::Push,
                        target,
                        gss.push(global(negative_to_positive_label(label) as u32)),
                    ));
                }
            }
        }
    }

    output
}

/// Apply a deterministic single-stack commit template to preallocated flat
/// stack scratch. `Some(true)` is an accepting result, `Some(false)` is an
/// empty result, and `None` means the template branches or scratch would grow.
#[cfg(test)]
mod tests {
    use super::advance_with_template;
    use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::parser::ParserGSS;
    use crate::runtime::CommitTemplateDfas;

    #[test]
    fn sparse_frontier_walk_matches_row_scanning_on_generated_dag_programs() {
        use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label};
        let mut seed = 608894u64;
        let mut next = || {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed
        };
        for case in 0..128 {
            let mut pop = UnweightedDfa::new();
            let mut read = UnweightedDfa::new();
            let mut push = UnweightedDfa::new();
            for _ in 1..5 { pop.add_state(); read.add_state(); push.add_state(); }
            for from in 0..5u32 {
                pop.set_accepting(from, next() % 5 == 0);
                read.set_accepting(from, next() % 5 == 0);
                push.set_accepting(from, next() % 3 == 0);
                if from == 4 { continue; }
                for label in 0..7u32 {
                    if next() % 3 == 0 { pop.add_transition(from, label as i32, from + 1 + (next() % (4-from) as u64) as u32); }
                    if next() % 3 == 0 { read.add_transition(from, label as i32, from + 1 + (next() % (4-from) as u64) as u32); }
                    if next() % 5 == 0 { push.add_transition(from, encode_negative_label(label), from + 1 + (next() % (4-from) as u64) as u32); }
                }
                if next() % 2 == 0 { pop.add_transition(from, DEFAULT_LABEL, from + 1); }
            }
            let mut link = || (0..5).map(|_| if next()%3==0 { Some((next()%5) as u32) } else { None }).collect();
            let template = CommitTemplateDfas { pop, read, push,
                pop_to_read:link(), pop_to_push:link(), read_to_push:link() };
            let clean = TerminalsDisallowed::new();
            let guarded = clean.with_insert(0, 5);
            let paths = [(vec![], clean.clone()), (vec![0], clean.clone()),
                (vec![0,1,4], clean.clone()), (vec![0,2,4], guarded.clone()),
                (vec![0,6,5], guarded.clone()), (vec![3,1], clean.clone())];
            for group in [&paths[..1], &paths[1..2], &paths[2..4], &paths[1..], &paths[..]] {
                let source = ParserGSS::from_stacks(group);
                let expected = super::advance_with_template_reference(&template, source.clone());
                let fast = crate::runtime::FastCommitTemplateDfas::from_template(&template);
                let accelerated = super::advance_with_prepared_template(&template, source.clone(), Some(&fast));
                let actual = advance_with_template(&template, source);
                assert_eq!(accelerated.semantically_eq(&expected, 16_384), Some(true),
                    "prepared PUSH differs, case={case}, paths={group:?}");
                assert_eq!(actual.semantically_eq(&expected, 16_384), Some(true),
                    "sparse frontier relation differs, case={case}, paths={group:?}");
            }
        }
    }

    #[test]
    fn template_advance_distributes_over_merged_branched_floor() {
        let mut pop = UnweightedDfa::new();
        let after_common = pop.add_state();
        let after_left = pop.add_state();
        let after_right = pop.add_state();
        pop.add_transition(pop.start_state, 10, after_common);
        pop.add_transition(after_common, 1, after_left);
        pop.add_transition(after_common, 2, after_right);
        pop.set_accepting(after_left, true);
        pop.set_accepting(after_right, true);

        let template = CommitTemplateDfas {
            pop,
            read: UnweightedDfa::default(),
            push: UnweightedDfa::default(),
            pop_to_read: vec![None; 4],
            pop_to_push: vec![None; 4],
            read_to_push: Vec::new(),
        };
        let acc = TerminalsDisallowed::new();
        let left = ParserGSS::from_single_stack(vec![0, 1, 10], acc.clone());
        let right = ParserGSS::from_single_stack(vec![0, 2, 10], acc);
        let merged = left.merge(&right);

        let expected = advance_with_template(&template, left)
            .merge(&advance_with_template(&template, right));
        let actual = advance_with_template(&template, merged);

        let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(actual_stacks, expected_stacks);
    }


    #[test]
    fn template_walker_retains_temporary_gss_identity_across_pop_read_and_push() {
        let mut pop = UnweightedDfa::new();
        let popped = pop.add_state();
        pop.add_transition(pop.start_state, 9, popped);

        let mut read = UnweightedDfa::new();
        let read_left = read.add_state();
        let read_right = read.add_state();
        read.add_transition(read.start_state, 1, read_left);
        read.add_transition(read.start_state, 2, read_right);

        let mut push = UnweightedDfa::new();
        let pushed_left = push.add_state();
        let pushed_right = push.add_state();
        push.add_transition(
            push.start_state,
            crate::compiler::glr::labels::encode_negative_label(20),
            pushed_left,
        );
        push.add_transition(
            push.start_state,
            crate::compiler::glr::labels::encode_negative_label(30),
            pushed_right,
        );
        push.set_accepting(pushed_left, true);
        push.set_accepting(pushed_right, true);

        let mut pop_to_read = vec![None; pop.states.len()];
        pop_to_read[popped as usize] = Some(read.start_state);
        let mut read_to_push = vec![None; read.states.len()];
        read_to_push[read_left as usize] = Some(push.start_state);
        read_to_push[read_right as usize] = Some(push.start_state);
        let template = CommitTemplateDfas {
            pop,
            read,
            push,
            pop_to_read,
            pop_to_push: vec![None; 2],
            read_to_push,
        };

        let acc_a = TerminalsDisallowed::new();
        let acc_b = TerminalsDisallowed::new().with_insert(0, 7);
        let input = ParserGSS::from_stacks(&[
            (vec![0, 1, 9], acc_a.clone()),
            (vec![0, 2, 9], acc_b.clone()),
        ]);
        let actual = advance_with_template(&template, input);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 1, 20], acc_a.clone()),
            (vec![0, 1, 30], acc_a),
            (vec![0, 2, 20], acc_b.clone()),
            (vec![0, 2, 30], acc_b),
        ]);
        assert!(
            actual
                .semantically_eq(&expected, 64)
                .expect("test output is explicitly bounded")
        );
    }

}
