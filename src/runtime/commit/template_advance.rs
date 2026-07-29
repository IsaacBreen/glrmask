use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::{
    DEFAULT_LABEL,
    is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::glr::parser::ParserGSS;
use crate::ds::leveled_gss::{GssSemanticKeyInterner, VirtualStack};
use crate::grammar::flat::TerminalID;
use crate::runtime::CommitTemplateDfas;
use crate::runtime::constraint::Constraint;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

pub(super) fn advance_stacks_template_dfa(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: TerminalID,
) -> Option<ParserGSS> {
    let dfa = constraint
        .template_dfas_by_terminal
        .get(terminal as usize)?
        .as_ref()?;
    Some(advance_with_template(dfa, stack.clone()))
}

pub(super) fn advance_stacks_template_dfa_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: TerminalID,
) -> Option<ParserGSS> {
    let dfa = constraint
        .template_dfas_by_terminal
        .get(terminal as usize)?
        .as_ref()?;
    Some(advance_with_template(dfa, stack))
}

pub(crate) struct TemplateAdvanceRuntime {
    interner: GssSemanticKeyInterner<u32, TerminalsDisallowed>,
    memo: FxHashMap<(u32, Phase, u32, u32), u32>,
    component_cache:
        FxHashMap<usize, (ParserGSS, Vec<(u32, TerminalsDisallowed)>)>,
    calls: u64,
    memo_hits: u64,
}

impl Default for TemplateAdvanceRuntime {
    fn default() -> Self {
        Self {
            interner: GssSemanticKeyInterner::new(),
            memo: FxHashMap::default(),
            component_cache: FxHashMap::default(),
            calls: 0,
            memo_hits: 0,
        }
    }
}

impl std::fmt::Debug for TemplateAdvanceRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemplateAdvanceRuntime")
            .field("memo", &self.memo.len())
            .field("component_cache", &self.component_cache.len())
            .field("calls", &self.calls)
            .field("memo_hits", &self.memo_hits)
            .finish_non_exhaustive()
    }
}

impl TemplateAdvanceRuntime {
    pub(crate) fn begin_commit(&mut self) {
        // The canonical language representation is intentionally token-local.
        // Rebuild it from the authoritative compact GSS for each selected token
        // rather than carrying a second semantic view across parser states.
        *self = Self::default();
    }

    pub(crate) fn reset_all(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn memo_summary(&self) -> (u64, u64, usize) {
        (self.calls, self.memo_hits, self.memo.len())
    }

    pub(crate) fn language_from_uniform_gss(
        &mut self,
        stack: &ParserGSS,
    ) -> Option<(u32, TerminalsDisallowed)> {
        let accumulator = stack.uniform_accumulator()?;
        Some((self.interner.key(stack), accumulator))
    }

    pub(crate) fn language_components_from_gss(
        &mut self,
        stack: &ParserGSS,
    ) -> Vec<(u32, TerminalsDisallowed)> {
        let ptr = stack.ptr_key();
        if let Some((_, cached)) = self.component_cache.get(&ptr) {
            return cached.clone();
        }

        let components = if let Some(component) = self.language_from_uniform_gss(stack) {
            vec![component]
        } else {
            stack
                .partition_by_accumulator()
                .into_iter()
                .map(|(paths, accumulator)| {
                    let restored = paths.apply(|_| accumulator.clone());
                    (self.interner.key(&restored), accumulator)
                })
                .collect()
        };
        self.register_components(stack, components.clone());
        components
    }

    pub(crate) fn register_components(
        &mut self,
        stack: &ParserGSS,
        components: Vec<(u32, TerminalsDisallowed)>,
    ) {
        self.component_cache
            .insert(stack.ptr_key(), (stack.clone(), components));
    }

    pub(crate) fn gss_from_language(
        &mut self,
        language: u32,
        accumulator: TerminalsDisallowed,
    ) -> ParserGSS {
        self.interner.gss_from_key(language, accumulator)
    }

    pub(crate) fn language_top_states(&self, language: u32) -> SmallVec<[u32; 8]> {
        self.interner
            .top_branches(language)
            .iter()
            .map(|(state, _)| *state)
            .collect()
    }

    pub(crate) fn union_languages(&mut self, left: u32, right: u32) -> u32 {
        self.interner.union_keys(left, right)
    }

    pub(crate) fn advance_language(
        &mut self,
        constraint: &Constraint,
        terminal: TerminalID,
        language: u32,
    ) -> Option<u32> {
        let template = constraint
            .template_dfas_by_terminal
            .get(terminal as usize)?
            .as_ref()?;
        Some(evaluate_template_language(
            template,
            terminal,
            Phase::Pop,
            template.pop.start_state,
            language,
            self,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Phase {
    Pop,
    Read,
    Push,
}

/// Homomorphic extension of one terminal's compiled stack transducer from
/// individual stacks to finite stack languages.
///
/// If `T_t(s)` is the table-equivalent template result for stack `s`, this
/// function computes `union_{s in L} T_t(s)` for the language ID `L`. Trie
/// branches are language union; target grouping applies the same DFA state to
/// the union of equal-target suffixes. Exactness follows by induction on the
/// acyclic product of template phase/state and canonical stack-trie node.
fn evaluate_template_language(
    template: &CommitTemplateDfas,
    terminal: TerminalID,
    phase: Phase,
    state_id: u32,
    language: u32,
    runtime: &mut TemplateAdvanceRuntime,
) -> u32 {
    if language == 0 {
        return 0;
    }
    runtime.calls += 1;
    let memo_key = (terminal, phase, state_id, language);
    if let Some(&cached) = runtime.memo.get(&memo_key) {
        runtime.memo_hits += 1;
        return cached;
    }

    let mut output = 0;

    fn merge_target(
        groups: &mut SmallVec<[(u32, u32); 8]>,
        target: u32,
        language: u32,
        runtime: &mut TemplateAdvanceRuntime,
    ) {
        if let Some((_, existing)) = groups
            .iter_mut()
            .find(|(candidate, _)| *candidate == target)
        {
            *existing = runtime.interner.union_keys(*existing, language);
        } else {
            groups.push((target, language));
        }
    }

    match phase {
        Phase::Pop => {
            let Some(dfa_state) = template.pop.states.get(state_id as usize) else {
                return 0;
            };
            if dfa_state.is_accepting {
                output = language;
            }

            let branches = runtime
                .interner
                .top_branches(language)
                .iter()
                .copied()
                .collect::<SmallVec<[(u32, u32); 8]>>();
            let default_target = dfa_state.transitions.get(&DEFAULT_LABEL).copied();
            let mut by_target = SmallVec::<[(u32, u32); 8]>::new();
            for (top, suffix) in branches {
                let target = dfa_state
                    .transitions
                    .get(&(top as i32))
                    .copied()
                    .or(default_target);
                if let Some(target) = target {
                    merge_target(&mut by_target, target, suffix, runtime);
                }
            }
            for (target, suffixes) in by_target {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Pop,
                    target,
                    suffixes,
                    runtime,
                );
                output = runtime.interner.union_keys(output, branch);
            }

            if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize) {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Read,
                    *read_state,
                    language,
                    runtime,
                );
                output = runtime.interner.union_keys(output, branch);
            }
            if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Push,
                    *push_state,
                    language,
                    runtime,
                );
                output = runtime.interner.union_keys(output, branch);
            }
        }
        Phase::Read => {
            let Some(dfa_state) = template.read.states.get(state_id as usize) else {
                return 0;
            };
            if dfa_state.is_accepting {
                output = language;
            }

            let branches = runtime
                .interner
                .top_branches(language)
                .iter()
                .copied()
                .collect::<SmallVec<[(u32, u32); 8]>>();
            let mut by_target = SmallVec::<[(u32, u32); 8]>::new();
            for (top, suffix) in branches {
                if let Some(&target) = dfa_state.transitions.get(&(top as i32)) {
                    let selected = runtime.interner.push_key(suffix, top);
                    merge_target(&mut by_target, target, selected, runtime);
                }
            }
            for (target, selected) in by_target {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Read,
                    target,
                    selected,
                    runtime,
                );
                output = runtime.interner.union_keys(output, branch);
            }

            if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Push,
                    *push_state,
                    language,
                    runtime,
                );
                output = runtime.interner.union_keys(output, branch);
            }
        }
        Phase::Push => {
            let Some(dfa_state) = template.push.states.get(state_id as usize) else {
                return 0;
            };
            if dfa_state.is_accepting {
                output = language;
            }
            for (&label, &target) in &dfa_state.transitions {
                if !is_negative_label(label) {
                    panic!(
                        "commit template push DFA contains non-push label {label} at state {state_id}"
                    );
                }
                let pushed = runtime
                    .interner
                    .push_key(language, negative_to_positive_label(label) as u32);
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Push,
                    target,
                    pushed,
                    runtime,
                );
                output = runtime.interner.union_keys(output, branch);
            }
        }
    }

    runtime.memo.insert(memo_key, output);
    output
}

fn advance_with_template(template: &CommitTemplateDfas, stack: ParserGSS) -> ParserGSS {

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

                for (&label, &target) in &dfa_state.transitions {
                    if is_negative_label(label) {
                        panic!(
                            "commit template pop DFA contains push label {label} at state {state_id}"
                        );
                    }
                    if label != DEFAULT_LABEL && label >= 0 {
                        let state = label as u32;
                        let branch = gss.isolate(Some(state)).popn(1);
                        if !branch.is_empty() {
                            worklist.push((Phase::Pop, target, branch));
                        }
                    }
                }
                if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                    for top in gss.peek_values() {
                        if dfa_state.transitions.contains_key(&(top as i32)) {
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

                for (&label, &target) in &dfa_state.transitions {
                    if label == DEFAULT_LABEL || is_negative_label(label) {
                        panic!(
                            "commit template read DFA contains non-read label {label} at state {state_id}"
                        );
                    }
                    let branch = gss.isolate(Some(label as u32));
                    if !branch.is_empty() {
                        worklist.push((Phase::Read, target, branch));
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
                        gss.push(negative_to_positive_label(label) as u32),
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
pub(super) fn advance_flat_stack_single_path(
    constraint: &Constraint,
    terminal: TerminalID,
    stack: &mut Vec<u32>,
) -> Option<bool> {
    let template = constraint
        .template_dfas_by_terminal
        .get(terminal as usize)?
        .as_ref()?;
    let mut phase = Phase::Pop;
    let mut state_id = template.pop.start_state;
    let total_states = template
        .pop
        .states
        .len()
        .saturating_add(template.read.states.len())
        .saturating_add(template.push.states.len());
    let max_steps = total_states.saturating_mul(2).saturating_add(8);
    let mut steps = 0usize;

    loop {
        let mut choice = None;
        let mut choices = 0usize;
        let accepting;
        match phase {
            Phase::Pop => {
                let dfa_state = template.pop.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;
                if let Some(&top) = stack.last() {
                    let label = top as i32;
                    if let Some(&target) = dfa_state.transitions.get(&label) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    } else if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    }
                    if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize)
                        && template
                            .read
                            .states
                            .get(*read_state as usize)
                            .is_some_and(|state| state.transitions.contains_key(&label))
                    {
                        choice = Some(SingleChoice::Read(*read_state));
                        choices += 1;
                    }
                }
                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Read => {
                let dfa_state = template.read.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;
                if let Some(&top) = stack.last() {
                    if let Some(&target) = dfa_state.transitions.get(&(top as i32)) {
                        choice = Some(SingleChoice::Read(target));
                        choices += 1;
                    }
                }
                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Push => {
                let dfa_state = template.push.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;
                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        return None;
                    }
                    choice = Some(SingleChoice::Push(
                        target,
                        Some(negative_to_positive_label(label) as u32),
                    ));
                    choices += 1;
                }
            }
        }

        if choices == 0 {
            return Some(accepting);
        }
        if accepting || choices > 1 {
            return None;
        }
        match choice? {
            SingleChoice::Pop(target) => {
                stack.pop()?;
                phase = Phase::Pop;
                state_id = target;
            }
            SingleChoice::Read(target) => {
                phase = Phase::Read;
                state_id = target;
            }
            SingleChoice::Push(target, pushed) => {
                if let Some(pushed) = pushed {
                    if stack.len() == stack.capacity() {
                        return None;
                    }
                    stack.push(pushed);
                }
                phase = Phase::Push;
                state_id = target;
            }
        }
        steps += 1;
        if steps > max_steps {
            return None;
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SingleChoice {
    Pop(u32),
    Read(u32),
    Push(u32, Option<u32>),
}

fn advance_virtual_stack_single_path(
    template: &CommitTemplateDfas,
    mut stack: VirtualStack<u32, TerminalsDisallowed>,
) -> Option<ParserGSS> {
    let mut phase = Phase::Pop;
    let mut state_id = template.pop.start_state;
    let total_states = template
        .pop
        .states
        .len()
        .saturating_add(template.read.states.len())
        .saturating_add(template.push.states.len());
    let max_steps = total_states.saturating_mul(2).saturating_add(8);
    let mut steps = 0usize;

    loop {
        if matches!(phase, Phase::Pop | Phase::Read)
            && stack.top().is_none()
            && stack.has_hidden_floor_values()
        {
            // The visible Segment prefix has been exhausted, but the GSS floor
            // still contains branch-specific parser states. Pop/read decisions
            // depend on those states, so the single-prefix fast path is no
            // longer exact. Return to the branch-aware template worklist.
            return None;
        }

        let mut choice = None;
        let mut choices = 0usize;
        let accepting;

        match phase {
            Phase::Pop => {
                let dfa_state = template.pop.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                debug_assert!(
                    dfa_state
                        .transitions
                        .keys()
                        .all(|&label| !is_negative_label(label)),
                    "commit template pop DFA contains push label at state {state_id}"
                );

                if let Some(top) = stack.top().copied() {
                    let label = top as i32;
                    if let Some(&target) = dfa_state.transitions.get(&label) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    } else if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    }

                    if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize)
                        && template
                            .read
                            .states
                            .get(*read_state as usize)
                            .is_some_and(|state| state.transitions.contains_key(&label))
                    {
                        choice = Some(SingleChoice::Read(*read_state));
                        choices += 1;
                    }
                }

                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Read => {
                let dfa_state = template.read.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                debug_assert!(
                    dfa_state
                        .transitions
                        .keys()
                        .all(|&label| label != DEFAULT_LABEL && !is_negative_label(label)),
                    "commit template read DFA contains non-read label at state {state_id}"
                );

                if let Some(top) = stack.top().copied() {
                    let label = top as i32;
                    if let Some(&target) = dfa_state.transitions.get(&label) {
                        choice = Some(SingleChoice::Read(target));
                        choices += 1;
                    }
                }

                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Push => {
                let dfa_state = template.push.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        panic!(
                            "commit template push DFA contains non-push label {label} at state {state_id}"
                        );
                    }
                    choice = Some(SingleChoice::Push(
                        target,
                        Some(negative_to_positive_label(label) as u32),
                    ));
                    choices += 1;
                }
            }
        }

        if choices == 0 {
            return Some(if accepting {
                stack.into_gss()
            } else {
                ParserGSS::empty()
            });
        }
        if accepting || choices > 1 {
            return None;
        }

        match choice.expect("single applicable split template transition") {
            SingleChoice::Pop(target) => {
                if stack.pop(1) != 0 {
                    return Some(ParserGSS::empty());
                }
                phase = Phase::Pop;
                state_id = target;
            }
            SingleChoice::Read(target) => {
                phase = Phase::Read;
                state_id = target;
            }
            SingleChoice::Push(target, pushed) => {
                if let Some(pushed) = pushed {
                    stack.push(pushed);
                }
                phase = Phase::Push;
                state_id = target;
            }
        }

        steps += 1;
        if steps > max_steps {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Phase, TemplateAdvanceRuntime, advance_with_template, evaluate_template_language};
    use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::parser::ParserGSS;
    use crate::runtime::CommitTemplateDfas;

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
    fn canonical_language_evaluation_matches_gss_template_walker() {
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
        let merged = ParserGSS::from_single_stack(vec![0, 1, 10], acc.clone()).merge(
            &ParserGSS::from_single_stack(vec![0, 2, 10], acc.clone()),
        );
        let expected = advance_with_template(&template, merged.clone());

        let mut runtime = TemplateAdvanceRuntime::default();
        let (language, accumulator) = runtime
            .language_from_uniform_gss(&merged)
            .expect("test GSS has one uniform accumulator");
        let output = evaluate_template_language(
            &template,
            0,
            Phase::Pop,
            template.pop.start_state,
            language,
            &mut runtime,
        );
        let actual = runtime.gss_from_language(output, accumulator);
        assert!(
            actual
                .semantically_eq(&expected, 4_096)
                .expect("test languages should remain explicitly bounded")
        );
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
