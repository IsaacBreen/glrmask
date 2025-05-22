use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};

use regex_automata::{
    dfa::{dense::DFA, Automaton, Text},
    meta::{MatchKind, Regex as MetaRegex, RegexBuilder},
    util::wire::{self, BoundedSerialize},
};
use regex_syntax::hir::{Hir, HirKind};

use crate::tokenizer::{LLMTokenID, TokenizerStateID, TokenizerToken};

use super::nfa_simulator::{Match, NFASimulator, SimulateResult};
use super::nfa::{NFAStateID, NFA};
use super::parsers::parse_groups_expr;
use super::expr::{Expr, ExprGroup};
use super::debug;

// -----------------------------------------------------------------------------
// Regex – the public compiled tokenizer type
// -----------------------------------------------------------------------------

/// Represents a compiled regular expression designed for tokenization, supporting
/// features necessary for grammar constraints (like partial matches, state persistence).
#[derive(Clone)]
pub struct Regex {
    pub(crate) dfa:       DFA,
    pub(crate) nfa:       NFA, // Storing the NFA for its initial state and states mapping
    pub(crate) group_map: Vec<usize>, // Maps DFA group ID to original group ID
}

impl Debug for Regex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Regex")
            .field("dfa", &self.dfa)
            .field("nfa", &self.nfa)
            .field("group_map", &self.group_map)
            .finish()
    }
}

impl Display for Regex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Regex ({} DFA states)", self.dfa.states.len())?;
        writeln!(f, "  DFA: {}", self.dfa)?;
        writeln!(f, "  NFA: {}", self.nfa)?;
        writeln!(f, "  Group Map: {:?}", self.group_map)?;
        Ok(())
    }
}

impl PartialEq for Regex {
    fn eq(&self, other: &Self) -> bool {
        // Simple comparison for now; deep equality of DFA and NFA might be complex.
        // This is likely sufficient for testing if compilation results are the same.
        // TODO: Implement proper deep equality for DFA and NFA if needed.
        self.dfa.states.len() == other.dfa.states.len() &&
        self.nfa.states.len() == other.nfa.states.len() &&
        self.group_map == other.group_map
    }
}

impl Eq for Regex {}


/// The result of attempting to match a suffix starting from a particular
/// tokenizer state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecuteResult {
    /// Zero or more token matches found at the *start* of the suffix.
    /// If multiple matches occur at the same starting position, all are included.
    pub matches:   Vec<TokenizerToken>,
    /// If the tokenizer finished processing the entire suffix and ended in
    /// a valid DFA state, this is the ID of that state. None otherwise.
    pub end_state: Option<usize>,
}

impl Regex {
    /// Creates a new `Regex` from a list of `ExprGroup`s.
    pub fn from_expr_groups(expr_groups: Vec<ExprGroup>) -> Self {
        let (nfa, group_map) = NFA::from_expr_groups(expr_groups);
        let dfa = DFA::from_nfa(&nfa, nfa.initial_state);

        Self { dfa, nfa, group_map }
    }

    /// Returns the ID of the initial state of the tokenizer's DFA.
    pub fn initial_state_id(&self) -> TokenizerStateID {
        TokenizerStateID(self.dfa.start_state)
    }

    /// Executes the tokenizer on a given byte slice starting from a specific
    /// DFA state.
    ///
    /// Returns a `ExecuteResult` containing any token matches found at the
    /// start of the slice and the resulting DFA state after processing the
    /// entire slice, if a valid state is reached.
    pub fn execute_from_state(&self, input_bytes: &[u8], state_id: TokenizerStateID) -> ExecuteResult {
        let mut matches = Vec::new();

        // 1. Handle potential matches at the current DFA state immediately.
        // The DFA tells us which token groups can end right at the current state.
        let (nfa_states_at_start, dfa_state) = self.get_nfa_states_from_dfa_state(state_id);
        for nfa_state_id in nfa_states_at_start.iter() {
             if let Some(group_id) = self.nfa.get_group_id_for_final_state(*nfa_state_id) {
                // This recognizes zero-width matches or matches that completed
                // at the end of the *previous* input chunk that led to `state_id`.
                // For non-zero width matches on the *current* `input_bytes`,
                // we rely on the NFA simulation below.
                // This part is mainly for correct handling of things like epsilon
                // productions or trailing whitespace matched at the end of the
                // *last* chunk, before processing `input_bytes`.
                // Let's add the group_id here, width will be 0 for DFA-only matches
                // at the state boundary. The NFASimulator will find non-zero width
                // matches within `input_bytes`.
                 let original_group_id = self.group_map.get(group_id).cloned().unwrap_or_else(|| {
                     debug!(0, "Warning: DFA group ID {} not found in group_map", group_id);
                     group_id // Fallback to the DFA group ID if not mapped
                 });
                 // Note: We add this with width 0. The NFASimulator finds non-zero width matches.
                 matches.push(TokenizerToken { id: original_group_id, width: 0 });
            }
        }


        // 2. Simulate NFA from the current DFA state using the input bytes.
        //    This finds matches within `input_bytes` and the end state.
        // We need to start the NFA simulation from the NFA states that
        // correspond to the *current* DFA state, PLUS the NFA's global
        // initial state. Including the global initial state ensures we
        // can start matching any token group at the beginning of `input_bytes`,
        // regardless of the DFA's current state. This is crucial for
        // tokenizers that need to identify *any* valid next token.
        let mut effective_initial_nfa_states = nfa_states_at_start;
        effective_initial_nfa_states.insert(self.nfa.initial_state);

        let mut sim = NFASimulator::new(&self.nfa, effective_initial_nfa_states);
        let SimulateResult { matches: sim_matches, end_state: sim_end_nfa_states } = sim.execute(input_bytes);

        // Add matches found by the simulator
        for m in sim_matches {
             let original_group_id = self.group_map.get(m.group_id).cloned().unwrap_or_else(|| {
                 debug!(0, "Warning: NFASimulator group ID {} not found in group_map", m.group_id);
                 m.group_id // Fallback to the NFASimulator group ID if not mapped
             });
             matches.push(TokenizerToken { id: original_group_id, width: m.width });
        }


        // 3. Determine the final DFA state.
        let final_dfa_state_id = self.get_dfa_state_from_nfa_states(&sim_end_nfa_states);

        ExecuteResult {
            matches,
            end_state: final_dfa_state_id.map(|id| id.0),
        }
    }

    /// Returns the set of original group IDs (TerminalIDs) that can be matched
    /// starting from the given DFA state.
    /// This is essentially asking the DFA "what tokens can I possibly see next?".
    pub fn tokens_accessible_from_state(&self, state_id: TokenizerStateID) -> BTreeSet<usize> {
        let (nfa_states, _dfa_state) = self.get_nfa_states_from_dfa_state(state_id);

        // We need to explore from the NFA states reachable from the current DFA state,
        // PLUS the NFA's global initial state (epsilon closure). This is because
        // any token *could* start from the initial state if it's a valid next
        // token sequence for the overall grammar.
        let mut reachable_nfa_states = nfa_states;
        reachable_nfa_states.insert(self.nfa.initial_state);
        let closure = self.nfa.epsilon_closure(&reachable_nfa_states);

        let mut possible_tokens = BTreeSet::new();
        for nfa_state_id in closure.iter() {
            // Check direct final states reachable by epsilon transitions
            if let Some(group_id) = self.nfa.get_group_id_for_final_state(*nfa_state_id) {
                 let original_group_id = self.group_map.get(group_id).cloned().unwrap_or_else(|| {
                     debug!(0, "Warning: DFA group ID {} not found in group_map", group_id);
                     group_id // Fallback
                 });
                possible_tokens.insert(original_group_id);
            }

            // Explore transitions to see what tokens can *start* from here
            // Any transition from these states on a non-epsilon character implies
            // that the target state is reachable by consuming that character.
            // If that target state eventually leads to a final state for a group,
            // that group is accessible.
            // A simpler approach is to find which group IDs are reachable *eventually*
            // from the initial states via at least one non-epsilon transition.

            // We can simulate one step with all possible characters to see which
            // states (and thus which potential tokens) are reachable.
            // Or, more efficiently, traverse the NFA from the reachable states.
            for &start_nfa_id in closure.iter() {
                for (char_range, target_nfa_id) in self.nfa.transitions_from(start_nfa_id) {
                     // Find any path from target_nfa_id to a final state.
                     // This is equivalent to checking if the NFA fragment starting at target_nfa_id
                     // can accept *any* non-empty string that ends in a final state.
                     // A simple reachability check to final states with group IDs is sufficient.
                     let reachable_final_states = self.nfa.find_reachable_final_states(*target_nfa_id);
                     for final_nfa_id in reachable_final_states {
                         if let Some(group_id) = self.nfa.get_group_id_for_final_state(final_nfa_id) {
                             let original_group_id = self.group_map.get(group_id).cloned().unwrap_or_else(|| {
                                 debug!(0, "Warning: DFA group ID {} not found in group_map", group_id);
                                 group_id // Fallback
                             });
                             possible_tokens.insert(original_group_id);
                         }
                     }
                }
            }
        }

        possible_tokens
    }

    /// Gets the set of NFA states that correspond to a given DFA state.
    fn get_nfa_states_from_dfa_state(&self, state_id: TokenizerStateID) -> (BTreeSet<NFAStateID>, &regex_automata::dfa::dense::State) {
        // Access the internal DFA state representation. This is implementation-dependent
        // and might break with regex-automata updates.
        // In current regex-automata (0.2.x), DFA state `i` corresponds to `self.dfa.states[i]`.
        // Each state object should contain the set of NFA states it represents.
        // Looking at the regex-automata `dense::State` struct, it contains a field
        // `nfa_states` which is a `BTreeSet<NFAState>`. NFAState struct contains an `id`.
        // This seems to be what we need.
        let dfa_state = &self.dfa.states[state_id.0];
        let nfa_states = dfa_state.nfa_states.iter().map(|s| NFAStateID(s.id)).collect();
        (nfa_states, dfa_state)
    }

    /// Gets the DFA state ID corresponding to a set of NFA states.
    /// Returns None if the set of NFA states does not correspond to any existing DFA state.
    fn get_dfa_state_from_nfa_states(&self, nfa_states: &BTreeSet<NFAStateID>) -> Option<TokenizerStateID> {
        // This requires mapping the set of NFA states back to a DFA state ID.
        // The DFA construction process typically builds a map from sets of NFA states to DFA states.
        // We don't have direct access to this internal map in `regex-automata`.
        // A workaround is to reconstruct a DFA state object and search for it. This is inefficient.
        // A better approach might require modifying `regex-automata` or storing the map ourselves
        // during DFA construction.

        // Alternative: If the NFA simulation ends in a state that *exactly* matches
        // the NFA state set of an existing DFA state, we can find that DFA state ID.
        // This is only guaranteed if the NFA simulation state set precisely matches
        // a state that was reachable and minimized during DFA construction.

        // Let's try iterating through DFA states and comparing NFA state sets.
        // This is inefficient but should work for correctness.
        // If the set of NFA states from simulation contains NFA_ACCEPT state,
        // it means one or more patterns matched. The DFA end state corresponds
        // to the set of *all* active NFA states after processing the input.

        let sim_nfa_states_set: BTreeSet<_> = nfa_states.iter().copied().collect();

        for (dfa_id, dfa_state) in self.dfa.states.iter().enumerate() {
            let dfa_nfa_states_set: BTreeSet<_> = dfa_state.nfa_states.iter().map(|s| NFAStateID(s.id)).collect();
            if dfa_nfa_states_set == sim_nfa_states_set {
                return Some(TokenizerStateID(dfa_id));
            }
        }

        // If no existing DFA state corresponds exactly to the final NFA state set,
        // it means the NFA simulation ended in a state that wasn't explicitly
        // represented as a unique state in the minimized DFA, or it represents
        // a state from which no further progress is possible in the original DFA.
        // In this context, it's usually equivalent to ending in a "dead" state
        // or a state that doesn't correspond to a valid grammar continuation.
        // So, returning None seems appropriate if a precise match isn't found.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finite_automata::expr::{eat_u8, repeat0, repeat1, choice, sequence};
    use crate::finite_automata::parsers::parse_expr;
    use crate::finite_automata::groups;

    #[test]
    fn test_regex_from_expr_groups() {
        // Test with a simple regex: (a|b)*c
        let expr_a = eat_u8(b'a');
        let expr_b = eat_u8(b'b');
        let expr_c = eat_u8(b'c');

        let choice_ab = choice(vec![expr_a, expr_b]);
        let repeat_ab = repeat0(choice_ab);
        let seq_expr = sequence(vec![repeat_ab, expr_c]);

        let expr_groups = groups![seq_expr];
        let regex = Regex::from_expr_groups(expr_groups);

        // Basic checks
        assert!(regex.dfa.states.len() > 1); // Should have more than just the initial state
        assert!(regex.nfa.states.len() > 1); // Should have more than just the initial state
        assert_eq!(regex.group_map.len(), 1); // Should map the single group

        // Test execution
        let initial_state = regex.initial_state_id();

        let result1 = regex.execute_from_state(b"abc", initial_state);
        // Should match "abc" as a whole token
        assert_eq!(result1.matches.len(), 1);
        assert_eq!(result1.matches[0].width, 3);
        assert_eq!(result1.matches[0].id, 0); // Assuming it's the first group

        // After matching "abc", it should end in a state that has processed "abc".
        // If the grammar rule is just "(a|b)*c", there isn't really a "next" valid token
        // unless EOF is considered. The end_state might be a final state or a state from
        // which no further grammar tokens are possible. The exact DFA state ID is hard to
        // predict without inspecting the generated DFA. Let's just check if there's an end state.
        assert!(result1.end_state.is_some());


        let result2 = regex.execute_from_state(b"ab", initial_state);
        // Should not match "ab" as a full token because 'c' is missing
        assert_eq!(result2.matches.len(), 0);
        // Should end in a state that has processed "ab" and is potentially looking for 'c'
        assert!(result2.end_state.is_some());

        let result3 = regex.execute_from_state(b"d", initial_state);
        // Should not match anything and end in a non-accepting state or None
        assert_eq!(result3.matches.len(), 0);
        assert!(result3.end_state.is_none() || regex.tokens_accessible_from_state(TokenizerStateID(result3.end_state.unwrap())).is_empty());


         let result4 = regex.execute_from_state(b"c", initial_state);
        // Should match "c" as a full token ((a|b)* can match empty string)
        assert_eq!(result4.matches.len(), 1);
        assert_eq!(result4.matches[0].width, 1);
        assert_eq!(result4.matches[0].id, 0);
        assert!(result4.end_state.is_some());
    }

    #[test]
    fn test_tokens_accessible_from_state() {
         // Regex: (a|b)+c
         let expr_a = eat_u8(b'a');
         let expr_b = eat_u8(b'b');
         let expr_c = eat_u8(b'c');

         let choice_ab = choice(vec![expr_a.clone(), expr_b.clone()]);
         let repeat1_ab = repeat1(choice_ab);
         let seq_expr = sequence(vec![repeat1_ab, expr_c.clone()]);

         let expr_groups = groups![seq_expr];
         let regex = Regex::from_expr_groups(expr_groups);

         let initial_state = regex.initial_state_id();

         // Initially, 'a' or 'b' should be accessible (to start (a|b)+)
         let initial_accessible = regex.tokens_accessible_from_state(initial_state);

         // Find the group IDs for 'a', 'b', 'c'. These should be 0, 1, 2 respectively based on definition order.
         // The `group_map` maps DFA group IDs back to original ExprGroup indices.
         // We need the original group IDs (indices in the `expr_groups` vector before DFA compilation).
         // Since we only have one ExprGroup containing the whole regex, the DFA group ID 0 maps to original group 0.
         // The internal tokens generated by the tokenizer will have IDs based on the Regex internal group IDs.
         // This is a bit tricky - the test expects to check for original terminal IDs.
         // Let's assume the tokenizer maps 'a' to ID 0, 'b' to ID 1, 'c' to ID 2 based on the internal DFA/NFA structure.
         // We need to map back from the internal DFA group ID to the original grammar terminal ID.
         // The `group_map` handles this.
         // Let's rebuild the tokenizer and inspect its internal structure or map.
         // Okay, the `group_map` maps DFA group ID to original `ExprGroup` index.
         // In this test, we have `groups![seq_expr]`. This is ONE ExprGroup. So DFA group ID 0 -> original group 0.
         // The `tokens_accessible_from_state` returns original group IDs.
         // So we need to know what original group IDs correspond to 'a', 'b', 'c' *within* the `seq_expr`.
         // The `seq_expr`'s components (`repeat1_ab`, `expr_c`) will generate internal regex groups.
         // `repeat1_ab` is based on `choice(vec![expr_a.clone(), expr_b.clone()])`. `choice` creates internal groups for its alternatives.
         // So 'a' and 'b' likely get internal group IDs within the regex.
         // The `Regex::from_expr_groups` flattens these and assigns DFA group IDs.
         // The `group_map` maps these DFA group IDs back to the *original* `ExprGroup` indices.
         // This test is checking `tokens_accessible_from_state` which should return the *original* group IDs defined in the grammar (TerminalIDs).
         // The `tokens_accessible_from_state` method already uses `self.group_map` to return original group IDs.
         // So we just need to know which original group IDs correspond to 'a', 'b', 'c'.
         // In `GrammarDefinition::get_terminal_expressions_for_tokenizer`, each terminal expression gets an ID.
         // Let's assume 'a' is TerminalID 0, 'b' is TerminalID 1, 'c' is TerminalID 2.
         // These TerminalIDs become the original group IDs passed to `Regex::from_expr_groups`.
         // So, let's simulate that mapping.
         let mut original_terminal_ids = HashMap::new();
         original_terminal_ids.insert(Expr::eat_u8(b'a'), 0);
         original_terminal_ids.insert(Expr::eat_u8(b'b'), 1);
         original_terminal_ids.insert(Expr::eat_u8(b'c'), 2);

         // Rebuild regex with explicit original group IDs
         let expr_groups_with_ids = vec![
             ExprGroup { expr: Expr::eat_u8(b'a'), id: 0, is_greedy: true },
             ExprGroup { expr: Expr::eat_u8(b'b'), id: 1, is_greedy: true },
             ExprGroup { expr: Expr::eat_u8(b'c'), id: 2, is_greedy: true },
             // We need groups for repeat1_ab and seq_expr as a whole as well,
             // but the tokenizer only needs the terminals grouped.
             // The grammar compiler handles assigning group IDs to terminals.
             // Let's use the logic from `GrammarDefinition::get_terminal_expressions_for_tokenizer`.
             // It gathers all unique terminal Exprs and assigns them contiguous IDs.
             // So, for `(a|b)+c`, the unique terminals are 'a', 'b', 'c'.
             // Let's say 'a' -> 0, 'b' -> 1, 'c' -> 2.
             // The Regex must be built from ExprGroups for these.
             ExprGroup { expr: Expr::eat_u8(b'a'), id: 0, is_greedy: true },
             ExprGroup { expr: Expr::eat_u8(b'b'), id: 1, is_greedy: true },
             ExprGroup { expr: Expr::eat_u8(b'c'), id: 2, is_greedy: true },
         ];
         // The `from_expr_groups` method expects ExprGroups as they come from
         // the grammar compiler, which includes internal ones.
         // Let's use the simpler approach of building the regex directly from
         // the top-level expression `seq_expr`. The internal groups are managed
         // by NFA::from_expr_groups. The `group_map` maps the *final* group IDs
         // assigned by DFA construction back to the *original* group IDs
         // provided in the input vector to NFA::from_expr_groups.
         // In `from_expr_groups`, the input `expr_groups` vector *are* the original groups.
         // So if `expr_groups` is `vec![group_a, group_b, group_c]`, then
         // DFA group ID `x` maps to original group ID `expr_groups[group_map[x]].id`.
         // The simplest way to test is to build the regex from the *terminal* expr groups.
         let terminal_expr_groups = groups![
             greedy_group(Expr::eat_u8(b'a')), // id 0
             greedy_group(Expr::eat_u8(b'b')), // id 1
             greedy_group(Expr::eat_u8(b'c')), // id 2
         ];
         let regex_terminals = Regex::from_expr_groups(terminal_expr_groups);
         let initial_state_terminals = regex_terminals.initial_state_id();

         // Now, the `tokens_accessible_from_state` method on `regex_terminals`
         // should tell us which of the original terminal group IDs (0, 1, 2)
         // are accessible from a given state.
         // This doesn't directly test the grammar logic (which combines terminals
         // into sequences/repeats), but tests the tokenizer's ability to report
         // which *individual* tokens (as defined by the original ExprGroups)
         // are accessible from a DFA state.

         // Let's define a simplified grammar for this test: S -> T+c, T -> a|b
         // Terminals: a, b, c.
         let terminal_a_group_id = 0;
         let terminal_b_group_id = 1;
         let terminal_c_group_id = 2;

         let term_groups = groups![
             greedy_group(eat_u8(b'a')).with_id(terminal_a_group_id),
             greedy_group(eat_u8(b'b')).with_id(terminal_b_group_id),
             greedy_group(eat_u8(b'c')).with_id(terminal_c_group_id),
         ];
         let regex = Regex::from_expr_groups(term_groups);

         let initial_state = regex.initial_state_id();

         // From initial state, should be able to start 'a' or 'b'.
         let initial_accessible = regex.tokens_accessible_from_state(initial_state);
         let expected_initial: BTreeSet<_> = vec![terminal_a_group_id, terminal_b_group_id].into_iter().collect();
         assert_eq!(initial_accessible, expected_initial);

         // Simulate consuming 'a'.
         let result_a = regex.execute_from_state(b"a", initial_state);
         assert!(result_a.end_state.is_some());
         let state_after_a = TokenizerStateID(result_a.end_state.unwrap());

         // From state after 'a', should be able to continue with 'a' or 'b' ((a|b)+)
         // OR end the (a|b)+ and match 'c'.
         let accessible_after_a = regex.tokens_accessible_from_state(state_after_a);
         let expected_after_a: BTreeSet<_> = vec![terminal_a_group_id, terminal_b_group_id, terminal_c_group_id].into_iter().collect();
         assert_eq!(accessible_after_a, expected_after_a);

         // Simulate consuming "ab".
         let result_ab = regex.execute_from_state(b"ab", initial_state);
         assert!(result_ab.end_state.is_some());
         let state_after_ab = TokenizerStateID(result_ab.end_state.unwrap());

         // From state after "ab", same as after "a": 'a', 'b', or 'c'.
         let accessible_after_ab = regex.tokens_accessible_from_state(state_after_ab);
         assert_eq!(accessible_after_ab, expected_after_a);

         // Simulate consuming "abc".
         let result_abc = regex.execute_from_state(b"abc", initial_state);
         assert!(result_abc.end_state.is_some());
         let state_after_abc = TokenizerStateID(result_abc.end_state.unwrap());

         // From state after "abc", where the whole pattern matched: no more tokens expected by the regex.
         // The DFA should be in a final state for the 'c' group.
         // `tokens_accessible_from_state` should return empty because no further tokens
         // are defined in this regex after 'c'.
         let accessible_after_abc = regex.tokens_accessible_from_state(state_after_abc);
         let expected_after_abc: BTreeSet<_> = vec![].into_iter().collect(); // Nothing accessible after matching the whole regex
         assert_eq!(accessible_after_abc, expected_after_abc);

          // Simulate consuming "ac".
          let result_ac = regex.execute_from_state(b"ac", initial_state);
          assert!(result_ac.end_state.is_some());
          let state_after_ac = TokenizerStateID(result_ac.end_state.unwrap());
          let accessible_after_ac = regex.tokens_accessible_from_state(state_after_ac);
          assert_eq!(accessible_after_ac, expected_after_abc, "After matching 'ac', should be final state like 'abc'");


         // Simulate consuming "c" (which shouldn't match the full pattern (a|b)+c)
         let result_c = regex.execute_from_state(b"c", initial_state);
         assert!(result_c.matches.is_empty()); // 'c' alone doesn't match (a|b)+c
         // The end state should be a state that consumed 'c' but isn't a valid
         // intermediate or final state for the pattern.
         // `tokens_accessible_from_state` from this state should yield nothing.
         assert!(result_c.end_state.is_none() || regex.tokens_accessible_from_state(TokenizerStateID(result_c.end_state.unwrap())).is_empty());
    }
}
