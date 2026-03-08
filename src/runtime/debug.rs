#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::Constraint;

// SEP1_MAP: there is no clean sep1 runtime-debug module equivalent.
// The nearest sep1 analogues are ad hoc debug helpers in
// `grammars2024/src/constraint.rs` plus tracing/profiling in
// `grammars2024/src/constraint_fns.rs`.
impl Constraint {
    // SEP1_MAP: `debug_dump()` has no direct sep1 counterpart; it is closest in
    // role to sep1's scattered inspection helpers such as GSS formatting and
    // terminal/DWA debug printing in `grammars2024/src/constraint.rs`.
    
    pub(crate) fn debug_dump(&self) {
        eprintln!("--- Constraint Debug Dump ---");
        eprintln!("Tokenizer DFA states: {}", self.tokenizer.dfa.num_states());
        for s in 0..self.tokenizer.dfa.num_states() {
            let fin = self.tokenizer.all_matched_terminals(s as u32);
            if !fin.is_empty() {
                eprintln!("  tok DFA state {}: finalizers={:?}", s, fin);
            }
            let mut trans = Vec::new();
            for b in 0u16..=255u16 {
                let next = self.tokenizer.dfa.get_transition(s as u32, b as u8);
                if next != crate::automata::dfa::DEAD {
                    trans.push((b as u8, next));
                }
            }
            if !trans.is_empty() && trans.len() <= 20 {
                eprintln!("  tok DFA state {}: transitions={:?}", s, trans);
            } else if !trans.is_empty() {
                eprintln!("  tok DFA state {}: {} transitions", s, trans.len());
            }
        }
        eprintln!("DWA states: {}", self.parser_dwa.states.len());
        for tokenizer_state in self.possible_matches.keys() {
            for (term, rs) in self.possible_matches_for_state(*tokenizer_state) {
                let vals: Vec<u32> = rs.iter().collect();
                eprintln!(
                    "possible_matches[state={}][term={}] = {:?}",
                    tokenizer_state,
                    term,
                    vals
                );
            }
        }
        eprintln!("--- End Debug Dump ---");
    }

    // SEP1_MAP: `debug_tokenizer()` is closest to manually probing sep1
    // `Tokenizer::execute_from_state()` in `grammars2024/src/dfa_u8/tokenizer_ops.rs`.
    // sep1 does not expose this exact runtime helper method.
    
    pub(crate) fn debug_tokenizer(&self, input: &[u8], start_state: u32) {
        let result = self.tokenizer.execute_all_matches(input, start_state);
        eprintln!(
            "[debug_tokenizer] input={:?} start={} -> end={} matches={:?}",
            input, start_state, result.end_state, result.matches
        );
        let mut state = start_state;
        for (i, &byte) in input.iter().enumerate() {
            let next = self.tokenizer.dfa.get_transition(state, byte);
            let is_dead = next == crate::automata::dfa::DEAD;
            let finals = if !is_dead {
                self.tokenizer
                    .all_matched_terminals(next)
                    .into_iter()
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };
            eprintln!(
                "  byte[{}]=0x{:02X} state {}->{}{}{}",
                i, byte, state, next,
                if is_dead { " DEAD" } else { "" },
                if !finals.is_empty() { format!(" finalizers={:?}", finals) } else { String::new() }
            );
            state = next;
            if is_dead { break; }
        }
    }

    // SEP1_MAP: nearest sep1 analogue is `Tokenizer::initial_state_id()` in
    // `grammars2024/src/dfa_u8/tokenizer_ops.rs`; glrmask exposes it through the
    // compiled constraint for runtime debugging.
    
    pub(crate) fn tokenizer_initial_state(&self) -> u32 {
        self.tokenizer.initial_state_id()
    }
}
