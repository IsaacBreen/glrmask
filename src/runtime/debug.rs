
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::Constraint;

impl Constraint {
    
    pub(crate) fn debug_dump(&self) {
        eprintln!("--- Constraint Debug Dump ---");
        eprintln!("Tokenizer DFA states: {}", self.tokenizer.dfa.num_states());
        for s in 0..self.tokenizer.dfa.num_states() {
            let fin = self.tokenizer.matched_terminals(s as u32);
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
        for (tokenizer_state, tsid_map) in &self.terminal_tokens_by_state {
            for (tsid, terminal_map) in tsid_map {
                for (term, rs) in terminal_map {
                    let vals: Vec<u32> = rs.iter().collect();
                    eprintln!(
                        "terminal_tokens_by_state[state={}][tsid={}][term={}] = {:?}",
                        tokenizer_state,
                        tsid,
                        term,
                        vals
                    );
                }
            }
        }
        eprintln!("--- End Debug Dump ---");
    }

    
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
                self.tokenizer.dfa.finalizers(next).iter().copied().collect::<Vec<_>>()
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

    
    pub(crate) fn tokenizer_initial_state(&self) -> u32 {
        unimplemented!()
    }
}
