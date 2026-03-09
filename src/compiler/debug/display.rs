
use std::collections::BTreeMap;

use crate::compiler::debug::artifacts::CompileDebug;
use crate::compiler::glr::analysis::EOF;

impl CompileDebug {
    fn terminal_name(
        &self,
        grammar: &crate::compiler::grammar::model::GrammarDef,
        id: crate::compiler::grammar::model::TerminalID,
    ) -> String {
        grammar.terminal_display_name(id)
    }

    fn nonterminal_str(
        &self,
        grammar: &crate::compiler::grammar::model::GrammarDef,
        nonterminal: crate::compiler::grammar::model::NonterminalID,
    ) -> String {
        match grammar.nonterminal_display_name(nonterminal) {
            Some(name) => format!("NT{}('{}')", nonterminal, name),
            None => format!("NT{}", nonterminal),
        }
    }

    fn symbol_str(
        &self,
        grammar: &crate::compiler::grammar::model::GrammarDef,
        sym: &crate::compiler::grammar::model::Symbol,
    ) -> String {
        match sym {
            crate::compiler::grammar::model::Symbol::Terminal(terminal) => {
                format!("T{}('{}')", terminal, self.terminal_name(grammar, *terminal))
            }
            crate::compiler::grammar::model::Symbol::Nonterminal(nonterminal) => {
                self.nonterminal_str(grammar, *nonterminal)
            }
        }
    }
}

impl std::fmt::Display for CompileDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        
        writeln!(f, "═══ GRAMMAR (original) ═══")?;
        writeln!(f, "Start: {}", self.nonterminal_str(&self.grammar_def, self.grammar_def.start))?;
        writeln!(f, "Terminals:")?;
        for t in &self.grammar_def.terminals {
            writeln!(f, "  T{}: name={:?} def={:?}", t.id(), t.name(), t)?;
        }
        writeln!(f, "Rules:")?;
        for (i, r) in self.grammar_def.rules.iter().enumerate() {
            let rhs: Vec<String> = r
                .rhs
                .iter()
                .map(|s| self.symbol_str(&self.grammar_def, s))
                .collect();
            writeln!(f, "  [{}] {} → {}", i, self.nonterminal_str(&self.grammar_def, r.lhs), rhs.join(" "))?;
        }

        writeln!(f, "\n═══ GRAMMAR (normalized for mask) ═══")?;
        writeln!(
            f,
            "Start: {}",
            self.nonterminal_str(&self.normalized_grammar_def, self.normalized_grammar_def.start)
        )?;
        if self.normalized_grammar_def.terminals.len() != self.grammar_def.terminals.len() {
            writeln!(
                f,
                "Terminals: {} (changed from {})",
                self.normalized_grammar_def.terminals.len(),
                self.grammar_def.terminals.len()
            )?;
        }
        writeln!(f, "Rules:")?;
        for (i, r) in self.normalized_grammar_def.rules.iter().enumerate() {
            let rhs: Vec<String> = r
                .rhs
                .iter()
                .map(|s| self.symbol_str(&self.normalized_grammar_def, s))
                .collect();
            writeln!(
                f,
                "  [{}] {} → {}",
                i,
                self.nonterminal_str(&self.normalized_grammar_def, r.lhs),
                rhs.join(" ")
            )?;
        }

        writeln!(
            f,
            "\n═══ GLR PARSE TABLE ({} states, {} rules) ═══",
            self.glr_table.num_states,
            self.glr_table.num_rules
        )?;
        for state in 0..self.glr_table.num_states {
            let actions = &self.glr_table.action[state as usize];
            let gotos = &self.glr_table.goto[state as usize];
            if actions.is_empty() && gotos.is_empty() {
                continue;
            }
            writeln!(f, "  State {state}:")?;
            for (tid, acts) in actions {
                let tname = if *tid == EOF {
                    "$".to_string()
                } else {
                    self.terminal_name(&self.normalized_grammar_def, *tid)
                };
                for a in acts {
                    let astr = match a {
                        crate::compiler::glr::table::Action::Shift(s) => format!("shift {s}"),
                        crate::compiler::glr::table::Action::Reduce(r) => {
                            let rule = &self.glr_table.rules[*r as usize];
                            format!(
                                "reduce r{r} ({} ← {} symbols)",
                                self.nonterminal_str(&self.normalized_grammar_def, rule.lhs),
                                rule.rhs.len()
                            )
                        }
                        crate::compiler::glr::table::Action::Accept => "accept".to_string(),
                    };
                    writeln!(f, "    on '{tname}': {astr}")?;
                }
            }
            for (nt, tgt) in gotos {
                writeln!(
                    f,
                    "    goto {} → state {tgt}",
                    self.nonterminal_str(&self.normalized_grammar_def, *nt)
                )?;
            }
        }

        writeln!(f, "\n═══ VOCABULARY ({} tokens) ═══", self.vocab_entries.len())?;
        if let Some(eos) = self.eos_token_id {
            writeln!(f, "EOS token: {eos}")?;
        }
        for (id, bytes) in &self.vocab_entries {
            let repr = String::from_utf8_lossy(bytes);
            writeln!(f, "  tok{id}: {repr:?} ({} bytes)", bytes.len())?;
        }

        writeln!(f, "\n═══ TOKENIZER STATE ID MAPPING ({}) ═══", self.id_map.tokenizer_states.num_internal_ids())?;
        for (internal, dfa_states) in self.id_map.tokenizer_states.internal_to_originals.iter().enumerate() {
            writeln!(f, "  TSID {internal} ↔ DFA states {:?}", dfa_states)?;
        }

        writeln!(f, "\n═══ VOCAB TOKEN ID MAPPING ({}) ═══", self.id_map.vocab_tokens.num_internal_ids())?;
        for (internal, token_ids) in self.id_map.vocab_tokens.internal_to_originals.iter().enumerate() {
            writeln!(f, "  token-class {internal} ↔ original token IDs {:?}", token_ids)?;
        }

        writeln!(f, "\n═══ TERMINAL CHARACTERIZATIONS ═══")?;
        for (tid, tc) in &self.characterizations {
            writeln!(
                f,
                "  Terminal '{}' (T{tid}):",
                self.terminal_name(&self.normalized_grammar_def, *tid)
            )?;
            for (from, to) in &tc.shifts {
                writeln!(f, "    shift: state {from} → state {to}")?;
            }
            for (from, pop, nt) in &tc.reduces {
                writeln!(f, "    reduce: state {from}, pop {pop}, → NT{nt}")?;
            }
            for (nt, revealed, goto, shift) in &tc.nt_escapes {
                writeln!(
                    f,
                    "    nt_escape: NT{nt}, revealed={revealed}, goto={goto}, shift={shift}"
                )?;
            }
            for (nt, revealed, pop, re_nt) in &tc.nt_rereduces {
                writeln!(
                    f,
                    "    nt_rereduce: NT{nt}, revealed={revealed}, pop={pop}, → NT{re_nt}"
                )?;
            }
        }

        writeln!(
            f,
            "\n═══ TEMPLATES ({} terminals) ═══",
            self.templates.by_terminal.len()
        )?;
        for (terminal, template_dfa) in &self.templates.by_terminal {
            writeln!(
                f,
                "  Terminal '{}'(T{terminal})",
                self.terminal_name(&self.normalized_grammar_def, *terminal)
            )?;
            writeln!(f, "    Template DFA:")?;
            write!(f, "{}", template_dfa)?;
        }

        let terminal_symbols: BTreeMap<i32, String> = self
            .normalized_grammar_def
            .terminals
            .iter()
            .map(|t| {
                (
                    t.id() as i32,
                    format!("'{}'", self.normalized_grammar_def.terminal_display_name(t.id())),
                )
            })
            .collect();

        // We pass an empty map here to coerce weight formatting to emit opaque TSIDs 
        // on the LHS (e.g. `0` instead of `tsid0/[0]`) while keeping meaningful LLM tokens on the RHS.
        let tsid_names: BTreeMap<u32, String> = BTreeMap::new();
        let token_names: BTreeMap<u32, String> = self
            .vocab_entries
            .iter()
            .map(|(id, bytes)| {
                let repr = String::from_utf8_lossy(bytes);
                (*id, format!("{repr:?}"))
            })
            .collect();

        writeln!(f, "\n═══ TERMINAL NWA — after build (raw) ═══")?;
        write!(
            f,
            "{}",
            self.terminal_debug
                .nwa_after_build
                .display_with_all_maps(&terminal_symbols, &tsid_names, &token_names)
        )?;

        writeln!(f, "\n═══ TERMINAL NWA — after collapse_always_allowed ═══")?;
        write!(
            f,
            "{}",
            self.terminal_debug
                .nwa_after_collapse
                .display_with_all_maps(&terminal_symbols, &tsid_names, &token_names)
        )?;

        writeln!(f, "\n═══ TERMINAL DWA — final ═══")?;
        write!(
            f,
            "{}",
            self.terminal_dwa
                .display_with_all_maps(&terminal_symbols, &tsid_names, &token_names)
        )?;

        writeln!(f, "\n═══ PARSER NWA — before resolve_negatives ═══")?;
        write!(f, "{}", self.parser_nwa_before_resolve)?;

        writeln!(f, "\n═══ PARSER NWA — after resolve_negatives ═══")?;
        write!(f, "{}", self.parser_nwa_after_resolve)?;

        writeln!(f, "\n═══ PARSER DWA — pre-minimize ═══")?;
        write!(f, "{}", self.parser_dwa_pre_minimize)?;

        writeln!(f, "\n═══ PARSER DWA — final (post-minimize) ═══")?;
        write!(f, "{}", self.parser_dwa)?;

        Ok(())
    }
}
