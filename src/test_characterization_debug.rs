#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use indoc::indoc;
    // use partial_debug::partial_debug; // Removed
    use crate::interface::GrammarDefinition;
    use crate::precompute4::characterize::compute_terminal_characterization;
    use crate::glr::table::{TerminalID, generate_glr_parser};

    #[test]
    fn test_debug_characterization() {
        let ebnf_grammar = indoc! {r#"
            s ::= A EOF;
            A ::= 'a';
            EOF ::= '$';
        "#};
        let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
        
        let nullable = grammar_definition.get_nullable_terminals();
        let ignore: std::collections::HashSet<TerminalID> = grammar_definition.ignore_terminal_ids.iter().cloned().collect();
        let parser = generate_glr_parser(
            &grammar_definition.productions,
            &nullable,
            ignore
        );
        
        // Terminals: A=0, EOF=1
        let char_a = compute_terminal_characterization(&parser, TerminalID(0));
        let char_eof = compute_terminal_characterization(&parser, TerminalID(1));
        
        println!("Characterization A (0):");
        println!("{}", char_a);
        
        println!("Characterization EOF (1):");
        println!("{}", char_eof);
        
        // Assertions
        let has_shift_0_for_a = char_a.initial_shifts.iter().any(|(init, _)| init.0 == 0);
        let has_shift_0_for_eof = char_eof.initial_shifts.iter().any(|(init, _)| init.0 == 0);
        
        println!("Has shift from 0 for A: {}", has_shift_0_for_a);
        println!("Has shift from 0 for EOF: {}", has_shift_0_for_eof);
        
        if has_shift_0_for_eof {
            println!("BUG: EOF allows shift from 0!");
        } else {
            println!("OK: EOF does not allow shift from 0.");
        }
    }
}
