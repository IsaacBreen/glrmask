#[cfg(test)]
mod tests {
    use crate::finite_automata::eat_u8;
    use crate::interface::{choice, sequence, regex, Grammar, IncrementalParser};
    use crate::tokenizer::TokenizerStateID; // Import necessary types

    #[test]
    fn test_incremental_parser_simple() {
        // Grammar: S -> 'a' 'b' | 'a' 'c'
        let exprs = vec![
            (
                "S".to_string(),
                choice(vec![
                    sequence(vec![regex(eat_u8(b'a')), regex(eat_u8(b'b'))]),
                    sequence(vec![regex(eat_u8(b'a')), regex(eat_u8(b'c'))]),
                ]),
            ),
        ];
        let grammar = Grammar::from_exprs(exprs);
        let mut parser = IncrementalParser::new(&grammar);

        assert!(parser.is_valid()); // Initial state is valid

        parser.feed(b"a");
        assert!(parser.is_valid()); // After 'a', still valid (expecting 'b' or 'c')
        // Check internal state (optional): should have one GLR state in tokenizer state 0
        // The tokenizer state after matching 'a' should reset to 0.
        assert_eq!(parser.state.len(), 1, "Expected 1 state after feeding 'a'");
        assert!(parser.state.contains_key(&TokenizerStateID(0)), "Expected tokenizer state 0 after 'a'");

        parser.feed(b"b");
        assert!(parser.is_valid()); // After 'ab', it's a valid complete parse

        // Reset and try the other path
        parser = IncrementalParser::new(&grammar);
        parser.feed(b"ac");
        assert!(parser.is_valid()); // After 'ac', also valid

        // Try invalid sequence
        parser = IncrementalParser::new(&grammar);
        parser.feed(b"ad");
        dbg!(&parser.state.keys().collect::<Vec<_>>());
        assert!(!parser.is_valid()); // After 'ad', invalid
    }
}
// This file can be used for tests specific to the interface module,
// although many tests are currently within interface.rs itself.


impl<'a> IncrementalParser<'a> {
    /// Checks if the current state is valid (i.e., there's at least one active parse path).
    /// A state is valid if the `state` map is not empty, meaning the input fed so far
    /// corresponds to a valid prefix according to the grammar and tokenizer.
    pub fn is_valid(&self) -> bool {
        !self.state.is_empty()
    }
}
