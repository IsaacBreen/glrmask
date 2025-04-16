// src/tests_apr25.rs
#![allow(unused_imports)] // Allow unused imports for clarity during development

#[cfg(test)]
mod tests_apr25 {
    use crate::datastructures::u8set::U8Set;
    use crate::finite_automata::*;
    use crate::{choice, groups, seq}; // Import macros
    use std::collections::BTreeMap;

    // Helper function to create Expr::U8Seq from string literal
    fn eat_string(s: &str) -> Expr {
        Expr::U8Seq(s.bytes().collect())
    }

    // Helper function to create Expr::U8Class from string literal (choice of chars)
    fn eat_u8_choice(s: &str) -> Expr {
        Expr::U8Class(U8Set::from_bytes(s.as_bytes()))
    }

    // Helper function to create Expr::U8Class for any byte except the given one
    fn eat_u8_negation_class(byte: u8) -> Expr {
        let mut set = U8Set::new();
        set.remove(byte);
        Expr::U8Class(set)
    }

    #[test]
    fn test_python_tokens() {
        // --- Define helper sets and expressions ---
        let digit_set = U8Set::from_byte_range(b'0'..=b'9');
        let alph_lower_set = U8Set::from_byte_range(b'a'..=b'z');
        let alph_upper_set = U8Set::from_byte_range(b'A'..=b'Z');
        let underscore_set = U8Set::from_byte(b'_');

        let name_start_set = alph_lower_set.union(&alph_upper_set).union(&underscore_set);
        let name_middle_set = name_start_set.union(&digit_set);

        let digit = Expr::U8Class(digit_set.clone());
        let name_start = Expr::U8Class(name_start_set);
        let name_middle = Expr::U8Class(name_middle_set);

        // --- Define the 'ignore' expression ---
        // ignore = rep(choice([
        //     eat_u8(ord(" ")),
        //     seq([eat_u8(ord("#")), rep(eat_u8_negation(ord("\n"))), eat_u8(ord("\n"))]),
        // ]))
        let ignore = rep(choice![
            eat_u8(b' '),
            seq![
                eat_u8(b'#'),
                rep(eat_u8_negation_class(b'\n')), // Any char except newline
                eat_u8(b'\n')
            ]
        ]);

        // --- Define core token expressions (without ignore prefix) ---
        let name_expr = seq![name_start, rep(name_middle)];
        let number_expr = choice![
            rep1(digit.clone()), // Use rep1 to match at least one digit, common in tokenizers
            seq![rep1(digit.clone()), eat_u8(b'.'), rep(digit.clone())], // Handle floats like 1., .1, 1.1
            seq![eat_u8(b'.'), rep1(digit.clone())]
        ];
        let newline_expr = eps(); // Represented by Epsilon in the Python code
        let indent_expr = eps();
        let dedent_expr = eps();
        let string_expr = choice![
            seq![
                eat_u8(b'"'),
                rep(eat_u8_negation_class(b'"')),
                eat_u8(b'"')
            ],
            seq![
                eat_u8(b'\''),
                rep(eat_u8_negation_class(b'\'')),
                eat_u8(b'\'')
            ],
        ];
        // Note: FSTRING parts are complex and might involve stateful parsing beyond simple regex.
        // We'll approximate them based on the Python code's regex structure.
        let fstring_start_expr = choice![eat_string("\"\"\""), eat_string("'''")];
        let fstring_end_expr = choice![eat_string("\"\"\""), eat_string("'''")];
        // FSTRING_MIDDLE = rep(choice([eat_u8_negation(ord("{")), eat_string("{{")]))
        let fstring_middle_expr = rep(choice![
            eat_u8_negation_class(b'{'),
            eat_string("{{")
        ]);
        let type_comment_expr = eps();
        let endmarker_expr = eps();

        // --- Combine core expressions with 'ignore' prefix ---
        // Map token names to GroupIDs for clarity in tests
        let mut token_map: BTreeMap<&str, GroupID> = BTreeMap::new();
        let mut token_exprs: Vec<ExprGroup> = Vec::new();

        let mut add_token = |name: &'static str, expr: Expr| {
            let group_id = token_exprs.len();
            token_map.insert(name, group_id);
            // Prepend the ignore pattern. Clone `ignore` as it's used multiple times.
            token_exprs.push(greedy_group(seq![ignore.clone(), expr]));
        };

        add_token("NAME", name_expr); // Group 0
        add_token("NUMBER", number_expr); // Group 1
        add_token("NEWLINE", newline_expr); // Group 2
        add_token("INDENT", indent_expr); // Group 3
        add_token("DEDENT", dedent_expr); // Group 4
        add_token("STRING", string_expr); // Group 5
        add_token("FSTRING_START", fstring_start_expr); // Group 6
        add_token("FSTRING_END", fstring_end_expr); // Group 7
        add_token("FSTRING_MIDDLE", fstring_middle_expr); // Group 8
        add_token("TYPE_COMMENT", type_comment_expr); // Group 9
        add_token("ENDMARKER", endmarker_expr); // Group 10

        // Add common operators/delimiters as separate tokens if needed for a full tokenizer
        // Example:
        // add_token("PLUS", eat_u8(b'+'));
        // add_token("EQUALS", eat_u8(b'='));
        // ... etc.

        // --- Build the Regex ---
        let expr_groups = groups(token_exprs);
        let regex = expr_groups.build();

        println!("Built Python Token Regex DFA:");
        // dbg!(®ex); // Print the DFA structure (can be very large)
        println!("Number of DFA states: {}", regex.dfa.states.len());

        // --- Test Cases ---
        let name_id = *token_map.get("NAME").unwrap();
        let number_id = *token_map.get("NUMBER").unwrap();
        let string_id = *token_map.get("STRING").unwrap();
        // Epsilon tokens like NEWLINE, INDENT, DEDENT, ENDMARKER are tricky to test
        // in isolation with execute(), as they match at position 0.
        // greedy_find_all is better suited for token streams.

        // Test 1: Simple name
        let mut state1 = regex.init();
        state1.execute(b"my_var");
        let m1 = state1.get_greedy_match();
        assert_eq!(m1, Some(Match { group_id: name_id, position: 6 }));
        assert!(state1.fully_matches_here()); // Should fully match the input

        // Test 2: Name with leading space
        let mut state2 = regex.init();
        state2.execute(b"  anotherVar");
        let m2 = state2.get_greedy_match();
        // Position includes the ignored spaces
        assert_eq!(m2, Some(Match { group_id: name_id, position: 12 }));
        assert!(state2.fully_matches_here());

        // Test 3: Simple number
        let mut state3 = regex.init();
        state3.execute(b"12345");
        let m3 = state3.get_greedy_match();
        assert_eq!(m3, Some(Match { group_id: number_id, position: 5 }));
        assert!(state3.fully_matches_here());

        // Test 4: Float number
        let mut state4 = regex.init();
        state4.execute(b" 3.14");
        let m4 = state4.get_greedy_match();
        assert_eq!(m4, Some(Match { group_id: number_id, position: 5 }));
        assert!(state4.fully_matches_here());

        // Test 5: Number starting with dot
        let mut state5 = regex.init();
        state5.execute(b".5");
        let m5 = state5.get_greedy_match();
        assert_eq!(m5, Some(Match { group_id: number_id, position: 2 }));
        assert!(state5.fully_matches_here());


        // Test 6: Simple string
        let mut state6 = regex.init();
        state6.execute(b"\"hello\"");
        let m6 = state6.get_greedy_match();
        assert_eq!(m6, Some(Match { group_id: string_id, position: 7 }));
        assert!(state6.fully_matches_here());

        // Test 7: String with leading comment
        let mut state7 = regex.init();
        state7.execute(b"# comment\n 'world'");
        let m7 = state7.get_greedy_match();
        assert_eq!(m7, Some(Match { group_id: string_id, position: 18 })); // Includes comment, newline, space
        assert!(state7.fully_matches_here());

        // Test 8: Incomplete input (should match prefix if possible)
        let mut state8 = regex.init();
        state8.execute(b" my_v"); // Incomplete name
        let m8 = state8.get_greedy_match();
        // It matches the space (via ignore) but not the partial name as a full token.
        // The behavior here depends on whether ignore itself is a token or just prefix.
        // In this setup, ignore isn't a token, so no match is expected unless an
        // epsilon token matches at pos 0. Let's check possible groups.
        // Depending on DFA minimization, it might match an epsilon token if available at pos 0.
        // Let's refine the test - does it *fail* correctly?
        assert!(!state8.definitely_fully_matches()); // It didn't fully match a token
        assert!(state8.could_match()); // It could potentially match NAME if more input arrived
        assert!(!state8.done()); // Not done, expects more input for NAME

        // Test 9: Invalid input
        let mut state9 = regex.init();
        state9.execute(b" $invalid");
        let m9 = state9.get_greedy_match();
         // Might match epsilon token at pos 0 or space via ignore, but not '$'
        assert!(!state9.definitely_fully_matches());
        assert!(!state9.could_match()); // Cannot match further after '$'
        assert!(state9.done()); // Execution stopped at '$'

        // Test 10: Sequence using greedy_find_all (simulating tokenization)
        let mut state10 = regex.init();
        let text10 = b"var1 = 10 # Assign\n print(var1)";
        let matches10 = state10.greedy_find_all(text10, true); // terminate=true

        println!("Tokenizing: {:?}", String::from_utf8_lossy(text10));
        println!("Matches: {:?}", matches10);

        // Expected sequence (approximate, depends on exact token definitions like operators):
        // NAME("var1"), ?, NUMBER("10"), ?, NAME("print"), ?, NAME("var1"), ?
        // The positions will be cumulative in the original string.
        // Let's check the first few:
        assert!(matches10.len() > 3); // Should find multiple tokens
        // Match 1: "var1" (NAME)
        assert_eq!(matches10[0], Match { group_id: name_id, position: 4 }); // "var1"
        // Match 2: " = " (Assuming '=' and surrounding spaces are handled, maybe by ignore or specific tokens)
        //            Let's assume '=' is not a token here, so ignore consumes " = "
        // Match 3: "10" (NUMBER) - position will be after "var1 = "
        //            The exact position depends on how greedy_find_all resets and consumes.
        //            Let's verify the token IDs found.
        let found_ids: Vec<GroupID> = matches10.iter().map(|m| m.group_id).collect();
        assert!(found_ids.contains(&name_id));
        assert!(found_ids.contains(&number_id));
        // The exact sequence and positions require adding operator tokens and refining ignore/whitespace handling.
        // This basic test confirms multiple tokens can be found.

        // Test 11: FString parts (basic check)
        let fstring_start_id = *token_map.get("FSTRING_START").unwrap();
        let fstring_middle_id = *token_map.get("FSTRING_MIDDLE").unwrap();
        let fstring_end_id = *token_map.get("FSTRING_END").unwrap();

        let mut state11_1 = regex.init();
        state11_1.execute(b"\"\"\"");
        assert_eq!(state11_1.get_greedy_match(), Some(Match { group_id: fstring_start_id, position: 3 }));

        let mut state11_2 = regex.init();
        state11_2.execute(b"some text {{escaped}}");
         assert_eq!(state11_2.get_greedy_match(), Some(Match { group_id: fstring_middle_id, position: 21 }));

        let mut state11_3 = regex.init();
        state11_3.execute(b"'''");
        assert_eq!(state11_3.get_greedy_match(), Some(Match { group_id: fstring_end_id, position: 3 }));

    }
}