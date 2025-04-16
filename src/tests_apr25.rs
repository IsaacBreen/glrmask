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
        // Let's refine ignore slightly: allow spaces, tabs, and comments.
        // A real Python tokenizer also handles line continuations (\) and form feeds.
        let whitespace_char = choice![eat_u8(b' '), eat_u8(b'\t')];
        let comment = seq![
            eat_u8(b'#'),
            rep(eat_u8_negation_class(b'\n')),
            // Optionally consume the newline ending the comment, or let a NEWLINE token handle it
            // For this regex, let's consume it as part of ignore.
            opt(eat_u8(b'\n'))
        ];
        // Ignore pattern: repetition of whitespace or comments
        let ignore = rep(choice![whitespace_char, comment]);


        // --- Define core token expressions (without ignore prefix) ---
        let name_expr = seq![name_start, rep(name_middle)];
        let number_expr = choice![
            // Integer
            rep1(digit.clone()),
            // Float variants
            seq![rep1(digit.clone()), eat_u8(b'.'), rep(digit.clone())], // 123. or 123.45
            seq![eat_u8(b'.'), rep1(digit.clone())]                     // .45
            // Exponent notation could be added here: e.g., seq![..., choice!['e', 'E'], opt(choice!['+', '-']), rep1(digit)]
        ];
        // Epsilon tokens removed from this regex compilation
        // let newline_expr = eps();
        // let indent_expr = eps();
        // let dedent_expr = eps();
        let string_expr = choice![
            seq![
                eat_u8(b'"'),
                rep(eat_u8_negation_class(b'"')), // Simplified: doesn't handle escapes like \"
                eat_u8(b'"')
            ],
            seq![
                eat_u8(b'\''),
                rep(eat_u8_negation_class(b'\'')), // Simplified: doesn't handle escapes like \'
                eat_u8(b'\'')
            ],
            // Triple-quoted strings could be added here
        ];
        let fstring_start_expr = choice![eat_string("\"\"\""), eat_string("'''")]; // Keep these as they consume chars
        let fstring_end_expr = choice![eat_string("\"\"\""), eat_string("'''")]; // Keep these
        let fstring_middle_expr = rep(choice![
            eat_u8_negation_class(b'{'), // Any char except {
            eat_string("{{")             // Escaped {{
        ]);
        // let type_comment_expr = eps();
        // let endmarker_expr = eps();

        // --- Combine core expressions with 'ignore' prefix ---
        let mut token_map: BTreeMap<&str, GroupID> = BTreeMap::new();
        let mut token_exprs: Vec<ExprGroup> = Vec::new();
        let mut current_group_id = 0; // Manual tracking after removing epsilons

        // Helper closure to add tokens and manage group IDs
        let mut add_token = |name: &'static str, expr: Expr, map: &mut BTreeMap<&str, GroupID>, expr_list: &mut Vec<ExprGroup>, id_counter: &mut usize| {
            map.insert(name, *id_counter);
            // Prepend the ignore pattern. Clone `ignore` as it's used multiple times.
            expr_list.push(greedy_group(seq![ignore.clone(), expr]));
            *id_counter += 1;
        };

        add_token("NAME", name_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 0
        add_token("NUMBER", number_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 1
        add_token("STRING", string_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 2
        add_token("FSTRING_START", fstring_start_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 3
        add_token("FSTRING_END", fstring_end_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 4
        add_token("FSTRING_MIDDLE", fstring_middle_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 5

        // Add common operators/delimiters if needed for more realistic tokenization
        // add_token("PLUS", eat_u8(b'+'), &mut token_map, &mut token_exprs, &mut current_group_id);
        // add_token("EQUALS", eat_u8(b'='), &mut token_map, &mut token_exprs, &mut current_group_id);
        // ... etc.

        // --- Build the Regex ---
        let expr_groups = groups(token_exprs);
        let regex = expr_groups.build();

        println!("Built Python Token Regex DFA (No Epsilon Tokens):");
        // dbg!(®ex); // Print the DFA structure (can be very large)
        println!("Number of DFA states: {}", regex.dfa.states.len());

        // --- Test Cases ---
        let name_id = *token_map.get("NAME").unwrap();
        let number_id = *token_map.get("NUMBER").unwrap();
        let string_id = *token_map.get("STRING").unwrap();
        let fstring_start_id = *token_map.get("FSTRING_START").unwrap();
        let fstring_middle_id = *token_map.get("FSTRING_MIDDLE").unwrap();
        let fstring_end_id = *token_map.get("FSTRING_END").unwrap();


        // Test 1: Simple name
        let mut state1 = regex.init();
        state1.execute(b"my_var");
        let m1 = state1.get_greedy_match();
        assert_eq!(m1, Some(Match { group_id: name_id, position: 6 }), "Test 1 Failed");
        assert!(state1.fully_matches_here(), "Test 1 Not Fully Matched");

        // Test 2: Name with leading space
        let mut state2 = regex.init();
        state2.execute(b"  anotherVar");
        let m2 = state2.get_greedy_match();
        assert_eq!(m2, Some(Match { group_id: name_id, position: 12 }), "Test 2 Failed");
        assert!(state2.fully_matches_here(), "Test 2 Not Fully Matched");

        // Test 3: Simple number
        let mut state3 = regex.init();
        state3.execute(b"12345");
        let m3 = state3.get_greedy_match();
        assert_eq!(m3, Some(Match { group_id: number_id, position: 5 }), "Test 3 Failed");
        assert!(state3.fully_matches_here(), "Test 3 Not Fully Matched");

        // Test 4: Float number
        let mut state4 = regex.init();
        state4.execute(b" 3.14");
        let m4 = state4.get_greedy_match();
        assert_eq!(m4, Some(Match { group_id: number_id, position: 5 }), "Test 4 Failed");
        assert!(state4.fully_matches_here(), "Test 4 Not Fully Matched");

        // Test 5: Number starting with dot
        let mut state5 = regex.init();
        state5.execute(b".5");
        let m5 = state5.get_greedy_match();
        assert_eq!(m5, Some(Match { group_id: number_id, position: 2 }), "Test 5 Failed");
        assert!(state5.fully_matches_here(), "Test 5 Not Fully Matched");


        // Test 6: Simple string
        let mut state6 = regex.init();
        state6.execute(b"\"hello\"");
        let m6 = state6.get_greedy_match();
        // *** This was the failing test ***
        assert_eq!(m6, Some(Match { group_id: string_id, position: 7 }), "Test 6 Failed");
        assert!(state6.fully_matches_here(), "Test 6 Not Fully Matched");

        // Test 7: String with leading comment
        let mut state7 = regex.init();
        // Use the updated ignore pattern which consumes the newline after comment
        state7.execute(b"# comment\n 'world'");
        let m7 = state7.get_greedy_match();
        assert_eq!(m7, Some(Match { group_id: string_id, position: 18 }), "Test 7 Failed");
        assert!(state7.fully_matches_here(), "Test 7 Not Fully Matched");

        // Test 8: Incomplete input
        let mut state8 = regex.init();
        state8.execute(b" my_v");
        let m8 = state8.get_greedy_match();
        // Ignore consumes the space " ". No full token matches.
        assert_eq!(m8, None, "Test 8 Failed - Expected None");
        assert!(!state8.definitely_fully_matches(), "Test 8 Not Def Fully Matched");
        assert!(state8.could_match(), "Test 8 Not Could Match"); // Could still match NAME
        assert!(!state8.done(), "Test 8 Is Done"); // Not done, expects more input for NAME

        // Test 9: Invalid input
        let mut state9 = regex.init();
        state9.execute(b" $invalid");
        let m9 = state9.get_greedy_match();
        // Ignore consumes the space " ". No token matches '$'.
        assert_eq!(m9, None, "Test 9 Failed - Expected None");
        assert!(!state9.definitely_fully_matches(), "Test 9 Not Def Fully Matched");
        assert!(!state9.could_match(), "Test 9 Not Could Match"); // Cannot match further after '$'
        assert!(state9.done(), "Test 9 Not Done"); // Execution stopped at '$'

        // Test 10: Sequence using greedy_find_all (simulating tokenization)
        // Requires adding operator tokens like '=' for a meaningful test.
        // Let's skip the complex assertion for now but run it to see output.
        let mut state10 = regex.init();
        let text10 = b"var1 = 10 # Assign\n print(var1)"; // Need '=' token etc.
        let matches10 = state10.greedy_find_all(text10, true); // terminate=true

        println!("Tokenizing (No Epsilon): {:?}", String::from_utf8_lossy(text10));
        println!("Matches (No Epsilon): {:?}", matches10);
        // Basic check: Found some matches, first is var1
        assert!(!matches10.is_empty(), "Test 10 Failed - No matches");
        assert_eq!(matches10[0], Match { group_id: name_id, position: 4 }, "Test 10 Failed - First match");


        // Test 11: FString parts
        let mut state11_1 = regex.init();
        state11_1.execute(b"\"\"\"");
        assert_eq!(state11_1.get_greedy_match(), Some(Match { group_id: fstring_start_id, position: 3 }), "Test 11_1 Failed");

        let mut state11_2 = regex.init();
        // FSTRING_MIDDLE allows empty match via rep(). Test non-empty.
        state11_2.execute(b"some text {{escaped}}");
        assert_eq!(state11_2.get_greedy_match(), Some(Match { group_id: fstring_middle_id, position: 21 }), "Test 11_2 Failed");

        let mut state11_3 = regex.init();
        state11_3.execute(b"'''");
        let m11_3 = state11_3.get_greedy_match();
        // FSTRING_START and FSTRING_END have identical patterns. The DFA might map ' to the first one defined (START).
        // Accept either START or END match here.
        assert!(
            m11_3 == Some(Match { group_id: fstring_start_id, position: 3 }) ||
            m11_3 == Some(Match { group_id: fstring_end_id, position: 3 }),
             "Test 11_3 Failed - Expected START or END, got {:?}", m11_3
        );
    }
}