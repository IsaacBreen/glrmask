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
        let mut set = U8Set::all();
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
        let whitespace_char = choice![eat_u8(b' '), eat_u8(b'\t')];
        let comment = seq![
            eat_u8(b'#'),
            rep(eat_u8_negation_class(b'\n')),
            opt(eat_u8(b'\n'))
        ];
        let ignore = rep(choice![whitespace_char, comment]);


        // --- Define core token expressions (without ignore prefix) ---
        let name_expr = seq![name_start, rep(name_middle)];
        let number_expr = choice![
            rep1(digit.clone()),
            seq![rep1(digit.clone()), eat_u8(b'.'), rep(digit.clone())],
            seq![eat_u8(b'.'), rep1(digit.clone())]
        ];
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
        let fstring_start_expr = choice![eat_string("\"\"\""), eat_string("'''")];
        let fstring_end_expr = choice![eat_string("\"\"\""), eat_string("'''")];
        let fstring_middle_expr = rep(choice![
            eat_u8_negation_class(b'{'),
            eat_string("{{")
        ]);

        // --- Combine core expressions with 'ignore' prefix ---
        let mut token_map: BTreeMap<&str, GroupID> = BTreeMap::new();
        let mut token_exprs: Vec<ExprGroup> = Vec::new();
        let mut current_group_id = 0;

        let mut add_token = |name: &'static str, expr: Expr, map: &mut BTreeMap<&str, GroupID>, expr_list: &mut Vec<ExprGroup>, id_counter: &mut usize| {
            map.insert(name, *id_counter);
            expr_list.push(greedy_group(seq![ignore.clone(), expr]));
            *id_counter += 1;
        };

        add_token("NAME", name_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 0
        add_token("NUMBER", number_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 1
        add_token("STRING", string_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 2
        add_token("FSTRING_START", fstring_start_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 3
        add_token("FSTRING_END", fstring_end_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 4
        add_token("FSTRING_MIDDLE", fstring_middle_expr, &mut token_map, &mut token_exprs, &mut current_group_id); // Group 5

        // --- Build the Regex ---
        let expr_groups = groups(token_exprs);
        let regex = expr_groups.build();

        println!("Built Python Token Regex DFA (No Epsilon Tokens, Direct Match Check):");
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
        let match_pos = state1.matches.get(&name_id).copied();
        assert_eq!(match_pos, Some(6), "Test 1 Failed - NAME");
        assert!(state1.fully_matches_here(), "Test 1 Not Fully Matched");

        // Test 2: Name with leading space
        let mut state2 = regex.init();
        state2.execute(b"  anotherVar");
        let match_pos = state2.matches.get(&name_id).copied();
        assert_eq!(match_pos, Some(12), "Test 2 Failed - NAME");
        assert!(state2.fully_matches_here(), "Test 2 Not Fully Matched");

        // Test 3: Simple number
        let mut state3 = regex.init();
        state3.execute(b"12345");
        let match_pos = state3.matches.get(&number_id).copied();
        assert_eq!(match_pos, Some(5), "Test 3 Failed - NUMBER");
        assert!(state3.fully_matches_here(), "Test 3 Not Fully Matched");

        // Test 4: Float number
        let mut state4 = regex.init();
        state4.execute(b" 3.14");
        let match_pos = state4.matches.get(&number_id).copied();
        assert_eq!(match_pos, Some(5), "Test 4 Failed - NUMBER");
        assert!(state4.fully_matches_here(), "Test 4 Not Fully Matched");

        // Test 5: Number starting with dot
        let mut state5 = regex.init();
        state5.execute(b".5");
        let match_pos = state5.matches.get(&number_id).copied();
        assert_eq!(match_pos, Some(2), "Test 5 Failed - NUMBER");
        assert!(state5.fully_matches_here(), "Test 5 Not Fully Matched");


        // Test 6: Simple string
        let mut state6 = regex.init();
        state6.execute(b"\"hello\"");
        // *** Check STRING token directly ***
        let match_pos = state6.matches.get(&string_id).copied();
        assert_eq!(match_pos, Some(7), "Test 6 Failed - STRING");
        // Check if the problematic FSTRING_MIDDLE also matched (it shouldn't be preferred, but might exist)
        let fstring_middle_match = state6.matches.get(&fstring_middle_id).copied();
        println!("Test 6 Matches: {:?}", state6.matches); // Debug print matches
        // We expect STRING to be the only *relevant* match here for full input.
        // If FSTRING_MIDDLE matched at pos 0, it's okay as long as STRING matched at 7.
        assert!(state6.fully_matches_here(), "Test 6 Not Fully Matched");


        // Test 7: String with leading comment
        let mut state7 = regex.init();
        state7.execute(b"# comment\n 'world'");
        let match_pos = state7.matches.get(&string_id).copied();
        assert_eq!(match_pos, Some(18), "Test 7 Failed - STRING");
        assert!(state7.fully_matches_here(), "Test 7 Not Fully Matched");

        // Test 8: Incomplete input
        let mut state8 = regex.init();
        state8.execute(b" my_v");
        // Assert that NAME was *not* fully matched
        let match_pos = state8.matches.get(&name_id).copied();
        assert_eq!(match_pos, None, "Test 8 Failed - NAME should not match");
        assert!(!state8.definitely_fully_matches(), "Test 8 Not Def Fully Matched");
        assert!(state8.could_match(), "Test 8 Not Could Match"); // Could still match NAME
        assert!(!state8.done(), "Test 8 Is Done"); // Not done, expects more input for NAME

        // Test 9: Invalid input
        let mut state9 = regex.init();
        state9.execute(b" $invalid");
        // Assert that no token we defined was matched after the space
        assert!(state9.matches.is_empty(), "Test 9 Failed - Expected no matches");
        assert!(!state9.definitely_fully_matches(), "Test 9 Not Def Fully Matched");
        assert!(!state9.could_match(), "Test 9 Not Could Match"); // Cannot match further after '$'
        assert!(state9.done(), "Test 9 Not Done"); // Execution stopped at '$'

        // Test 10: Sequence using greedy_find_all (simulating tokenization)
        // This test still relies on greedy matching logic within greedy_find_all
        let mut state10 = regex.init();
        let text10 = b"var1 = 10 # Assign\n print(var1)"; // Need '=' token etc.
        let matches10 = state10.greedy_find_all(text10, true); // terminate=true

        println!("Tokenizing (No Epsilon): {:?}", String::from_utf8_lossy(text10));
        println!("Matches (No Epsilon): {:?}", matches10);
        assert!(!matches10.is_empty(), "Test 10 Failed - No matches");
        assert_eq!(matches10[0], Match { group_id: name_id, position: 4 }, "Test 10 Failed - First match");


        // Test 11: FString parts
        let mut state11_1 = regex.init();
        state11_1.execute(b"\"\"\"");
        let match_pos = state11_1.matches.get(&fstring_start_id).copied();
        assert_eq!(match_pos, Some(3), "Test 11_1 Failed - FSTRING_START");

        let mut state11_2 = regex.init();
        state11_2.execute(b"some text {{escaped}}");
        let match_pos = state11_2.matches.get(&fstring_middle_id).copied();
        assert_eq!(match_pos, Some(21), "Test 11_2 Failed - FSTRING_MIDDLE");

        let mut state11_3 = regex.init();
        state11_3.execute(b"'''");
        let start_match_pos = state11_3.matches.get(&fstring_start_id).copied();
        let end_match_pos = state11_3.matches.get(&fstring_end_id).copied();
        // Check if either START or END matched at position 3
        assert!(
            start_match_pos == Some(3) || end_match_pos == Some(3),
             "Test 11_3 Failed - Expected START or END at pos 3, got START={:?}, END={:?}", start_match_pos, end_match_pos
        );
    }
}