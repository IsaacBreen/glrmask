use std::collections::BTreeMap;
use std::sync::Arc;

use sep1::constraint::{GrammarConstraint, GrammarConstraintConfig};
use sep1::interface::GrammarDefinition;
use sep1::dfa_u8::LLMTokenID;

fn birds_schema_lark() -> &'static str {
    r##"
start: ws object ws
object: "{" ws pairs_0 ws "}"
pairs_0: id_pair ws "," ws pairs_1
pairs_1: name_pair ws "," ws pairs_2
pairs_2: family_pair ws "," ws pairs_3
pairs_3: continents_pair ws "," ws pairs_4
pairs_4: added_pair ws "," ws pairs_5
pairs_5: visible_pair
id_pair: QUOTE "id" QUOTE ws ":" ws id_val
name_pair: QUOTE "name" QUOTE ws ":" ws name_val
family_pair: QUOTE "family" QUOTE ws ":" ws family_val
continents_pair: QUOTE "continents" QUOTE ws ":" ws continents_val
added_pair: QUOTE "added" QUOTE ws ":" ws added_val
visible_pair: QUOTE "visible" QUOTE ws ":" ws visible_val
id_val: QUOTE id_chars QUOTE
id_chars: STR_CHAR*
name_val: QUOTE name_chars QUOTE
name_chars: STR_CHAR*
family_val: QUOTE family_chars QUOTE
family_chars: STR_CHAR*
continents_item_val: QUOTE continents_item_chars QUOTE
continents_item_chars: STR_CHAR*
continents_val: "[" ws continents_item_val (ws "," ws continents_item_val)* ws "]"
added_val: QUOTE added_chars QUOTE
added_chars: STR_CHAR*
visible_val: BOOL
QUOTE: "\""
ws: WS*
WS: " " | "\n" | "\t" | "\r"
BOOL: "true" | "false"
STR_CHAR: " " | "!" | "#" | "$" | "%" | "&" | "'" | "(" | ")" | "*" | "+" | "," | "-" | "." | "/" | "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | ":" | ";" | "<" | "=" | ">" | "?" | "@" | "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z" | "[" | "]" | "^" | "_" | "`" | "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z" | "{" | "|" | "}" | "~"
"##
}

fn tiny_vocab() -> (BTreeMap<Vec<u8>, LLMTokenID>, usize) {
    let mut map = BTreeMap::new();
    let tokens = vec![
        (0usize, b"{".to_vec()),
        (1usize, b"\"".to_vec()),
        (2usize, b"id".to_vec()),
        (3usize, b":".to_vec()),
        (4usize, b"b".to_vec()),
        (5usize, b"}".to_vec()),
    ];
    for (id, bytes) in tokens {
        map.insert(bytes, LLMTokenID(id));
    }
    (map, 5)
}


#[test]
fn test_birds_is_complete_mid_string() {
    let grammar_definition = GrammarDefinition::from_lark(birds_schema_lark())
        .expect("Failed to parse birds schema_lark");

    let (llm_token_map, max_id) = tiny_vocab();
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_id,
        &GrammarConstraintConfig::default(),
    );
    let eos_id = constraint.eos_token_id;
    eprintln!("eos_token_id={:?}", eos_id);
    let mut state = constraint.init();

    for token in [0usize, 1, 2, 1, 3, 1, 4] {
        state.commit(LLMTokenID(token)).expect("commit failed");
    }

    assert!(!state.is_complete(), "should not be complete inside string value");
}

