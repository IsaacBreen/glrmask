use sep1::interface::{
    choice, choice_fast, eat_bytestring_fast, eat_u8_fast, eat_u8_range_fast, r#ref, repeat,
    repeat0_fast, repeat1_fast, seq_fast, sequence, GrammarDefinition,
};

fn main() {
    // Define terminals using tokenizer combinators (Expr)
    let terminals = vec![
        (
            "NUMBER".to_string(),
            repeat1_fast(eat_u8_range_fast(b'0', b'9')),
        ),
        (
            "IDENT".to_string(),
            seq_fast(vec![
                eat_u8_range_fast(b'a', b'z'),
                repeat0_fast(choice_fast(vec![
                    eat_u8_range_fast(b'a', b'z'),
                    eat_u8_range_fast(b'0', b'9'),
                    eat_u8_fast(b'_'),
                ])),
            ]),
        ),
        ("PLUS".to_string(), eat_u8_fast(b'+')),
        ("MINUS".to_string(), eat_u8_fast(b'-')),
        ("LPAREN".to_string(), eat_u8_fast(b'(')),
        ("RPAREN".to_string(), eat_u8_fast(b')')),
        ("IF".to_string(), eat_bytestring_fast(b"if".to_vec())),
        ("ELSE".to_string(), eat_bytestring_fast(b"else".to_vec())),
    ];

    // Define grammar rules using GrammarExpr
    // Expr -> Term { (+|-) Term }
    // Term -> Factor
    // Factor -> NUMBER | IDENT | ( Expr ) | IF Expr ELSE Expr
    
    let expr_rule = sequence(vec![
        r#ref("Term"),
        repeat(sequence(vec![
            choice(vec![r#ref("PLUS"), r#ref("MINUS")]),
            r#ref("Term"),
        ])),
    ]);

    let term_rule = r#ref("Factor");

    let factor_rule = choice(vec![
        r#ref("NUMBER"),
        r#ref("IDENT"),
        sequence(vec![
            r#ref("LPAREN"),
            r#ref("Expr"),
            r#ref("RPAREN"),
        ]),
        sequence(vec![
            r#ref("IF"),
            r#ref("Expr"),
            r#ref("ELSE"),
            r#ref("Expr"),
        ]),
    ]);

    let rules = vec![
        ("Expr".to_string(), expr_rule),
        ("Term".to_string(), term_rule),
        ("Factor".to_string(), factor_rule),
    ];

    println!("Building grammar...");
    let grammar_def = GrammarDefinition::from_exprs(rules, terminals).expect("Failed to create grammar definition");

    println!("\n{}", grammar_def);
}
