#[cfg(test)]
mod tests {
    use crate::glr::analyze::resolve_right_recursion;
    use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
    use std::collections::BTreeSet;

    fn create_nt(name: &str) -> Symbol {
        Symbol::NonTerminal(NonTerminal(name.to_string()))
    }

    fn create_term(name: &str) -> Symbol {
        Symbol::Terminal(Terminal::Literal(name.as_bytes().to_vec()))
    }

    #[test]
    fn test_repro_yield_expression_cycle() {
        // expression ::= assignment_expression
        // assignment_expression ::= yield_expression | conditional_expression
        // yield_expression ::= 'yield' expression_opt
        // expression_opt ::= expression | ε
        // conditional_expression ::= 'x'

        let mut productions = vec![
            Production {
                lhs: NonTerminal("expression".to_string()),
                rhs: vec![create_nt("assignment_expression")],
            },
            Production {
                lhs: NonTerminal("assignment_expression".to_string()),
                rhs: vec![create_nt("yield_expression")],
            },
            Production {
                lhs: NonTerminal("assignment_expression".to_string()),
                rhs: vec![create_nt("conditional_expression")],
            },
            Production {
                lhs: NonTerminal("yield_expression".to_string()),
                rhs: vec![create_term("yield"), create_nt("expression_opt")],
            },
            Production {
                lhs: NonTerminal("expression_opt".to_string()),
                rhs: vec![create_nt("expression")],
            },
            Production {
                lhs: NonTerminal("expression_opt".to_string()),
                rhs: vec![],
            },
            Production {
                lhs: NonTerminal("conditional_expression".to_string()),
                rhs: vec![create_term("x")],
            },
        ];

        let mut name_counter = 0;
        let mut name_gen = |base: &str| {
            name_counter += 1;
            format!("{}_{}", base, name_counter)
        };

        resolve_right_recursion(&mut productions, &mut name_gen);

        // Check that right recursion is eliminated.
        // We can check this by running the analysis again and seeing if any right-recursive NTs remain.
        // Or just check that the cycle is broken.

        // For now, let's just print the productions to see what happened
        for p in &productions {
            println!("{}", p);
        }

        // We can reuse the logic from resolve_right_recursion to check for remaining recursion
        // But since that function is what we're testing, we assume if it returns, it thinks it's done.
        // We should verify it didn't give up early.
    }
}
