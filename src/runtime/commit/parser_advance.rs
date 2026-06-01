//! Parser-stack transition dispatch for Commit.
//!
//! This module is the boundary between Commit and parser semantics.  It chooses
//! between the template-DFA stack-effect recognizer and the reference GLR-table
//! advance, while preserving the same mathematical transition relation.

use crate::parser::glr::advance::{
    advance_stacks,
    advance_stacks_owned,
    advance_stacks_profiled,
    AdvanceProfile,
    ParserGSS,
};
use crate::runtime::constraint::Constraint;

use super::options::{template_advance_enabled, validate_template_advance_enabled};
use crate::runtime::template_dfa::advance::{
    advance_stacks_template_dfa,
    advance_stacks_template_dfa_owned,
};

pub(super) fn advance_parser_stacks(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if template_advance_enabled()
        && let Some(template_advanced) = advance_stacks_template_dfa(constraint, stack, terminal)
    {
        if validate_template_advance_enabled() {
            let table_advanced = advance_stacks(&constraint.table, stack, terminal);
            assert_eq!(
                template_advanced,
                table_advanced,
                "template-DFA advance mismatch for terminal {terminal}"
            );
        }
        return template_advanced;
    }

    advance_stacks(&constraint.table, stack, terminal)
}

pub(super) fn advance_parser_stacks_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if template_advance_enabled()
        && let Some(template_advanced) =
            advance_stacks_template_dfa_owned(constraint, stack.clone(), terminal)
    {
        if validate_template_advance_enabled() {
            let table_advanced = advance_stacks_owned(&constraint.table, stack, terminal);
            assert_eq!(
                template_advanced,
                table_advanced,
                "template-DFA advance mismatch for terminal {terminal}"
            );
        }
        return template_advanced;
    }

    advance_stacks_owned(&constraint.table, stack, terminal)
}

pub(super) fn advance_parser_stacks_profiled(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> (ParserGSS, AdvanceProfile) {
    if template_advance_enabled()
        && let Some(template_advanced) = advance_stacks_template_dfa(constraint, stack, terminal)
    {
        if validate_template_advance_enabled() {
            let (table_advanced, table_profile) =
                advance_stacks_profiled(&constraint.table, stack, terminal);
            assert_eq!(
                template_advanced,
                table_advanced,
                "template-DFA advance mismatch for terminal {terminal}"
            );
            return (template_advanced, table_profile);
        }
        return (template_advanced, AdvanceProfile::default());
    }

    advance_stacks_profiled(&constraint.table, stack, terminal)
}

