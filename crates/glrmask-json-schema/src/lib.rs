#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub type Error = glrmask_grammar::Error;
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) type GlrMaskError = Error;

pub(crate) mod grammar {
    pub(crate) use glrmask_grammar::__private::grammar::*;
}

pub(crate) mod automata {
    pub(crate) use glrmask_lexer::__private::automata::lexer;
}

pub(crate) mod import {
    pub(crate) use glrmask_grammar::__private::grammar::ast;
    pub(crate) mod numeric_range;
}

mod json_schema;

pub use json_schema::{
    prepare_named_grammar, prepare_named_grammar_for_dump, schema_to_named_grammar,
    schema_to_named_grammar_with_dynamic_value_token,
    schema_to_named_grammar_with_programmatic_value_token,
    schema_to_named_grammar_with_programmatic_value_tokens,
};

/// Implementation details shared with the `glrmask` facade.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod ast {
        pub use crate::json_schema::ast::*;
    }
    pub mod config {
        pub use crate::json_schema::config::*;
    }
    pub mod load {
        pub use crate::json_schema::load::*;
    }
    pub mod lower {
        pub use crate::json_schema::lower::*;
    }
    pub mod preflight {
        pub use crate::json_schema::preflight::*;
    }
    pub mod string {
        pub use crate::json_schema::string::*;
    }

    pub use crate::json_schema::{
        GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS_ENV, finalize_lexer_partitions,
        lower_exact_subtractions_enabled, split_literal_terminals_enabled,
        swap_split_literal_terminals_test_override,
    };

    pub fn set_test_compat_mode(enabled: bool) {
        crate::json_schema::string::TEST_COMPAT_MODE.with(|cell| {
            cell.set(if enabled {
                crate::json_schema::string::JsonStringCompatMode::LlGuidanceNative
            } else {
                crate::json_schema::string::JsonStringCompatMode::JsonSchema
            });
        });
    }
}
