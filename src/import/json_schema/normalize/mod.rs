//! JSON Schema semantic normalization helpers.
//!
//! This module should contain schema-level algebra.  It is allowed to inspect
//! `schema::*` values and to ask the lowerer to lower branches, but its public
//! contract is semantic: preserve or explicitly document the relationship
//! between schema denotations before and after a rewrite.

pub(crate) mod combinators;

pub(crate) use self::combinators::{
    all_of_schema,
    open_object_any_of_covers_json_object,
    try_merge_all_of_objects,
};
