//! Shape predicates over loaded schema nodes.
//!
//! These helpers answer questions about schema syntax and assertion shapes.
//! They do not read raw JSON object keys and do not construct grammar IR.

use super::super::schema::{Schema, SchemaAssertions, SchemaKind, SchemaType};

pub(super) fn singleton_all_of_ref_without_siblings(assertions: &SchemaAssertions) -> Option<&str> {
    if assertions.all_of.len() != 1 {
        return None;
    }

    let mut siblings = assertions.clone();
    siblings.all_of.clear();
    if !siblings.is_empty() {
        return None;
    }

    match &assertions.all_of[0].kind {
        SchemaKind::Ref(reference) => Some(reference.as_str()),
        _ => None,
    }
}

pub(super) fn one_of_mixes_ref_and_inline_branches(branches: &[Schema]) -> bool {
    branches.len() > 1
        && branches
            .iter()
            .any(|branch| matches!(branch.kind, SchemaKind::Ref(_)))
        && branches
            .iter()
            .any(|branch| {
                !matches!(branch.kind, SchemaKind::Ref(_))
                    && !schema_is_null_only_inline_branch(branch)
            })
}

pub(super) fn schema_is_null_only_inline_branch(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };

    matches!(assertions.types.as_deref(), Some([SchemaType::Null]))
        && assertions.const_value.is_none()
        && assertions.enum_values.is_none()
        && assertions.object.is_none()
        && assertions.array.is_none()
        && assertions.string.is_none()
        && assertions.number.is_none()
        && assertions.any_of.is_empty()
        && assertions.one_of.is_empty()
        && assertions.all_of.is_empty()
}

