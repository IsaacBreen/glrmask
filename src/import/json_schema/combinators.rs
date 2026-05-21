use std::collections::BTreeSet;

use crate::import::ast::GrammarExpr;

use super::ast::{
    AdditionalProperties, ObjectSchema, Schema, SchemaAssertions, SchemaKind,
    SchemaType,
};
use super::error::ImportResult;
use super::lower::{choice, r, Lowerer, JSON_VALUE_RULE};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_any_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {
        if let Some((object, any_required_names)) = try_factor_required_property_any_of(assertions) {
            return self.lower_object_requiring_any_property(&object, &any_required_names);
        }
        if let Some((object, exclusive_names, require_one)) =
            try_factor_closed_object_variant_any_of(assertions)
        {
            return self.lower_object_with_exclusive_properties(&object, &exclusive_names, require_one);
        }

        let siblings = sibling_assertion_schema(assertions);
        let alternatives = assertions
            .any_of
            .iter()
            .map(|branch| self.lower_schema(&branch_with_siblings(branch.clone(), siblings.clone())))
            .collect::<ImportResult<Vec<_>>>()?;
        Ok(choice(alternatives))
    }

    pub(crate) fn lower_one_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {
        let siblings = sibling_assertion_schema(assertions);
        let alternatives = assertions
            .one_of
            .iter()
            .map(|branch| self.lower_schema(&branch_with_siblings(branch.clone(), siblings.clone())))
            .collect::<ImportResult<Vec<_>>>()?;
        Ok(choice(alternatives))
    }

    pub(crate) fn lower_all_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {
        let mut branches = assertions.all_of.clone();
        let siblings = assertions.clone_without_combinators();
        if siblings.has_value_assertions_without_combinators() {
            branches.push(Schema::assertions("<allOf-siblings>", siblings));
        }
        branches = self.inline_all_of_refs(&branches)?;

        if branches.is_empty() {
            return Ok(r(JSON_VALUE_RULE));
        }
        if let Some(object) = try_merge_all_of_objects(&branches) {
            return self.lower_object(&object);
        }

        let mut lowered = branches
            .iter()
            .map(|branch| self.lower_schema(branch))
            .collect::<ImportResult<Vec<_>>>()?;
        if lowered.is_empty() {
            return Ok(r(JSON_VALUE_RULE));
        }
        let first = lowered.remove(0);
        Ok(lowered.into_iter().fold(first, |left, right| GrammarExpr::Intersect {
            expr: Box::new(left),
            intersect: Box::new(right),
        }))
    }

    fn inline_all_of_refs(&self, branches: &[Schema]) -> ImportResult<Vec<Schema>> {
        branches
            .iter()
            .map(|branch| match &branch.kind {
                SchemaKind::Ref(pointer) => self.resolve_ref_target(pointer).map(Clone::clone),
                _ => Ok(branch.clone()),
            })
            .collect()
    }
}

fn try_factor_required_property_any_of(
    assertions: &SchemaAssertions,
) -> Option<(ObjectSchema, BTreeSet<String>)> {
    if assertions.any_of.len() < 2 {
        return None;
    }

    let siblings = assertions.clone_without_combinators();
    if siblings.const_value.is_some()
        || siblings.enum_values.is_some()
        || siblings.array.is_some()
        || siblings.string.is_some()
        || siblings.number.is_some()
    {
        return None;
    }
    if let Some(types) = &siblings.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return None;
        }
    }

    let object = siblings.object.clone()?;
    if !object.pattern_properties.is_empty() || object.properties.is_empty() {
        return None;
    }

    let fixed_property_names = object
        .properties
        .iter()
        .map(|property| property.name.clone())
        .collect::<BTreeSet<_>>();
    let mut any_required_names = BTreeSet::new();
    for branch in &assertions.any_of {
        let required_name = single_required_object_branch_name(branch)?;
        if !fixed_property_names.contains(required_name) {
            return None;
        }
        if !any_required_names.insert(required_name.to_string()) {
            return None;
        }
    }

    Some((object, any_required_names))
}

fn try_factor_closed_object_variant_any_of(
    assertions: &SchemaAssertions,
) -> Option<(ObjectSchema, BTreeSet<String>, bool)> {
    if assertions.any_of.len() < 2 {
        return None;
    }

    let siblings = assertions.clone_without_combinators();
    if siblings.object.is_some()
        || siblings.const_value.is_some()
        || siblings.enum_values.is_some()
        || siblings.array.is_some()
        || siblings.string.is_some()
        || siblings.number.is_some()
    {
        return None;
    }
    if let Some(types) = &siblings.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return None;
        }
    }

    let branch_objects = assertions
        .any_of
        .iter()
        .map(closed_object_variant_branch)
        .collect::<Option<Vec<_>>>()?;

    let mut common_names = branch_objects[0]
        .properties
        .iter()
        .map(|property| property.name.clone())
        .collect::<BTreeSet<_>>();
    for object in branch_objects.iter().skip(1) {
        let names = object
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();
        common_names = common_names
            .intersection(&names)
            .cloned()
            .collect::<BTreeSet<_>>();
    }

    for common_name in &common_names {
        let expected = property_schema_by_name(&branch_objects[0], common_name)?;
        if !branch_objects.iter().skip(1).all(|object| {
            property_schema_by_name(object, common_name)
                .is_some_and(|actual| schemas_shape_equivalent(expected, actual))
        }) {
            return None;
        }
    }

    let mut merged_properties = branch_objects[0]
        .properties
        .iter()
        .filter(|property| common_names.contains(&property.name))
        .cloned()
        .collect::<Vec<_>>();
    let mut exclusive_names = BTreeSet::new();
    let mut require_one = true;
    let mut saw_variant = false;

    for object in &branch_objects {
        let variant_properties = object
            .properties
            .iter()
            .filter(|property| !common_names.contains(&property.name))
            .cloned()
            .collect::<Vec<_>>();
        if variant_properties.len() > 1 {
            return None;
        }
        if let Some(variant) = variant_properties.into_iter().next() {
            if !exclusive_names.insert(variant.name.clone()) {
                return None;
            }
            merged_properties.push(variant);
            saw_variant = true;
        } else {
            require_one = false;
        }
    }

    if !saw_variant {
        return None;
    }

    Some((
        ObjectSchema {
            properties: merged_properties,
            required: BTreeSet::new(),
            pattern_properties: Vec::new(),
            additional_properties: AdditionalProperties::Deny,
        },
        exclusive_names,
        require_one,
    ))
}

fn single_required_object_branch_name(schema: &Schema) -> Option<&str> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.array.is_some()
        || assertions.string.is_some()
        || assertions.number.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return None;
    }
    if let Some(types) = &assertions.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return None;
        }
    }

    let object = assertions.object.as_ref()?;
    if !object.properties.is_empty()
        || !object.pattern_properties.is_empty()
        || !matches!(object.additional_properties, AdditionalProperties::AllowAny)
        || object.required.len() != 1
    {
        return None;
    }

    object.required.iter().next().map(String::as_str)
}

fn closed_object_variant_branch(schema: &Schema) -> Option<&ObjectSchema> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.array.is_some()
        || assertions.string.is_some()
        || assertions.number.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return None;
    }
    if let Some(types) = &assertions.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return None;
        }
    }

    let object = assertions.object.as_ref()?;
    if !matches!(object.additional_properties, AdditionalProperties::Deny)
        || !object.required.is_empty()
        || !object.pattern_properties.is_empty()
        || object.properties.is_empty()
    {
        return None;
    }

    Some(object)
}

fn property_schema_by_name<'a>(object: &'a ObjectSchema, name: &str) -> Option<&'a Schema> {
    object
        .properties
        .iter()
        .find(|property| property.name == name)
        .map(|property| &property.schema)
}

fn schemas_shape_equivalent(left: &Schema, right: &Schema) -> bool {
    match (&left.kind, &right.kind) {
        (SchemaKind::Any, SchemaKind::Any) | (SchemaKind::Never, SchemaKind::Never) => true,
        (SchemaKind::Ref(left), SchemaKind::Ref(right)) => left == right,
        (SchemaKind::Assertions(left), SchemaKind::Assertions(right)) => {
            left.types == right.types
                && left.const_value == right.const_value
                && left.enum_values == right.enum_values
                && option_objects_shape_equivalent(left.object.as_ref(), right.object.as_ref())
                && option_arrays_shape_equivalent(left.array.as_ref(), right.array.as_ref())
                && option_strings_shape_equivalent(left.string.as_ref(), right.string.as_ref())
                && option_numbers_shape_equivalent(left.number.as_ref(), right.number.as_ref())
                && schema_slices_shape_equivalent(&left.any_of, &right.any_of)
                && schema_slices_shape_equivalent(&left.one_of, &right.one_of)
                && schema_slices_shape_equivalent(&left.all_of, &right.all_of)
        }
        _ => false,
    }
}

fn option_objects_shape_equivalent(left: Option<&ObjectSchema>, right: Option<&ObjectSchema>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => object_schemas_shape_equivalent(left, right),
        _ => false,
    }
}

fn object_schemas_shape_equivalent(left: &ObjectSchema, right: &ObjectSchema) -> bool {
    additional_properties_shape_equivalent(&left.additional_properties, &right.additional_properties)
        && left.required == right.required
        && left.pattern_properties.len() == right.pattern_properties.len()
        && left
            .pattern_properties
            .iter()
            .zip(&right.pattern_properties)
            .all(|(left, right)| {
                left.pattern == right.pattern && schemas_shape_equivalent(&left.schema, &right.schema)
            })
        && left.properties.len() == right.properties.len()
        && left
            .properties
            .iter()
            .zip(&right.properties)
            .all(|(left, right)| {
                left.name == right.name && schemas_shape_equivalent(&left.schema, &right.schema)
            })
}

fn additional_properties_shape_equivalent(
    left: &AdditionalProperties,
    right: &AdditionalProperties,
) -> bool {
    match (left, right) {
        (AdditionalProperties::AllowAny, AdditionalProperties::AllowAny)
        | (AdditionalProperties::Deny, AdditionalProperties::Deny) => true,
        (AdditionalProperties::Schema(left), AdditionalProperties::Schema(right)) => {
            schemas_shape_equivalent(left, right)
        }
        _ => false,
    }
}

fn option_arrays_shape_equivalent(
    left: Option<&super::ast::ArraySchema>,
    right: Option<&super::ast::ArraySchema>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.min_items == right.min_items
                && left.max_items == right.max_items
                && schemas_shape_equivalent(&left.items, &right.items)
                && schema_slices_shape_equivalent(&left.prefix_items, &right.prefix_items)
        }
        _ => false,
    }
}

fn option_strings_shape_equivalent(
    left: Option<&super::ast::StringSchema>,
    right: Option<&super::ast::StringSchema>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.min_length == right.min_length
                && left.max_length == right.max_length
                && left.pattern == right.pattern
                && left.format == right.format
        }
        _ => false,
    }
}

fn option_numbers_shape_equivalent(
    left: Option<&super::ast::NumberSchema>,
    right: Option<&super::ast::NumberSchema>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.integer == right.integer
                && left.minimum == right.minimum
                && left.maximum == right.maximum
                && left.exclusive_minimum == right.exclusive_minimum
                && left.exclusive_maximum == right.exclusive_maximum
                && left.multiple_of == right.multiple_of
        }
        _ => false,
    }
}

fn schema_slices_shape_equivalent(left: &[Schema], right: &[Schema]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| schemas_shape_equivalent(left, right))
}

fn sibling_assertion_schema(assertions: &SchemaAssertions) -> Option<Schema> {
    let siblings = assertions.clone_without_combinators();
    if siblings.is_empty() {
        None
    } else {
        Some(Schema::assertions("<combinator-siblings>", siblings))
    }
}

fn branch_with_siblings(branch: Schema, siblings: Option<Schema>) -> Schema {
    let Some(siblings) = siblings else {
        return branch;
    };
    all_of_schema(branch, siblings)
}

fn try_merge_all_of_objects(branches: &[Schema]) -> Option<ObjectSchema> {
    let mut objects = branches.iter().map(plain_object_schema).collect::<Option<Vec<_>>>()?;
    let mut merged = objects.remove(0).clone();
    for object in objects {
        merged = merge_two_objects(&merged, object);
    }
    Some(merged)
}

fn plain_object_schema(schema: &Schema) -> Option<&ObjectSchema> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return None;
    }
    if let Some(types) = &assertions.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return None;
        }
    }
    assertions.object.as_ref()
}

fn merge_two_objects(left: &ObjectSchema, right: &ObjectSchema) -> ObjectSchema {
    let mut merged = left.clone();

    for required in &right.required {
        merged.required.insert(required.clone());
    }

    for property in &right.properties {
        if let Some(existing) = merged.properties.iter_mut().find(|candidate| candidate.name == property.name) {
            existing.schema = all_of_schema(existing.schema.clone(), property.schema.clone());
        } else {
            merged.properties.push(property.clone());
        }
    }

    merged.pattern_properties.extend(right.pattern_properties.clone());
    let additional_properties = merge_additional_properties(
        &merged.additional_properties,
        &right.additional_properties,
    );
    merged.additional_properties = additional_properties;
    merged
}

fn merge_additional_properties(
    left: &AdditionalProperties,
    right: &AdditionalProperties,
) -> AdditionalProperties {
    match (left, right) {
        (AdditionalProperties::Deny, _) | (_, AdditionalProperties::Deny) => AdditionalProperties::Deny,
        (AdditionalProperties::AllowAny, AdditionalProperties::AllowAny) => AdditionalProperties::AllowAny,
        (AdditionalProperties::Schema(schema), AdditionalProperties::AllowAny)
        | (AdditionalProperties::AllowAny, AdditionalProperties::Schema(schema)) => {
            AdditionalProperties::Schema(schema.clone())
        }
        (AdditionalProperties::Schema(left), AdditionalProperties::Schema(right)) => {
            AdditionalProperties::Schema(Box::new(all_of_schema(
                left.as_ref().clone(),
                right.as_ref().clone(),
            )))
        }
    }
}

pub(crate) fn all_of_schema(left: Schema, right: Schema) -> Schema {
    Schema::assertions(
        "<merged-allOf-property>",
        SchemaAssertions {
            all_of: vec![left, right],
            ..SchemaAssertions::default()
        },
    )
}
