use std::collections::BTreeSet;

use crate::import::ast::GrammarExpr;

use super::ast::{
    AdditionalProperties, ObjectSchema, Schema, SchemaAssertions, SchemaKind,
    SchemaType,
};
use super::error::ImportResult;
use super::lower::{choice, never, normalize_local_ref, r, Lowerer, JSON_VALUE_RULE};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_any_of(
        &mut self,
        schema: &Schema,
        assertions: &SchemaAssertions,
    ) -> ImportResult<GrammarExpr> {
        if let Some((object, any_required_names)) = try_factor_required_property_any_of(assertions) {
            return self.lower_object_requiring_any_property(&object, &any_required_names);
        }

        let siblings = sibling_assertion_schema(assertions);
        let branches = assertions
            .any_of
            .iter()
            .cloned()
            .map(|branch| branch_with_siblings(branch, siblings.clone()))
            .collect::<Vec<_>>();
        let resolved_branches;
        let has_ref_branch = branches.iter().any(schema_contains_ref);
        let factoring_branches = if has_ref_branch {
            resolved_branches = self.inline_all_of_refs(&branches)?;
            &resolved_branches
        } else {
            &branches
        };
        if let Some(expr) =
            self.try_lower_closed_object_any_of_variants(factoring_branches, has_ref_branch)?
        {
            return Ok(expr);
        }
        if let Some(expr) = self.try_lower_open_object_any_of_variants(factoring_branches)? {
            return Ok(expr);
        }

        if let Some((object, exclusive_names, require_one)) =
            try_factor_closed_object_variant_any_of(assertions)
        {
            return self.lower_object_with_exclusive_properties(&object, &exclusive_names, require_one);
        }

        if let Some(expr) = self.try_lower_ref_string_path_object_any_of(schema, &branches)? {
            return Ok(expr);
        }

        let alternatives = branches
            .iter()
            .map(|branch| self.lower_schema(branch))
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
        if let Some(expr) = self.try_lower_single_ref_with_object_siblings(assertions)? {
            return Ok(expr);
        }

        let mut branches = assertions.all_of.clone();
        let siblings = assertions.clone_without_combinators();
        if siblings.has_value_assertions_without_combinators() {
            branches.push(Schema::assertions("<allOf-siblings>", siblings));
        }
        branches = self.inline_all_of_refs(&branches)?;
        branches = flatten_pure_all_of_branches(branches);
        branches = self.inline_all_of_refs(&branches)?;
        branches = collapse_pure_single_any_of_branches(branches);
        branches = self.inline_all_of_refs(&branches)?;
        if let Some(filtered) = drop_vacuous_untyped_family_branches(branches) {
            branches = filtered;
        } else {
            return Ok(never());
        }

        if branches.is_empty() {
            return Ok(r(JSON_VALUE_RULE));
        }
        if let Some(object) = try_merge_all_of_objects(&branches) {
            return self.lower_object(&object);
        }
        if let Some(object) = self.try_merge_all_of_single_ref_object_branches(&branches)? {
            return self.lower_object(&object);
        }
        if let Some(distributed) = distribute_all_of_over_single_object_any_of(&branches) {
            let alternatives = distributed
                .iter()
                .map(|branch| self.lower_schema(branch))
                .collect::<ImportResult<Vec<_>>>()?;
            return Ok(choice(alternatives));
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

    fn try_lower_single_ref_with_object_siblings(
        &mut self,
        assertions: &SchemaAssertions,
    ) -> ImportResult<Option<GrammarExpr>> {
        if assertions.all_of.len() != 1 {
            return Ok(None);
        }

        let SchemaKind::Ref(pointer) = &assertions.all_of[0].kind else {
            return Ok(None);
        };

        let Ok(target) = self.resolve_ref_target(pointer) else {
            return Ok(None);
        };
        if plain_object_schema(target).is_none() {
            return Ok(None);
        }

        let siblings = assertions.clone_without_combinators();
        if siblings.const_value.is_some()
            || siblings.enum_values.is_some()
            || siblings.array.is_some()
            || siblings.string.is_some()
            || siblings.number.is_some()
        {
            return Ok(None);
        }
        if let Some(types) = &siblings.types
            && !types.iter().all(|schema_type| *schema_type == SchemaType::Object)
        {
            return Ok(None);
        }

        let sibling_object = siblings.object.unwrap_or_default();
        if !sibling_object.pattern_properties.is_empty()
            || !matches!(sibling_object.additional_properties, AdditionalProperties::AllowAny)
        {
            return Ok(None);
        }

        if sibling_object.properties.is_empty() && sibling_object.required.is_empty() {
            return self.lower_ref(pointer).map(Some);
        }

        Ok(None)
    }

    fn inline_all_of_refs(&self, branches: &[Schema]) -> ImportResult<Vec<Schema>> {
        branches
            .iter()
            .map(|branch| self.inline_refs_in_all_of_branch(branch))
            .collect()
    }

    fn schema_transitively_refs_pointer(
        &self,
        schema: &Schema,
        wanted: &str,
        seen_refs: &mut BTreeSet<String>,
    ) -> ImportResult<bool> {
        match &schema.kind {
            SchemaKind::Any | SchemaKind::Never => Ok(false),
            SchemaKind::Ref(pointer) => {
                let normalized = normalize_local_ref(pointer)?;
                if normalized == wanted {
                    return Ok(true);
                }
                if !seen_refs.insert(normalized.clone()) {
                    return Ok(false);
                }
                let target = self.resolve_ref_target(pointer)?;
                self.schema_transitively_refs_pointer(target, wanted, seen_refs)
            }
            SchemaKind::Assertions(assertions) => {
                if let Some(object) = &assertions.object {
                    for property in &object.properties {
                        if self.schema_transitively_refs_pointer(&property.schema, wanted, seen_refs)? {
                            return Ok(true);
                        }
                    }
                    for property in &object.pattern_properties {
                        if self.schema_transitively_refs_pointer(&property.schema, wanted, seen_refs)? {
                            return Ok(true);
                        }
                    }
                    if let AdditionalProperties::Schema(schema) = &object.additional_properties
                        && self.schema_transitively_refs_pointer(schema, wanted, seen_refs)?
                    {
                        return Ok(true);
                    }
                }

                if let Some(array) = &assertions.array {
                    if self.schema_transitively_refs_pointer(&array.items, wanted, seen_refs)? {
                        return Ok(true);
                    }
                    for item in &array.prefix_items {
                        if self.schema_transitively_refs_pointer(item, wanted, seen_refs)? {
                            return Ok(true);
                        }
                    }
                }

                for branch in assertions
                    .any_of
                    .iter()
                    .chain(assertions.one_of.iter())
                    .chain(assertions.all_of.iter())
                {
                    if self.schema_transitively_refs_pointer(branch, wanted, seen_refs)? {
                        return Ok(true);
                    }
                }

                Ok(false)
            }
        }
    }

    fn inline_all_of_ref_target(&self, pointer: &str, fallback: &Schema) -> ImportResult<Schema> {
        let normalized = normalize_local_ref(pointer)?;
        let target = self.resolve_ref_target(pointer)?;
        if self.schema_transitively_refs_pointer(target, &normalized, &mut BTreeSet::new())? {
            Ok(fallback.clone())
        } else {
            Ok(target.clone())
        }
    }

    fn inline_refs_in_all_of_branch(&self, branch: &Schema) -> ImportResult<Schema> {
        match &branch.kind {
            SchemaKind::Ref(pointer) => self.inline_all_of_ref_target(pointer, branch),
            SchemaKind::Assertions(assertions) if !assertions.all_of.is_empty() => {
                let mut inlined = assertions.as_ref().clone();
                inlined.all_of = assertions
                    .all_of
                    .iter()
                    .map(|part| match &part.kind {
                        SchemaKind::Ref(pointer) => self.inline_all_of_ref_target(pointer, part),
                        _ => Ok(part.clone()),
                    })
                    .collect::<ImportResult<Vec<_>>>()?;
                Ok(Schema::assertions(branch.location.clone(), inlined))
            }
            _ => Ok(branch.clone()),
        }
    }

    fn try_merge_all_of_single_ref_object_branches(
        &self,
        branches: &[Schema],
    ) -> ImportResult<Option<ObjectSchema>> {
        let mut merged: Option<ObjectSchema> = None;
        let mut saw_ref_branch = false;

        for branch in branches {
            let object = match &branch.kind {
                SchemaKind::Ref(pointer) => {
                    if saw_ref_branch {
                        return Ok(None);
                    }
                    saw_ref_branch = true;
                    let target = self.resolve_ref_target(pointer)?;
                    let Some(object) = plain_object_schema(target) else {
                        return Ok(None);
                    };
                    object
                }
                _ => {
                    let Some(object) = plain_object_schema(branch) else {
                        return Ok(None);
                    };
                    object
                }
            };

            merged = Some(match merged {
                Some(current) => merge_two_objects(&current, object),
                None => object.clone(),
            });
        }

        Ok(saw_ref_branch.then_some(merged).flatten())
    }
}

fn explicit_all_of_type_intersection(branches: &[Schema]) -> Option<BTreeSet<SchemaType>> {
    let mut intersection: Option<BTreeSet<SchemaType>> = None;

    for branch in branches {
        let SchemaKind::Assertions(assertions) = &branch.kind else {
            continue;
        };
        let Some(types) = &assertions.types else {
            continue;
        };

        let branch_types = types.iter().cloned().collect::<BTreeSet<_>>();
        intersection = Some(match intersection {
            Some(existing) => existing.intersection(&branch_types).cloned().collect(),
            None => branch_types,
        });
    }

    intersection
}

fn untyped_single_family_assertion(schema: &Schema) -> Option<SchemaType> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.types.is_some()
        || assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return None;
    }

    let mut family = None;
    for candidate in [
        (assertions.object.is_some(), SchemaType::Object),
        (assertions.array.is_some(), SchemaType::Array),
        (assertions.string.is_some(), SchemaType::String),
        (assertions.number.is_some(), SchemaType::Number),
    ] {
        if !candidate.0 {
            continue;
        }
        if family.is_some() {
            return None;
        }
        family = Some(candidate.1);
    }

    family
}

fn family_overlaps_types(family: SchemaType, types: &BTreeSet<SchemaType>) -> bool {
    match family {
        SchemaType::Number => {
            types.contains(&SchemaType::Number) || types.contains(&SchemaType::Integer)
        }
        other => types.contains(&other),
    }
}

fn drop_vacuous_untyped_family_branches(branches: Vec<Schema>) -> Option<Vec<Schema>> {
    let Some(explicit_types) = explicit_all_of_type_intersection(&branches) else {
        return Some(branches);
    };
    if explicit_types.is_empty() {
        return None;
    }

    Some(
        branches
            .into_iter()
            .filter(|branch| {
                untyped_single_family_assertion(branch)
                    .is_none_or(|family| family_overlaps_types(family, &explicit_types))
            })
            .collect(),
    )
}

fn flatten_pure_all_of_branches(branches: Vec<Schema>) -> Vec<Schema> {
    let mut out = Vec::new();
    for branch in branches {
        match &branch.kind {
            SchemaKind::Assertions(assertions)
                if !assertions.all_of.is_empty()
                    && assertions.clone_without_combinators().is_empty() =>
            {
                out.extend(flatten_pure_all_of_branches(assertions.all_of.clone()));
            }
            _ => out.push(branch),
        }
    }
    out
}

fn collapse_pure_single_any_of_branches(branches: Vec<Schema>) -> Vec<Schema> {
    branches
        .into_iter()
        .map(|branch| {
            if let Some([single]) = pure_any_of_branch(&branch) {
                single.clone()
            } else {
                branch
            }
        })
        .collect()
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
            min_properties: 0,
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

fn schema_contains_ref(schema: &Schema) -> bool {
    match &schema.kind {
        SchemaKind::Ref(_) => true,
        SchemaKind::Assertions(assertions) => {
            assertions.all_of.iter().any(schema_contains_ref)
                || assertions.any_of.iter().any(schema_contains_ref)
                || assertions.one_of.iter().any(schema_contains_ref)
        }
        _ => false,
    }
}

pub(crate) fn try_merge_all_of_objects(branches: &[Schema]) -> Option<ObjectSchema> {
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

fn pure_any_of_branch(schema: &Schema) -> Option<&[Schema]> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.any_of.is_empty()
        || assertions.types.is_some()
        || assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.string.is_some()
        || assertions.number.is_some()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return None;
    }
    Some(&assertions.any_of)
}

fn distribute_all_of_over_single_object_any_of(branches: &[Schema]) -> Option<Vec<Schema>> {
    let mut any_of_index = None;
    for (index, branch) in branches.iter().enumerate() {
        if let Some(alternatives) = pure_any_of_branch(branch) {
            if any_of_index.is_some() || alternatives.iter().any(|alternative| plain_object_schema(alternative).is_none()) {
                return None;
            }
            any_of_index = Some(index);
        } else if plain_object_schema(branch).is_none() {
            return None;
        }
    }

    let any_of_index = any_of_index?;
    let alternatives = pure_any_of_branch(&branches[any_of_index])?;
    let object_siblings = branches
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != any_of_index)
        .map(|(_, branch)| branch.clone())
        .collect::<Vec<_>>();

    Some(
        alternatives
            .iter()
            .map(|alternative| {
                let mut all_of = Vec::with_capacity(object_siblings.len() + 1);
                all_of.push(alternative.clone());
                all_of.extend(object_siblings.iter().cloned());
                Schema::assertions(
                    "<distributed-allOf-anyOf>",
                    SchemaAssertions { all_of, ..SchemaAssertions::default() },
                )
            })
            .collect(),
    )
}

fn merge_two_objects(left: &ObjectSchema, right: &ObjectSchema) -> ObjectSchema {
    let mut merged = left.clone();
    merged.min_properties = merged.min_properties.max(right.min_properties);

    for required in &right.required {
        merged.required.insert(required.clone());
    }

    for property in &right.properties {
        if let Some(existing) = merged.properties.iter_mut().find(|candidate| candidate.name == property.name) {
            existing.schema = merge_property_schemas(existing.schema.clone(), property.schema.clone());
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

fn merge_property_schemas(left: Schema, right: Schema) -> Schema {
    if is_vacuous_json_value_schema(&left) || is_vacuous_object_schema(&left) {
        right
    } else if is_vacuous_json_value_schema(&right) || is_vacuous_object_schema(&right) {
        left
    } else {
        all_of_schema(left, right)
    }
}

fn is_vacuous_json_value_schema(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return matches!(schema.kind, SchemaKind::Any);
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return false;
    }
    if !option_objects_shape_equivalent(assertions.object.as_ref(), Some(&ObjectSchema::default()))
        || !option_arrays_shape_equivalent(assertions.array.as_ref(), Some(&super::ast::ArraySchema::default()))
        || !option_strings_shape_equivalent(assertions.string.as_ref(), Some(&super::ast::StringSchema::default()))
        || !option_numbers_shape_equivalent(assertions.number.as_ref(), Some(&super::ast::NumberSchema::default()))
    {
        return false;
    }
    let Some(types) = &assertions.types else {
        return true;
    };
    types.contains(&SchemaType::Null)
        && types.contains(&SchemaType::Boolean)
        && types.contains(&SchemaType::Object)
        && types.contains(&SchemaType::Array)
        && types.contains(&SchemaType::String)
        && types.contains(&SchemaType::Number)
}

fn is_vacuous_object_schema(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return false;
    }
    let Some(types) = &assertions.types else {
        return false;
    };
    if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
        return false;
    }
    option_objects_shape_equivalent(assertions.object.as_ref(), Some(&ObjectSchema::default()))
        && assertions.array.is_none()
        && assertions.string.is_none()
        && assertions.number.is_none()
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
