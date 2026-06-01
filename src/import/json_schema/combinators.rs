use std::collections::BTreeSet;

use crate::import::ast::GrammarExpr;

use super::ast::{
    AdditionalProperties, ArraySchema, ObjectSchema, PropertySchema, Schema, SchemaAssertions,
    SchemaKind, SchemaType,
};
use super::error::ImportResult;
use super::lower::{
    choice, never, normalize_local_ref, r, Lowerer, JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE,
    JSON_ADDITIONAL_KEY_COLON_SHARED_RULE, JSON_BOOL_RULE, JSON_INTEGER_RULE,
    JSON_ITEM_SEPARATOR_RULE, JSON_KEY_SEPARATOR_RULE, JSON_NULL_RULE, JSON_NUMBER_RULE,
    JSON_OBJECT_RULE, JSON_STRING_CHAR_RULE, JSON_STRING_RULE, JSON_VALUE_RULE,
};
use super::string::string_value_satisfies_schema;

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_any_of(
        &mut self,
        schema: &Schema,
        assertions: &SchemaAssertions,
    ) -> ImportResult<GrammarExpr> {
        if let Some((object, any_required_names)) = try_factor_required_property_any_of(assertions) {
            return self.lower_object_requiring_any_property(&object, &any_required_names);
        }
        if let Some(object) = self.try_merge_single_object_any_of_with_siblings(assertions)? {
            return self.lower_object(&object);
        }

        let siblings = sibling_assertion_schema(assertions);
        let branches = assertions
            .any_of
            .iter()
            .cloned()
            .map(|branch| branch_with_siblings(branch, siblings.clone()))
            .collect::<Vec<_>>();
        let has_ref_branch = branches.iter().any(schema_contains_ref);
        let factoring_branches = if has_ref_branch {
            self.inline_all_of_refs_for_any_of_factoring(&branches)?
        } else {
            branches.clone()
        };
        let suppress_untyped_non_object_alts = has_ref_branch
            || assertions.types.as_ref().is_some_and(|types| {
                types.iter().all(|schema_type| *schema_type == SchemaType::Object)
            });
        if open_object_any_of_covers_json_object(&factoring_branches) {
            return Ok(r(JSON_OBJECT_RULE));
        }
        let factoring_branches = self.drop_subsumed_open_object_any_of_branches(factoring_branches)?;
        if let Some(expr) =
            self.try_lower_closed_object_any_of_variants(
                &factoring_branches,
                suppress_untyped_non_object_alts,
            )?
        {
            return Ok(expr);
        }
        if let Some(expr) = self.try_lower_open_object_any_of_variants(&factoring_branches)? {
            return Ok(expr);
        }

        if let Some((object, exclusive_names, require_one)) =
            try_factor_mutually_exclusive_property_not_any_of(assertions)
        {
            return self.lower_object_with_exclusive_properties(&object, &exclusive_names, require_one);
        }

        if let Some((object, exclusive_names, require_one)) =
            try_factor_closed_object_variant_any_of(assertions)
        {
            return self.lower_object_with_exclusive_properties(&object, &exclusive_names, require_one);
        }

        if let Some(expr) = self.try_lower_ref_string_path_object_any_of(schema, &factoring_branches)? {
            return Ok(expr);
        }

        let alternatives = factoring_branches
            .iter()
            .map(|branch| self.lower_schema(branch))
            .collect::<ImportResult<Vec<_>>>()?;
        Ok(choice(alternatives))
    }

    fn try_merge_single_object_any_of_with_siblings(
        &self,
        assertions: &SchemaAssertions,
    ) -> ImportResult<Option<ObjectSchema>> {
        if assertions.any_of.len() != 1 {
            return Ok(None);
        }

        let siblings = assertions.clone_without_combinators();
        if siblings.is_empty() {
            return Ok(None);
        }

        let branch = match &assertions.any_of[0].kind {
            SchemaKind::Ref(pointer) => self.resolve_ref_target(pointer)?.clone(),
            _ => assertions.any_of[0].clone(),
        };
        if !schema_has_explicit_object_only_type(&branch) {
            return Ok(None);
        }
        let sibling_schema = Schema::assertions("<single-anyOf-siblings>", siblings);
        Ok(try_merge_all_of_objects(&[branch, sibling_schema]))
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
        if let [left, right] = branches.as_slice() {
            if is_vacuous_object_schema(left)
                && let Some(branch) = push_object_only_type_into_branch(right)
            {
                return self.lower_schema(&branch);
            }
            if is_vacuous_object_schema(right)
                && let Some(branch) = push_object_only_type_into_branch(left)
            {
                return self.lower_schema(&branch);
            }
        }
        if branches.len() > 1 && branches.iter().any(schema_has_explicit_object_only_type) {
            branches.retain(|branch| !is_vacuous_object_schema(branch));
            if branches.is_empty() {
                return Ok(r(JSON_OBJECT_RULE));
            }
            if branches.len() == 1 {
                return self.lower_schema(&branches[0]);
            }
        }
        branches = flatten_pure_all_of_branches(branches);
        branches = self.inline_all_of_refs(&branches)?;
        branches = collapse_pure_single_choice_branches(branches);
        branches = self.inline_all_of_refs(&branches)?;
        if let Some(filtered) = drop_vacuous_untyped_family_branches(branches) {
            branches = filtered;
        } else {
            return Ok(never());
        }
        branches = drop_vacuous_string_branches(branches);

        if let Some(explicit_types) = explicit_all_of_type_intersection(&branches) {
            let explicit_types_vec = explicit_types.into_iter().collect::<Vec<_>>();
            for branch in &mut branches {
                if let SchemaKind::Assertions(assertions) = &mut branch.kind {
                    if assertions.any_of.is_empty()
                        && assertions.one_of.is_empty()
                        && assertions.all_of.is_empty()
                    {
                        if assertions.types.is_none() {
                            assertions.types = Some(explicit_types_vec.clone());
                        } else if let Some(types) = &mut assertions.types {
                            types.retain(|t| explicit_types_vec.contains(t));
                        }
                    }
                }
            }
        }

        if branches.is_empty() {
            return Ok(r(JSON_VALUE_RULE));
        }
        if let Some(merged) = merge_all_of_object_like_schema(&branches) {
            return self.lower_schema(&merged);
        }
        if let Some(merged) = merge_all_of_array_like_schema(&branches) {
            return self.lower_schema(&merged);
        }
        if let Some(object) = try_merge_all_of_objects(&branches) {
            return self.lower_object(&object);
        }
        if let Some(object) = self.try_merge_all_of_single_ref_object_branches(&branches)? {
            return self.lower_object(&object);
        }
        if let Some((object, any_required_names)) =
            self.try_factor_all_of_required_property_any_of(&branches)?
        {
            return self.lower_object_requiring_any_property(&object, &any_required_names);
        }
        if let Some((kind, distributed)) = distribute_all_of_over_single_object_choice(&branches) {
            return match kind {
                ChoiceKind::AnyOf => {
                    if let Some(expr) = self.try_lower_open_object_any_of_variants(&distributed)? {
                        Ok(expr)
                    } else {
                        let alternatives = distributed
                            .iter()
                            .map(|branch| self.lower_schema(branch))
                            .collect::<ImportResult<Vec<_>>>()?;
                        Ok(choice(alternatives))
                    }
                }
                ChoiceKind::OneOf => {
                    let alternatives = distributed
                        .iter()
                        .map(|branch| self.lower_schema(branch))
                        .collect::<ImportResult<Vec<_>>>()?;
                    Ok(choice(alternatives))
                }
            };
        }

        let mut lowered = branches
            .iter()
            .map(|branch| self.lower_schema(branch))
            .collect::<ImportResult<Vec<_>>>()?;
        if lowered.is_empty() {
            return Ok(r(JSON_VALUE_RULE));
        }
        if !lowered.iter().all(all_of_intersection_terminal_safe) {
            // The generic grammar lowerer treats Intersect as terminal-ish. Parser-shaped
            // object/array allOf operands can contain nonterminal refs or SeparatedSequence,
            // so overapproximate them for build parity instead of emitting an invalid terminal.
            return Ok(choice(lowered));
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

    fn inline_all_of_refs_for_any_of_factoring(
        &self,
        branches: &[Schema],
    ) -> ImportResult<Vec<Schema>> {
        // Object-anyOf factoring needs short alias chains such as
        // `$ref -> allOf([$ref -> allOf(...)])` to expose their object branches.
        // Keep this bounded and local to factoring so general ref lowering stays conservative.
        let mut current = branches.to_vec();
        for _ in 0..4 {
            current = self.inline_all_of_refs(&current)?;
        }
        Ok(current)
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
                if let Some(schema) = &assertions.not {
                    if self.schema_transitively_refs_pointer(schema, wanted, seen_refs)? {
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
            if let Some(rewritten) = self.try_rewrite_all_of_object_choice_target(target)? {
                Ok(rewritten)
            } else {
                Ok(fallback.clone())
            }
        } else if let SchemaKind::Assertions(assertions) = &target.kind
            && let Some(object) = self.try_merge_single_object_any_of_with_siblings(assertions)?
        {
            Ok(Schema::assertions(
                target.location.clone(),
                SchemaAssertions {
                    types: Some(vec![SchemaType::Object]),
                    object: Some(object),
                    ..SchemaAssertions::default()
                },
            ))
        } else if let Some(rewritten) = self.try_rewrite_all_of_object_choice_target(target)? {
            Ok(rewritten)
        } else if let Some(merged) = self.try_inline_object_like_all_of_target(target)? {
            Ok(merged)
        } else {
            Ok(target.clone())
        }
    }

    fn try_inline_object_like_all_of_target(&self, target: &Schema) -> ImportResult<Option<Schema>> {
        let SchemaKind::Assertions(assertions) = &target.kind else {
            return Ok(None);
        };
        if assertions.all_of.is_empty() || assertions.has_value_assertions_without_combinators() {
            return Ok(None);
        }

        let inlined = self.inline_refs_in_all_of_branch(target)?;
        let SchemaKind::Assertions(inlined_assertions) = &inlined.kind else {
            return Ok(None);
        };
        Ok(merge_all_of_object_like_schema(&inlined_assertions.all_of))
    }

    fn try_rewrite_all_of_object_choice_target(&self, target: &Schema) -> ImportResult<Option<Schema>> {
        let SchemaKind::Assertions(assertions) = &target.kind else {
            return Ok(None);
        };
        if assertions.all_of.is_empty() {
            return Ok(None);
        }

        let mut branches = assertions.all_of.clone();
        let siblings = assertions.clone_without_combinators();
        if siblings.has_value_assertions_without_combinators() {
            branches.push(Schema::assertions("<allOf-siblings>", siblings));
        }

        branches = self.inline_all_of_refs(&branches)?;
        branches = flatten_pure_all_of_branches(branches);
        branches = collapse_pure_single_choice_branches(branches);

        let Some((kind, distributed)) = distribute_all_of_over_single_object_choice(&branches) else {
            return Ok(None);
        };
        let alternatives = distributed
            .into_iter()
            .map(|branch| {
                let SchemaKind::Assertions(assertions) = &branch.kind else {
                    return branch;
                };
                if assertions.all_of.is_empty() || !assertions.clone_without_combinators().is_empty() {
                    return branch;
                }
                merge_all_of_object_like_schema(&assertions.all_of).unwrap_or(branch)
            })
            .collect::<Vec<_>>();

        Ok(Some(Schema::assertions(
            target.location.clone(),
            match kind {
                ChoiceKind::AnyOf => SchemaAssertions {
                    any_of: alternatives,
                    ..SchemaAssertions::default()
                },
                ChoiceKind::OneOf => SchemaAssertions {
                    one_of: alternatives,
                    ..SchemaAssertions::default()
                },
            },
        )))
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

    fn try_factor_all_of_required_property_any_of(
        &self,
        branches: &[Schema],
    ) -> ImportResult<Option<(ObjectSchema, BTreeSet<String>)>> {
        let mut merged: Option<ObjectSchema> = None;
        let mut any_required_names: Option<BTreeSet<String>> = None;

        for branch in branches {
            if let Some(names) = required_property_any_of_names(branch) {
                if any_required_names.replace(names).is_some() {
                    return Ok(None);
                }
                continue;
            }

            let Some(object) = self.object_branch_resolved(branch)? else {
                return Ok(None);
            };
            merged = Some(match merged {
                Some(current) => merge_two_objects(&current, object),
                None => object.clone(),
            });
        }

        let (object, any_required_names) = match (merged, any_required_names) {
            (Some(object), Some(any_required_names)) => (object, any_required_names),
            _ => return Ok(None),
        };
        if !object.pattern_properties.is_empty() || object.properties.is_empty() {
            return Ok(None);
        }

        let fixed_property_names = object
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();
        if any_required_names
            .iter()
            .any(|name| !fixed_property_names.contains(name))
        {
            return Ok(None);
        }

        Ok(Some((object, any_required_names)))
    }

    fn drop_subsumed_open_object_any_of_branches(
        &self,
        branches: Vec<Schema>,
    ) -> ImportResult<Vec<Schema>> {
        let keep = branches
            .iter()
            .enumerate()
            .map(|(index, branch)| {
                let Some(branch_object) = self.object_branch_resolved(branch)? else {
                    return Ok(true);
                };

                for (other_index, other_branch) in branches.iter().enumerate() {
                    if index == other_index {
                        continue;
                    }

                    let Some(other_object) = self.object_branch_resolved(other_branch)? else {
                        continue;
                    };

                    if !self.object_schema_subsumes(
                        other_object,
                        branch_object,
                        &mut BTreeSet::new(),
                    )? {
                        continue;
                    }

                    if !self.object_schema_subsumes(
                        branch_object,
                        other_object,
                        &mut BTreeSet::new(),
                    )? || other_index < index
                    {
                        return Ok(false);
                    }
                }

                Ok(true)
            })
            .collect::<ImportResult<Vec<_>>>()?;

        Ok(branches
            .into_iter()
            .zip(keep)
            .filter_map(|(branch, keep)| keep.then_some(branch))
            .collect())
    }

    fn object_branch_resolved<'schema>(
        &'schema self,
        schema: &'schema Schema,
    ) -> ImportResult<Option<&'schema ObjectSchema>> {
        match &schema.kind {
            SchemaKind::Ref(pointer) => self.object_branch_resolved(self.resolve_ref_target(pointer)?),
            _ => Ok(object_branch(schema)),
        }
    }

    fn object_schema_subsumes(
        &self,
        superset: &ObjectSchema,
        subset: &ObjectSchema,
        seen_pairs: &mut BTreeSet<(String, String)>,
    ) -> ImportResult<bool> {
        if !matches!(superset.additional_properties, AdditionalProperties::AllowAny)
            || !superset.pattern_properties.is_empty()
            || (!subset.pattern_properties.is_empty()
                && !matches!(superset.additional_properties, AdditionalProperties::AllowAny))
        {
            return Ok(false);
        }

        if !superset
            .required
            .iter()
            .all(|required| subset.required.contains(required))
        {
            return Ok(false);
        }

        if superset.min_properties > subset.min_properties {
            return Ok(false);
        }

        if let Some(superset_max) = superset.max_properties {
            let Some(subset_max) = subset.max_properties else {
                return Ok(false);
            };
            if subset_max > superset_max {
                return Ok(false);
            }
        }

        for property in &superset.properties {
            let Some(actual) = property_schema_by_name(subset, &property.name) else {
                return Ok(false);
            };
            if !self.schema_subsumes(&property.schema, actual, seen_pairs)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn schema_subsumes(
        &self,
        superset: &Schema,
        subset: &Schema,
        seen_pairs: &mut BTreeSet<(String, String)>,
    ) -> ImportResult<bool> {
        if schemas_shape_equivalent(superset, subset) {
            return Ok(true);
        }
        if matches!(superset.kind, SchemaKind::Any) || matches!(subset.kind, SchemaKind::Never) {
            return Ok(true);
        }
        if matches!(superset.kind, SchemaKind::Never) {
            return Ok(false);
        }

        let pair = (
            schema_subsumption_key(superset)?,
            schema_subsumption_key(subset)?,
        );
        if !seen_pairs.insert(pair.clone()) {
            return Ok(true);
        }

        let result = match (&superset.kind, &subset.kind) {
            (SchemaKind::Ref(pointer), _) => {
                self.schema_subsumes(self.resolve_ref_target(pointer)?, subset, seen_pairs)?
            }
            (_, SchemaKind::Ref(pointer)) => {
                self.schema_subsumes(superset, self.resolve_ref_target(pointer)?, seen_pairs)?
            }
            (SchemaKind::Assertions(superset_assertions), _)
                if pure_any_of_assertions(superset_assertions) =>
            {
                let mut subsumes = false;
                for branch in &superset_assertions.any_of {
                    if self.schema_subsumes(branch, subset, seen_pairs)? {
                        subsumes = true;
                        break;
                    }
                }
                subsumes
            }
            (_, SchemaKind::Assertions(subset_assertions))
                if pure_any_of_assertions(subset_assertions) =>
            {
                let mut all_subsumed = true;
                for branch in &subset_assertions.any_of {
                    if !self.schema_subsumes(superset, branch, seen_pairs)? {
                        all_subsumed = false;
                        break;
                    }
                }
                all_subsumed
            }
            (SchemaKind::Assertions(superset_assertions), SchemaKind::Assertions(subset_assertions)) => {
                if let (Some(string_schema), Some(values)) = (
                    broad_string_assertions(superset_assertions),
                    string_literal_values(subset_assertions),
                ) {
                    values
                        .iter()
                        .all(|value| string_value_satisfies_schema(value, string_schema).unwrap_or(false))
                } else if let (Some(superset_object), Some(subset_object)) =
                    (object_branch(superset), object_branch(subset))
                {
                    self.object_schema_subsumes(superset_object, subset_object, seen_pairs)?
                } else {
                    false
                }
            }
            _ => false,
        };

        seen_pairs.remove(&pair);
        Ok(result)
    }
}

fn all_of_intersection_terminal_safe(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte
        | GrammarExpr::Epsilon => true,
        GrammarExpr::Ref(name) => matches!(
            name.as_str(),
            JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE
                | JSON_ADDITIONAL_KEY_COLON_SHARED_RULE
                | JSON_BOOL_RULE
                | JSON_INTEGER_RULE
                | JSON_ITEM_SEPARATOR_RULE
                | JSON_KEY_SEPARATOR_RULE
                | JSON_NULL_RULE
                | JSON_NUMBER_RULE
                | JSON_STRING_CHAR_RULE
                | JSON_STRING_RULE
        ),
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => all_of_intersection_terminal_safe(inner),
        GrammarExpr::RepeatRange { expr, .. } => all_of_intersection_terminal_safe(expr),
        GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
            parts.iter().all(all_of_intersection_terminal_safe)
        }
        GrammarExpr::Intersect { expr, intersect } | GrammarExpr::Exclude { expr, exclude: intersect } => {
            all_of_intersection_terminal_safe(expr) && all_of_intersection_terminal_safe(intersect)
        }
        GrammarExpr::SeparatedSequence { .. } | GrammarExpr::ExprNFA(_) => false,
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

fn collapse_pure_single_choice_branches(branches: Vec<Schema>) -> Vec<Schema> {
    branches
        .into_iter()
        .map(|branch| {
            if let Some((_, [single])) = pure_choice_branch(&branch) {
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

fn required_property_any_of_names(schema: &Schema) -> Option<BTreeSet<String>> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if !pure_any_of_assertions(assertions) {
        return None;
    }

    let mut names = BTreeSet::new();
    for branch in &assertions.any_of {
        let required_name = single_required_object_branch_name(branch)?;
        if !names.insert(required_name.to_string()) {
            return None;
        }
    }
    Some(names)
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
            max_properties: None,
            pattern_properties: Vec::new(),
            additional_properties: AdditionalProperties::Deny,
        },
        exclusive_names,
        require_one,
    ))
}

fn try_factor_mutually_exclusive_property_not_any_of(
    assertions: &SchemaAssertions,
) -> Option<(ObjectSchema, BTreeSet<String>, bool)> {
    if assertions.any_of.len() != 2 {
        return None;
    }

    let siblings = assertions.clone_without_combinators();
    if siblings.const_value.is_some()
        || siblings.enum_values.is_some()
        || siblings.array.is_some()
        || siblings.string.is_some()
        || siblings.number.is_some()
        || siblings.object.is_some()
        || siblings.types.as_ref().is_some_and(|types| {
            !types.iter().all(|schema_type| *schema_type == SchemaType::Object)
        })
    {
        return None;
    }

    let mut properties = Vec::<PropertySchema>::new();
    let mut property_names = BTreeSet::<String>::new();
    let mut forbidden_names = BTreeSet::<String>::new();

    for branch in &assertions.any_of {
        let (property, forbidden_name) = mutually_exclusive_property_not_branch(branch)?;
        if !property_names.insert(property.name.clone()) {
            return None;
        }
        forbidden_names.insert(forbidden_name);
        properties.push(property.clone());
    }

    if property_names != forbidden_names {
        return None;
    }

    Some((
        ObjectSchema {
            properties,
            required: BTreeSet::new(),
            min_properties: 0,
            max_properties: None,
            pattern_properties: Vec::new(),
            additional_properties: AdditionalProperties::AllowAny,
        },
        property_names,
        false,
    ))
}

fn mutually_exclusive_property_not_branch(schema: &Schema) -> Option<(&PropertySchema, String)> {
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
    if object.properties.len() != 1
        || !object.required.is_empty()
        || !object.pattern_properties.is_empty()
        || !matches!(object.additional_properties, AdditionalProperties::AllowAny)
    {
        return None;
    }
    let property = &object.properties[0];
    let forbidden_name = single_required_object_not_name(assertions.not.as_ref()?)?;
    if forbidden_name == property.name {
        return None;
    }
    Some((property, forbidden_name.to_string()))
}

fn single_required_object_not_name(schema: &Schema) -> Option<&str> {
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
        || assertions.not.is_some()
    {
        return None;
    }
    if let Some(types) = &assertions.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return None;
        }
    }

    let object = assertions.object.as_ref()?;
    if object.required.len() != 1
        || !object.pattern_properties.is_empty()
        || !matches!(object.additional_properties, AdditionalProperties::AllowAny)
    {
        return None;
    }
    object.required.iter().next().map(String::as_str)
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

pub(super) fn open_object_any_of_covers_json_object(branches: &[Schema]) -> bool {
    if branches.len() < 2 {
        return false;
    }

    let Some(objects) = branches
        .iter()
        .map(object_branch)
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };

    if objects.iter().any(|object| {
        !matches!(object.additional_properties, AdditionalProperties::AllowAny)
            || !object.required.is_empty()
            || object.min_properties != 0
            || object.max_properties.is_some()
            || !object.pattern_properties.is_empty()
    }) {
        return false;
    }

    let property_names = objects
        .iter()
        .flat_map(|object| object.properties.iter().map(|property| property.name.as_str()))
        .collect::<BTreeSet<_>>();

    property_names.into_iter().all(|name| {
        objects
            .iter()
            .any(|object| property_schema_by_name(object, name).is_none())
    })
}

fn object_branch(schema: &Schema) -> Option<&ObjectSchema> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.array.is_some()
        || assertions.string.is_some()
        || assertions.number.is_some()
        || assertions.not.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return None;
    }
    if let Some(types) = &assertions.types
        && !types.iter().all(|schema_type| *schema_type == SchemaType::Object)
    {
        return None;
    }

    assertions.object.as_ref()
}

fn property_schema_by_name<'a>(object: &'a ObjectSchema, name: &str) -> Option<&'a Schema> {
    object
        .properties
        .iter()
        .find(|property| property.name == name)
        .map(|property| &property.schema)
}

fn schema_subsumption_key(schema: &Schema) -> ImportResult<String> {
    match &schema.kind {
        SchemaKind::Ref(pointer) => normalize_local_ref(pointer).map(|pointer| format!("ref:{pointer}")),
        _ => Ok(format!("loc:{}", schema.location)),
    }
}

fn pure_any_of_assertions(assertions: &SchemaAssertions) -> bool {
    !assertions.any_of.is_empty()
        && assertions.types.is_none()
        && assertions.const_value.is_none()
        && assertions.enum_values.is_none()
        && assertions.object.is_none()
        && assertions.array.is_none()
        && assertions.string.is_none()
        && assertions.number.is_none()
        && assertions.one_of.is_empty()
        && assertions.all_of.is_empty()
        && assertions.not.is_none()
}

fn broad_string_assertions(assertions: &SchemaAssertions) -> Option<&super::ast::StringSchema> {
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.number.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
        || assertions.not.is_some()
    {
        return None;
    }
    if !assertions
        .types
        .as_ref()
        .is_some_and(|types| types.iter().all(|schema_type| *schema_type == SchemaType::String))
    {
        return None;
    }
    assertions.string.as_ref()
}

fn string_literal_values(assertions: &SchemaAssertions) -> Option<Vec<&serde_json::Value>> {
    if assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.number.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
        || assertions.not.is_some()
    {
        return None;
    }
    if let Some(types) = &assertions.types
        && !types.iter().all(|schema_type| *schema_type == SchemaType::String)
    {
        return None;
    }
    if let Some(value) = &assertions.const_value {
        return value.is_string().then_some(vec![value]);
    }
    let values = assertions.enum_values.as_ref()?;
    values.iter().all(|value| value.is_string()).then_some(values.iter().collect())
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
                && option_schemas_shape_equivalent(left.not.as_ref(), right.not.as_ref())
        }
        _ => false,
    }
}

fn option_schemas_shape_equivalent(left: Option<&Schema>, right: Option<&Schema>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => schemas_shape_equivalent(left, right),
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
        && left.min_properties == right.min_properties
        && left.max_properties == right.max_properties
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
    if is_vacuous_object_schema(&siblings)
        && let Some(branch) = push_object_only_type_into_branch(&branch)
    {
        return branch;
    }
    all_of_schema(branch, siblings)
}

fn push_object_only_type_into_branch(branch: &Schema) -> Option<Schema> {
    let SchemaKind::Assertions(assertions) = &branch.kind else {
        return None;
    };
    if assertions.const_value.is_some() || assertions.enum_values.is_some() {
        return None;
    }
    if let Some(types) = &assertions.types
        && !types.iter().all(|schema_type| *schema_type == SchemaType::Object)
    {
        return None;
    }
    if assertions.object.is_none()
        && assertions.all_of.is_empty()
        && assertions.any_of.is_empty()
        && assertions.one_of.is_empty()
    {
        return None;
    }

    let mut updated = assertions.as_ref().clone();
    updated.types = Some(vec![SchemaType::Object]);
    Some(Schema::assertions(branch.location.clone(), updated))
}

fn schema_contains_ref(schema: &Schema) -> bool {
    match &schema.kind {
        SchemaKind::Ref(_) => true,
        SchemaKind::Assertions(assertions) => {
            assertions.all_of.iter().any(schema_contains_ref)
                || assertions.any_of.iter().any(schema_contains_ref)
                || assertions.one_of.iter().any(schema_contains_ref)
                || assertions.not.as_ref().is_some_and(schema_contains_ref)
        }
        _ => false,
    }
}

fn schema_has_explicit_object_only_type(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };
    assertions
        .types
        .as_ref()
        .is_some_and(|types| types.iter().all(|schema_type| *schema_type == SchemaType::Object))
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

#[derive(Clone, Copy)]
enum ChoiceKind {
    AnyOf,
    OneOf,
}

fn pure_choice_branch(schema: &Schema) -> Option<(ChoiceKind, &[Schema])> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.types.is_some()
        || assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.string.is_some()
        || assertions.number.is_some()
        || !assertions.all_of.is_empty()
    {
        return None;
    }

    match (assertions.any_of.is_empty(), assertions.one_of.is_empty()) {
        (false, true) => Some((ChoiceKind::AnyOf, &assertions.any_of)),
        (true, false) => Some((ChoiceKind::OneOf, &assertions.one_of)),
        _ => None,
    }
}

fn distribute_all_of_over_single_object_choice(
    branches: &[Schema],
) -> Option<(ChoiceKind, Vec<Schema>)> {
    let mut choice_branch = None;
    for branch in branches {
        if let Some((kind, alternatives)) = pure_choice_branch(branch) {
            if choice_branch.is_some()
                || alternatives
                    .iter()
                    .any(|alternative| !schema_is_object_like(alternative))
            {
                return None;
            }
            choice_branch = Some((kind, alternatives));
        } else if !schema_is_object_like(branch) {
            return None;
        }
    }

    let (kind, alternatives) = choice_branch?;
    let object_siblings = branches
        .iter()
        .filter(|branch| pure_choice_branch(branch).is_none())
        .cloned()
        .collect::<Vec<_>>();

    Some((
        kind,
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
    ))
}

fn schema_is_object_like(schema: &Schema) -> bool {
    object_like_schema(schema).is_some()
}

fn merge_all_of_object_like_schema(branches: &[Schema]) -> Option<Schema> {
    let mut objects = Vec::new();
    let has_explicit_object_only_type = branches.iter().any(schema_has_explicit_object_only_type);

    for branch in branches {
        let object_like = object_like_schema(branch)?;
        let SchemaKind::Assertions(assertions) = object_like.kind else {
            return None;
        };
        if let Some(object) = assertions.object {
            objects.push(object.clone());
        }
    }

    if objects.is_empty() {
        return has_explicit_object_only_type.then(|| {
            Schema::assertions(
                "<merged-allOf-object-like>",
                SchemaAssertions {
                    types: Some(vec![SchemaType::Object]),
                    ..SchemaAssertions::default()
                },
            )
        });
    }

    let mut merged = objects.remove(0);
    for object in objects {
        merged = merge_two_objects(&merged, &object);
    }

    Some(Schema::assertions(
        "<merged-allOf-object-like>",
        SchemaAssertions {
            types: has_explicit_object_only_type.then_some(vec![SchemaType::Object]),
            object: Some(merged),
            ..SchemaAssertions::default()
        },
    ))
}

fn object_like_schema(schema: &Schema) -> Option<Schema> {
    if let Some(object) = plain_object_schema(schema) {
        return Some(Schema::assertions(
            schema.location.clone(),
            SchemaAssertions {
                types: schema_has_explicit_object_only_type(schema).then_some(vec![SchemaType::Object]),
                object: Some(object.clone()),
                ..SchemaAssertions::default()
            },
        ));
    }
    if is_vacuous_object_schema(schema) {
        return Some(Schema::assertions(
            schema.location.clone(),
            SchemaAssertions {
                types: Some(vec![SchemaType::Object]),
                ..SchemaAssertions::default()
            },
        ));
    }

    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.string.is_some()
        || assertions.number.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || assertions.all_of.is_empty()
    {
        return None;
    }
    if let Some(types) = &assertions.types
        && !types.iter().all(|schema_type| *schema_type == SchemaType::Object)
    {
        return None;
    }

    merge_all_of_object_like_schema(&assertions.all_of)
}

fn merge_all_of_array_like_schema(branches: &[Schema]) -> Option<Schema> {
    let mut merged = None;
    let mut pending_bounds = None;
    let mut saw_array_shape = false;

    for branch in branches {
        let (array, constrains_to_array) = plain_array_schema(branch)?;
        if array_is_bounds_only(array) {
            if let Some(existing) = &mut pending_bounds {
                merge_array_bounds(existing, array);
            } else {
                pending_bounds = Some(array.clone());
            }
            continue;
        }

        if !constrains_to_array || saw_array_shape {
            return None;
        }
        merged = Some(array.clone());
        saw_array_shape = true;
    }

    if let (Some(array), Some(bounds)) = (&mut merged, &pending_bounds) {
        merge_array_bounds(array, bounds);
    }

    saw_array_shape.then(|| {
        Schema::assertions(
            "<merged-allOf-array-like>",
            SchemaAssertions {
                types: Some(vec![SchemaType::Array]),
                array: merged,
                ..SchemaAssertions::default()
            },
        )
    })
}

fn plain_array_schema(schema: &Schema) -> Option<(&ArraySchema, bool)> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.object.is_some()
        || assertions.string.is_some()
        || assertions.number.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return None;
    }
    let constrains_to_array = match &assertions.types {
        Some(types) if types.iter().all(|schema_type| *schema_type == SchemaType::Array) => true,
        Some(_) => return None,
        None => false,
    };
    Some((assertions.array.as_ref()?, constrains_to_array))
}

fn array_is_bounds_only(array: &ArraySchema) -> bool {
    schemas_shape_equivalent(&array.items, &Schema::any("<implicit-array-items>"))
        && array.prefix_items.is_empty()
}

fn merge_array_bounds(left: &mut ArraySchema, right: &ArraySchema) {
    left.min_items = left.min_items.max(right.min_items);
    left.max_items = match (left.max_items, right.max_items) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(max), None) | (None, Some(max)) => Some(max),
        (None, None) => None,
    };
}


fn merge_two_objects(left: &ObjectSchema, right: &ObjectSchema) -> ObjectSchema {
    let mut merged = left.clone();
    merged.min_properties = merged.min_properties.max(right.min_properties);
    merged.max_properties = match (merged.max_properties, right.max_properties) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(max), None) | (None, Some(max)) => Some(max),
        (None, None) => None,
    };

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

fn drop_vacuous_string_branches(branches: Vec<Schema>) -> Vec<Schema> {
    let has_non_vacuous_string_branch = branches
        .iter()
        .any(|branch| !is_vacuous_string_schema(branch) && schema_has_string_family(branch));
    if !has_non_vacuous_string_branch {
        return branches;
    }
    branches
        .into_iter()
        .filter(|branch| !is_vacuous_string_schema(branch))
        .collect()
}

fn is_vacuous_string_schema(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
        || assertions.not.is_some()
    {
        return false;
    }
    let Some(types) = &assertions.types else {
        return false;
    };
    if !types.iter().all(|schema_type| *schema_type == SchemaType::String) {
        return false;
    }
    assertions.object.is_none()
        && assertions.array.is_none()
        && option_strings_shape_equivalent(
            assertions.string.as_ref(),
            Some(&super::ast::StringSchema::default()),
        )
        && assertions.number.is_none()
}

fn schema_has_string_family(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };
    assertions.string.is_some()
        || assertions
            .types
            .as_ref()
            .is_some_and(|types| types.iter().any(|schema_type| *schema_type == SchemaType::String))
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
