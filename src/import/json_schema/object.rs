use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::grammar::expr_nfa::ExprNfaBuilder;
use crate::import::ast::GrammarExpr;

use super::ast::{
    AdditionalProperties, ObjectSchema, PatternPropertySchema,
    PropertySchema, Schema, SchemaAssertions, SchemaKind, SchemaType,
};
use super::combinators::{all_of_schema, try_merge_all_of_objects};
use super::error::{ImportResult, SchemaImportError};
use super::lower::{
    choice, lit, r, seq, Lowerer, JSON_ARRAY_RULE, JSON_BOOL_RULE,
    JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE,
    JSON_ADDITIONAL_KEY_COLON_SHARED_RULE, JSON_NULL_RULE,
    JSON_NUMBER_RULE, JSON_OBJECT_RULE, JSON_STRING_RULE, JSON_VALUE_RULE,
};
use super::string::property_name_matches_pattern;

const POINT_PATH_PATTERN: &str = "^(/[^/]+)+$";
const LARGE_OBJECT_LITERAL_KEY_TRIE_MIN_ITEMS: usize = 64;
const SNOWPLOW_CONTEXTS_PATTERN: &str = "^contexts_.*";
const SNOWPLOW_UNSTRUCT_EVENT_PATTERN: &str = "^unstruct_event_.*";
const SNOWPLOW_KEY_TRIE_PREFIX_SPLIT_BYTES: usize = 1;

struct ObjectItem {
    pair: GrammarExpr,
    separator_pair: GrammarExpr,
    required: bool,
    satisfies_any_group: bool,
    exclusive_group: bool,
}

const ANYOF_FIXED_OBJECT_EXPR_NFA_MAX_STATES: usize = 4096;

struct AnyOfFixedObjectItem {
    key: String,
    value_expr: GrammarExpr,
    required: bool,
}

struct AnyOfFixedObjectVariant {
    items: Vec<AnyOfFixedObjectItem>,
}

struct AnyOfObjectVariant {
    items: Vec<AnyOfFixedObjectItem>,
    fixed_keys: BTreeSet<String>,
    pattern_pairs: Vec<GrammarExpr>,
    pattern_keys: Vec<String>,
    additional_value_expr: Option<GrammarExpr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AnyOfFixedObjectState {
    variant_idx: u16,
    cursor: u16,
    has_content: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum AnyOfObjectPhase {
    Fixed,
    Pattern,
    Additional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AnyOfObjectState {
    variant_idx: u16,
    cursor: u16,
    has_content: bool,
    phase: AnyOfObjectPhase,
}

fn is_obviously_object_valued_property(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };

    assertions.object.is_some()
        || assertions
            .types
            .as_ref()
            .is_some_and(|types| types.iter().any(|schema_type| *schema_type == SchemaType::Object))
}

fn obvious_object_valued_property_count(schema: &ObjectSchema) -> usize {
    schema
        .properties
        .iter()
        .filter(|property| is_obviously_object_valued_property(&property.schema))
        .count()
}

impl AnyOfFixedObjectVariant {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn advance_cursor(&self, cursor: usize, key: &str) -> Option<usize> {
        let idx = self.items[cursor..]
            .iter()
            .position(|item| item.key == key)
            .map(|offset| cursor + offset)?;
        if self.items[cursor..idx].iter().any(|item| item.required) {
            return None;
        }
        Some(idx + 1)
    }

    fn close_allowed(&self, cursor: usize) -> bool {
        !self.items[cursor..].iter().any(|item| item.required)
    }

    fn legal_next_keys(&self, cursor: usize) -> Vec<&str> {
        let mut keys = Vec::new();
        for item in &self.items[cursor..] {
            keys.push(item.key.as_str());
            if item.required {
                break;
            }
        }
        keys
    }

    fn value_expr_for_key(&self, key: &str) -> Option<GrammarExpr> {
        self.items
            .iter()
            .find(|item| item.key == key)
            .map(|item| item.value_expr.clone())
    }
}

impl AnyOfObjectVariant {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn advance_cursor(&self, cursor: usize, key: &str) -> Option<usize> {
        let idx = self.items[cursor..]
            .iter()
            .position(|item| item.key == key)
            .map(|offset| cursor + offset)?;
        if self.items[cursor..idx].iter().any(|item| item.required) {
            return None;
        }
        Some(idx + 1)
    }

    fn close_allowed(&self, cursor: usize) -> bool {
        !self.items[cursor..].iter().any(|item| item.required)
    }

    fn legal_next_keys(&self, cursor: usize) -> Vec<&str> {
        let mut keys = Vec::new();
        for item in &self.items[cursor..] {
            keys.push(item.key.as_str());
            if item.required {
                break;
            }
        }
        keys
    }

    fn value_expr_for_key(&self, key: &str) -> Option<GrammarExpr> {
        self.items
            .iter()
            .find(|item| item.key == key)
            .map(|item| item.value_expr.clone())
    }
}

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_object(&mut self, schema: &ObjectSchema) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, None, None)
    }

    pub(crate) fn lower_object_requiring_any_property(
        &mut self,
        schema: &ObjectSchema,
        any_required_names: &BTreeSet<String>,
    ) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, Some(any_required_names), None)
    }

    pub(crate) fn lower_object_with_exclusive_properties(
        &mut self,
        schema: &ObjectSchema,
        exclusive_names: &BTreeSet<String>,
        require_one: bool,
    ) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, None, Some((exclusive_names, require_one)))
    }

    pub(crate) fn try_lower_closed_object_any_of_variants(
        &mut self,
        branches: &[Schema],
        suppress_untyped_non_object_alts: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        if branches.len() < 2 {
            return Ok(None);
        }

        let mut variants = Vec::with_capacity(branches.len());
        let mut include_untyped_non_object_alts = false;
        for branch in branches {
            let Some((variant, branch_requires_untyped_non_object_alts)) =
                self.collect_closed_any_of_object_variant(branch)?
            else {
                return Ok(None);
            };
            include_untyped_non_object_alts |=
                branch_requires_untyped_non_object_alts && !suppress_untyped_non_object_alts;
            variants.push(variant);
        }

        self.lower_closed_any_of_object_variants_expr_nfa(
            &variants,
            include_untyped_non_object_alts,
        )
    }

    pub(crate) fn try_lower_open_object_any_of_variants(
        &mut self,
        branches: &[Schema],
    ) -> ImportResult<Option<GrammarExpr>> {
        if branches.len() < 2 {
            return Ok(None);
        }

        let mut variants = Vec::with_capacity(branches.len());
        let mut include_untyped_non_object_alts = false;
        for branch in branches {
            let Some((variant, branch_requires_untyped_non_object_alts)) =
                self.collect_open_any_of_object_variant(branch)?
            else {
                return Ok(None);
            };
            include_untyped_non_object_alts |= branch_requires_untyped_non_object_alts;
            variants.push(variant);
        }

        self.lower_open_any_of_object_variants_expr_nfa(
            &variants,
            include_untyped_non_object_alts,
        )
    }

    pub(crate) fn try_lower_ref_string_path_object_any_of(
        &mut self,
        current_schema: &Schema,
        branches: &[Schema],
    ) -> ImportResult<Option<GrammarExpr>> {
        if branches.len() != 3 {
            return Ok(None);
        }

        let mut array_branch = None;
        let mut saw_ref_string_object = false;
        let mut saw_path_object = false;

        for branch in branches {
            let resolved = self.resolve_branch_schema(branch)?;
            if is_ref_string_open_object_branch(resolved) {
                if saw_ref_string_object {
                    return Ok(None);
                }
                saw_ref_string_object = true;
                continue;
            }
            if self.is_path_recursive_open_object_branch(current_schema, resolved)? {
                if saw_path_object {
                    return Ok(None);
                }
                saw_path_object = true;
                continue;
            }
            if is_plain_array_branch(resolved) {
                if array_branch.is_some() {
                    return Ok(None);
                }
                array_branch = Some(branch.clone());
                continue;
            }
            return Ok(None);
        }

        if !saw_ref_string_object || !saw_path_object {
            return Ok(None);
        }

        let Some(array_branch) = array_branch else {
            return Ok(None);
        };

        // This recognizer is intentionally tied to the observed Point-like
        // overlap shape: the open "$ref": string object branch and the open
        // path-key object branch together already cover the full object
        // language, so we can collapse just that object subset and leave the
        // array sibling untouched.
        Ok(Some(choice(vec![r(JSON_OBJECT_RULE), self.lower_schema(&array_branch)?])))
    }

    fn resolve_branch_schema<'b>(&'b self, schema: &'b Schema) -> ImportResult<&'b Schema> {
        match &schema.kind {
            SchemaKind::Ref(pointer) => self.resolve_ref_target(pointer),
            _ => Ok(schema),
        }
    }

    fn is_path_recursive_open_object_branch(
        &self,
        current_schema: &Schema,
        schema: &Schema,
    ) -> ImportResult<bool> {
        let SchemaKind::Assertions(assertions) = &schema.kind else {
            return Ok(false);
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
            return Ok(false);
        }
        if let Some(types) = &assertions.types {
            if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
                return Ok(false);
            }
        }

        let Some(object) = &assertions.object else {
            return Ok(false);
        };
        if !object.properties.is_empty() || object.pattern_properties.len() != 1 {
            return Ok(false);
        }
        if !matches!(object.additional_properties, AdditionalProperties::AllowAny) {
            return Ok(false);
        }

        let pattern_property = &object.pattern_properties[0];
        if pattern_property.pattern != POINT_PATH_PATTERN {
            return Ok(false);
        }

        let SchemaKind::Ref(pointer) = &pattern_property.schema.kind else {
            return Ok(false);
        };
        let target = self.resolve_ref_target(pointer)?;
        Ok(target.location == current_schema.location)
    }

    fn lower_object_internal(
        &mut self,
        schema: &ObjectSchema,
        any_required_names: Option<&BTreeSet<String>>,
        exclusive_group: Option<(&BTreeSet<String>, bool)>,
    ) -> ImportResult<GrammarExpr> {
        let normalized = self.object_with_required_synthetic_properties(schema)?;
        let min_property_group_required = if normalized.min_properties <= normalized.required.len() {
            false
        } else if normalized.min_properties == 1
            && normalized.required.is_empty()
            && matches!(normalized.additional_properties, AdditionalProperties::Deny)
            && normalized.pattern_properties.is_empty()
            && !normalized.properties.is_empty()
            && any_required_names.is_none()
            && exclusive_group.is_none()
        {
            true
        } else {
            return Err(SchemaImportError::new(
                "minProperties is only supported when it is redundant or requires at least one fixed property on a closed object"
                    .to_string(),
            ));
        };
        let fixed_names = normalized
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();
        let mut items = normalized
            .properties
            .iter()
            .map(|property| {
                let required = normalized.required.contains(&property.name);
                self.lower_property_item(
                    property,
                    &normalized.pattern_properties,
                    required,
                    any_required_names
                        .is_some_and(|names| names.contains(&property.name))
                        || min_property_group_required,
                    exclusive_group
                        .is_some_and(|(names, _)| names.contains(&property.name)),
                )
            })
            .collect::<ImportResult<Vec<_>>>()?;

        // Keep the large-open ordered-prefix special case narrowly scoped so
        // fixed-key continuations stay deterministic for that hotspot without
        // changing ordinary fixed-object ExprNFA paths.
        let use_large_optional_open_object_prefix_chain = normalized.pattern_properties.is_empty()
            && normalized.required.is_empty()
            && any_required_names.is_none()
            && exclusive_group.is_none()
            && match &normalized.additional_properties {
                AdditionalProperties::Schema(_) => true,
                AdditionalProperties::AllowAny => {
                    obvious_object_valued_property_count(&normalized) < 32
                }
                AdditionalProperties::Deny => false,
            }
            && normalized.properties.len() >= 16;

        if normalized.pattern_properties.is_empty() && !normalized.properties.is_empty() {
            let use_large_closed_object_fixed_pair_loop =
                matches!(normalized.additional_properties, AdditionalProperties::Deny)
                    && normalized.required.is_empty()
                    && any_required_names.is_none()
                    && exclusive_group.is_none()
                    && normalized.properties.len() >= LARGE_OBJECT_LITERAL_KEY_TRIE_MIN_ITEMS;
            let use_closed_object_prefix_chain =
                matches!(normalized.additional_properties, AdditionalProperties::Deny)
                    && any_required_names.is_none()
                    && exclusive_group.is_none()
                    && normalized
                        .properties
                        .iter()
                        .any(|property| self.schema_has_huge_bounded_string(&property.schema));

            let tail_pair = match &normalized.additional_properties {
                AdditionalProperties::Deny => None,
                AdditionalProperties::AllowAny => Some(seq(vec![
                    self.lower_additional_key_colon(&fixed_names, &[])?,
                    r(JSON_VALUE_RULE),
                ])),
                AdditionalProperties::Schema(value_schema) => {
                    let value = self.lower_schema(value_schema)?;
                    Some(seq(vec![
                        self.lower_additional_key_colon(&fixed_names, &[])?,
                        value,
                    ]))
                }
            };
            if use_large_optional_open_object_prefix_chain {
                if let Some(tail_pair_expr) = tail_pair {
                    return self.lower_large_optional_open_object_fused_prefix_chain(&items, tail_pair_expr);
                }
            }
            if use_large_closed_object_fixed_pair_loop {
                return self.lower_large_closed_object_fixed_pair_loop(&items);
            }
            if use_closed_object_prefix_chain {
                return Ok(self.lower_large_closed_object_prefix_chain(&items));
            }
            return self.lower_fixed_object_body_exprnfa(
                &items,
                tail_pair,
                any_required_names.is_some() || min_property_group_required,
                exclusive_group.is_some_and(|(_, require_one)| require_one),
            );
        }

        if any_required_names.is_some() || exclusive_group.is_some() {
            return Err(SchemaImportError::new(
                "grouped anyOf object factoring requires fixed object properties without patternProperties"
                    .to_string(),
            ));
        }

        if normalized.properties.is_empty()
            && normalized.required.is_empty()
            && matches!(normalized.additional_properties, AdditionalProperties::Deny)
            && !normalized.pattern_properties.is_empty()
            && let Some(expr) = self.try_lower_pattern_map_pair_list_object(&normalized)?
        {
            return Ok(expr);
        }

        let pattern_keys = normalized
            .pattern_properties
            .iter()
            .map(|pattern_property| pattern_property.pattern.clone())
            .collect::<Vec<_>>();

        let mut tail_pairs = Vec::new();
        for pattern_property in &normalized.pattern_properties {
            let key = self.lower_pattern_key_colon_appearance(&pattern_property.pattern, &fixed_names)?;
            let value = self.lower_schema(&pattern_property.schema)?;
            tail_pairs.push(seq(vec![key, value]));
        }

        match &normalized.additional_properties {
            AdditionalProperties::AllowAny => {
                tail_pairs.push(seq(vec![
                    self.lower_additional_key_colon(&fixed_names, &pattern_keys)?,
                    r(JSON_VALUE_RULE),
                ]));
            }
            AdditionalProperties::Deny => {}
            AdditionalProperties::Schema(value_schema) => {
                let value = self.lower_schema(value_schema)?;
                tail_pairs.push(seq(vec![
                    self.lower_additional_key_colon(&fixed_names, &pattern_keys)?,
                    value,
                ]));
            }
        }

        if !tail_pairs.is_empty() {
            let required_prefix_len = items.iter().take_while(|item| item.required).count();
            let required_count = items.iter().filter(|item| item.required).count();
            let use_required_prefix_open_object_pair_loop = required_prefix_len > 0
                && required_prefix_len == required_count
                && items.len().saturating_sub(required_prefix_len) + tail_pairs.len() >= 8
                && any_required_names.is_none()
                && exclusive_group.is_none();

            if use_required_prefix_open_object_pair_loop {
                return self.lower_required_prefix_open_object_pair_loop(
                    &items,
                    required_prefix_len,
                    &tail_pairs,
                );
            }

            let is_snowplow_large_closed_pattern_object = normalized.properties.len()
                >= LARGE_OBJECT_LITERAL_KEY_TRIE_MIN_ITEMS
                && normalized.required.is_empty()
                && any_required_names.is_none()
                && exclusive_group.is_none()
                && matches!(normalized.additional_properties, AdditionalProperties::Deny)
                && normalized.pattern_properties.len() == 2
                && normalized
                    .pattern_properties
                    .iter()
                    .all(|pattern_property| {
                        pattern_property.pattern == SNOWPLOW_CONTEXTS_PATTERN
                            || pattern_property.pattern == SNOWPLOW_UNSTRUCT_EVENT_PATTERN
                    });

            if is_snowplow_large_closed_pattern_object {
                return self.lower_snowplow_large_pattern_object_key_trie(&items, &tail_pairs);
            }

            let pair = GrammarExpr::RepeatOne(Box::new(choice(tail_pairs)));
            items.push(ObjectItem {
                separator_pair: seq(vec![self.item_separator_expr(), pair.clone()]),
                pair,
                required: false,
                satisfies_any_group: false,
                exclusive_group: false,
            });
        }

        let body = if items.is_empty() {
            GrammarExpr::Epsilon
        } else {
            GrammarExpr::SeparatedSequence {
                items: items.into_iter().map(|item| (item.pair, item.required)).collect(),
                separator: Box::new(self.item_separator_expr()),
                allow_empty: true,
            }
        };

        Ok(seq(vec![lit("{"), body, lit("}")]))
    }

    fn schema_has_huge_bounded_string(&self, schema: &Schema) -> bool {
        let SchemaKind::Assertions(assertions) = &schema.kind else {
            return false;
        };
        assertions
            .string
            .as_ref()
            .and_then(|string| string.max_length)
            .is_some_and(|max_length| max_length >= 10_000)
            || assertions
                .any_of
                .iter()
                .chain(assertions.one_of.iter())
                .chain(assertions.all_of.iter())
                .any(|branch| self.schema_has_huge_bounded_string(branch))
    }

    fn try_lower_pattern_map_pair_list_object(
        &mut self,
        schema: &ObjectSchema,
    ) -> ImportResult<Option<GrammarExpr>> {
        if !schema.properties.is_empty()
            || !schema.required.is_empty()
            || schema.pattern_properties.len() != 1
            || !matches!(schema.additional_properties, AdditionalProperties::Deny)
        {
            return Ok(None);
        }

        let pattern_property = &schema.pattern_properties[0];
        let Some(value) = self.try_lower_wrapper_pattern_map_anyof_value(&pattern_property.schema)? else {
            return Ok(None);
        };

        let fixed_names = BTreeSet::new();
        let key = self.lower_pattern_key_colon_appearance(&pattern_property.pattern, &fixed_names)?;
        let pair_expr = seq(vec![key, value]);

        let pair_name = self.fresh_rule_name("json_pattern_map_pair");
        self.add_nonterminal_rule(&pair_name, pair_expr);

        let list_name = self.fresh_rule_name("json_pattern_map_list");
        self.add_nonterminal_rule(
            &list_name,
            choice(vec![
                r(&pair_name),
                seq(vec![r(&list_name), self.item_separator_expr(), r(&pair_name)]),
            ]),
        );

        let body_name = self.fresh_rule_name("json_pattern_map_body");
        self.add_nonterminal_rule(&body_name, choice(vec![GrammarExpr::Epsilon, r(&list_name)]));

        Ok(Some(seq(vec![lit("{"), r(&body_name), lit("}")])))
    }

    fn try_lower_wrapper_pattern_map_anyof_value(
        &mut self,
        schema: &Schema,
    ) -> ImportResult<Option<GrammarExpr>> {
        let SchemaKind::Assertions(assertions) = &schema.kind else {
            return Ok(None);
        };
        if assertions.any_of.len() < 2
            || assertions.const_value.is_some()
            || assertions.enum_values.is_some()
            || assertions.object.is_some()
            || assertions.array.is_some()
            || assertions.string.is_some()
            || assertions.number.is_some()
            || assertions.types.is_some()
            || !assertions.one_of.is_empty()
            || !assertions.all_of.is_empty()
        {
            return Ok(None);
        }

        self.try_lower_open_object_any_of_variants(&assertions.any_of)
    }

    fn lower_fixed_object_body_exprnfa(
        &mut self,
        items: &[ObjectItem],
        tail_pair: Option<GrammarExpr>,
        any_group_required: bool,
        exclusive_require_one: bool,
    ) -> ImportResult<GrammarExpr> {
        if items.iter().all(|item| !item.satisfies_any_group && !item.exclusive_group) {
            return self.lower_fixed_object_body_exprnfa_without_group(items, tail_pair);
        }

        let mut builder = ExprNfaBuilder::new();
        let mut states = vec![[[[0u32; 2]; 2]; 2]; items.len() + 1];
        for state_set in &mut states {
            for separator_seen in 0..=1 {
                for any_group_seen in 0..=1 {
                    for exclusive_seen in 0..=1 {
                        state_set[separator_seen][any_group_seen][exclusive_seen] = builder.add_state();
                    }
                }
            }
        }
        states[0][0][0][0] = builder.start_state();

        for (index, item) in items.iter().enumerate() {
            let separator_pair = seq(vec![self.item_separator_expr(), item.pair.clone()]);
            for separator_seen in 0..=1 {
                for any_group_seen in 0..=1 {
                    for exclusive_seen in 0..=1 {
                        if !item.required {
                            builder.add_epsilon(
                                states[index][separator_seen][any_group_seen][exclusive_seen],
                                states[index + 1][separator_seen][any_group_seen][exclusive_seen],
                            );
                        }
                        if item.exclusive_group && exclusive_seen == 1 {
                            continue;
                        }
                        let next_any_group_seen =
                            usize::from(any_group_seen == 1 || item.satisfies_any_group);
                        let next_exclusive_seen =
                            usize::from(exclusive_seen == 1 || item.exclusive_group);
                        let transition_expr = if separator_seen == 0 {
                            item.pair.clone()
                        } else {
                            separator_pair.clone()
                        };
                        builder.add_transition(
                            states[index][separator_seen][any_group_seen][exclusive_seen],
                            transition_expr,
                            states[index + 1][1][next_any_group_seen][next_exclusive_seen],
                        );
                    }
                }
            }
        }

        for separator_seen in 0..=1 {
            for any_group_seen in 0..=1 {
                if any_group_required && any_group_seen == 0 {
                    continue;
                }
                for exclusive_seen in 0..=1 {
                    if exclusive_require_one && exclusive_seen == 0 {
                        continue;
                    }
                    builder.set_accepting(states[items.len()][separator_seen][any_group_seen][exclusive_seen]);
                }
            }
        }

        if let Some(tail_pair_expr) = tail_pair {
            for any_group_seen in 0..=1 {
                for exclusive_seen in 0..=1 {
                    let tail_state = builder.add_state();
                    if (!any_group_required || any_group_seen == 1)
                        && (!exclusive_require_one || exclusive_seen == 1)
                    {
                        builder.set_accepting(tail_state);
                    }
                    builder.add_transition(
                        states[items.len()][0][any_group_seen][exclusive_seen],
                        tail_pair_expr.clone(),
                        tail_state,
                    );
                    builder.add_transition(
                        states[items.len()][1][any_group_seen][exclusive_seen],
                        seq(vec![self.item_separator_expr(), tail_pair_expr.clone()]),
                        tail_state,
                    );
                    builder.add_transition(
                        tail_state,
                        seq(vec![self.item_separator_expr(), tail_pair_expr.clone()]),
                        tail_state,
                    );
                }
            }
        }

        let rule_name = self.fresh_rule_name("json_closed_object_body");
        let body = GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized()));
        self.add_nonterminal_rule(&rule_name, body);

        Ok(seq(vec![lit("{"), r(&rule_name), lit("}")]))
    }

    fn collect_closed_any_of_object_variant(
        &mut self,
        branch: &Schema,
    ) -> ImportResult<Option<(AnyOfFixedObjectVariant, bool)>> {
        let SchemaKind::Assertions(assertions) = &branch.kind else {
            return Ok(None);
        };
        if assertions.const_value.is_some()
            || assertions.enum_values.is_some()
            || assertions.array.is_some()
            || assertions.string.is_some()
            || assertions.number.is_some()
            || !assertions.any_of.is_empty()
            || !assertions.one_of.is_empty()
        {
            return Ok(None);
        }
        if let Some(types) = &assertions.types {
            if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
                return Ok(None);
            }
        }
        let merged_object;
        let (object, include_untyped_non_object_alts) = if !assertions.all_of.is_empty() {
            if assertions.object.as_ref().is_some_and(|object| {
                !object.properties.is_empty()
                    || !object.required.is_empty()
                    || !object.pattern_properties.is_empty()
                    || !matches!(object.additional_properties, AdditionalProperties::AllowAny)
            }) {
                return Ok(None);
            }
            merged_object = match try_merge_all_of_objects(&assertions.all_of) {
                Some(object) => object,
                None => return Ok(None),
            };
            (
                &merged_object,
                assertions.types.is_none()
                    && !all_of_has_explicit_object_only_type(&assertions.all_of)
                    && !assertions
                        .all_of
                        .iter()
                        .any(|schema| matches!(schema.kind, SchemaKind::Ref(_))),
            )
        } else {
            match &assertions.object {
                Some(object) => (object, assertions.types.is_none()),
                None => return Ok(None),
            }
        };
        if !matches!(object.additional_properties, AdditionalProperties::Deny)
            || !object.pattern_properties.is_empty()
        {
            return Ok(None);
        }
        if object
            .required
            .iter()
            .any(|required_name| !object.properties.iter().any(|property| property.name == *required_name))
        {
            return Ok(None);
        }

        let mut items = Vec::with_capacity(object.properties.len());
        for property in &object.properties {
            items.push(AnyOfFixedObjectItem {
                key: property.name.clone(),
                value_expr: self.lower_schema(&property.schema)?,
                required: object.required.contains(&property.name),
            });
        }

        Ok(Some((
            AnyOfFixedObjectVariant { items },
            include_untyped_non_object_alts,
        )))
    }

    fn collect_open_any_of_object_variant(
        &mut self,
        branch: &Schema,
    ) -> ImportResult<Option<(AnyOfObjectVariant, bool)>> {
        let SchemaKind::Assertions(assertions) = &branch.kind else {
            return Ok(None);
        };
        if assertions.const_value.is_some()
            || assertions.enum_values.is_some()
            || assertions.array.is_some()
            || assertions.string.is_some()
            || assertions.number.is_some()
            || !assertions.any_of.is_empty()
            || !assertions.one_of.is_empty()
        {
            return Ok(None);
        }
        if let Some(types) = &assertions.types {
            if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
                return Ok(None);
            }
        }
        let merged_object;
        let (object, include_untyped_non_object_alts) = if !assertions.all_of.is_empty() {
            if assertions.object.as_ref().is_some_and(|object| {
                !object.properties.is_empty()
                    || !object.required.is_empty()
                    || !object.pattern_properties.is_empty()
                    || !matches!(object.additional_properties, AdditionalProperties::AllowAny)
            }) {
                return Ok(None);
            }
            merged_object = match try_merge_all_of_objects(&assertions.all_of) {
                Some(object) => object,
                None => return Ok(None),
            };
            (
                &merged_object,
                assertions.types.is_none()
                    && !all_of_has_explicit_object_only_type(&assertions.all_of)
                    && !assertions
                        .all_of
                        .iter()
                        .any(|schema| matches!(schema.kind, SchemaKind::Ref(_))),
            )
        } else {
            match &assertions.object {
                Some(object) => (object, assertions.types.is_none()),
                None => return Ok(None),
            }
        };

        let normalized = self.object_with_required_synthetic_properties(object)?;
        let fixed_keys = normalized
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();

        let mut items = Vec::with_capacity(normalized.properties.len());
        for property in &normalized.properties {
            let mut effective_schema = property.schema.clone();
            for pattern_property in &normalized.pattern_properties {
                if property_matches_pattern(&pattern_property.pattern, &property.name)? {
                    let pattern_schema =
                        pattern_schema_for_property(&effective_schema, &pattern_property.schema);
                    effective_schema = all_of_schema(effective_schema, pattern_schema);
                }
            }
            items.push(AnyOfFixedObjectItem {
                key: property.name.clone(),
                value_expr: self.lower_schema(&effective_schema)?,
                required: normalized.required.contains(&property.name),
            });
        }

        let mut pattern_pairs = Vec::with_capacity(normalized.pattern_properties.len());
        for pattern_property in &normalized.pattern_properties {
            let key = self.lower_pattern_key_colon_appearance(&pattern_property.pattern, &fixed_keys)?;
            let value = self.lower_schema(&pattern_property.schema)?;
            pattern_pairs.push(seq(vec![key, value]));
        }

        let additional_value_expr = match &normalized.additional_properties {
            AdditionalProperties::AllowAny => Some(r(JSON_VALUE_RULE)),
            AdditionalProperties::Deny => None,
            AdditionalProperties::Schema(value_schema) => Some(self.lower_schema(value_schema)?),
        };

        Ok(Some((
            AnyOfObjectVariant {
                items,
                fixed_keys,
                pattern_pairs,
                pattern_keys: normalized
                    .pattern_properties
                    .iter()
                    .map(|pattern_property| pattern_property.pattern.clone())
                    .collect(),
                additional_value_expr,
            },
            include_untyped_non_object_alts,
        )))
    }

    fn add_expr_nfa_symbol_path(
        builder: &mut ExprNfaBuilder,
        from: u32,
        symbols: Vec<GrammarExpr>,
        to: u32,
    ) {
        if symbols.is_empty() {
            builder.add_epsilon(from, to);
            return;
        }

        let mut current = from;
        let last = symbols.len() - 1;
        for (index, symbol) in symbols.into_iter().enumerate() {
            let next = if index == last { to } else { builder.add_state() };
            for transition_symbol in Self::split_additional_key_colon_transition(symbol) {
                builder.add_transition(current, transition_symbol, next);
            }
            current = next;
        }
    }

    fn split_additional_key_colon_transition(symbol: GrammarExpr) -> Vec<GrammarExpr> {
        match symbol {
            GrammarExpr::Choice(alternatives)
                if Self::is_shared_additional_key_colon_choice(&alternatives) =>
            {
                alternatives
            }
            other => vec![other],
        }
    }

    fn is_shared_additional_key_colon_choice(alternatives: &[GrammarExpr]) -> bool {
        if alternatives.len() != 2 {
            return false;
        }

        let has_shared_base = alternatives
            .iter()
            .any(Self::is_shared_additional_key_colon_base_ref);
        let has_shared_excluded_addback = alternatives
            .iter()
            .any(Self::is_shared_additional_key_colon_addback);

        has_shared_base && has_shared_excluded_addback
    }

    fn is_shared_additional_key_colon_base_ref(expr: &GrammarExpr) -> bool {
        matches!(expr, GrammarExpr::Ref(rule_name) if rule_name == JSON_ADDITIONAL_KEY_COLON_SHARED_RULE)
    }

    fn is_shared_additional_key_colon_addback(expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Ref(rule_name) => {
                rule_name == JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE
            }
            GrammarExpr::Exclude { expr, .. } => {
                matches!(
                    expr.as_ref(),
                    GrammarExpr::Ref(rule_name)
                        if rule_name == JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE
                )
            }
            _ => false,
        }
    }

    fn split_object_pair_symbols(pair: &GrammarExpr) -> ImportResult<[GrammarExpr; 2]> {
        match pair {
            GrammarExpr::Sequence(parts) if parts.len() == 2 => {
                Ok([parts[0].clone(), parts[1].clone()])
            }
            _ => Err(SchemaImportError::new(
                "expected object pair to lower as key-colon/value sequence".to_string(),
            )),
        }
    }

    fn split_object_pair_symbol_paths(pair: &GrammarExpr) -> ImportResult<Vec<[GrammarExpr; 2]>> {
        match pair {
            GrammarExpr::Choice(alternatives) => alternatives
                .iter()
                .map(Self::split_object_pair_symbols)
                .collect(),
            _ => Ok(vec![Self::split_object_pair_symbols(pair)?]),
        }
    }

    fn lower_closed_any_of_object_variants_expr_nfa(
        &mut self,
        variants: &[AnyOfFixedObjectVariant],
        include_untyped_non_object_alts: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        if variants.is_empty() {
            return Ok(None);
        }

        let mut builder = ExprNfaBuilder::new();
        let accept = builder.add_state();
        builder.set_accepting(accept);
        let mut state_ids = BTreeMap::<AnyOfFixedObjectState, u32>::new();
        let mut queue = VecDeque::<AnyOfFixedObjectState>::new();
        for (variant_idx, _) in variants.iter().enumerate() {
            let state = AnyOfFixedObjectState {
                variant_idx: variant_idx as u16,
                cursor: 0,
                has_content: false,
            };
            let state_id = builder.add_state();
            builder.add_epsilon(builder.start_state(), state_id);
            state_ids.insert(state, state_id);
            queue.push_back(state);
        }

        while let Some(state) = queue.pop_front() {
            if state_ids.len() > ANYOF_FIXED_OBJECT_EXPR_NFA_MAX_STATES {
                return Ok(None);
            }

            let state_id = state_ids[&state];
            let variant = &variants[state.variant_idx as usize];
            let cursor = state.cursor as usize;

            if variant.close_allowed(cursor) {
                builder.add_epsilon(state_id, accept);
            }

            for key in variant.legal_next_keys(cursor) {
                let Some(next_cursor) = variant.advance_cursor(cursor, key) else {
                    continue;
                };
                let Some(value_expr) = variant.value_expr_for_key(key) else {
                    continue;
                };

                let next_state = AnyOfFixedObjectState {
                    variant_idx: state.variant_idx,
                    cursor: next_cursor as u16,
                    has_content: true,
                };
                let next_state_id = if let Some(&existing) = state_ids.get(&next_state) {
                    existing
                } else {
                    let id = builder.add_state();
                    state_ids.insert(next_state, id);
                    queue.push_back(next_state);
                    id
                };

                let mut symbols = Vec::new();
                if state.has_content {
                    symbols.push(self.item_separator_expr());
                }
                symbols.push(self.lower_literal_key_colon(key));
                symbols.push(value_expr);
                Self::add_expr_nfa_symbol_path(&mut builder, state_id, symbols, next_state_id);
            }
        }

        let rule_name = self.fresh_rule_name("json_anyof_object_body");
        let body = GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized()));
        self.add_nonterminal_rule(&rule_name, body);

        let object_expr = seq(vec![lit("{"), r(&rule_name), lit("}")]);
        if include_untyped_non_object_alts {
            return Ok(Some(choice(vec![
                object_expr,
                r(JSON_ARRAY_RULE),
                r(JSON_STRING_RULE),
                r(JSON_NUMBER_RULE),
                r(JSON_BOOL_RULE),
                r(JSON_NULL_RULE),
            ])));
        }

        Ok(Some(object_expr))
    }

    fn lower_open_any_of_object_variants_expr_nfa(
        &mut self,
        variants: &[AnyOfObjectVariant],
        include_untyped_non_object_alts: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        if variants.is_empty() {
            return Ok(None);
        }

        let mut builder = ExprNfaBuilder::new();
        let accept = builder.add_state();
        builder.set_accepting(accept);
        let anyof_fixed_keys = variants
            .iter()
            .flat_map(|variant| variant.fixed_keys.iter().cloned())
            .collect::<BTreeSet<_>>();

        let mut state_ids = BTreeMap::<AnyOfObjectState, u32>::new();
        let mut queue = VecDeque::<AnyOfObjectState>::new();
        for (variant_idx, _) in variants.iter().enumerate() {
            let state = AnyOfObjectState {
                variant_idx: variant_idx as u16,
                cursor: 0,
                has_content: false,
                phase: AnyOfObjectPhase::Fixed,
            };
            let state_id = builder.add_state();
            builder.add_epsilon(builder.start_state(), state_id);
            state_ids.insert(state, state_id);
            queue.push_back(state);
        }

        while let Some(state) = queue.pop_front() {
            if state_ids.len() > ANYOF_FIXED_OBJECT_EXPR_NFA_MAX_STATES {
                return Ok(None);
            }

            let state_id = state_ids[&state];
            let variant = &variants[state.variant_idx as usize];
            let cursor = state.cursor as usize;
            let fixed_phase = state.phase == AnyOfObjectPhase::Fixed;
            let can_leave_fixed_tail = !fixed_phase || variant.close_allowed(cursor);

            if can_leave_fixed_tail {
                builder.add_epsilon(state_id, accept);
            }

            if can_leave_fixed_tail
                && state.phase != AnyOfObjectPhase::Additional
                && !variant.pattern_pairs.is_empty()
            {
                let next_state = AnyOfObjectState {
                    variant_idx: state.variant_idx,
                    cursor: variant.len() as u16,
                    has_content: true,
                    phase: AnyOfObjectPhase::Pattern,
                };
                let next_state_id = if let Some(&existing) = state_ids.get(&next_state) {
                    existing
                } else {
                    let id = builder.add_state();
                    state_ids.insert(next_state, id);
                    queue.push_back(next_state);
                    id
                };
                for pattern_pair in &variant.pattern_pairs {
                    for [key_symbol, value_symbol] in Self::split_object_pair_symbol_paths(pattern_pair)? {
                        let mut symbols = Vec::new();
                        if state.has_content {
                            symbols.push(self.item_separator_expr());
                        }
                        symbols.extend(Self::object_pair_path_symbols(key_symbol, value_symbol));
                        Self::add_expr_nfa_symbol_path(
                            &mut builder,
                            state_id,
                            symbols,
                            next_state_id,
                        );
                    }
                }
            }

            if can_leave_fixed_tail
                && let Some(value_expr) = &variant.additional_value_expr
            {
                let next_state = AnyOfObjectState {
                    variant_idx: state.variant_idx,
                    cursor: variant.len() as u16,
                    has_content: true,
                    phase: AnyOfObjectPhase::Additional,
                };
                let next_state_id = if let Some(&existing) = state_ids.get(&next_state) {
                    existing
                } else {
                    let id = builder.add_state();
                    state_ids.insert(next_state, id);
                    queue.push_back(next_state);
                    id
                };

                let mut symbols = Vec::new();
                if state.has_content {
                    symbols.push(self.item_separator_expr());
                }
                symbols.push(self.lower_additional_key_colon(
                    &anyof_fixed_keys,
                    &variant.pattern_keys,
                )?);
                symbols.push(value_expr.clone());
                Self::add_expr_nfa_symbol_path(&mut builder, state_id, symbols, next_state_id);
            }

            if !fixed_phase {
                continue;
            }

            for key in variant.legal_next_keys(cursor) {
                let Some(next_cursor) = variant.advance_cursor(cursor, key) else {
                    continue;
                };
                let Some(value_expr) = variant.value_expr_for_key(key) else {
                    continue;
                };

                let next_state = AnyOfObjectState {
                    variant_idx: state.variant_idx,
                    cursor: next_cursor as u16,
                    has_content: true,
                    phase: AnyOfObjectPhase::Fixed,
                };
                let next_state_id = if let Some(&existing) = state_ids.get(&next_state) {
                    existing
                } else {
                    let id = builder.add_state();
                    state_ids.insert(next_state, id);
                    queue.push_back(next_state);
                    id
                };

                let mut symbols = Vec::new();
                if state.has_content {
                    symbols.push(self.item_separator_expr());
                }
                symbols.push(self.lower_literal_key_colon(key));
                symbols.push(value_expr);
                Self::add_expr_nfa_symbol_path(&mut builder, state_id, symbols, next_state_id);
            }
        }

        let rule_name = self.fresh_rule_name("json_anyof_object_body");
        let body = GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized()));
        self.add_nonterminal_rule(&rule_name, body);

        let object_expr = seq(vec![lit("{"), r(&rule_name), lit("}")]);
        if include_untyped_non_object_alts {
            return Ok(Some(choice(vec![
                object_expr,
                r(JSON_ARRAY_RULE),
                r(JSON_STRING_RULE),
                r(JSON_NUMBER_RULE),
                r(JSON_BOOL_RULE),
                r(JSON_NULL_RULE),
            ])));
        }

        Ok(Some(object_expr))
    }

    fn lower_fixed_object_body_exprnfa_without_group(
        &mut self,
        items: &[ObjectItem],
        tail_pair: Option<GrammarExpr>,
    ) -> ImportResult<GrammarExpr> {
        let mut builder = ExprNfaBuilder::new();
        let mut states = vec![[0u32; 2]; items.len() + 1];
        states[0][0] = builder.start_state();
        states[0][1] = builder.add_state();
        for state_pair in states.iter_mut().skip(1) {
            state_pair[0] = builder.add_state();
            state_pair[1] = builder.add_state();
        }

        let use_separator_states = tail_pair.is_some() && items.iter().any(|item| !item.required);
        let post_separator_states = if use_separator_states {
            Some((0..=items.len()).map(|_| builder.add_state()).collect::<Vec<_>>())
        } else {
            None
        };
        let mut item_symbols = Vec::with_capacity(items.len());
        for item in items {
            item_symbols.push(Self::split_object_pair_symbols(&item.pair)?);
        }
        let tail_symbols = tail_pair
            .as_ref()
            .map(Self::split_object_pair_symbols)
            .transpose()?;

        for (index, item) in items.iter().enumerate() {
            if !item.required {
                builder.add_epsilon(states[index][0], states[index + 1][0]);
                builder.add_epsilon(states[index][1], states[index + 1][1]);
                if let Some(post_separator_states) = &post_separator_states {
                    builder.add_epsilon(post_separator_states[index], post_separator_states[index + 1]);
                }
            }
            Self::add_expr_nfa_symbol_path(
                &mut builder,
                states[index][0],
                item_symbols[index].to_vec(),
                states[index + 1][1],
            );
            if let Some(post_separator_states) = &post_separator_states {
                builder.add_transition(
                    states[index][1],
                    self.item_separator_expr(),
                    post_separator_states[index],
                );
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    post_separator_states[index],
                    item_symbols[index].to_vec(),
                    states[index + 1][1],
                );
            } else {
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    states[index][1],
                    vec![
                        self.item_separator_expr(),
                        item_symbols[index][0].clone(),
                        item_symbols[index][1].clone(),
                    ],
                    states[index + 1][1],
                );
            }
        }

        builder.set_accepting(states[items.len()][0]);
        builder.set_accepting(states[items.len()][1]);

        if let Some(tail_pair_expr) = tail_pair {
            let tail_state = builder.add_state();
            builder.set_accepting(tail_state);
            let tail_symbols = tail_symbols
                .as_ref()
                .expect("tail pair must lower as key-colon/value sequence");
            Self::add_expr_nfa_symbol_path(
                &mut builder,
                states[items.len()][0],
                tail_symbols.to_vec(),
                tail_state,
            );
            if let Some(post_separator_states) = &post_separator_states {
                builder.add_transition(
                    states[items.len()][1],
                    self.item_separator_expr(),
                    post_separator_states[items.len()],
                );
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    post_separator_states[items.len()],
                    tail_symbols.to_vec(),
                    tail_state,
                );

                let tail_post_separator_state = builder.add_state();
                builder.add_transition(
                    tail_state,
                    self.item_separator_expr(),
                    tail_post_separator_state,
                );
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    tail_post_separator_state,
                    tail_symbols.to_vec(),
                    tail_state,
                );
            } else {
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    states[items.len()][1],
                    vec![
                        self.item_separator_expr(),
                        tail_symbols[0].clone(),
                        tail_symbols[1].clone(),
                    ],
                    tail_state,
                );
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    tail_state,
                    vec![
                        self.item_separator_expr(),
                        tail_symbols[0].clone(),
                        tail_symbols[1].clone(),
                    ],
                    tail_state,
                );
            }
        }

        let rule_name = self.fresh_rule_name("json_closed_object_body");
        let body = GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized()));
        self.add_nonterminal_rule(&rule_name, body);

        Ok(seq(vec![lit("{"), r(&rule_name), lit("}")]))
    }

    fn split_literal_key_symbol(symbol: GrammarExpr) -> Vec<GrammarExpr> {
        match symbol {
            GrammarExpr::Literal(bytes) if bytes.len() > 1 => {
                let split_len = SNOWPLOW_KEY_TRIE_PREFIX_SPLIT_BYTES.min(bytes.len());
                if split_len >= bytes.len() {
                    bytes
                        .into_iter()
                        .map(|byte| GrammarExpr::Literal(vec![byte]))
                        .collect()
                } else {
                    let mut symbols = bytes[..split_len]
                        .iter()
                        .copied()
                        .map(|byte| GrammarExpr::Literal(vec![byte]))
                        .collect::<Vec<_>>();
                    symbols.push(GrammarExpr::Literal(bytes[split_len..].to_vec()));
                    symbols
                }
            }
            other => Self::split_additional_key_colon_transition(other),
        }
    }

    fn object_pair_path_symbols(
        key_symbol: GrammarExpr,
        value_symbol: GrammarExpr,
    ) -> Vec<GrammarExpr> {
        let mut symbols = Self::split_literal_key_symbol(key_symbol);
        symbols.push(value_symbol);
        symbols
    }

    fn lower_snowplow_large_pattern_object_key_trie(
        &mut self,
        items: &[ObjectItem],
        tail_pairs: &[GrammarExpr],
    ) -> ImportResult<GrammarExpr> {
        let mut builder = ExprNfaBuilder::new();
        let mut states = vec![[0u32; 2]; items.len() + 1];
        states[0][0] = builder.start_state();
        states[0][1] = builder.add_state();
        for state_pair in states.iter_mut().skip(1) {
            state_pair[0] = builder.add_state();
            state_pair[1] = builder.add_state();
        }

        let use_separator_states = items.iter().any(|item| !item.required);
        let post_separator_states = if use_separator_states {
            Some((0..=items.len()).map(|_| builder.add_state()).collect::<Vec<_>>())
        } else {
            None
        };
        let mut item_symbols = Vec::with_capacity(items.len());
        for item in items {
            item_symbols.push(Self::split_object_pair_symbols(&item.pair)?);
        }
        let tail_symbol_paths = tail_pairs
            .iter()
            .map(Self::split_object_pair_symbol_paths)
            .collect::<ImportResult<Vec<_>>>()?;

        for (index, item) in items.iter().enumerate() {
            if !item.required {
                builder.add_epsilon(states[index][0], states[index + 1][0]);
                builder.add_epsilon(states[index][1], states[index + 1][1]);
                if let Some(post_separator_states) = &post_separator_states {
                    builder.add_epsilon(post_separator_states[index], post_separator_states[index + 1]);
                }
            }
            Self::add_expr_nfa_symbol_path(
                &mut builder,
                states[index][0],
                Self::object_pair_path_symbols(
                    item_symbols[index][0].clone(),
                    item_symbols[index][1].clone(),
                ),
                states[index + 1][1],
            );
            if let Some(post_separator_states) = &post_separator_states {
                builder.add_transition(
                    states[index][1],
                    self.item_separator_expr(),
                    post_separator_states[index],
                );
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    post_separator_states[index],
                    Self::object_pair_path_symbols(
                        item_symbols[index][0].clone(),
                        item_symbols[index][1].clone(),
                    ),
                    states[index + 1][1],
                );
            } else {
                let mut symbols = vec![self.item_separator_expr()];
                symbols.extend(Self::object_pair_path_symbols(
                    item_symbols[index][0].clone(),
                    item_symbols[index][1].clone(),
                ));
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    states[index][1],
                    symbols,
                    states[index + 1][1],
                );
            }
        }

        builder.set_accepting(states[items.len()][0]);
        builder.set_accepting(states[items.len()][1]);

        let tail_state = builder.add_state();
        builder.set_accepting(tail_state);
        for tail_symbol_paths in &tail_symbol_paths {
            for tail_symbols in tail_symbol_paths {
                Self::add_expr_nfa_symbol_path(
                    &mut builder,
                    states[items.len()][0],
                    Self::object_pair_path_symbols(
                        tail_symbols[0].clone(),
                        tail_symbols[1].clone(),
                    ),
                    tail_state,
                );
            }
        }
        if let Some(post_separator_states) = &post_separator_states {
            builder.add_transition(
                states[items.len()][1],
                self.item_separator_expr(),
                post_separator_states[items.len()],
            );
            for tail_symbol_paths in &tail_symbol_paths {
                for tail_symbols in tail_symbol_paths {
                    Self::add_expr_nfa_symbol_path(
                        &mut builder,
                        post_separator_states[items.len()],
                        Self::object_pair_path_symbols(
                            tail_symbols[0].clone(),
                            tail_symbols[1].clone(),
                        ),
                        tail_state,
                    );
                }
            }

            let tail_post_separator_state = builder.add_state();
            builder.add_transition(
                tail_state,
                self.item_separator_expr(),
                tail_post_separator_state,
            );
            for tail_symbol_paths in &tail_symbol_paths {
                for tail_symbols in tail_symbol_paths {
                    Self::add_expr_nfa_symbol_path(
                        &mut builder,
                        tail_post_separator_state,
                        Self::object_pair_path_symbols(
                            tail_symbols[0].clone(),
                            tail_symbols[1].clone(),
                        ),
                        tail_state,
                    );
                }
            }
        } else {
            for tail_symbol_paths in &tail_symbol_paths {
                for tail_symbols in tail_symbol_paths {
                    let mut symbols = vec![self.item_separator_expr()];
                    symbols.extend(Self::object_pair_path_symbols(
                        tail_symbols[0].clone(),
                        tail_symbols[1].clone(),
                    ));
                    Self::add_expr_nfa_symbol_path(
                        &mut builder,
                        states[items.len()][1],
                        symbols,
                        tail_state,
                    );
                    let mut loop_symbols = vec![self.item_separator_expr()];
                    loop_symbols.extend(Self::object_pair_path_symbols(
                        tail_symbols[0].clone(),
                        tail_symbols[1].clone(),
                    ));
                    Self::add_expr_nfa_symbol_path(
                        &mut builder,
                        tail_state,
                        loop_symbols,
                        tail_state,
                    );
                }
            }
        }

        let rule_name = self.fresh_rule_name("json_closed_object_body");
        let body = GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized()));
        self.add_nonterminal_rule(&rule_name, body);

        Ok(seq(vec![lit("{"), r(&rule_name), lit("}")]))
    }

    fn lower_large_optional_open_object_fused_prefix_chain(
        &mut self,
        items: &[ObjectItem],
        tail_pair_expr: GrammarExpr,
    ) -> ImportResult<GrammarExpr> {
        let mut prefix_rule_names: Vec<String> = Vec::with_capacity(items.len());
        for end_exclusive in 1..=items.len() {
            let mut alternatives = Vec::new();
            for start in 0..end_exclusive {
                if items[start..end_exclusive - 1].iter().any(|item| item.required) {
                    continue;
                }
                if start == 0 {
                    alternatives.push(items[end_exclusive - 1].pair.clone());
                } else {
                    alternatives.push(seq(vec![
                        r(&prefix_rule_names[start - 1]),
                        items[end_exclusive - 1].separator_pair.clone(),
                    ]));
                }
            }
            let rule_name = self.fresh_rule_name("json_open_object_prefix");
            self.add_nonterminal_rule(&rule_name, choice(alternatives));
            prefix_rule_names.push(rule_name);
        }

        let free_nonempty_rule = self.fresh_rule_name("json_open_object_free_nonempty");
        self.add_nonterminal_rule(
            &free_nonempty_rule,
            seq(vec![
                tail_pair_expr.clone(),
                GrammarExpr::Repeat(Box::new(seq(vec![
                    self.item_separator_expr(),
                    tail_pair_expr,
                ]))),
            ]),
        );

        let mut body_alternatives = vec![GrammarExpr::Epsilon, r(&free_nonempty_rule)];
        for (index, prefix_rule_name) in prefix_rule_names.iter().enumerate() {
            if items[index + 1..].iter().any(|item| item.required) {
                continue;
            }
            body_alternatives.push(r(prefix_rule_name));
            body_alternatives.push(seq(vec![
                r(prefix_rule_name),
                self.item_separator_expr(),
                r(&free_nonempty_rule),
            ]));
        }

        Ok(seq(vec![lit("{"), choice(body_alternatives), lit("}")]))
    }

    fn lower_large_closed_object_prefix_chain(&mut self, items: &[ObjectItem]) -> GrammarExpr {
        let mut prefix_rule_names: Vec<String> = Vec::with_capacity(items.len());
        for end_exclusive in 1..=items.len() {
            let mut alternatives = Vec::new();
            for start in 0..end_exclusive {
                if items[start..end_exclusive - 1].iter().any(|item| item.required) {
                    continue;
                }
                if start == 0 {
                    alternatives.push(items[end_exclusive - 1].pair.clone());
                } else {
                    alternatives.push(seq(vec![
                        r(&prefix_rule_names[start - 1]),
                        items[end_exclusive - 1].separator_pair.clone(),
                    ]));
                }
            }
            let rule_name = self.fresh_rule_name("json_closed_object_prefix");
            self.add_nonterminal_rule(&rule_name, choice(alternatives));
            prefix_rule_names.push(rule_name);
        }

        let mut body_alternatives = Vec::new();
        if items.iter().all(|item| !item.required) {
            body_alternatives.push(GrammarExpr::Epsilon);
        }
        for (index, prefix_rule_name) in prefix_rule_names.iter().enumerate() {
            if items[index + 1..].iter().any(|item| item.required) {
                continue;
            }
            body_alternatives.push(r(prefix_rule_name));
        }

        seq(vec![lit("{"), choice(body_alternatives), lit("}")])
    }

    fn lower_large_closed_object_fixed_pair_loop(
        &mut self,
        items: &[ObjectItem],
    ) -> ImportResult<GrammarExpr> {
        let mut grouped_keys: Vec<(GrammarExpr, Vec<GrammarExpr>)> = Vec::new();
        for item in items {
            let [key_symbol, value_symbol] = Self::split_object_pair_symbols(&item.pair)?;
            if let Some((_, key_symbols)) = grouped_keys
                .iter_mut()
                .find(|(group_value_symbol, _)| *group_value_symbol == value_symbol)
            {
                key_symbols.push(key_symbol);
            } else {
                grouped_keys.push((value_symbol, vec![key_symbol]));
            }
        }

        let mut grouped_pairs = Vec::with_capacity(grouped_keys.len());
        for (value_symbol, key_symbols) in grouped_keys {
            let mut key_builder = ExprNfaBuilder::new();
            let key_start = key_builder.start_state();
            let key_end = key_builder.add_state();
            key_builder.set_accepting(key_end);
            for key_symbol in key_symbols {
                Self::add_expr_nfa_symbol_path(
                    &mut key_builder,
                    key_start,
                    Self::split_literal_key_symbol(key_symbol),
                    key_end,
                );
            }

            let key_rule_name = self.fresh_rule_name("json_closed_object_key_group");
            self.add_nonterminal_rule(
                &key_rule_name,
                GrammarExpr::ExprNFA(Box::new(
                    key_builder.build().into_determinized_and_minimized(),
                )),
            );
            grouped_pairs.push((r(&key_rule_name), value_symbol));
        }

        let mut builder = ExprNfaBuilder::new();
        let start_state = builder.start_state();
        let seen_state = builder.add_state();
        builder.set_accepting(start_state);
        builder.set_accepting(seen_state);

        for (key_group_ref, value_symbol) in grouped_pairs {
            let pair_path = vec![key_group_ref.clone(), value_symbol.clone()];
            Self::add_expr_nfa_symbol_path(
                &mut builder,
                start_state,
                pair_path.clone(),
                seen_state,
            );
            let mut loop_symbols = vec![self.item_separator_expr()];
            loop_symbols.extend(pair_path);
            Self::add_expr_nfa_symbol_path(&mut builder, seen_state, loop_symbols, seen_state);
        }

        let body_rule_name = self.fresh_rule_name("json_closed_object_fixed_pair_loop_body");
        self.add_nonterminal_rule(
            &body_rule_name,
            GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized())),
        );

        Ok(seq(vec![
            lit("{"),
            r(&body_rule_name),
            lit("}"),
        ]))
    }

    fn lower_required_prefix_open_object_pair_loop(
        &mut self,
        items: &[ObjectItem],
        required_prefix_len: usize,
        tail_pairs: &[GrammarExpr],
    ) -> ImportResult<GrammarExpr> {
        let mut builder = ExprNfaBuilder::new();
        let start_state = builder.start_state();
        let mut required_state = start_state;

        for (index, item) in items.iter().take(required_prefix_len).enumerate() {
            let [key_symbol, value_symbol] = Self::split_object_pair_symbols(&item.pair)?;
            let next_state = builder.add_state();
            let path = if index == 0 {
                Self::object_pair_path_symbols(key_symbol, value_symbol)
            } else {
                let mut symbols = vec![self.item_separator_expr()];
                symbols.extend(Self::object_pair_path_symbols(key_symbol, value_symbol));
                symbols
            };
            Self::add_expr_nfa_symbol_path(&mut builder, required_state, path, next_state);
            required_state = next_state;
        }

        builder.set_accepting(required_state);

        let mut pair_paths = Vec::new();
        let mut grouped_keys: Vec<(GrammarExpr, Vec<GrammarExpr>)> = Vec::new();
        for item in &items[required_prefix_len..] {
            let [key_symbol, value_symbol] = Self::split_object_pair_symbols(&item.pair)?;
            if let Some((_, key_symbols)) = grouped_keys
                .iter_mut()
                .find(|(group_value_symbol, _)| *group_value_symbol == value_symbol)
            {
                key_symbols.push(key_symbol);
            } else {
                grouped_keys.push((value_symbol, vec![key_symbol]));
            }
        }

        for (value_symbol, key_symbols) in grouped_keys {
            let mut key_builder = ExprNfaBuilder::new();
            let key_start = key_builder.start_state();
            let key_end = key_builder.add_state();
            key_builder.set_accepting(key_end);
            for key_symbol in key_symbols {
                Self::add_expr_nfa_symbol_path(
                    &mut key_builder,
                    key_start,
                    Self::split_literal_key_symbol(key_symbol),
                    key_end,
                );
            }

            let key_rule_name = self.fresh_rule_name("json_closed_object_key_group");
            self.add_nonterminal_rule(
                &key_rule_name,
                GrammarExpr::ExprNFA(Box::new(
                    key_builder.build().into_determinized_and_minimized(),
                )),
            );
            pair_paths.push(vec![r(&key_rule_name), value_symbol]);
        }

        for tail_pair in tail_pairs {
            for [key_symbol, value_symbol] in Self::split_object_pair_symbol_paths(tail_pair)? {
                pair_paths.push(Self::object_pair_path_symbols(key_symbol, value_symbol));
            }
        }

        let loop_state = builder.add_state();
        builder.set_accepting(loop_state);

        for pair_path in pair_paths {
            let mut first_loop_symbols = vec![self.item_separator_expr()];
            first_loop_symbols.extend(pair_path.clone());
            Self::add_expr_nfa_symbol_path(
                &mut builder,
                required_state,
                first_loop_symbols,
                loop_state,
            );
            let mut loop_symbols = vec![self.item_separator_expr()];
            loop_symbols.extend(pair_path);
            Self::add_expr_nfa_symbol_path(
                &mut builder,
                loop_state,
                loop_symbols,
                loop_state,
            );
        }

        let body_rule_name =
            self.fresh_rule_name("json_required_prefix_open_object_pair_loop_body");
        self.add_nonterminal_rule(
            &body_rule_name,
            GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized())),
        );

        Ok(seq(vec![lit("{"), r(&body_rule_name), lit("}")]))
    }

    fn lower_property_item(
        &mut self,
        property: &PropertySchema,
        pattern_properties: &[PatternPropertySchema],
        required: bool,
        satisfies_any_group: bool,
        exclusive_group: bool,
    ) -> ImportResult<ObjectItem> {
        let key = self.lower_literal_key_colon(&property.name);
        let separator_key = self.lower_literal_key_colon_with_prefix(b", ", &property.name);
        let mut effective_schema = property.schema.clone();
        for pattern_property in pattern_properties {
            if property_matches_pattern(&pattern_property.pattern, &property.name)? {
                let pattern_schema = pattern_schema_for_property(&effective_schema, &pattern_property.schema);
                effective_schema = all_of_schema(effective_schema, pattern_schema);
            }
        }
        let value = self.lower_schema(&effective_schema)?;
        Ok(ObjectItem {
            pair: seq(vec![key, value.clone()]),
            separator_pair: seq(vec![separator_key, value]),
            required,
            satisfies_any_group,
            exclusive_group,
        })
    }

    fn object_with_required_synthetic_properties(
        &self,
        schema: &ObjectSchema,
    ) -> ImportResult<ObjectSchema> {
        let mut normalized = schema.clone();
        let mut known = normalized
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();

        for required_name in &schema.required {
            if known.contains(required_name) {
                continue;
            }

            let synthetic_schema = match &schema.additional_properties {
                AdditionalProperties::AllowAny => Schema::any(format!("<required:{required_name}>")),
                AdditionalProperties::Schema(schema) => schema.as_ref().clone(),
                AdditionalProperties::Deny => {
                    return Err(SchemaImportError::new(format!(
                        "required property {required_name:?} is not listed in properties and additionalProperties is false"
                    )));
                }
            };
            normalized.properties.push(PropertySchema {
                name: required_name.clone(),
                schema: synthetic_schema,
            });
            known.insert(required_name.clone());
        }

        Ok(normalized)
    }
}

fn is_ref_string_open_object_branch(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.array.is_some()
        || assertions.number.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return false;
    }
    if let Some(types) = &assertions.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return false;
        }
    }

    let Some(object) = &assertions.object else {
        return false;
    };
    if object.pattern_properties.len() != 0 || object.properties.len() != 1 {
        return false;
    }
    if object.properties[0].name != "$ref" {
        return false;
    }
    if !matches!(object.additional_properties, AdditionalProperties::AllowAny) {
        return false;
    }

    is_string_schema(&object.properties[0].schema)
}

fn all_of_has_explicit_object_only_type(branches: &[Schema]) -> bool {
    branches.iter().any(schema_has_explicit_object_only_type)
}

fn schema_has_explicit_object_only_type(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };
    if assertions
        .types
        .as_ref()
        .is_some_and(|types| types.iter().all(|schema_type| *schema_type == SchemaType::Object))
    {
        return true;
    }
    all_of_has_explicit_object_only_type(&assertions.all_of)
}

fn is_plain_array_branch(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
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
        return false;
    }
    if let Some(types) = &assertions.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::Array) {
            return false;
        }
        return true;
    }

    assertions.array.is_some()
}

fn is_string_schema(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };
    if assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.number.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
    {
        return false;
    }
    if let Some(types) = &assertions.types {
        if !types.iter().all(|schema_type| *schema_type == SchemaType::String) {
            return false;
        }
        return true;
    }

    assertions.string.is_some()
}

fn property_matches_pattern(pattern: &str, property_name: &str) -> ImportResult<bool> {
    property_name_matches_pattern(pattern, property_name)
}

fn pattern_schema_for_property(property_schema: &Schema, pattern_schema: &Schema) -> Schema {
    let Some(property_type) = single_numeric_property_type(property_schema) else {
        return pattern_schema.clone();
    };

    let SchemaKind::Assertions(assertions) = &pattern_schema.kind else {
        return pattern_schema.clone();
    };
    if assertions.types.is_some() || assertions.number.is_none() || has_non_numeric_assertions(assertions) {
        return pattern_schema.clone();
    }

    let mut typed = assertions.as_ref().clone();
    typed.types = Some(vec![property_type]);
    Schema::assertions(pattern_schema.location.clone(), typed)
}

fn single_numeric_property_type(property_schema: &Schema) -> Option<SchemaType> {
    let SchemaKind::Assertions(assertions) = &property_schema.kind else {
        return None;
    };
    match assertions.types.as_deref() {
        Some([SchemaType::Integer]) => Some(SchemaType::Integer),
        Some([SchemaType::Number]) => Some(SchemaType::Number),
        _ => None,
    }
}

fn has_non_numeric_assertions(assertions: &SchemaAssertions) -> bool {
    assertions.const_value.is_some()
        || assertions.enum_values.is_some()
        || assertions.object.is_some()
        || assertions.array.is_some()
        || assertions.string.is_some()
        || !assertions.any_of.is_empty()
        || !assertions.one_of.is_empty()
        || !assertions.all_of.is_empty()
}
