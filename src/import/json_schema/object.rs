use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::grammar::expr_nfa::{ExprNFA, ExprNfaBuilder};
use crate::import::ast::{GrammarExpr, Quantifier};

use super::ast::{
    AdditionalProperties, ObjectSchema, PatternPropertySchema,
    PropertySchema, Schema, SchemaAssertions, SchemaKind, SchemaType,
};
use super::combinators::{
    all_of_schema, merge_two_objects, open_object_any_of_covers_json_object,
    try_merge_all_of_objects,
};
use super::error::{ImportResult, SchemaImportError};
use super::split_literal_terminals_enabled;
use super::lower::{
    choice, lit, lit_bytes, never, normalize_local_ref, r, seq, FixedObjectTemplateKey,
    Lowerer, JSON_ARRAY_RULE,
    json_key_string_rule,
    JSON_BOOL_RULE,
    JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE,
    JSON_ADDITIONAL_KEY_COLON_SHARED_RULE, JSON_NULL_RULE,
    JSON_KEY_SEPARATOR_RULE, JSON_KEY_SUFFIX_RULE, JSON_QUOTE_RULE,
    JSON_NUMBER_RULE, JSON_OBJECT_RULE, JSON_STRING_RULE, JSON_VALUE_RULE,
};
use super::string::property_name_matches_pattern;

const LARGE_OBJECT_LITERAL_KEY_TRIE_MIN_ITEMS: usize = 64;
const LARGE_OBJECT_KEY_TRIE_PREFIX_SPLIT_BYTES: usize = 1;

struct ObjectItem {
    key: String,
    pair: GrammarExpr,
    separator_pair: GrammarExpr,
    required: bool,
    satisfies_any_group: bool,
    exclusive_group: bool,
}

const ANYOF_FIXED_OBJECT_EXPR_NFA_MAX_STATES: usize = 4096;
const PROPERTY_DEPENDENCY_MAX_FIXED_PROPERTIES: usize = 12;

fn discriminator_anyof_fastpath_disabled() -> bool {
    std::env::var_os("GLRMASK_DISABLE_DISCRIMINATOR_ANYOF_FASTPATH").is_some()
}

struct AnyOfFixedObjectItem {
    key: String,
    value_expr: GrammarExpr,
    value_identity: Option<String>,
    schema: Schema,
    required: bool,
}

struct AnyOfFixedObjectVariant {
    items: Vec<AnyOfFixedObjectItem>,
}

#[derive(Default)]
struct ClosedAnyOfCollectionProfile {
    lower_property_ms: f64,
    identity_ms: f64,
    schema_clone_ms: f64,
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
struct AnyOfFixedObjectCursor {
    variant_idx: u16,
    cursor: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AnyOfFixedObjectMergedState {
    cursors: Vec<AnyOfFixedObjectCursor>,
    has_content: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum AnyOfObjectPhase {
    Fixed,
    Pattern,
    Additional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ShadowOwnerState {
    None,
    Impossible,
    Possible {
        variant_idx: u16,
        cursor: u16,
        phase: AnyOfObjectPhase,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AnyOfObjectState {
    variant_idx: u16,
    cursor: u16,
    has_content: bool,
    phase: AnyOfObjectPhase,
    shadow_owner: ShadowOwnerState,
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

fn is_unconstrained_open_object_schema(schema: &ObjectSchema) -> bool {
    schema.properties.is_empty()
        && schema.required.is_empty()
        && schema.min_properties == 0
        && schema.max_properties.is_none()
        && schema.pattern_properties.is_empty()
        && matches!(schema.additional_properties, AdditionalProperties::AllowAny)
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

    fn item_for_key(&self, key: &str) -> Option<&AnyOfFixedObjectItem> {
        self.items.iter().find(|item| item.key == key)
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

    fn has_required_items(&self) -> bool {
        self.items.iter().any(|item| item.required)
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

    fn schema_contains_any_ref(schema: &Schema) -> bool {
        match &schema.kind {
            SchemaKind::Ref(_) => true,
            SchemaKind::Any | SchemaKind::Never => false,
            SchemaKind::Assertions(assertions) => {
                assertions.object.as_ref().is_some_and(|object| {
                    object.properties.iter().any(|property| Self::schema_contains_any_ref(&property.schema))
                        || object.pattern_properties.iter().any(|property| {
                            Self::schema_contains_any_ref(&property.schema)
                        })
                        || object.property_names.as_ref().is_some_and(Self::schema_contains_any_ref)
                        || matches!(
                            &object.additional_properties,
                            AdditionalProperties::Schema(schema) if Self::schema_contains_any_ref(schema)
                        )
                }) || assertions.array.as_ref().is_some_and(|array| {
                    Self::schema_contains_any_ref(&array.items)
                        || array.prefix_items.iter().any(Self::schema_contains_any_ref)
                }) || assertions.any_of.iter().any(Self::schema_contains_any_ref)
                    || assertions.one_of.iter().any(Self::schema_contains_any_ref)
                    || assertions.all_of.iter().any(Self::schema_contains_any_ref)
                    || assertions.not.as_ref().is_some_and(Self::schema_contains_any_ref)
            }
        }
    }

    pub(crate) fn try_lower_closed_object_any_of_variants(
        &mut self,
        branches: &[Schema],
        suppress_untyped_non_object_alts: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        let profile_enabled = branches.len() >= 32
            && (std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some());
        let profile_started_at = profile_enabled.then(std::time::Instant::now);
        if branches.len() < 2 {
            return Ok(None);
        }
        if branches.iter().any(Self::schema_contains_any_ref)
            && self.has_recursive_ref_branch(branches)?
        {
            return Ok(None);
        }
        if self.has_duplicate_recursive_ref_branches(branches)? {
            return Ok(None);
        }

        if let Some(expr) = self.try_lower_ordered_string_discriminator_closed_anyof(branches)? {
            return Ok(Some(expr));
        }

        let collect_started_at = profile_enabled.then(std::time::Instant::now);
        let mut variants = Vec::with_capacity(branches.len());
        let mut collection_profile = profile_enabled.then(ClosedAnyOfCollectionProfile::default);
        let mut include_untyped_non_object_alts = false;
        for branch in branches {
            let Some((variant, branch_requires_untyped_non_object_alts)) =
                self.collect_closed_any_of_object_variant(branch, collection_profile.as_mut())?
            else {
                return Ok(None);
            };
            include_untyped_non_object_alts |=
                branch_requires_untyped_non_object_alts && !suppress_untyped_non_object_alts;
            variants.push(variant);
        }

        let collect_ms = collect_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        let build_started_at = profile_enabled.then(std::time::Instant::now);
        let result = self.lower_closed_any_of_object_variants_expr_nfa(
            &variants,
            include_untyped_non_object_alts,
        )?;
        if let (Some(profile_started_at), Some(collect_ms), Some(build_started_at), Some(collection_profile)) =
            (profile_started_at, collect_ms, build_started_at, collection_profile)
        {
            eprintln!(
                "[glrmask/profile][json_schema_closed_anyof] branches={} variants={} outcome={} collect_ms={:.3} property_lower_ms={:.3} identity_ms={:.3} schema_clone_ms={:.3} nfa_build_ms={:.3} elapsed_ms={:.3}",
                branches.len(),
                variants.len(),
                result.is_some(),
                collect_ms,
                collection_profile.lower_property_ms,
                collection_profile.identity_ms,
                collection_profile.schema_clone_ms,
                build_started_at.elapsed().as_secs_f64() * 1000.0,
                profile_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(result)
    }

    /// Fast path for a closed object union whose first, required property is a
    /// unique singleton string discriminator in every branch.  The generic
    /// closed-union builder initially creates one NFA state per branch and
    /// discovers this shared key only during determinization.  Here the shared
    /// key is represented directly, while the discriminator value selects the
    /// branch-specific remainder.  This is the same language, including the
    /// declaration-order restriction used by the ordinary closed-object path.
    fn try_lower_ordered_string_discriminator_closed_anyof(
        &mut self,
        branches: &[Schema],
    ) -> ImportResult<Option<GrammarExpr>> {
        let profile_enabled = branches.len() >= 32
            && (std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some());
        let profile_started_at = profile_enabled.then(std::time::Instant::now);
        if branches.len() < 2 {
            return Ok(None);
        }

        let mut discriminator_key: Option<&str> = None;
        let mut payload_key: Option<&str> = None;
        let mut cases: Vec<(String, &Schema)> = Vec::with_capacity(branches.len());
        let mut seen_values = BTreeSet::new();

        for branch in branches {
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
                || !assertions.all_of.is_empty()
                || assertions.not.is_some()
            {
                return Ok(None);
            }
            let Some(types) = assertions.types.as_ref() else {
                return Ok(None);
            };
            if types.is_empty() || !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
                return Ok(None);
            }
            let Some(object) = assertions.object.as_ref() else {
                return Ok(None);
            };
            if !matches!(object.additional_properties, AdditionalProperties::Deny)
                || !object.pattern_properties.is_empty()
                || object.property_names.is_some()
                || !object.property_dependencies.is_empty()
                || object.min_properties != 0
                || object.max_properties.is_some()
                || object.properties.len() != 2
                || object.required.len() != 2
            {
                return Ok(None);
            }

            let discriminator = &object.properties[0];
            let payload = &object.properties[1];
            if !object.required.contains(&discriminator.name)
                || !object.required.contains(&payload.name)
            {
                return Ok(None);
            }
            let Some(discriminator_value) = plain_singleton_string_enum_value(&discriminator.schema) else {
                return Ok(None);
            };
            match discriminator_key {
                Some(key) if key != discriminator.name => return Ok(None),
                None => discriminator_key = Some(&discriminator.name),
                _ => {}
            }
            match payload_key {
                Some(key) if key != payload.name => return Ok(None),
                None => payload_key = Some(&payload.name),
                _ => {}
            }
            if !seen_values.insert(discriminator_value.clone()) {
                return Ok(None);
            }
            cases.push((discriminator_value, &payload.schema));
        }

        let discriminator_key = discriminator_key.expect("nonempty branch set has a discriminator key");
        let payload_key = payload_key.expect("nonempty branch set has a payload key");
        let validation_ms = profile_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        let mut builder = ExprNfaBuilder::new();
        let accept = builder.add_state();
        builder.set_accepting(accept);
        let after_discriminator_key = builder.add_state();
        builder.add_transition(
            builder.start_state(),
            self.lower_literal_key_colon(discriminator_key),
            after_discriminator_key,
        );
        let payload_key_prefix = self.lower_literal_key_colon_with_prefix(b", ", payload_key);
        let payload_lower_started_at = profile_enabled.then(std::time::Instant::now);
        let mut slow_payloads = profile_enabled.then(Vec::new);

        for (discriminator_value, payload_schema) in cases {
            let after_discriminator_value = builder.add_state();
            builder.add_transition(
                after_discriminator_key,
                self.lower_string_literal(&discriminator_value),
                after_discriminator_value,
            );
            let payload_started_at = profile_enabled.then(std::time::Instant::now);
            let payload_expr = self.lower_schema(payload_schema)?;
            if let (Some(payload_started_at), Some(slow_payloads)) =
                (payload_started_at, slow_payloads.as_mut())
            {
                let shape = match &payload_schema.kind {
                    SchemaKind::Assertions(assertions) => assertions.object.as_ref().map(|object| {
                        (object.properties.len(), object.required.len())
                    }),
                    _ => None,
                };
                slow_payloads.push((
                    payload_started_at.elapsed().as_secs_f64() * 1000.0,
                    discriminator_value.clone(),
                    shape,
                ));
            }
            Self::add_expr_nfa_symbol_path(
                &mut builder,
                after_discriminator_value,
                vec![
                    payload_key_prefix.clone(),
                    payload_expr,
                ],
                accept,
            );
        }

        let payload_lower_ms = payload_lower_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        let nfa_finalize_started_at = profile_enabled.then(std::time::Instant::now);
        let rule_name = self.fresh_rule_name("json_anyof_object_body");
        self.add_nonterminal_rule(
            &rule_name,
            GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized())),
        );
        if let (Some(profile_started_at), Some(validation_ms), Some(payload_lower_ms), Some(nfa_finalize_started_at)) =
            (
                profile_started_at,
                validation_ms,
                payload_lower_ms,
                nfa_finalize_started_at,
            )
        {
            eprintln!(
                "[glrmask/profile][ordered_discriminator] branches={} validation_ms={:.3} payload_lower_ms={:.3} nfa_finalize_ms={:.3} elapsed_ms={:.3}",
                branches.len(),
                validation_ms,
                payload_lower_ms,
                nfa_finalize_started_at.elapsed().as_secs_f64() * 1000.0,
                profile_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if let Some(slow_payloads) = &mut slow_payloads {
            slow_payloads.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
            slow_payloads.truncate(20);
            eprintln!(
                "[glrmask/profile][ordered_discriminator_slowest_payloads] payloads={:?}",
                slow_payloads,
            );
        }
        Ok(Some(seq(vec![lit("{"), r(&rule_name), lit("}")])) )
    }

    pub(crate) fn try_lower_open_object_any_of_variants(
        &mut self,
        branches: &[Schema],
    ) -> ImportResult<Option<GrammarExpr>> {
        if branches.len() < 2 {
            return Ok(None);
        }
        if branches.iter().any(Self::schema_contains_any_ref)
            && self.has_recursive_ref_branch(branches)?
        {
            return Ok(None);
        }
        if self.has_duplicate_recursive_ref_branches(branches)? {
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

    fn has_duplicate_recursive_ref_branches(&self, branches: &[Schema]) -> ImportResult<bool> {
        let mut seen = BTreeSet::new();
        for branch in branches {
            let SchemaKind::Ref(pointer) = &branch.kind else {
                continue;
            };
            let normalized = normalize_local_ref(pointer)?;
            if seen.insert(normalized.clone()) {
                continue;
            }
            let target = self.resolve_ref_target(pointer)?;
            if self.schema_transitively_refs_pointer(target, &normalized, &mut BTreeSet::new())? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn has_recursive_ref_branch(&self, branches: &[Schema]) -> ImportResult<bool> {
        for branch in branches {
            let normalized = match &branch.kind {
                SchemaKind::Ref(pointer) => normalize_local_ref(pointer)?,
                _ if branch.location.starts_with('#') => normalize_local_ref(&branch.location)?,
                _ => continue,
            };
            if self.schema_transitively_refs_pointer(branch, &normalized, &mut BTreeSet::new())? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn try_lower_ref_string_path_object_any_of(
        &mut self,
        _current_schema: &Schema,
        _branches: &[Schema],
    ) -> ImportResult<Option<GrammarExpr>> {
        // Do not recognize exact benchmark-observed path-recursive anyOf
        // shapes here. This used to collapse a "$ref" string object branch,
        // a recursive path-key object branch, and an array branch into
        // JSON_OBJECT | array after matching the exact pattern
        // `^(/[^/]+)+$`. That is too schema-tailored for the production JSON
        // importer. If this optimization is needed, reintroduce it as a
        // separately reviewed, shape-generic proof that the object branches
        // cover the full JSON object language, without matching literal
        // benchmark regexes.
        Ok(None)
    }

    fn lower_object_internal(
        &mut self,
        schema: &ObjectSchema,
        any_required_names: Option<&BTreeSet<String>>,
        exclusive_group: Option<(&BTreeSet<String>, bool)>,
    ) -> ImportResult<GrammarExpr> {
        let normalized = self.object_with_required_synthetic_properties(schema)?;
        // llguidance-compatible broad fallback: when `propertyNames` and
        // `patternProperties` are both present, keep patternProperties lowering
        // and ignore propertyNames.  Enforcing the name pattern here rejects
        // tokens llguidance keeps admissible for these schemas.
        let property_name_pattern = if normalized.pattern_properties.is_empty() {
            self.resolve_property_names_pattern(&normalized)?
        } else {
            None
        };
        if any_required_names.is_none()
            && exclusive_group.is_none()
            && is_unconstrained_open_object_schema(&normalized)
        {
            return Ok(r(JSON_OBJECT_RULE));
        }
        if normalized
            .max_properties
            .is_some_and(|max_properties| max_properties < normalized.min_properties)
        {
            return Ok(never());
        }
        let mut max_property_group_exclusive = false;
        let mut drop_optional_fixed_properties = false;
        let max_properties_filled_by_required = normalized
            .max_properties
            .is_some_and(|max_properties| max_properties == normalized.required.len())
            && any_required_names.is_none()
            && exclusive_group.is_none();
        if let Some(max_properties) = normalized.max_properties {
            let fixed_closed = normalized.pattern_properties.is_empty()
                && matches!(normalized.additional_properties, AdditionalProperties::Deny);
            if max_properties < normalized.required.len() {
                return Ok(never());
            }
            if max_properties_filled_by_required {
                drop_optional_fixed_properties = true;
            }
            let fixed_closed_redundant = fixed_closed && max_properties >= normalized.properties.len();
            let fixed_closed_optional_cap = fixed_closed
                && max_properties < normalized.properties.len()
                && any_required_names.is_none()
                && exclusive_group.is_none();
            let pattern_map_candidate = normalized.properties.is_empty()
                && normalized.required.is_empty()
                && matches!(normalized.additional_properties, AdditionalProperties::Deny)
                && normalized.pattern_properties.len() == 1;
            let open_dynamic_map_candidate = normalized.properties.is_empty()
                && normalized.required.is_empty()
                && !matches!(normalized.additional_properties, AdditionalProperties::Deny);
            if fixed_closed_optional_cap {
                match max_properties - normalized.required.len() {
                    0 => drop_optional_fixed_properties = true,
                    1 => max_property_group_exclusive = true,
                    _ => {
                        // Broad build-parity fallback: unsupported fixed-property
                        // caps are overapproximated by ignoring maxProperties.
                    }
                }
            } else if !fixed_closed_redundant && !pattern_map_candidate && !open_dynamic_map_candidate {
                // Broad build-parity fallback: preserve the rest of the object
                // constraints and ignore unsupported maxProperties shapes.
            }
        }
        let extra_min_properties =
            normalized.min_properties.saturating_sub(normalized.required.len());
        let min_property_group_required = if extra_min_properties == 0 {
            false
        } else if normalized.properties.is_empty()
            && normalized.required.is_empty()
            && !matches!(normalized.additional_properties, AdditionalProperties::Deny)
        {
            false
        } else if normalized.properties.is_empty()
            && normalized.required.is_empty()
            && matches!(normalized.additional_properties, AdditionalProperties::Deny)
            && normalized.pattern_properties.len() == 1
        {
            false
        } else if extra_min_properties == 1
            && matches!(normalized.additional_properties, AdditionalProperties::Deny)
            && normalized.pattern_properties.is_empty()
            && normalized.properties.len() > normalized.required.len()
            && any_required_names.is_none()
            && exclusive_group.is_none()
        {
            true
        } else {
            // Broad build-parity fallback: preserve the rest of the object
            // constraints and ignore unsupported minProperties shapes.
            false
        };
        let fixed_names = (property_name_pattern.is_some()
            || !normalized.pattern_properties.is_empty()
            || !matches!(normalized.additional_properties, AdditionalProperties::Deny))
        .then(|| {
            normalized
                .properties
                .iter()
                .map(|property| property.name.clone())
                .collect::<BTreeSet<_>>()
        });
        let implicit_ap_default_false = self.llguidance_compat_enabled()
            && normalized.required.len() > normalized.properties.len()
            && matches!(normalized.additional_properties, AdditionalProperties::AllowAny);
        if let Some(pattern) = &property_name_pattern {
            for key in fixed_names
                .as_ref()
                .expect("propertyNames requires fixed property names")
            {
                if !property_name_matches_pattern(pattern, key)? {
                    return Err(SchemaImportError::new(format!(
                        "propertyNames pattern {pattern:?} does not allow fixed property {key:?}"
                    )));
                }
            }
        }
        let items = normalized
            .properties
            .iter()
            .filter(|property| {
                !drop_optional_fixed_properties || normalized.required.contains(&property.name)
            })
            .map(|property| {
                let required = normalized.required.contains(&property.name);
                self.lower_property_item(
                    property,
                    &normalized.pattern_properties,
                    required,
                    any_required_names
                        .is_some_and(|names| names.contains(&property.name))
                        || (min_property_group_required && !required),
                    exclusive_group
                        .is_some_and(|(names, _)| names.contains(&property.name))
                        || (max_property_group_exclusive && !required),
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
                AdditionalProperties::AllowAny => false,
                AdditionalProperties::Deny => false,
            }
            && normalized.properties.len() >= 16;

        if normalized.pattern_properties.is_empty() && !normalized.properties.is_empty() {
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
                AdditionalProperties::AllowAny if implicit_ap_default_false || max_properties_filled_by_required => None,
                AdditionalProperties::AllowAny => Some(seq(vec![
                    self.lower_object_additional_key_colon(
                        fixed_names
                            .as_ref()
                            .expect("open objects require fixed property names"),
                        &[],
                        property_name_pattern.as_deref(),
                    )?,
                    r(JSON_VALUE_RULE),
                ])),
                AdditionalProperties::Schema(value_schema) => {
                    let value = self.lower_schema(value_schema)?;
                    Some(seq(vec![
                        self.lower_object_additional_key_colon(
                            fixed_names
                                .as_ref()
                                .expect("open objects require fixed property names"),
                            &[],
                            property_name_pattern.as_deref(),
                        )?,
                        value,
                    ]))
                }
            };
            if !normalized.property_dependencies.is_empty() {
                if any_required_names.is_some()
                    || exclusive_group.is_some()
                    || min_property_group_required
                    || max_property_group_exclusive
                    || drop_optional_fixed_properties
                {
                    return Err(SchemaImportError::new(
                        "property dependencies are not supported with grouped object constraints"
                            .to_string(),
                    ));
                }
                if !matches!(normalized.additional_properties, AdditionalProperties::Deny) {
                    return Err(SchemaImportError::new(
                        "property dependencies are only supported for fixed-property objects"
                            .to_string(),
                    ));
                }
                return self.lower_fixed_object_body_exprnfa_with_property_dependencies(
                    &items,
                    &normalized.property_dependencies,
                );
            }
            if use_large_optional_open_object_prefix_chain {
                if let Some(tail_pair_expr) = tail_pair {
                    return self.lower_large_optional_open_object_fused_prefix_chain(&items, tail_pair_expr);
                }
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

        // If `propertyNames` and `patternProperties` were both present,
        // `property_name_pattern` has intentionally been dropped above as a
        // broad llguidance-compatible fallback.
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
            let key = self.lower_pattern_key_colon_appearance(
                &pattern_property.pattern,
                fixed_names
                    .as_ref()
                    .expect("pattern properties require fixed property names"),
            )?;
            let value = self.lower_schema(&pattern_property.schema)?;
            tail_pairs.push(seq(vec![key, value]));
        }

        match &normalized.additional_properties {
            AdditionalProperties::AllowAny if implicit_ap_default_false || max_properties_filled_by_required => {}
            AdditionalProperties::AllowAny => {
                let key_colon = if fixed_names
                    .as_ref()
                    .expect("pattern properties require fixed property names")
                    .is_empty()
                    && pattern_keys.is_empty()
                    && property_name_pattern.is_none()
                    && super::string::json_string_compat_mode()
                        == super::string::JsonStringCompatMode::LlGuidanceNative
                {
                    seq(vec![r(json_key_string_rule()), r(JSON_KEY_SEPARATOR_RULE)])
                } else {
                    self.lower_object_additional_key_colon(
                        fixed_names
                            .as_ref()
                            .expect("open objects require fixed property names"),
                        &pattern_keys,
                        property_name_pattern.as_deref(),
                    )?
                };
                tail_pairs.push(seq(vec![key_colon, r(JSON_VALUE_RULE)]));
            }
            AdditionalProperties::Deny => {}
            AdditionalProperties::Schema(_) if max_properties_filled_by_required => {}
            AdditionalProperties::Schema(value_schema) => {
                let value = self.lower_schema(value_schema)?;
                let key_colon = if fixed_names
                    .as_ref()
                    .expect("pattern properties require fixed property names")
                    .is_empty()
                    && pattern_keys.is_empty()
                    && property_name_pattern.is_none()
                    && super::string::json_string_compat_mode()
                        == super::string::JsonStringCompatMode::LlGuidanceNative
                {
                    // Keep map-only typed additionalProperties aligned with
                    // llguidance's strict key handling.
                    seq(vec![r(json_key_string_rule()), r(JSON_KEY_SEPARATOR_RULE)])
                } else {
                    self.lower_object_additional_key_colon(
                        fixed_names
                            .as_ref()
                            .expect("open objects require fixed property names"),
                        &pattern_keys,
                        property_name_pattern.as_deref(),
                    )?
                };
                tail_pairs.push(seq(vec![
                    key_colon,
                    value,
                ]));
            }
        }

        let mut tail_pair_repetition = None;
        if !tail_pairs.is_empty() {
            if items.is_empty() {
                let pair = choice(tail_pairs);
                let body = self.dynamic_pair_list_body(
                    pair,
                    normalized.min_properties,
                    normalized.max_properties,
                );
                return Ok(seq(vec![lit("{"), body, lit("}")]));
            }

            let required_prefix_len = items.iter().take_while(|item| item.required).count();
            let required_count = items.iter().filter(|item| item.required).count();
            let use_large_optional_open_object_prefix_chain = normalized.required.is_empty()
                && any_required_names.is_none()
                && exclusive_group.is_none()
                && !matches!(normalized.additional_properties, AdditionalProperties::Deny)
                && normalized.properties.len() >= 16;
            if use_large_optional_open_object_prefix_chain {
                return self.lower_large_optional_open_object_fused_prefix_chain(
                    &items,
                    choice(tail_pairs),
                );
            }

            let use_required_prefix_large_optional_open_object_prefix_chain = required_prefix_len > 0
                && required_prefix_len == required_count
                && any_required_names.is_none()
                && exclusive_group.is_none()
                && !matches!(normalized.additional_properties, AdditionalProperties::Deny)
                && items.len().saturating_sub(required_prefix_len) + tail_pairs.len() >= 16;
            if use_required_prefix_large_optional_open_object_prefix_chain {
                return self.lower_required_prefix_large_optional_open_object_fused_prefix_chain(
                    &items,
                    required_prefix_len,
                    choice(tail_pairs),
                );
            }

            let use_large_closed_pattern_object_key_trie = normalized.properties.len()
                >= LARGE_OBJECT_LITERAL_KEY_TRIE_MIN_ITEMS
                && normalized.required.is_empty()
                && any_required_names.is_none()
                && exclusive_group.is_none()
                && matches!(normalized.additional_properties, AdditionalProperties::Deny)
                && !tail_pairs.is_empty();

            if use_large_closed_pattern_object_key_trie {
                return self.lower_large_closed_pattern_object_key_trie(&items, &tail_pairs);
            }

            // The dynamic tail is a repeated object entry.  Keep the repetition
            // on the SeparatedSequence item so the SeparatedSequence lowerer
            // threads JSON_ITEM_SEPARATOR between tail entries.  Putting `+`
            // inside the item expression would produce adjacent entries with no
            // separator, e.g. `entry entry`, and reject valid `entry, entry` maps.
            tail_pair_repetition = Some(choice(tail_pairs));
        }

        let mut separated_items = items
            .into_iter()
            .map(|item| {
                (
                    item.pair,
                    if item.required { None } else { Some(Quantifier::Optional) },
                )
            })
            .collect::<Vec<_>>();
        if let Some(tail_pair) = tail_pair_repetition {
            separated_items.push((tail_pair, Some(Quantifier::ZeroPlus)));
        }

        let body = if separated_items.is_empty() {
            GrammarExpr::Epsilon
        } else {
            GrammarExpr::SeparatedSequence {
                items: separated_items,
                separator: Box::new(self.item_separator_expr()),
                allow_empty: true,
            }
        };

        Ok(seq(vec![lit("{"), body, lit("}")]))
    }

    fn dynamic_pair_list_body(
        &self,
        pair: GrammarExpr,
        min_properties: usize,
        max_properties: Option<usize>,
    ) -> GrammarExpr {
        let separator_pair = seq(vec![self.item_separator_expr(), pair.clone()]);
        match max_properties {
            Some(0) => GrammarExpr::Epsilon,
            Some(max_properties) => {
                let tail = GrammarExpr::Quantified(Box::new(separator_pair), Quantifier::Range(min_properties.saturating_sub(1), Some(max_properties - 1)));
                if min_properties == 0 {
                    choice(vec![GrammarExpr::Epsilon, seq(vec![pair, tail])])
                } else {
                    seq(vec![pair, tail])
                }
            }
            None => {
                let required_tail = (0..min_properties.saturating_sub(1))
                    .map(|_| separator_pair.clone())
                    .collect::<Vec<_>>();
                let nonempty = seq(vec![
                    pair,
                    seq(required_tail),
                    GrammarExpr::Quantified(Box::new(separator_pair), Quantifier::ZeroPlus),
                ]);
                if min_properties == 0 {
                    choice(vec![GrammarExpr::Epsilon, nonempty])
                } else {
                    nonempty
                }
            }
        }
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

        if schema.max_properties.is_some_and(|max| max < schema.min_properties) {
            return Ok(Some(never()));
        }

        let body = match schema.max_properties {
            Some(0) => GrammarExpr::Epsilon,
            Some(max) => {
                let separator_pair = seq(vec![self.item_separator_expr(), r(&pair_name)]);
                if schema.min_properties == 0 {
                    GrammarExpr::Quantified(Box::new(seq(vec![
                        r(&pair_name),
                        GrammarExpr::Quantified(Box::new(separator_pair), Quantifier::Range(0, Some(max - 1))),
                    ])), Quantifier::Optional)
                } else {
                    seq(vec![
                        r(&pair_name),
                        GrammarExpr::Quantified(Box::new(separator_pair), Quantifier::Range(schema.min_properties - 1, Some(max - 1))),
                    ])
                }
            }
            None => {
                let list_name = self.fresh_rule_name("json_pattern_map_list");
                self.add_nonterminal_rule(
                    &list_name,
                    choice(vec![
                        r(&pair_name),
                        seq(vec![r(&list_name), self.item_separator_expr(), r(&pair_name)]),
                    ]),
                );

                if schema.min_properties == 0 {
                    choice(vec![GrammarExpr::Epsilon, r(&list_name)])
                } else {
                    r(&list_name)
                }
            }
        };

        let body_name = self.fresh_rule_name("json_pattern_map_body");
        self.add_nonterminal_rule(&body_name, body);

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

        if open_object_any_of_covers_json_object(&assertions.any_of) {
            if super::string::json_string_compat_mode()
                == super::string::JsonStringCompatMode::JsonSchema
            {
                return Ok(Some(r(JSON_OBJECT_RULE)));
            }

            // In llguidance mode, preserve open anyOf lowering so unknown keys
            // use additional-key semantics instead of strict JSON_OBJECT keys.
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

    fn lower_fixed_object_body_exprnfa_with_property_dependencies(
        &mut self,
        items: &[ObjectItem],
        property_dependencies: &BTreeMap<String, BTreeSet<String>>,
    ) -> ImportResult<GrammarExpr> {
        if items.len() > PROPERTY_DEPENDENCY_MAX_FIXED_PROPERTIES {
            return Err(SchemaImportError::new(format!(
                "property dependencies support at most {PROPERTY_DEPENDENCY_MAX_FIXED_PROPERTIES} fixed properties"
            )));
        }

        let mut key_indexes = BTreeMap::new();
        let mut item_symbols = Vec::with_capacity(items.len());
        let mut required_mask = 0u64;
        for (index, item) in items.iter().enumerate() {
            key_indexes.insert(item.key.as_str(), index);
            item_symbols.push(Self::split_object_pair_symbols(&item.pair)?);
            if item.required {
                required_mask |= 1u64 << index;
            }
        }

        let mut dependency_masks = vec![0u64; items.len()];
        let mut impossible_trigger_mask = 0u64;
        for (trigger, dependents) in property_dependencies {
            let Some(&trigger_index) = key_indexes.get(trigger.as_str()) else {
                continue;
            };
            for dependent in dependents {
                if let Some(&dependent_index) = key_indexes.get(dependent.as_str()) {
                    dependency_masks[trigger_index] |= 1u64 << dependent_index;
                } else {
                    impossible_trigger_mask |= 1u64 << trigger_index;
                }
            }
        }

        let accepts = |seen_mask: u64| {
            if (seen_mask & required_mask) != required_mask {
                return false;
            }
            if seen_mask & impossible_trigger_mask != 0 {
                return false;
            }
            dependency_masks.iter().enumerate().all(|(index, dependency_mask)| {
                seen_mask & (1u64 << index) == 0 || (seen_mask & dependency_mask) == *dependency_mask
            })
        };

        let mut builder = ExprNfaBuilder::new();
        let mut states = Vec::with_capacity(items.len() + 1);
        for index in 0..=items.len() {
            let prefix_masks = 1usize << index;
            let mut mask_states = Vec::with_capacity(prefix_masks);
            for _ in 0..prefix_masks {
                mask_states.push([builder.add_state(), builder.add_state()]);
            }
            states.push(mask_states);
        }
        states[0][0][0] = builder.start_state();

        for (index, item) in items.iter().enumerate() {
            let item_bit = 1usize << index;
            for seen_mask in 0..(1usize << index) {
                for has_content in 0..=1 {
                    let state_id = states[index][seen_mask][has_content];
                    if !item.required {
                        builder.add_epsilon(state_id, states[index + 1][seen_mask][has_content]);
                    }
                    let next_mask = seen_mask | item_bit;
                    let mut transition_symbols = Vec::new();
                    if has_content == 1 {
                        transition_symbols.push(self.item_separator_expr());
                    }
                    transition_symbols.extend(item_symbols[index].iter().cloned());
                    Self::add_expr_nfa_symbol_path(
                        &mut builder,
                        state_id,
                        transition_symbols,
                        states[index + 1][next_mask][1],
                    );
                }
            }
        }

        for seen_mask in 0..(1usize << items.len()) {
            if accepts(seen_mask as u64) {
                builder.set_accepting(states[items.len()][seen_mask][0]);
                builder.set_accepting(states[items.len()][seen_mask][1]);
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
        profile: Option<&mut ClosedAnyOfCollectionProfile>,
    ) -> ImportResult<Option<(AnyOfFixedObjectVariant, bool)>> {
        self.collect_closed_any_of_object_variant_inner(branch, 0, profile)
    }

    fn collect_closed_any_of_object_variant_inner(
        &mut self,
        branch: &Schema,
        ref_depth: usize,
        profile: Option<&mut ClosedAnyOfCollectionProfile>,
    ) -> ImportResult<Option<(AnyOfFixedObjectVariant, bool)>> {
        let mut profile = profile;
        if let SchemaKind::Ref(pointer) = &branch.kind {
            if ref_depth >= 4 {
                return Ok(None);
            }
            let target = self.resolve_ref_target(pointer)?.clone();
            return self.collect_closed_any_of_object_variant_inner(&target, ref_depth + 1, profile);
        }

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
            merged_object = match self.try_merge_all_of_objects_resolving_refs(&assertions.all_of)?
            {
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
            let lower_started_at = profile.as_ref().map(|_| std::time::Instant::now());
            let value_expr = self.lower_schema(&property.schema)?;
            if let (Some(lower_started_at), Some(profile)) = (lower_started_at, profile.as_deref_mut()) {
                profile.lower_property_ms += lower_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            let identity_started_at = profile.as_ref().map(|_| std::time::Instant::now());
            let value_identity = exact_property_value_identity(&property.schema);
            if let (Some(identity_started_at), Some(profile)) = (identity_started_at, profile.as_deref_mut()) {
                profile.identity_ms += identity_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            let clone_started_at = profile.as_ref().map(|_| std::time::Instant::now());
            let item_schema = property.schema.clone();
            if let (Some(clone_started_at), Some(profile)) = (clone_started_at, profile.as_deref_mut()) {
                profile.schema_clone_ms += clone_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            items.push(AnyOfFixedObjectItem {
                key: property.name.clone(),
                value_expr,
                value_identity,
                schema: item_schema,
                required: object.required.contains(&property.name),
            });
        }

        Ok(Some((
            AnyOfFixedObjectVariant { items },
            include_untyped_non_object_alts,
        )))
    }

    fn try_merge_all_of_objects_resolving_refs(
        &self,
        branches: &[Schema],
    ) -> ImportResult<Option<ObjectSchema>> {
        let mut objects = Vec::with_capacity(branches.len());
        for branch in branches {
            if let Some(object) = plain_object_schema_for_closed_any_of(branch) {
                objects.push(object.clone());
                continue;
            }
            let SchemaKind::Ref(pointer) = &branch.kind else {
                return Ok(None);
            };
            let Some(object) = plain_object_schema_for_closed_any_of(self.resolve_ref_target(pointer)?)
            else {
                return Ok(None);
            };
            objects.push(object.clone());
        }

        let mut merged = objects.into_iter();
        let Some(mut object) = merged.next() else {
            return Ok(None);
        };
        for next in merged {
            object = merge_two_objects(&object, &next);
        }
        Ok(Some(object))
    }

    fn collect_open_any_of_object_variant(
        &mut self,
        branch: &Schema,
    ) -> ImportResult<Option<(AnyOfObjectVariant, bool)>> {
        self.collect_open_any_of_object_variant_inner(branch, 0)
    }

    fn collect_open_any_of_object_variant_inner(
        &mut self,
        branch: &Schema,
        ref_depth: usize,
    ) -> ImportResult<Option<(AnyOfObjectVariant, bool)>> {
        if let SchemaKind::Ref(pointer) = &branch.kind {
            if ref_depth >= 4 {
                return Ok(None);
            }
            let target = self.resolve_ref_target(pointer)?;
            return self.collect_open_any_of_object_variant_inner(target, ref_depth + 1);
        }

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
                value_identity: exact_property_value_identity(&effective_schema),
                schema: effective_schema,
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

        for symbols in Self::expand_structured_literal_key_paths(symbols) {
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
    }

    fn split_additional_key_colon_transition(symbol: GrammarExpr) -> Vec<GrammarExpr> {
        const MAX_SPLIT_ADDITIONAL_KEY_COLON_ALTERNATIVES: usize = 32;

        match symbol {
            GrammarExpr::Choice(alternatives) if alternatives.is_empty() => Vec::new(),
            GrammarExpr::Choice(alternatives)
                if Self::is_additional_key_colon_choice(&alternatives) =>
            {
                if alternatives.len() <= MAX_SPLIT_ADDITIONAL_KEY_COLON_ALTERNATIVES {
                    alternatives
                } else {
                    vec![choice(alternatives)]
                }
            }
            other => vec![other],
        }
    }

    fn is_additional_key_colon_choice(alternatives: &[GrammarExpr]) -> bool {
        alternatives.iter().any(Self::is_shared_additional_key_colon_base_ref)
            || alternatives.iter().any(Self::is_shared_additional_key_colon_addback)
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


    fn split_key_colon_symbol_paths(symbol: GrammarExpr) -> Vec<Vec<GrammarExpr>> {
        match symbol {
            GrammarExpr::Choice(alternatives) => alternatives
                .into_iter()
                .map(Self::split_literal_key_symbol)
                .collect(),
            other => vec![Self::split_literal_key_symbol(other)],
        }
    }

    fn split_object_pair_symbols(pair: &GrammarExpr) -> ImportResult<[GrammarExpr; 2]> {
        match pair {
            GrammarExpr::Sequence(parts) if parts.len() >= 2 => {
                let value = if parts.len() == 2 {
                    parts[1].clone()
                } else {
                    seq(parts[1..].to_vec())
                };
                Ok([parts[0].clone(), value])
            }
            other => Ok([other.clone(), GrammarExpr::Epsilon]),
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
        if let Some(expr) = self.try_lower_common_property_closed_any_of_object_variants_expr_nfa(
            variants,
            include_untyped_non_object_alts,
        )? {
            return Ok(Some(expr));
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
                    symbols.push(self.lower_literal_key_colon_with_prefix(b", ", key));
                } else {
                    symbols.push(self.lower_literal_key_colon(key));
                }
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

    fn try_lower_common_property_closed_any_of_object_variants_expr_nfa(
        &mut self,
        variants: &[AnyOfFixedObjectVariant],
        include_untyped_non_object_alts: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        if variants.len() < 2 || !closed_any_of_variants_have_shareable_property(variants) {
            return Ok(None);
        }

        let mut builder = ExprNfaBuilder::new();
        let accept = builder.add_state();
        builder.set_accepting(accept);
        let initial = AnyOfFixedObjectMergedState {
            cursors: (0..variants.len())
                .map(|variant_idx| AnyOfFixedObjectCursor {
                    variant_idx: variant_idx as u16,
                    cursor: 0,
                })
                .collect(),
            has_content: false,
        };
        let start = builder.add_state();
        builder.add_epsilon(builder.start_state(), start);

        let mut state_ids = BTreeMap::<AnyOfFixedObjectMergedState, u32>::new();
        let mut queue = VecDeque::<AnyOfFixedObjectMergedState>::new();
        state_ids.insert(initial.clone(), start);
        queue.push_back(initial);

        while let Some(state) = queue.pop_front() {
            if state_ids.len() > ANYOF_FIXED_OBJECT_EXPR_NFA_MAX_STATES {
                return Ok(None);
            }

            let state_id = state_ids[&state];
            if state.cursors.iter().any(|cursor| {
                variants[cursor.variant_idx as usize].close_allowed(cursor.cursor as usize)
            }) {
                builder.add_epsilon(state_id, accept);
            }

            let mut legal_keys = BTreeSet::new();
            for cursor in &state.cursors {
                let variant = &variants[cursor.variant_idx as usize];
                for key in variant.legal_next_keys(cursor.cursor as usize) {
                    legal_keys.insert(key.to_owned());
                }
            }

            for key in legal_keys {
                if let Some((value_expr, next_cursors)) =
                    shared_closed_any_of_key_transition(variants, &state.cursors, &key)
                {
                    let next_state = AnyOfFixedObjectMergedState {
                        cursors: next_cursors,
                        has_content: true,
                    };
                    let next_state_id = merged_closed_any_of_state_id(
                        &mut builder,
                        &mut state_ids,
                        &mut queue,
                        next_state,
                    );
                    let key_expr = if state.has_content {
                        self.lower_literal_key_colon_with_prefix(b", ", &key)
                    } else {
                        self.lower_literal_key_colon(&key)
                    };
                    Self::add_expr_nfa_symbol_path(
                        &mut builder,
                        state_id,
                        vec![key_expr, value_expr],
                        next_state_id,
                    );
                    continue;
                }

                for cursor in &state.cursors {
                    let variant = &variants[cursor.variant_idx as usize];
                    let Some(next_cursor) = variant.advance_cursor(cursor.cursor as usize, &key)
                    else {
                        continue;
                    };
                    let Some(value_expr) = variant.value_expr_for_key(&key) else {
                        continue;
                    };
                    let next_state = AnyOfFixedObjectMergedState {
                        cursors: vec![AnyOfFixedObjectCursor {
                            variant_idx: cursor.variant_idx,
                            cursor: next_cursor as u16,
                        }],
                        has_content: true,
                    };
                    let next_state_id = merged_closed_any_of_state_id(
                        &mut builder,
                        &mut state_ids,
                        &mut queue,
                        next_state,
                    );
                    let key_expr = if state.has_content {
                        self.lower_literal_key_colon_with_prefix(b", ", &key)
                    } else {
                        self.lower_literal_key_colon(&key)
                    };
                    Self::add_expr_nfa_symbol_path(
                        &mut builder,
                        state_id,
                        vec![key_expr, value_expr],
                        next_state_id,
                    );
                }
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

    fn is_json_value_expr(expr: &GrammarExpr) -> bool {
        matches!(expr, GrammarExpr::Ref(rule_name) if rule_name == JSON_VALUE_RULE)
    }

    fn is_json_string_expr(expr: &GrammarExpr) -> bool {
        matches!(expr, GrammarExpr::Ref(rule_name) if rule_name == JSON_STRING_RULE)
    }

    fn is_json_string_constrained_expr(expr: &GrammarExpr) -> bool {
        matches!(expr, GrammarExpr::Ref(rule_name) if rule_name.starts_with("json_string_constrained"))
    }

    fn non_string_json_value_expr() -> GrammarExpr {
        choice(vec![
            r(JSON_NULL_RULE),
            r(JSON_BOOL_RULE),
            r(JSON_NUMBER_RULE),
            r(JSON_OBJECT_RULE),
            r(JSON_ARRAY_RULE),
        ])
    }

    fn invalid_residual_value_for_owner(owner_value_expr: &GrammarExpr) -> Option<GrammarExpr> {
        if Self::is_json_string_expr(owner_value_expr) {
            return Some(Self::non_string_json_value_expr());
        }
        if Self::is_json_string_constrained_expr(owner_value_expr) {
            return Some(choice(vec![
                Self::non_string_json_value_expr(),
                GrammarExpr::Exclude {
                    expr: Box::new(r(JSON_STRING_RULE)),
                    exclude: Box::new(owner_value_expr.clone()),
                },
            ]));
        }
        None
    }

    fn select_shadow_owner_for_variant(
        variants: &[AnyOfObjectVariant],
        residual_idx: usize,
    ) -> Option<usize> {
        let residual = &variants[residual_idx];
        if !residual.pattern_pairs.is_empty()
            || !residual.pattern_keys.is_empty()
            || !residual
                .additional_value_expr
                .as_ref()
                .is_some_and(Self::is_json_value_expr)
            || residual.items.iter().any(|item| item.required)
        {
            return None;
        }

        let mut candidate = None;
        for (owner_idx, owner) in variants.iter().enumerate() {
            if owner_idx == residual_idx
                || !owner.pattern_pairs.is_empty()
                || !owner.pattern_keys.is_empty()
                || !owner
                    .additional_value_expr
                    .as_ref()
                    .is_some_and(Self::is_json_value_expr)
                || !owner.has_required_items()
            {
                continue;
            }
            if owner
                .items
                .iter()
                .any(|item| item.required && Self::invalid_residual_value_for_owner(&item.value_expr).is_none())
            {
                continue;
            }
            if candidate.replace(owner_idx).is_some() {
                return None;
            }
        }
        candidate
    }

    fn shadow_owner_suppresses_close(
        variants: &[AnyOfObjectVariant],
        shadow_owner: ShadowOwnerState,
    ) -> bool {
        match shadow_owner {
            ShadowOwnerState::Possible {
                variant_idx,
                cursor,
                phase,
            } => {
                let owner = &variants[variant_idx as usize];
                phase != AnyOfObjectPhase::Fixed || owner.close_allowed(cursor as usize)
            }
            ShadowOwnerState::None | ShadowOwnerState::Impossible => false,
        }
    }

    fn shadow_owner_can_take_additional(
        owner: &AnyOfObjectVariant,
        cursor: usize,
        phase: AnyOfObjectPhase,
    ) -> bool {
        owner.additional_value_expr.as_ref().is_some_and(Self::is_json_value_expr)
            && match phase {
                AnyOfObjectPhase::Fixed => owner.close_allowed(cursor),
                AnyOfObjectPhase::Additional => true,
                AnyOfObjectPhase::Pattern => false,
            }
    }

    fn advance_shadow_owner_on_key(
        variants: &[AnyOfObjectVariant],
        shadow_owner: ShadowOwnerState,
        key: &str,
        value_expr: &GrammarExpr,
    ) -> ShadowOwnerState {
        let ShadowOwnerState::Possible {
            variant_idx,
            cursor,
            phase,
        } = shadow_owner
        else {
            return shadow_owner;
        };
        if !Self::is_json_value_expr(value_expr) {
            return ShadowOwnerState::Impossible;
        }

        let owner = &variants[variant_idx as usize];
        let cursor = cursor as usize;
        if phase == AnyOfObjectPhase::Fixed
            && let Some(next_cursor) = owner.advance_cursor(cursor, key)
            && owner.value_expr_for_key(key).is_some()
        {
            return ShadowOwnerState::Possible {
                variant_idx,
                cursor: next_cursor as u16,
                phase: AnyOfObjectPhase::Fixed,
            };
        }

        if !owner.fixed_keys.contains(key)
            && Self::shadow_owner_can_take_additional(owner, cursor, phase)
        {
            return ShadowOwnerState::Possible {
                variant_idx,
                cursor: owner.len() as u16,
                phase: AnyOfObjectPhase::Additional,
            };
        }

        ShadowOwnerState::Impossible
    }


    fn try_lower_discriminator_value_open_any_of_object(
        &mut self,
        variants: &[AnyOfObjectVariant],
        include_untyped_non_object_alts: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        if include_untyped_non_object_alts || variants.len() < 2 {
            return Ok(None);
        }
        if variants.iter().any(|variant| {
            !variant.pattern_pairs.is_empty()
                || !variant.pattern_keys.is_empty()
                || !variant
                    .additional_value_expr
                    .as_ref()
                    .is_some_and(Self::is_json_value_expr)
                || variant.items.len() != 2
                || !variant.items.iter().all(|item| item.required)
        }) {
            return Ok(None);
        }

        let first_keys = variants[0]
            .items
            .iter()
            .map(|item| item.key.as_str())
            .collect::<Vec<_>>();
        if variants.iter().skip(1).any(|variant| {
            variant
                .items
                .iter()
                .map(|item| item.key.as_str())
                .collect::<Vec<_>>()
                != first_keys
        }) {
            return Ok(None);
        }

        let mut discriminator_index = None;
        for index in 0..first_keys.len() {
            let mut seen = BTreeSet::new();
            let mut all_singleton = true;
            for variant in variants {
                let Some(value) = singleton_string_enum_value(&variant.items[index].schema) else {
                    all_singleton = false;
                    break;
                };
                if !seen.insert(value) {
                    all_singleton = false;
                    break;
                }
            }
            if all_singleton {
                if discriminator_index.replace(index).is_some() {
                    return Ok(None);
                }
            }
        }
        let Some(discriminator_index) = discriminator_index else {
            return Ok(None);
        };
        if discriminator_index != 0 {
            return Ok(None);
        }
        let value_index = 1;

        let fixed_keys = variants[0]
            .fixed_keys
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut alternatives = Vec::with_capacity(variants.len());
        for variant in variants {
            let discriminator = &variant.items[discriminator_index];
            let payload = &variant.items[value_index];
            let Some(discriminator_value) = singleton_string_enum_value(&discriminator.schema) else {
                return Ok(None);
            };
            let payload_expr = payload.value_expr.clone();
            alternatives.push(seq(vec![
                self.lower_literal_key_colon(&discriminator.key),
                self.lower_string_literal(&discriminator_value),
                self.item_separator_expr(),
                self.lower_literal_key_colon(&payload.key),
                payload_expr,
            ]));
        }

        let additional_pair = seq(vec![
            self.item_separator_expr(),
            self.lower_additional_key_colon(&fixed_keys, &[])?,
            r(JSON_VALUE_RULE),
        ]);
        let body = seq(vec![
            choice(alternatives),
            GrammarExpr::Quantified(Box::new(additional_pair), Quantifier::ZeroPlus),
        ]);
        let rule_name = self.fresh_rule_name("json_discriminator_anyof_object_body");
        self.add_nonterminal_rule(&rule_name, body);
        Ok(Some(seq(vec![lit("{"), r(&rule_name), lit("}")])))
    }

    fn lower_open_any_of_object_variants_expr_nfa(
        &mut self,
        variants: &[AnyOfObjectVariant],
        include_untyped_non_object_alts: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        if variants.is_empty() {
            return Ok(None);
        }
        if !discriminator_anyof_fastpath_disabled() {
            if let Some(expr) = self.try_lower_discriminator_value_open_any_of_object(
                variants,
                include_untyped_non_object_alts,
            )? {
                return Ok(Some(expr));
            }
        }

        let mut builder = ExprNfaBuilder::new();
        let accept = builder.add_state();
        builder.set_accepting(accept);
        let shadow_owners = (0..variants.len())
            .map(|variant_idx| Self::select_shadow_owner_for_variant(variants, variant_idx))
            .collect::<Vec<_>>();
        let mut state_ids = BTreeMap::<AnyOfObjectState, u32>::new();
        let mut queue = VecDeque::<AnyOfObjectState>::new();
        for (variant_idx, _) in variants.iter().enumerate() {
            let shadow_owner = shadow_owners[variant_idx]
                .map(|owner_idx| ShadowOwnerState::Possible {
                    variant_idx: owner_idx as u16,
                    cursor: 0,
                    phase: AnyOfObjectPhase::Fixed,
                })
                .unwrap_or(ShadowOwnerState::None);
            let state = AnyOfObjectState {
                variant_idx: variant_idx as u16,
                cursor: 0,
                has_content: false,
                phase: AnyOfObjectPhase::Fixed,
                shadow_owner,
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

            if can_leave_fixed_tail
                && !Self::shadow_owner_suppresses_close(variants, state.shadow_owner)
            {
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
                    shadow_owner: ShadowOwnerState::Impossible,
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
                let excluded_fixed_keys = variant.fixed_keys.clone();
                let mut owner_fixed_keys = Vec::<(String, bool)>::new();
                if let ShadowOwnerState::Possible {
                    variant_idx: owner_idx,
                    cursor: owner_cursor,
                    phase: owner_phase,
                } = state.shadow_owner
                {
                    let owner = &variants[owner_idx as usize];
                    match owner_phase {
                        AnyOfObjectPhase::Fixed => {
                            for owner_key in owner.legal_next_keys(owner_cursor as usize) {
                                if !variant.fixed_keys.contains(owner_key) {
                                    let can_split = owner
                                        .value_expr_for_key(owner_key)
                                        .as_ref()
                                        .and_then(Self::invalid_residual_value_for_owner)
                                        .is_some();
                                    owner_fixed_keys.push((owner_key.to_string(), can_split));
                                }
                            }
                        }
                        AnyOfObjectPhase::Additional => {
                            for owner_key in &owner.fixed_keys {
                                if !variant.fixed_keys.contains(owner_key) {
                                    owner_fixed_keys.push((owner_key.clone(), false));
                                }
                            }
                        }
                        AnyOfObjectPhase::Pattern => {}
                    }
                }

                let broad_key_shadow = match state.shadow_owner {
                    ShadowOwnerState::Possible {
                        variant_idx,
                        cursor,
                        phase,
                    } if Self::shadow_owner_can_take_additional(
                        &variants[variant_idx as usize],
                        cursor as usize,
                        phase,
                    ) =>
                    {
                        ShadowOwnerState::Possible {
                            variant_idx,
                            cursor: variants[variant_idx as usize].len() as u16,
                            phase: AnyOfObjectPhase::Additional,
                        }
                    }
                    ShadowOwnerState::Possible { .. } => ShadowOwnerState::Impossible,
                    other => other,
                };
                let next_state = AnyOfObjectState {
                    variant_idx: state.variant_idx,
                    cursor: variant.len() as u16,
                    has_content: true,
                    phase: AnyOfObjectPhase::Additional,
                    shadow_owner: broad_key_shadow,
                };
                let next_state_id = if let Some(&existing) = state_ids.get(&next_state) {
                    existing
                } else {
                    let id = builder.add_state();
                    state_ids.insert(next_state, id);
                    queue.push_back(next_state);
                    id
                };

                let key_symbol = self.lower_additional_key_colon(
                    &excluded_fixed_keys,
                    &variant.pattern_keys,
                )?;
                for key_path in Self::split_key_colon_symbol_paths(key_symbol) {
                    let mut symbols = Vec::new();
                    if state.has_content {
                        symbols.push(self.item_separator_expr());
                    }
                    symbols.extend(key_path);
                    symbols.push(value_expr.clone());
                    Self::add_expr_nfa_symbol_path(&mut builder, state_id, symbols, next_state_id);
                }

                for (owner_key, owner_fixed_key_can_advance) in owner_fixed_keys {
                    let ShadowOwnerState::Possible {
                        variant_idx: owner_idx,
                        cursor: owner_cursor,
                        phase: owner_phase,
                    } = state.shadow_owner
                    else {
                        continue;
                    };
                    let owner = &variants[owner_idx as usize];
                    if !owner_fixed_key_can_advance || owner_phase != AnyOfObjectPhase::Fixed {
                        let invalid_state = AnyOfObjectState {
                            variant_idx: state.variant_idx,
                            cursor: variant.len() as u16,
                            has_content: true,
                            phase: AnyOfObjectPhase::Additional,
                            shadow_owner: ShadowOwnerState::Impossible,
                        };
                        let invalid_state_id =
                            if let Some(&existing) = state_ids.get(&invalid_state) {
                                existing
                            } else {
                                let id = builder.add_state();
                                state_ids.insert(invalid_state, id);
                                queue.push_back(invalid_state);
                                id
                            };
                        let mut symbols = Vec::new();
                        if state.has_content {
                            symbols.push(self.item_separator_expr());
                        }
                        symbols.push(self.lower_literal_key_colon(&owner_key));
                        symbols.push(value_expr.clone());
                        Self::add_expr_nfa_symbol_path(
                            &mut builder,
                            state_id,
                            symbols,
                            invalid_state_id,
                        );
                        continue;
                    }
                    let Some(next_owner_cursor) = owner.advance_cursor(owner_cursor as usize, &owner_key) else {
                        continue;
                    };
                    let Some(owner_value_expr) = owner.value_expr_for_key(&owner_key) else {
                        continue;
                    };
                    let Some(invalid_residual_value_expr) =
                        Self::invalid_residual_value_for_owner(&owner_value_expr)
                    else {
                        continue;
                    };

                    let owner_valid_state = AnyOfObjectState {
                        variant_idx: state.variant_idx,
                        cursor: variant.len() as u16,
                        has_content: true,
                        phase: AnyOfObjectPhase::Additional,
                        shadow_owner: ShadowOwnerState::Possible {
                            variant_idx: owner_idx,
                            cursor: next_owner_cursor as u16,
                            phase: AnyOfObjectPhase::Fixed,
                        },
                    };
                    let owner_valid_state_id =
                        if let Some(&existing) = state_ids.get(&owner_valid_state) {
                            existing
                        } else {
                            let id = builder.add_state();
                            state_ids.insert(owner_valid_state, id);
                            queue.push_back(owner_valid_state);
                            id
                        };
                    let mut symbols = Vec::new();
                    if state.has_content {
                        symbols.push(self.item_separator_expr());
                    }
                    symbols.push(self.lower_literal_key_colon(&owner_key));
                    symbols.push(owner_value_expr);
                    Self::add_expr_nfa_symbol_path(
                        &mut builder,
                        state_id,
                        symbols,
                        owner_valid_state_id,
                    );

                    let invalid_state = AnyOfObjectState {
                        variant_idx: state.variant_idx,
                        cursor: variant.len() as u16,
                        has_content: true,
                        phase: AnyOfObjectPhase::Additional,
                        shadow_owner: ShadowOwnerState::Impossible,
                    };
                    let invalid_state_id = if let Some(&existing) = state_ids.get(&invalid_state) {
                        existing
                    } else {
                        let id = builder.add_state();
                        state_ids.insert(invalid_state, id);
                        queue.push_back(invalid_state);
                        id
                    };
                    let mut symbols = Vec::new();
                    if state.has_content {
                        symbols.push(self.item_separator_expr());
                    }
                    symbols.push(self.lower_literal_key_colon(&owner_key));
                    symbols.push(invalid_residual_value_expr);
                    Self::add_expr_nfa_symbol_path(&mut builder, state_id, symbols, invalid_state_id);
                }
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
                    shadow_owner: Self::advance_shadow_owner_on_key(
                        variants,
                        state.shadow_owner,
                        key,
                        &value_expr,
                    ),
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

    fn fixed_object_template_symbols(
        items: &[ObjectItem],
        item_symbols: &[[GrammarExpr; 2]],
        separator: GrammarExpr,
    ) -> (FixedObjectTemplateKey, Vec<GrammarExpr>) {
        let mut symbols = Vec::<GrammarExpr>::new();
        let mut labels = HashMap::<GrammarExpr, i32>::new();
        let mut occurrences = Vec::<i32>::with_capacity(item_symbols.len() * 5);
        let mut intern = |expr: GrammarExpr| {
            if let Some(&label) = labels.get(&expr) {
                return label;
            }
            let label = i32::try_from(symbols.len())
                .expect("fixed-object expression symbol table exceeded i32 labels");
            symbols.push(expr.clone());
            labels.insert(expr, label);
            label
        };

        for item in item_symbols {
            let key = intern(item[0].clone());
            let value = intern(item[1].clone());
            let separator = intern(separator.clone());
            // The normal builder emits the first-item path, followed by the
            // separator-prefixed path, in this exact order.
            occurrences.extend([key, value, separator, key, value]);
        }

        (
            FixedObjectTemplateKey {
                required: items.iter().map(|item| item.required).collect(),
                symbol_occurrences: occurrences,
            },
            symbols,
        )
    }

    fn lower_fixed_object_body_exprnfa_without_group(
        &mut self,
        items: &[ObjectItem],
        tail_pair: Option<GrammarExpr>,
    ) -> ImportResult<GrammarExpr> {
        let profile_started_at = self
            .fixed_object_profile
            .as_ref()
            .map(|_| std::time::Instant::now());
        let required_count = items.iter().filter(|item| item.required).count();
        let item_symbols_started_at = profile_started_at.map(|_| std::time::Instant::now());
        let mut item_symbols = Vec::with_capacity(items.len());
        for item in items {
            item_symbols.push(Self::split_object_pair_symbols(&item.pair)?);
        }
        let item_symbol_ms = item_symbols_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let template_symbols = tail_pair.is_none().then(|| {
            Self::fixed_object_template_symbols(
                items,
                &item_symbols,
                self.item_separator_expr(),
            )
        });
        if let Some((template_key, symbols)) = &template_symbols
            && let Some(template_nfa) = self.fixed_object_nfa_templates.get(template_key).cloned()
        {
            let rule_name = self.fresh_rule_name("json_closed_object_body");
            let body = GrammarExpr::ExprNFA(Box::new(ExprNFA {
                nfa: template_nfa.nfa,
                symbols: symbols.clone(),
                is_determinized_and_minimized: template_nfa.is_determinized_and_minimized,
                canonical_dfa: template_nfa.canonical_dfa,
            }));
            self.add_nonterminal_rule(&rule_name, body);

            let total_ms = profile_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            if let Some(profile) = self.fixed_object_profile.as_mut() {
                let shape = profile.shapes.entry((items.len(), required_count)).or_default();
                profile.calls += 1;
                profile.total_items += items.len();
                profile.template_hits += 1;
                shape.calls += 1;
                shape.item_symbol_ms += item_symbol_ms;
                shape.total_ms += total_ms;
            }

            return Ok(seq(vec![lit("{"), r(&rule_name), lit("}")]));
        }
        let graph_build_started_at = profile_started_at.map(|_| std::time::Instant::now());
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

        let graph_build_ms = graph_build_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let determinize_minimize_started_at =
            profile_started_at.map(|_| std::time::Instant::now());
        let rule_name = self.fresh_rule_name("json_closed_object_body");
        let expr_nfa = builder.build().into_determinized_and_minimized();
        if let Some((template_key, symbols)) = &template_symbols {
            debug_assert_eq!(expr_nfa.symbols, *symbols);
            self.fixed_object_nfa_templates
                .insert(template_key.clone(), expr_nfa.clone());
        }
        let body = GrammarExpr::ExprNFA(Box::new(expr_nfa));
        self.add_nonterminal_rule(&rule_name, body);

        let determinize_minimize_ms = determinize_minimize_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let total_ms = profile_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if std::env::var("GLRMASK_PROFILE_FIXED_OBJECT_DETAIL")
            .map(|value| value == "1")
            .unwrap_or(false)
            && total_ms >= 0.5
        {
            eprintln!(
                "[glrmask/profile][fixed_object_slow] properties={} required={} keys={:?} symbols_ms={:.3} graph_ms={:.3} detmin_ms={:.3} total_ms={:.3}",
                items.len(),
                required_count,
                items.iter().map(|item| item.key.as_str()).collect::<Vec<_>>(),
                item_symbol_ms,
                graph_build_ms,
                determinize_minimize_ms,
                total_ms,
            );
        }
        if let Some(profile) = self.fixed_object_profile.as_mut() {
            let shape = profile.shapes.entry((items.len(), required_count)).or_default();
            profile.calls += 1;
            profile.total_items += items.len();
            profile.template_misses += usize::from(template_symbols.is_some());
            shape.calls += 1;
            shape.item_symbol_ms += item_symbol_ms;
            shape.graph_build_ms += graph_build_ms;
            shape.determinize_minimize_ms += determinize_minimize_ms;
            shape.total_ms += total_ms;
        }

        Ok(seq(vec![lit("{"), r(&rule_name), lit("}")]))
    }

    fn is_structured_literal_key_symbol(symbol: &GrammarExpr) -> bool {
        fn contains_ref(expr: &GrammarExpr, wanted: &str) -> bool {
            match expr {
                GrammarExpr::Ref(name) => name == wanted,
                GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                    parts.iter().any(|part| contains_ref(part, wanted))
                }
                GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
                    contains_ref(inner, wanted)
                }
                _ => false,
            }
        }

        contains_ref(symbol, JSON_QUOTE_RULE) && contains_ref(symbol, JSON_KEY_SUFFIX_RULE)
    }

    /// Expand a grammar symbol that contains a literal object-key path into
    /// grammar-visible terminal paths.  A `Choice` must be expanded path-wise:
    /// treating a choice of `"key": value` sequences as one ExprNFA symbol
    /// would silently reintroduce a unique terminal for every property.
    fn structured_literal_key_symbol_paths(symbol: GrammarExpr) -> Option<Vec<Vec<GrammarExpr>>> {
        if !Self::is_structured_literal_key_symbol(&symbol) {
            return None;
        }

        match symbol {
            GrammarExpr::Sequence(parts) => {
                let mut paths = vec![Vec::new()];
                for part in parts {
                    let part_paths = Self::structured_literal_key_symbol_paths(part.clone())
                        .unwrap_or_else(|| vec![vec![part]]);
                    let mut combined = Vec::with_capacity(paths.len() * part_paths.len());
                    for path in paths {
                        for part_path in &part_paths {
                            let mut full_path = path.clone();
                            full_path.extend(part_path.iter().cloned());
                            combined.push(full_path);
                        }
                    }
                    paths = combined;
                }
                Some(paths)
            }
            GrammarExpr::Choice(alternatives) => {
                let mut paths = Vec::new();
                for alternative in alternatives {
                    paths.extend(
                        Self::structured_literal_key_symbol_paths(alternative.clone())
                            .unwrap_or_else(|| vec![vec![alternative]]),
                    );
                }
                Some(paths)
            }
            other => Some(vec![vec![other]]),
        }
    }

    fn expand_structured_literal_key_paths(symbols: Vec<GrammarExpr>) -> Vec<Vec<GrammarExpr>> {
        if !split_literal_terminals_enabled() {
            return vec![symbols];
        }

        let mut paths = vec![Vec::new()];
        for symbol in symbols {
            let symbol_paths = Self::structured_literal_key_symbol_paths(symbol.clone())
                .unwrap_or_else(|| vec![vec![symbol]]);
            let mut combined = Vec::with_capacity(paths.len() * symbol_paths.len());
            for path in paths {
                for symbol_path in &symbol_paths {
                    let mut full_path = path.clone();
                    full_path.extend(symbol_path.iter().cloned());
                    combined.push(full_path);
                }
            }
            paths = combined;
        }
        paths
    }

    fn flatten_structured_literal_key_parts(symbol: GrammarExpr, out: &mut Vec<GrammarExpr>) {
        match symbol {
            GrammarExpr::Sequence(parts) => {
                for part in parts {
                    Self::flatten_structured_literal_key_parts(part, out);
                }
            }
            other => out.push(other),
        }
    }

    fn split_literal_key_symbol(symbol: GrammarExpr) -> Vec<GrammarExpr> {
        if Self::is_structured_literal_key_symbol(&symbol) {
            let mut symbols = Vec::new();
            Self::flatten_structured_literal_key_parts(symbol, &mut symbols);
            return symbols;
        }

        match symbol {
            GrammarExpr::Literal(bytes) if bytes.len() > 1 => {
                let split_len = LARGE_OBJECT_KEY_TRIE_PREFIX_SPLIT_BYTES.min(bytes.len());
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

    fn lower_large_closed_pattern_object_key_trie(
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
                GrammarExpr::Quantified(Box::new(seq(vec![
                    self.item_separator_expr(),
                    tail_pair_expr,
                ])), Quantifier::ZeroPlus),
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

    fn lower_required_prefix_large_optional_open_object_fused_prefix_chain(
        &mut self,
        items: &[ObjectItem],
        required_prefix_len: usize,
        tail_pair_expr: GrammarExpr,
    ) -> ImportResult<GrammarExpr> {
        debug_assert!(required_prefix_len > 0);
        debug_assert!(required_prefix_len <= items.len());

        let required_prefix = seq(
            items[..required_prefix_len]
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    if index == 0 {
                        item.pair.clone()
                    } else {
                        item.separator_pair.clone()
                    }
                })
                .collect(),
        );

        let optional_items = &items[required_prefix_len..];
        let mut prefix_rule_names: Vec<String> = Vec::with_capacity(optional_items.len());
        for end_exclusive in 1..=optional_items.len() {
            let mut alternatives = Vec::new();
            for start in 0..end_exclusive {
                if start == 0 {
                    alternatives.push(optional_items[end_exclusive - 1].pair.clone());
                } else {
                    alternatives.push(seq(vec![
                        r(&prefix_rule_names[start - 1]),
                        optional_items[end_exclusive - 1].separator_pair.clone(),
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
                GrammarExpr::Quantified(
                    Box::new(seq(vec![self.item_separator_expr(), tail_pair_expr])),
                    Quantifier::ZeroPlus,
                ),
            ]),
        );

        let mut suffix_alternatives = vec![r(&free_nonempty_rule)];
        for prefix_rule_name in &prefix_rule_names {
            suffix_alternatives.push(r(prefix_rule_name));
            suffix_alternatives.push(seq(vec![
                r(prefix_rule_name),
                self.item_separator_expr(),
                r(&free_nonempty_rule),
            ]));
        }

        Ok(seq(vec![
            lit("{"),
            choice(vec![
                required_prefix.clone(),
                seq(vec![
                    required_prefix,
                    self.item_separator_expr(),
                    choice(suffix_alternatives),
                ]),
            ]),
            lit("}"),
        ]))
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
        let mut intersected_schema = None;
        for pattern_property in pattern_properties {
            if property_matches_pattern(&pattern_property.pattern, &property.name)? {
                let current_schema = intersected_schema
                    .take()
                    .unwrap_or_else(|| property.schema.clone());
                let pattern_schema =
                    pattern_schema_for_property(&current_schema, &pattern_property.schema);
                intersected_schema = Some(all_of_schema(current_schema, pattern_schema));
            }
        }
        let effective_schema = intersected_schema.as_ref().unwrap_or(&property.schema);
        if let Some(item) = self.lower_string_property_item(
            &property.name,
            effective_schema,
            required,
            satisfies_any_group,
            exclusive_group,
        )? {
            return Ok(item);
        }
        let value = self.lower_object_property_value_schema(effective_schema)?;
        if let Some(non_string_value) = Self::without_json_string_branch(value.clone()) {
            let (non_string_value, has_null_branch) =
                if let Some(non_string_value) = non_string_value {
                    if let Some(non_null_value) =
                        Self::without_ref_branch(non_string_value.clone(), JSON_NULL_RULE)
                    {
                        (non_null_value, true)
                    } else {
                        (Some(non_string_value), false)
                    }
                } else {
                    (None, false)
                };
            let string_key =
                self.lower_literal_key_colon_with_prefix_and_json_string(b"", &property.name);
            let separator_string_key =
                self.lower_literal_key_colon_with_prefix_and_json_string(b", ", &property.name);
            let null_key = self.lower_literal_key_colon_with_prefix_and_literal_value(
                b"",
                &property.name,
                b"null",
            );
            let separator_null_key = self.lower_literal_key_colon_with_prefix_and_literal_value(
                b", ",
                &property.name,
                b"null",
            );
            let pair = if let Some(non_string_value) = non_string_value.clone() {
                let mut alternatives = vec![string_key];
                if has_null_branch {
                    alternatives.push(null_key);
                }
                alternatives.push(seq(vec![key, non_string_value]));
                seq(vec![
                    choice(alternatives),
                    GrammarExpr::Epsilon,
                ])
            } else if has_null_branch {
                seq(vec![
                    choice(vec![string_key, null_key]),
                    GrammarExpr::Epsilon,
                ])
            } else {
                seq(vec![string_key, GrammarExpr::Epsilon])
            };
            let separator_pair = if let Some(non_string_value) = non_string_value {
                let mut alternatives = vec![separator_string_key];
                if has_null_branch {
                    alternatives.push(separator_null_key);
                }
                alternatives.push(seq(vec![separator_key, non_string_value]));
                seq(vec![
                    choice(alternatives),
                    GrammarExpr::Epsilon,
                ])
            } else if has_null_branch {
                seq(vec![
                    choice(vec![separator_string_key, separator_null_key]),
                    GrammarExpr::Epsilon,
                ])
            } else {
                seq(vec![separator_string_key, GrammarExpr::Epsilon])
            };
            return Ok(ObjectItem {
                key: property.name.clone(),
                pair,
                separator_pair,
                required,
                satisfies_any_group,
                exclusive_group,
            });
        }
        if let Some(non_null_value) = Self::without_ref_branch(value.clone(), JSON_NULL_RULE) {
            let null_key = self.lower_literal_key_colon_with_prefix_and_literal_value(
                b"",
                &property.name,
                b"null",
            );
            let separator_null_key = self.lower_literal_key_colon_with_prefix_and_literal_value(
                b", ",
                &property.name,
                b"null",
            );
            let pair = if let Some(non_null_value) = non_null_value.clone() {
                seq(vec![
                    choice(vec![null_key, seq(vec![key, non_null_value])]),
                    GrammarExpr::Epsilon,
                ])
            } else {
                seq(vec![null_key, GrammarExpr::Epsilon])
            };
            let separator_pair = if let Some(non_null_value) = non_null_value {
                seq(vec![
                    choice(vec![
                        separator_null_key,
                        seq(vec![separator_key, non_null_value]),
                    ]),
                    GrammarExpr::Epsilon,
                ])
            } else {
                seq(vec![separator_null_key, GrammarExpr::Epsilon])
            };
            return Ok(ObjectItem {
                key: property.name.clone(),
                pair,
                separator_pair,
                required,
                satisfies_any_group,
                exclusive_group,
            });
        }
        let (key, separator_key, value) = if let Some(rest) =
            Self::strip_leading_literal_byte(value.clone(), b'[')
        {
            (
                self.lower_literal_key_colon_with_prefix_and_suffix(b"", &property.name, b'['),
                self.lower_literal_key_colon_with_prefix_and_suffix(b", ", &property.name, b'['),
                rest,
            )
        } else if let Some(rest) = Self::strip_leading_literal_byte(value.clone(), b'{') {
            (
                self.lower_literal_key_colon_with_prefix_and_suffix(b"", &property.name, b'{'),
                self.lower_literal_key_colon_with_prefix_and_suffix(b", ", &property.name, b'{'),
                rest,
            )
        } else {
            (key, separator_key, value)
        };
        Ok(ObjectItem {
            key: property.name.clone(),
            pair: seq(vec![key, value.clone()]),
            separator_pair: seq(vec![separator_key, value]),
            required,
            satisfies_any_group,
            exclusive_group,
        })
    }

    fn without_json_string_branch(expr: GrammarExpr) -> Option<Option<GrammarExpr>> {
        Self::without_ref_branch(expr, JSON_STRING_RULE)
    }

    fn without_ref_branch(expr: GrammarExpr, branch_rule: &str) -> Option<Option<GrammarExpr>> {
        match expr {
            GrammarExpr::Ref(name) if name == branch_rule => Some(None),
            GrammarExpr::Choice(alternatives) => {
                let original_len = alternatives.len();
                let alternatives = alternatives
                    .into_iter()
                    .filter(|expr| !matches!(expr, GrammarExpr::Ref(name) if name == branch_rule))
                    .collect::<Vec<_>>();
                if alternatives.len() == original_len {
                    None
                } else if alternatives.is_empty() {
                    Some(None)
                } else {
                    Some(Some(choice(alternatives)))
                }
            }
            _ => None,
        }
    }

    fn strip_leading_literal_byte(expr: GrammarExpr, byte: u8) -> Option<GrammarExpr> {
        match expr {
            GrammarExpr::Literal(bytes) => {
                let rest = bytes.strip_prefix(&[byte])?;
                if rest.is_empty() {
                    Some(GrammarExpr::Epsilon)
                } else {
                    Some(lit_bytes(rest.to_vec()))
                }
            }
            GrammarExpr::Sequence(mut parts) => {
                let GrammarExpr::Literal(bytes) = parts.first()? else {
                    return None;
                };
                let rest = bytes.strip_prefix(&[byte])?.to_vec();
                if rest.is_empty() {
                    parts.remove(0);
                } else {
                    parts[0] = lit_bytes(rest);
                }
                Some(seq(parts))
            }
            _ => None,
        }
    }

    fn lower_string_property_item(
        &mut self,
        key_name: &str,
        schema: &Schema,
        required: bool,
        satisfies_any_group: bool,
        exclusive_group: bool,
    ) -> ImportResult<Option<ObjectItem>> {
        let SchemaKind::Assertions(assertions) = &schema.kind else {
            return Ok(None);
        };
        if assertions.const_value.is_some()
            || assertions.enum_values.is_some()
            || !assertions.any_of.is_empty()
            || !assertions.one_of.is_empty()
            || !assertions.all_of.is_empty()
            || assertions.not.is_some()
        {
            return Ok(None);
        }
        let untyped = assertions.types.is_none();
        let explicitly_allows_string = assertions.types.as_ref().is_some_and(|types| {
            types.iter().any(|schema_type| *schema_type == SchemaType::String)
        });
        if !explicitly_allows_string && !untyped {
            return Ok(None);
        }
        if untyped
            && (assertions.object.is_some()
                || assertions.array.is_some()
                || assertions.number.is_some())
        {
            return Ok(None);
        }
        let default_string;
        let string = if let Some(string) = assertions.string.as_ref() {
            string
        } else if explicitly_allows_string {
            default_string = super::ast::StringSchema::default();
            &default_string
        } else {
            return Ok(None);
        };

        let string_only = assertions.types.as_ref().is_some_and(|types| {
            !types.is_empty()
                && types
                    .iter()
                    .all(|schema_type| *schema_type == SchemaType::String)
        });
        let string_branch_needs_isolation = string.pattern.is_some()
            || super::string::recognized_string_format_body_regex_for_lowering(
                string.format.as_deref(),
            )
            .is_some();
        // Keep ordinary typed string unions on the established generic path.
        // The specialized branch below exists to isolate a patterned or
        // recognized-format string branch while preserving its non-string
        // alternatives. Applying it to every `string | null` property changes
        // the object shape of large nullable-record schemas without adding any
        // lexical isolation benefit.
        if assertions.types.is_some() && !string_only && !string_branch_needs_isolation {
            return Ok(None);
        }

        let string_pair = self.lower_literal_key_colon_with_prefix_and_string_schema(
            b"",
            key_name,
            string,
        )?;
        // The key is the useful lexical context: it keeps this value DFA
        // disjoint from unrelated string values.  Do not also compile a
        // second copy of a costly key/value terminal merely to absorb the
        // preceding comma.  Keep the comma as grammar-visible JSON separator
        // and reuse the same fused key/value terminal in either position.
        let separator_string_pair = if string_branch_needs_isolation {
            seq(vec![self.item_separator_expr(), string_pair.clone()])
        } else {
            self.lower_literal_key_colon_with_prefix_and_string_schema(
                b", ",
                key_name,
                string,
            )?
        };

        let (pair, separator_pair) = if let Some(types) = &assertions.types {
            let mut pair_alternatives = vec![string_pair];
            let mut separator_pair_alternatives = vec![separator_string_pair];
            for schema_type in types.iter().copied() {
                if schema_type == SchemaType::String {
                    continue;
                }
                if schema_type == SchemaType::Null {
                    pair_alternatives.push(
                        self.lower_literal_key_colon_with_prefix_and_literal_value(
                            b"",
                            key_name,
                            b"null",
                        ),
                    );
                    separator_pair_alternatives.push(
                        self.lower_literal_key_colon_with_prefix_and_literal_value(
                            b", ",
                            key_name,
                            b"null",
                        ),
                    );
                } else {
                    pair_alternatives.push(seq(vec![
                        self.lower_literal_key_colon(key_name),
                        self.lower_for_type(schema_type, assertions)?,
                    ]));
                    separator_pair_alternatives.push(seq(vec![
                        self.lower_literal_key_colon_with_prefix(b", ", key_name),
                        self.lower_for_type(schema_type, assertions)?,
                    ]));
                }
            }
            (choice(pair_alternatives), choice(separator_pair_alternatives))
        } else {
            let non_string_value = choice(vec![
                r(JSON_OBJECT_RULE),
                r(JSON_ARRAY_RULE),
                r(JSON_NUMBER_RULE),
                r(JSON_BOOL_RULE),
                r(JSON_NULL_RULE),
            ]);
            (
                choice(vec![
                    string_pair,
                    seq(vec![self.lower_literal_key_colon(key_name), non_string_value.clone()]),
                ]),
                choice(vec![
                    separator_string_pair,
                    seq(vec![
                        self.lower_literal_key_colon_with_prefix(b", ", key_name),
                        non_string_value,
                    ]),
                ]),
            )
        };

        Ok(Some(ObjectItem {
            key: key_name.to_string(),
            pair,
            separator_pair,
            required,
            satisfies_any_group,
            exclusive_group,
        }))
    }

    fn lower_object_property_value_schema(&mut self, schema: &Schema) -> ImportResult<GrammarExpr> {
        let SchemaKind::Assertions(assertions) = &schema.kind else {
            return self.lower_schema(schema);
        };
        if assertions.types.is_some()
            || assertions.const_value.is_some()
            || assertions.enum_values.is_some()
            || assertions.object.is_some()
            || assertions.array.is_some()
            || !assertions.any_of.is_empty()
            || !assertions.one_of.is_empty()
            || !assertions.all_of.is_empty()
        {
            return self.lower_schema(schema);
        }
        if let Some(number) = &assertions.number
            && assertions.string.is_none()
        {
            return Ok(choice(vec![
                self.lower_number(number)?,
                r(JSON_OBJECT_RULE),
                r(JSON_ARRAY_RULE),
                r(JSON_STRING_RULE),
                r(JSON_BOOL_RULE),
                r(JSON_NULL_RULE),
            ]));
        }
        if let Some(string) = &assertions.string
            && assertions.number.is_none()
        {
            return Ok(choice(vec![
                self.lower_string(string)?,
                r(JSON_OBJECT_RULE),
                r(JSON_ARRAY_RULE),
                r(JSON_NUMBER_RULE),
                r(JSON_BOOL_RULE),
                r(JSON_NULL_RULE),
            ]));
        }
        self.lower_schema(schema)
    }

    fn resolve_property_names_pattern(
        &self,
        schema: &ObjectSchema,
    ) -> ImportResult<Option<String>> {
        let Some(property_names) = &schema.property_names else {
            return Ok(None);
        };

        let resolved = match &property_names.kind {
            SchemaKind::Ref(pointer) => self.resolve_ref_target(pointer)?,
            _ => property_names,
        };
        let SchemaKind::Assertions(assertions) = &resolved.kind else {
            return Err(SchemaImportError::at(
                &resolved.location,
                "propertyNames is only supported for inline/local-ref string pattern schemas",
            ));
        };

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
            return Err(SchemaImportError::at(
                &resolved.location,
                "propertyNames is only supported for inline/local-ref string pattern schemas",
            ));
        }
        if let Some(types) = &assertions.types
            && !types.iter().all(|schema_type| *schema_type == SchemaType::String)
        {
            return Err(SchemaImportError::at(
                &resolved.location,
                "propertyNames is only supported for inline/local-ref string pattern schemas",
            ));
        }

        let Some(string) = &assertions.string else {
            return Err(SchemaImportError::at(
                &resolved.location,
                "propertyNames is only supported for inline/local-ref string pattern schemas",
            ));
        };
        if string.min_length != 0 || string.max_length.is_some() || string.format.is_some() {
            return Err(SchemaImportError::at(
                &resolved.location,
                "propertyNames is only supported for inline/local-ref string pattern schemas",
            ));
        }

        let Some(pattern) = &string.pattern else {
            return Err(SchemaImportError::at(
                &resolved.location,
                "propertyNames is only supported for inline/local-ref string pattern schemas",
            ));
        };
        Ok(Some(pattern.clone()))
    }

    fn lower_object_additional_key_colon(
        &mut self,
        fixed_keys: &BTreeSet<String>,
        local_patterns: &[String],
        property_name_pattern: Option<&str>,
    ) -> ImportResult<GrammarExpr> {
        let Some(pattern) = property_name_pattern else {
            return self.lower_additional_key_colon(fixed_keys, local_patterns);
        };

        let mut expr = self.lower_pattern_key_colon_terminal(pattern)?;
        if !fixed_keys.is_empty() {
            let excluded = fixed_keys
                .iter()
                .map(|key| self.lower_literal_key_colon(key))
                .collect::<Vec<_>>();
            expr = GrammarExpr::Exclude {
                expr: Box::new(expr),
                exclude: Box::new(choice(excluded)),
            };
        }
        Ok(expr)
    }

    fn object_with_required_synthetic_properties<'schema>(
        &self,
        schema: &'schema ObjectSchema,
    ) -> ImportResult<Cow<'schema, ObjectSchema>> {
        let all_required_are_declared = schema
            .required_order
            .iter()
            .chain(schema.required.iter())
            .all(|required_name| schema.properties.iter().any(|property| property.name == *required_name));
        if all_required_are_declared {
            return Ok(Cow::Borrowed(schema));
        }

        let mut normalized = schema.clone();
        let mut known = normalized
            .properties
            .iter()
            .map(|property| property.name.clone())
            .collect::<BTreeSet<_>>();

        let mut required_names = schema.required_order.clone();
        for required_name in &schema.required {
            if !required_names.contains(required_name) {
                required_names.push(required_name.clone());
            }
        }

        for required_name in &required_names {
            if known.contains(required_name) {
                continue;
            }

            let mut matching_pattern_schema = None;
            for pattern_property in &schema.pattern_properties {
                if property_matches_pattern(&pattern_property.pattern, required_name)? {
                    matching_pattern_schema = Some(match matching_pattern_schema {
                        Some(schema) => all_of_schema(schema, pattern_property.schema.clone()),
                        None => pattern_property.schema.clone(),
                    });
                }
            }

            let synthetic_schema = if let Some(schema) = matching_pattern_schema {
                schema
            } else {
                match &schema.additional_properties {
                    AdditionalProperties::AllowAny => {
                        Schema::any(format!("<required:{required_name}>"))
                    }
                    AdditionalProperties::Schema(schema) => schema.as_ref().clone(),
                    AdditionalProperties::Deny => {
                        Schema::never(format!("<required:{required_name}:unsatisfiable>"))
                    }
                }
            };
            normalized.properties.push(PropertySchema {
                name: required_name.clone(),
                schema: synthetic_schema,
            });
            known.insert(required_name.clone());
        }

        Ok(Cow::Owned(normalized))
    }
}


fn singleton_string_enum_value(schema: &Schema) -> Option<String> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    match assertions.enum_values.as_deref() {
        Some([serde_json::Value::String(value)]) => Some(value.clone()),
        _ => None,
    }
}

/// A singleton string enum with no sibling assertion that can narrow or widen
/// its language.  Unlike `singleton_string_enum_value`, this is suitable for a
/// direct language-preserving discriminator construction.
fn plain_singleton_string_enum_value(schema: &Schema) -> Option<String> {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return None;
    };
    let value = match assertions.enum_values.as_deref() {
        Some([serde_json::Value::String(value)]) => value,
        _ => return None,
    };
    if assertions.const_value.is_none()
        && assertions.object.is_none()
        && assertions.array.is_none()
        && assertions.string.as_ref().is_none_or(|string| {
            string.min_length == 0
                && string.max_length.is_none()
                && string.pattern.is_none()
                && string.format.is_none()
        })
        && assertions.number.is_none()
        && assertions.not.is_none()
        && assertions.any_of.is_empty()
        && assertions.one_of.is_empty()
        && assertions.all_of.is_empty()
        && assertions.types.as_ref().is_some_and(|types| {
            !types.is_empty()
                && types.iter().all(|schema_type| *schema_type == SchemaType::String)
        })
    {
        Some(value.clone())
    } else {
        None
    }
}

fn exact_property_value_identity(schema: &Schema) -> Option<String> {
    match &schema.kind {
        SchemaKind::Ref(pointer) => Some(format!("ref:{pointer}")),
        SchemaKind::Assertions(assertions) => {
            let literal = match assertions.enum_values.as_deref() {
                Some([serde_json::Value::String(value)]) => Some(value),
                _ => None,
            }?;
            if assertions.const_value.is_none()
                && assertions.object.is_none()
                && assertions.array.is_none()
                && assertions.string.is_none()
                && assertions.number.is_none()
                && assertions.not.is_none()
                && assertions.any_of.is_empty()
                && assertions.one_of.is_empty()
                && assertions.all_of.is_empty()
                && assertions.types.as_ref().is_none_or(|types| {
                    types.iter().all(|schema_type| *schema_type == SchemaType::String)
                })
            {
                Some(format!(
                    "string:{}",
                    serde_json::to_string(literal).unwrap_or_else(|_| "\"\"".to_string())
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn singleton_string_identity_literal_expr(identity: &str) -> Option<GrammarExpr> {
    let encoded = identity.strip_prefix("string:")?;
    Some(lit_bytes(encoded.as_bytes().to_vec()))
}

fn schemas_structurally_equal(lhs: &Schema, rhs: &Schema) -> bool {
    match (&lhs.kind, &rhs.kind) {
        (SchemaKind::Any, SchemaKind::Any) | (SchemaKind::Never, SchemaKind::Never) => true,
        (SchemaKind::Ref(left), SchemaKind::Ref(right)) => left == right,
        (SchemaKind::Assertions(left), SchemaKind::Assertions(right)) => {
            left.types == right.types
                && left.const_value == right.const_value
                && left.enum_values == right.enum_values
                && object_schemas_structurally_equal(left.object.as_ref(), right.object.as_ref())
                && array_schemas_structurally_equal(left.array.as_ref(), right.array.as_ref())
                && left.string == right.string
                && left.number == right.number
                && schema_slices_structurally_equal(&left.any_of, &right.any_of)
                && schema_slices_structurally_equal(&left.one_of, &right.one_of)
                && schema_slices_structurally_equal(&left.all_of, &right.all_of)
                && optional_schemas_structurally_equal(left.not.as_ref(), right.not.as_ref())
        }
        _ => false,
    }
}

fn optional_schemas_structurally_equal(lhs: Option<&Schema>, rhs: Option<&Schema>) -> bool {
    match (lhs, rhs) {
        (Some(left), Some(right)) => schemas_structurally_equal(left, right),
        (None, None) => true,
        _ => false,
    }
}

fn schema_slices_structurally_equal(lhs: &[Schema], rhs: &[Schema]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs)
            .all(|(left, right)| schemas_structurally_equal(left, right))
}

fn object_schemas_structurally_equal(lhs: Option<&ObjectSchema>, rhs: Option<&ObjectSchema>) -> bool {
    match (lhs, rhs) {
        (Some(left), Some(right)) => {
            left.required == right.required
                && left.property_dependencies == right.property_dependencies
                && left.min_properties == right.min_properties
                && left.max_properties == right.max_properties
                && left.properties.len() == right.properties.len()
                && left
                    .properties
                    .iter()
                    .zip(&right.properties)
                    .all(|(lprop, rprop)| {
                        lprop.name == rprop.name
                            && schemas_structurally_equal(&lprop.schema, &rprop.schema)
                    })
                && left.pattern_properties.len() == right.pattern_properties.len()
                && left
                    .pattern_properties
                    .iter()
                    .zip(&right.pattern_properties)
                    .all(|(lpat, rpat)| {
                        lpat.pattern == rpat.pattern
                            && schemas_structurally_equal(&lpat.schema, &rpat.schema)
                    })
                && optional_schemas_structurally_equal(
                    left.property_names.as_ref(),
                    right.property_names.as_ref(),
                )
                && additional_properties_structurally_equal(
                    &left.additional_properties,
                    &right.additional_properties,
                )
        }
        (None, None) => true,
        _ => false,
    }
}

fn array_schemas_structurally_equal(lhs: Option<&super::ast::ArraySchema>, rhs: Option<&super::ast::ArraySchema>) -> bool {
    match (lhs, rhs) {
        (Some(left), Some(right)) => {
            schemas_structurally_equal(&left.items, &right.items)
                && left.prefix_items.len() == right.prefix_items.len()
                && left
                    .prefix_items
                    .iter()
                    .zip(&right.prefix_items)
                    .all(|(litem, ritem)| schemas_structurally_equal(litem, ritem))
                && left.min_items == right.min_items
                && left.max_items == right.max_items
        }
        (None, None) => true,
        _ => false,
    }
}

fn additional_properties_structurally_equal(
    lhs: &AdditionalProperties,
    rhs: &AdditionalProperties,
) -> bool {
    match (lhs, rhs) {
        (AdditionalProperties::AllowAny, AdditionalProperties::AllowAny)
        | (AdditionalProperties::Deny, AdditionalProperties::Deny) => true,
        (AdditionalProperties::Schema(left), AdditionalProperties::Schema(right)) => {
            schemas_structurally_equal(left, right)
        }
        _ => false,
    }
}

fn shareable_property_schemas_match(lhs: &AnyOfFixedObjectItem, rhs: &AnyOfFixedObjectItem) -> bool {
    match (lhs.value_identity.as_ref(), rhs.value_identity.as_ref()) {
        (Some(left), Some(right)) => left == right,
        _ => schemas_structurally_equal(&lhs.schema, &rhs.schema),
    }
}

fn plain_object_schema_for_closed_any_of(schema: &Schema) -> Option<&ObjectSchema> {
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
    if let Some(types) = &assertions.types
        && !types.iter().all(|schema_type| *schema_type == SchemaType::Object)
    {
        return None;
    }
    assertions.object.as_ref()
}

fn closed_any_of_variants_have_shareable_property(variants: &[AnyOfFixedObjectVariant]) -> bool {
    let Some(first) = variants.first() else {
        return false;
    };
    first.items.iter().any(|first_item| {
        variants.iter().skip(1).all(|variant| {
            variant
                .item_for_key(&first_item.key)
                .is_some_and(|candidate| shareable_property_schemas_match(first_item, candidate))
        })
    })
}

fn shared_closed_any_of_key_transition(
    variants: &[AnyOfFixedObjectVariant],
    cursors: &[AnyOfFixedObjectCursor],
    key: &str,
) -> Option<(GrammarExpr, Vec<AnyOfFixedObjectCursor>)> {
    let mut identity = None;
    let mut value_expr = None;
    let mut next_cursors = Vec::with_capacity(cursors.len());

    for cursor in cursors {
        let variant = &variants[cursor.variant_idx as usize];
        let next_cursor = variant.advance_cursor(cursor.cursor as usize, key)?;
        let item = variant.item_for_key(key)?;
        if let Some(expected) = identity {
            if !shareable_property_schemas_match(expected, item) {
                return None;
            }
        } else {
            identity = Some(item);
            value_expr = Some(item.value_expr.clone());
        }
        next_cursors.push(AnyOfFixedObjectCursor {
            variant_idx: cursor.variant_idx,
            cursor: next_cursor as u16,
        });
    }

    if next_cursors.len() < 2 {
        return None;
    }
    Some((value_expr?, next_cursors))
}

fn merged_closed_any_of_state_id(
    builder: &mut ExprNfaBuilder,
    state_ids: &mut BTreeMap<AnyOfFixedObjectMergedState, u32>,
    queue: &mut VecDeque<AnyOfFixedObjectMergedState>,
    state: AnyOfFixedObjectMergedState,
) -> u32 {
    if let Some(&existing) = state_ids.get(&state) {
        existing
    } else {
        let id = builder.add_state();
        state_ids.insert(state.clone(), id);
        queue.push_back(state);
        id
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
