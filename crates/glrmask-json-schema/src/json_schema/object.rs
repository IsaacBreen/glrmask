use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use rayon::prelude::*;

use crate::grammar::expr_nfa::ExprNfaBuilder;
use crate::import::ast::{GrammarExpr, Quantifier};

use super::ast::{
    AdditionalProperties, ObjectSchema, PatternPropertySchema,
    PropertySchema, Schema, SchemaAssertions, SchemaKind, SchemaType,
};
use super::combinators::{
    all_of_schema, open_object_any_of_covers_json_object,
    simplify_finite_string_allof_schema, try_merge_all_of_objects,
    validate_object_all_of_merge_bounds,
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

const EXACT_UNORDERED_SMALL_OBJECT_MAX_ITEMS: usize = 3;
const LARGE_OBJECT_LITERAL_KEY_TRIE_MIN_ITEMS: usize = 64;
const LARGE_OBJECT_KEY_TRIE_PREFIX_SPLIT_BYTES: usize = 1;
const SPARSE_FIXED_OBJECT_MIN_OPTIONAL_PROPERTIES: usize = 64;

struct ObjectItem {
    key: String,
    pair: GrammarExpr,
    separator_pair: GrammarExpr,
    required: bool,
    satisfies_any_group: bool,
    exclusive_group: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FixedObjectSelection {
    OrderedDense,
    OrderedSparse,
    Unordered,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum FixedObjectPhase {
    First,
    After,
    Select,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FixedObjectObligations {
    cursor: usize,
    seen: u64,
    groups: u8,
    phase: FixedObjectPhase,
}

const ANYOF_FIXED_OBJECT_EXPR_NFA_MAX_STATES: usize = 4096;
const PROPERTY_DEPENDENCY_MAX_FIXED_PROPERTIES: usize = 12;

fn discriminator_anyof_fastpath_disabled() -> bool {
    std::env::var_os("GLRMASK_DISABLE_DISCRIMINATOR_ANYOF_FASTPATH").is_some()
}

struct AnyOfFixedObjectItem {
    key: String,
    value_expr: GrammarExpr,
    schema: Schema,
    required: bool,
}

type AnyOfFixedObjectVariant = AnyOfObjectVariant;

#[derive(Default)]
struct ClosedAnyOfCollectionProfile {
    lower_property_ms: f64,
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
enum ObjectPhase { Fixed, Pattern, Additional }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ObjectCursor {
    variant: u16,
    cursor: u16,
    phase: ObjectPhase,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ObjectFrontier {
    cursors: Vec<ObjectCursor>,
    has_content: bool,
}

/// A member transition label is the entire emitted key/value path. It is
/// safe to merge target cursors only for exactly equal labels, including
/// the separator and any lexical key policy.
#[derive(Clone, Copy)]
struct MemberLabels([usize; 2]);

impl MemberLabels {
    fn select(self, has_content: bool) -> usize { self.0[usize::from(has_content)] }
}

struct ObjectVariantPlan {
    fixed: Vec<MemberLabels>,
    next_required: Vec<usize>,
    patterns: Vec<MemberLabels>,
    additional: Vec<MemberLabels>,
}

struct ObjectUnionPlan {
    variants: Vec<ObjectVariantPlan>,
    labels: Vec<Vec<GrammarExpr>>,
}

impl ObjectVariantPlan {
    fn can_close(&self, cursor: ObjectCursor) -> bool {
        cursor.phase != ObjectPhase::Fixed
            || self.next_required[cursor.cursor as usize] == self.fixed.len()
    }

    fn for_each_member(&self, cursor: ObjectCursor, has_content: bool,
        mut visit: impl FnMut(usize, ObjectCursor))
    {
        if cursor.phase == ObjectPhase::Fixed {
            let start = cursor.cursor as usize;
            let end = (self.next_required[start] + 1).min(self.fixed.len());
            for i in start..end {
                visit(self.fixed[i].select(has_content), ObjectCursor {
                    cursor: (i + 1) as u16, ..cursor
                });
            }
        }
        if !self.can_close(cursor) { return; }
        if cursor.phase != ObjectPhase::Additional {
            for &label in &self.patterns {
                visit(label.select(has_content), ObjectCursor {
                    cursor: self.fixed.len() as u16, phase: ObjectPhase::Pattern, ..cursor
                });
            }
        }
        for &label in &self.additional {
            visit(label.select(has_content), ObjectCursor {
                cursor: self.fixed.len() as u16, phase: ObjectPhase::Additional, ..cursor
            });
        }
    }
}

fn is_obviously_object_valued_property(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };

    assertions.object.is_some()
        || assertions
            .types
            .as_ref()
            .is_some_and(|types| types.contains(&SchemaType::Object))
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
        && schema.property_names.is_none()
        && schema.property_dependencies.is_empty()
        && matches!(schema.additional_properties, AdditionalProperties::AllowAny)
}

impl<'a> Lowerer<'a> {
    const PARALLEL_DISCRIMINATOR_PAYLOAD_MIN_CASES: usize = 64;
    const PARALLEL_DISCRIMINATOR_PAYLOAD_MAX_WORKERS: usize = 4;
    const ISOLATED_PAYLOAD_RULE_ID_BASE: usize = 1_000_000_000;
    const ISOLATED_PAYLOAD_RULE_ID_STRIDE: usize = 1_000_000;

    pub fn lower_object(&mut self, schema: &ObjectSchema) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, None, None)
    }

    pub fn lower_object_requiring_any_property(
        &mut self,
        schema: &ObjectSchema,
        any_required_names: &BTreeSet<String>,
    ) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, Some(any_required_names), None)
    }

    pub fn lower_object_with_exclusive_properties(
        &mut self,
        schema: &ObjectSchema,
        exclusive_names: &BTreeSet<String>,
        require_one: bool,
    ) -> ImportResult<GrammarExpr> {
        self.lower_object_internal(schema, None, Some((exclusive_names, require_one)))
    }

    fn schema_contains_any_ref(schema: &Schema) -> bool {
        schema.descendants().any(|node| matches!(node.kind, SchemaKind::Ref(_)))
    }
    pub fn try_lower_closed_object_any_of_variants(
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
                "[glrmask/profile][json_schema_closed_anyof] branches={} variants={} outcome={} collect_ms={:.3} property_lower_ms={:.3} schema_clone_ms={:.3} nfa_build_ms={:.3} elapsed_ms={:.3}",
                branches.len(),
                variants.len(),
                result.is_some(),
                collect_ms,
                collection_profile.lower_property_ms,
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
    pub(super) fn try_lower_ordered_string_discriminator_closed_anyof(
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
        let payload_exprs = self.lower_discriminator_payloads(&cases)?;
        debug_assert_eq!(cases.len(), payload_exprs.len());

        for ((discriminator_value, payload_schema), payload_expr) in
            cases.into_iter().zip(payload_exprs)
        {
            let after_discriminator_value = builder.add_state();
            builder.add_transition(
                after_discriminator_key,
                self.lower_string_literal(&discriminator_value),
                after_discriminator_value,
            );
            let payload_started_at = profile_enabled.then(std::time::Instant::now);
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

    fn lower_discriminator_payloads(
        &mut self,
        cases: &[(String, &Schema)],
    ) -> ImportResult<Vec<GrammarExpr>> {
        // The lowerer has intentionally thread-local test compatibility state.
        // Keep unit tests serial, while production imports can parallelize only
        // non-recursive payloads whose generated rules are fully self-contained.
        let can_parallelize = !cfg!(test)
            && cases.len() >= Self::PARALLEL_DISCRIMINATOR_PAYLOAD_MIN_CASES
            && !cases
                .iter()
                .any(|(_, payload_schema)| Self::schema_contains_any_ref(payload_schema));
        let worker_count = rayon::current_num_threads()
            .min(Self::PARALLEL_DISCRIMINATOR_PAYLOAD_MAX_WORKERS)
            .min(cases.len());
        if !can_parallelize || worker_count < 2 {
            return cases
                .iter()
                .map(|(_, payload_schema)| self.lower_schema(payload_schema))
                .collect();
        }

        let chunk_size = cases.len().div_ceil(worker_count);
        let isolated = {
            let parent = &*self;
            cases
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_index, chunk)| {
                    let namespace = Self::ISOLATED_PAYLOAD_RULE_ID_BASE
                        .checked_add(
                            Self::ISOLATED_PAYLOAD_RULE_ID_STRIDE
                                .checked_mul(chunk_index)
                                .expect("isolated JSON Schema payload namespace overflow"),
                        )
                        .expect("isolated JSON Schema payload namespace overflow");
                    let mut lowerer = parent.isolated_fragment_lowerer(namespace);
                    let expressions = chunk
                        .iter()
                        .map(|(_, payload_schema)| lowerer.lower_schema(payload_schema))
                        .collect::<ImportResult<Vec<_>>>()?;
                    Ok::<_, SchemaImportError>((
                        expressions,
                        lowerer.rules,
                        lowerer.terminal_partition_classes,
                        lowerer.terminal_pattern_partition_keys,
                    ))
                })
                .collect::<ImportResult<Vec<_>>>()?
        };

        let mut expressions = Vec::with_capacity(cases.len());
        for (
            chunk_expressions,
            rules,
            terminal_partition_classes,
            terminal_pattern_partition_keys,
        ) in isolated
        {
            self.append_isolated_rules(
                rules,
                terminal_partition_classes,
                terminal_pattern_partition_keys,
            )?;
            expressions.extend(chunk_expressions);
        }
        Ok(expressions)
    }

    pub fn try_lower_open_object_any_of_variants(
        &mut self,
        branches: &[Schema],
    ) -> ImportResult<Option<GrammarExpr>> {
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let total_started = profile.then(std::time::Instant::now);
        if branches.len() < 2 {
            return Ok(None);
        }
        let recursive_started = profile.then(std::time::Instant::now);
        if self.has_duplicate_recursive_ref_branches(branches)? {
            return Ok(None);
        }
        let recursive_ms = recursive_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let collect_started = profile.then(std::time::Instant::now);
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
        let collect_ms = collect_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let lower_started = profile.then(std::time::Instant::now);
        let result = self.lower_open_any_of_object_variants_expr_nfa(
            &variants,
            include_untyped_non_object_alts,
        )?;
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][json_schema_open_anyof] branches={} recursive_check_ms={:.3} collect_ms={:.3} lower_ms={:.3} total_ms={:.3} success={}",
                branches.len(),
                recursive_ms,
                collect_ms,
                lower_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started.elapsed().as_secs_f64() * 1000.0,
                result.is_some(),
            );
        }
        Ok(result)
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

    pub fn try_lower_ref_string_path_object_any_of(
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
        let mut normalized = self.object_with_required_synthetic_properties(schema)?;
        if let Some(property_names) = &normalized.property_names {
            let resolved = match &property_names.kind {
                SchemaKind::Ref(pointer) => self.resolve_ref_target(pointer)?,
                _ => property_names,
            };
            if matches!(resolved.kind, SchemaKind::Never) {
                return Ok(if normalized.required.is_empty()
                    && normalized.min_properties == 0
                    && any_required_names.is_none()
                    && !exclusive_group.is_some_and(|(_, require_one)| require_one)
                {
                    seq(vec![lit("{"), lit("}")])
                } else {
                    never()
                });
            }
        }
        let property_name_pattern = self.resolve_property_names_pattern(&normalized)?;
        // A forbidden optional name may simply be absent. Keep it excluded from
        // additional keys, but give its fixed property an empty value language.
        if let Some(pattern) = &property_name_pattern {
            let mut forbidden = Vec::new();
            for (index, property) in normalized.properties.iter().enumerate() {
                if !self.property_name_matches_pattern_cached(pattern, &property.name)? {
                    if normalized.required.contains(&property.name) {
                        return Ok(never());
                    }
                    forbidden.push(index);
                }
            }
            for index in forbidden {
                let property = &mut normalized.to_mut().properties[index];
                property.schema = Schema::never(property.schema.location.clone());
            }
            if !normalized.pattern_properties.is_empty() {
                return Err(SchemaImportError::at(
                    &normalized.property_names.as_ref().unwrap().location,
                    "unsupported propertyNames combined with patternProperties: key predicate intersection is not implemented",
                ));
            }
        }
        if !normalized.property_dependencies.is_empty()
            && (!normalized.pattern_properties.is_empty()
                || !matches!(normalized.additional_properties, AdditionalProperties::Deny))
        {
            return Err(SchemaImportError::new(
                "unsupported property dependencies: only fixed-property objects with additionalProperties: false are supported",
            ));
        }
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
        // Dynamic key lists may repeat a name. Counting serialized members
        // cannot establish a lower bound on distinct data-model properties.
        if normalized.min_properties > 1
            && normalized.min_properties > normalized.required.len()
            && (!normalized.pattern_properties.is_empty()
                || !matches!(normalized.additional_properties, AdditionalProperties::Deny))
        {
            return Err(SchemaImportError::new(
                "unsupported minProperties: dynamic property names cannot guarantee distinct keys for minima greater than one",
            ));
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
                        return Err(SchemaImportError::new(
                            "unsupported maxProperties: fixed objects currently support at most one optional property below the cap",
                        ));
                    }
                }
            } else if !fixed_closed_redundant && !pattern_map_candidate && !open_dynamic_map_candidate
                && !max_properties_filled_by_required {
                    return Err(SchemaImportError::new(
                        "unsupported maxProperties combined with fixed properties or patternProperties",
                    ));
                }
        }
        let extra_min_properties =
            normalized.min_properties.saturating_sub(normalized.required.len());
        let min_property_group_required = if extra_min_properties == 0
            || (normalized.properties.is_empty()
                && normalized.required.is_empty()
                && (!matches!(normalized.additional_properties, AdditionalProperties::Deny)
                    || normalized.pattern_properties.len() == 1))
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
            if matches!(normalized.additional_properties, AdditionalProperties::Deny)
                && normalized.pattern_properties.is_empty()
                && normalized.min_properties > normalized.properties.len()
            {
                return Ok(never());
            }
            return Err(SchemaImportError::new(
                "unsupported minProperties combined with fixed properties or patternProperties",
            ));
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
                    && !min_property_group_required
                    && !max_property_group_exclusive
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
                AdditionalProperties::Schema(_) if max_properties_filled_by_required => None,
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
            if use_large_optional_open_object_prefix_chain
                && let Some(tail_pair_expr) = tail_pair {
                    return self.lower_large_optional_open_object_fused_prefix_chain(&items, tail_pair_expr);
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
        for pattern_property in normalized.pattern_properties.iter().filter(|_| !max_properties_filled_by_required) {
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

        let body = self.dynamic_pair_list_body(
            r(&pair_name), schema.min_properties, schema.max_properties,
        );

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
            || assertions.not.is_some()
        {
            return Ok(None);
        }

        if open_object_any_of_covers_json_object(&assertions.any_of)
            && super::string::json_string_compat_mode()
                == super::string::JsonStringCompatMode::JsonSchema
            {
                return Ok(Some(r(JSON_OBJECT_RULE)));
            }

            // In llguidance mode, preserve open anyOf lowering so unknown keys
            // use additional-key semantics instead of strict JSON_OBJECT keys.

        self.try_lower_open_object_any_of_variants(&assertions.any_of)
    }

    fn lower_fixed_object_body_exprnfa(
        &mut self,
        items: &[ObjectItem],
        tail_pair: Option<GrammarExpr>,
        any_group_required: bool,
        exclusive_require_one: bool,
    ) -> ImportResult<GrammarExpr> {
        self.lower_fixed_object_plan(items, tail_pair, any_group_required, exclusive_require_one, None)
    }

    fn lower_fixed_object_body_exprnfa_with_property_dependencies(
        &mut self,
        items: &[ObjectItem],
        property_dependencies: &BTreeMap<String, BTreeSet<String>>,
    ) -> ImportResult<GrammarExpr> {
        if items.len() > PROPERTY_DEPENDENCY_MAX_FIXED_PROPERTIES {
            return Err(SchemaImportError::new(format!(
                "resource limit: property dependencies support at most {PROPERTY_DEPENDENCY_MAX_FIXED_PROPERTIES} fixed properties"
            )));
        }
        self.lower_fixed_object_plan(items, None, false, false, Some(property_dependencies))
    }
    fn collect_closed_any_of_object_variant(
        &mut self,
        branch: &Schema,
        profile: Option<&mut ClosedAnyOfCollectionProfile>,
    ) -> ImportResult<Option<(AnyOfFixedObjectVariant, bool)>> {
        self.collect_object_variant(branch, true, 0, profile)
    }

    fn collect_open_any_of_object_variant(
        &mut self,
        branch: &Schema,
    ) -> ImportResult<Option<(AnyOfObjectVariant, bool)>> {
        self.collect_object_variant(branch, false, 0, None)
    }

    /// A closed variant is simply one with an empty dynamic tail. Resolve
    /// aliases and select the object predicate once; the caller chooses the
    /// applicable shape policy before any property grammar is emitted.
    fn collect_object_variant(
        &mut self,
        branch: &Schema,
        closed: bool,
        depth: usize,
        mut profile: Option<&mut ClosedAnyOfCollectionProfile>,
    ) -> ImportResult<Option<(AnyOfObjectVariant, bool)>> {
        if let SchemaKind::Ref(pointer) = &branch.kind {
            if depth >= 4 { return Ok(None); }
            let normalized = normalize_local_ref(pointer)?;
            let target = self.resolve_ref_target(pointer)?;
            if !self.object_variant_ref_stack.insert(normalized.clone()) {
                return Ok(None);
            }
            let result = self.collect_object_variant(target, closed, depth + 1, profile);
            self.object_variant_ref_stack.remove(&normalized);
            return result;
        }
        let SchemaKind::Assertions(assertions) = &branch.kind else { return Ok(None); };
        if assertions.const_value.is_some() || assertions.enum_values.is_some()
            || assertions.array.is_some() || assertions.string.is_some() || assertions.number.is_some()
            || !assertions.any_of.is_empty() || !assertions.one_of.is_empty() || assertions.not.is_some()
            || assertions.types.as_ref().is_some_and(|types| {
                super::ast::TypeDomain::from_types(types) != super::ast::TypeDomain::of(SchemaType::Object)
            })
        {
            return Ok(None);
        }
        let merged;
        let (object, include_nonobjects) = if assertions.all_of.is_empty() {
            let Some(object) = &assertions.object else { return Ok(None); };
            (object, assertions.types.is_none())
        } else {
            if assertions.object.as_ref().is_some_and(|object| !is_unconstrained_open_object_schema(object)) {
                return Ok(None);
            }
            // Closed variants may inline one reference per conjunct, as
            // before. Open variants retain their existing plain-shape gate.
            let mut resolved = Vec::with_capacity(assertions.all_of.len());
            for child in &assertions.all_of {
                resolved.push(match &child.kind {
                    SchemaKind::Ref(pointer) if closed => self.resolve_ref_target(pointer)?.clone(),
                    _ => child.clone(),
                });
            }
            validate_object_all_of_merge_bounds(&resolved)?;
            merged = match try_merge_all_of_objects(&resolved) {
                Some(object) => object,
                None => return Ok(None),
            };
            (&merged, assertions.types.is_none() && !all_of_has_explicit_object_only_type(&resolved))
        };
        if object.property_names.is_some() || !object.property_dependencies.is_empty()
            || object.min_properties != 0 || object.max_properties.is_some()
            || closed && (!matches!(object.additional_properties, AdditionalProperties::Deny)
                || !object.pattern_properties.is_empty()
                || object.required.iter().any(|name| !object.properties.iter().any(|p| p.name == *name)))
        {
            return Ok(None);
        }
        let normalized = self.object_with_required_synthetic_properties(object)?;
        if normalized.properties.len() > u16::MAX as usize { return Ok(None); }
        let fixed_keys = normalized.properties.iter().map(|property| property.name.clone()).collect::<BTreeSet<_>>();
        let mut items = Vec::with_capacity(normalized.properties.len());
        for property in &normalized.properties {
            let mut effective = Cow::Borrowed(&property.schema);
            for pattern in &normalized.pattern_properties {
                if self.property_name_matches_pattern_cached(&pattern.pattern, &property.name)? {
                    let extra = pattern_schema_for_property(effective.as_ref(), &pattern.schema);
                    effective = Cow::Owned(all_of_schema(effective.into_owned(), extra));
                }
            }
            if let Some(simplified) = simplify_finite_string_allof_schema(effective.as_ref()) {
                effective = Cow::Owned(simplified);
            }
            let started = profile.as_ref().map(|_| std::time::Instant::now());
            let value_expr = self.lower_schema(effective.as_ref())?;
            if let (Some(started), Some(profile)) = (started, profile.as_deref_mut()) {
                profile.lower_property_ms += started.elapsed().as_secs_f64() * 1000.0;
            }
            let started = profile.as_ref().map(|_| std::time::Instant::now());
            let schema = effective.into_owned();
            if let (Some(started), Some(profile)) = (started, profile.as_deref_mut()) {
                profile.schema_clone_ms += started.elapsed().as_secs_f64() * 1000.0;
            }
            items.push(AnyOfFixedObjectItem {
                key: property.name.clone(), value_expr, schema,
                required: normalized.required.contains(&property.name),
            });
        }
        let mut pattern_pairs = Vec::with_capacity(normalized.pattern_properties.len());
        for property in &normalized.pattern_properties {
            let key = self.lower_pattern_key_colon_appearance(&property.pattern, &fixed_keys)?;
            let value = self.lower_schema(&property.schema)?;
            pattern_pairs.push(seq(vec![key, value]));
        }
        let additional_value_expr = match &normalized.additional_properties {
            AdditionalProperties::AllowAny => Some(r(JSON_VALUE_RULE)),
            AdditionalProperties::Deny => None,
            AdditionalProperties::Schema(value_schema) => Some(self.lower_schema(value_schema)?),
        };
        Ok(Some((AnyOfObjectVariant {
            items, fixed_keys, pattern_pairs, additional_value_expr,
            pattern_keys: normalized.pattern_properties.iter().map(|p| p.pattern.clone()).collect(),
        }, include_nonobjects)))
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
        self.lower_object_union(variants, include_untyped_non_object_alts, true)
    }

    fn prepare_object_union(
        &mut self,
        variants: &[AnyOfObjectVariant],
        closed: bool,
    ) -> ImportResult<ObjectUnionPlan> {
        let mut plan = ObjectUnionPlan { variants: Vec::with_capacity(variants.len()), labels: Vec::new() };
        let mut ids = HashMap::<Vec<GrammarExpr>, usize>::new();
        let mut intern = |path: Vec<GrammarExpr>| {
            *ids.entry(path).or_insert_with_key(|path| {
                let label = plan.labels.len();
                plan.labels.push(path.clone());
                label
            })
        };
        let separator = self.item_separator_expr();
        let mut key_paths = HashMap::<String, [GrammarExpr; 2]>::new();
        for variant in variants {
            let n = variant.items.len();
            let mut next_required = vec![n; n + 1];
            for i in (0..n).rev() {
                next_required[i] = if variant.items[i].required { i } else { next_required[i + 1] };
            }
            let mut fixed = Vec::with_capacity(n);
            for item in &variant.items {
                let keys = key_paths.entry(item.key.clone()).or_insert_with(|| {
                    let first = self.lower_literal_key_colon(&item.key);
                    let next = if closed {
                        self.lower_literal_key_colon_with_prefix(b", ", &item.key)
                    } else { first.clone() };
                    [first, next]
                });
                let first = intern(vec![keys[0].clone(), item.value_expr.clone()]);
                let mut next = Vec::with_capacity(3);
                if !closed { next.push(separator.clone()); }
                next.extend([keys[1].clone(), item.value_expr.clone()]);
                fixed.push(MemberLabels([first, intern(next)]));
            }
            let mut tail_labels = |path: Vec<GrammarExpr>| {
                let mut next = Vec::with_capacity(path.len() + 1);
                next.push(separator.clone());
                next.extend(path.iter().cloned());
                MemberLabels([intern(path), intern(next)])
            };
            let mut patterns = Vec::new();
            for pair in &variant.pattern_pairs {
                for [key, value] in Self::split_object_pair_symbol_paths(pair)? {
                    patterns.push(tail_labels(Self::object_pair_path_symbols(key, value)));
                }
            }
            let mut additional = Vec::new();
            if let Some(value) = &variant.additional_value_expr {
                let key = if self.llguidance_compat_enabled() && variant.fixed_keys.is_empty()
                    && variant.pattern_keys.is_empty()
                {
                    seq(vec![r(json_key_string_rule()), self.key_separator_expr()])
                } else {
                    self.lower_additional_key_colon(&variant.fixed_keys, &variant.pattern_keys)?
                };
                for mut path in Self::split_key_colon_symbol_paths(key) {
                    path.push(value.clone());
                    additional.push(tail_labels(path));
                }
            }
            plan.variants.push(ObjectVariantPlan { fixed, next_required, patterns, additional });
        }
        Ok(plan)
    }

    fn build_object_union(plan: &ObjectUnionPlan, share: bool) -> Option<ExprNfaBuilder> {
        const MAX_FRONTIER_CELLS: usize = 262_144;
        let mut builder = ExprNfaBuilder::new();
        let mut state_ids = BTreeMap::new();
        let mut queue = VecDeque::new();
        let mut cells = 0usize;
        let mut intern = |builder: &mut ExprNfaBuilder, ids: &mut BTreeMap<ObjectFrontier, u32>,
            pending: &mut VecDeque<ObjectFrontier>, frontier: ObjectFrontier| -> Option<u32> {
            if let Some(&id) = ids.get(&frontier) { return Some(id); }
            cells = cells.checked_add(frontier.cursors.len())?;
            if cells > MAX_FRONTIER_CELLS || ids.len() >= ANYOF_FIXED_OBJECT_EXPR_NFA_MAX_STATES {
                return None;
            }
            let id = builder.add_state();
            ids.insert(frontier.clone(), id);
            pending.push_back(frontier);
            Some(id)
        };
        let cursors = (0..plan.variants.len()).map(|i| ObjectCursor {
            variant: i as u16, cursor: 0, phase: ObjectPhase::Fixed,
        }).collect::<Vec<_>>();
        let starts = if share { vec![cursors] } else {
            cursors.into_iter().map(|cursor| vec![cursor]).collect()
        };
        for cursors in starts {
            let start = intern(&mut builder, &mut state_ids, &mut queue,
                ObjectFrontier { cursors, has_content: false })?;
            builder.add_epsilon(builder.start_state(), start);
        }
        while let Some(frontier) = queue.pop_front() {
            let source = state_ids[&frontier];
            let mut outgoing = BTreeMap::<usize, Vec<ObjectCursor>>::new();
            let mut accepts = false;
            for cursor in frontier.cursors {
                let variant = &plan.variants[cursor.variant as usize];
                accepts |= variant.can_close(cursor);
                variant.for_each_member(cursor, frontier.has_content, |label, target| {
                    outgoing.entry(label).or_default().push(target);
                });
            }
            if accepts { builder.set_accepting(source); }
            for (label, mut cursors) in outgoing {
                cursors.sort_unstable();
                cursors.dedup();
                let groups = if share { vec![cursors] } else {
                    cursors.into_iter().map(|cursor| vec![cursor]).collect()
                };
                for cursors in groups {
                    let target = intern(&mut builder, &mut state_ids, &mut queue,
                        ObjectFrontier { cursors, has_content: true })?;
                    Self::add_expr_nfa_symbol_path(&mut builder, source, plan.labels[label].clone(), target);
                }
            }
        }
        Some(builder)
    }

    fn lower_object_union(
        &mut self,
        variants: &[AnyOfObjectVariant],
        include_nonobjects: bool,
        closed: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        if variants.is_empty() || variants.len() > u16::MAX as usize { return Ok(None); }
        let plan = self.prepare_object_union(variants, closed)?;
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let started = profile.then(std::time::Instant::now);
        // Ordinary anyOf is union, not subtraction. The old shadow-owner
        // route suppressed acceptance using a speculative match of a broader
        // value/key against an owner's narrower language. No such implication
        // follows from a shared key. Instead, preserve both cursors until a
        // complete identical emitted label proves a safe subset transition.
        let Some(builder) = Self::build_object_union(&plan, true)
            .or_else(|| Self::build_object_union(&plan, false)) else { return Ok(None); };
        let graph_ms = started.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        let started = profile.then(std::time::Instant::now);
        let body = GrammarExpr::ExprNFA(Box::new(builder.build().into_determinized_and_minimized()));
        if let Some(t) = started {
            eprintln!("[glrmask/profile][json_schema_object_union_build] graph_ms={:.3} determinize_minimize_ms={:.3}",
                graph_ms, t.elapsed().as_secs_f64() * 1000.0);
        }
        let name = self.fresh_rule_name("json_anyof_object_body");
        self.add_nonterminal_rule(&name, body);
        Ok(Some(Self::object_union_result(&name, include_nonobjects)))
    }

    fn object_union_result(rule_name: &str, include_nonobjects: bool) -> GrammarExpr {
        let object = seq(vec![lit("{"), r(rule_name), lit("}")]);
        if !include_nonobjects { return object; }
        choice(vec![object, r(JSON_ARRAY_RULE), r(JSON_STRING_RULE),
            r(JSON_NUMBER_RULE), r(JSON_BOOL_RULE), r(JSON_NULL_RULE)])
    }

    fn is_json_value_expr(expr: &GrammarExpr) -> bool {
        matches!(expr, GrammarExpr::Ref(rule_name) if rule_name == JSON_VALUE_RULE)
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
                let Some(value) = plain_singleton_string_enum_value(&variant.items[index].schema) else {
                    all_singleton = false;
                    break;
                };
                if !seen.insert(value) {
                    all_singleton = false;
                    break;
                }
            }
            if all_singleton
                && discriminator_index.replace(index).is_some() {
                    return Ok(None);
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
            let Some(discriminator_value) = plain_singleton_string_enum_value(&discriminator.schema) else {
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
        if variants.is_empty() { return Ok(None); }
        if !discriminator_anyof_fastpath_disabled()
            && let Some(expr) = self.try_lower_discriminator_value_open_any_of_object(
                variants, include_untyped_non_object_alts,
            )? {
                return Ok(Some(expr));
            }
        self.lower_object_union(variants, include_untyped_non_object_alts, false)
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

    /// The state is the remaining property obligations and whether the next
    /// member requires a separator. Epsilon skips are an encoding of ordered
    /// optional selection, not a different object language.
    fn emit_fixed_object_plan(
        &self,
        builder: &mut ExprNfaBuilder,
        items: &[ObjectItem],
        symbols: &[[GrammarExpr; 2]],
        tail: Option<&GrammarExpr>,
        any_required: bool,
        exclusive_required: bool,
        dependencies: Option<&BTreeMap<String, BTreeSet<String>>>,
        mode: FixedObjectSelection,
    ) -> ImportResult<()> {
        let unordered = mode == FixedObjectSelection::Unordered;
        let sparse = mode == FixedObjectSelection::OrderedSparse;
        let track_seen = unordered || dependencies.is_some();
        let separator = self.item_separator_expr();
        let n = items.len();
        let mut next_required = vec![n; n + 1];
        for i in (0..n).rev() {
            next_required[i] = if items[i].required { i } else { next_required[i + 1] };
        }
        let required = if track_seen {
            items.iter().enumerate().fold(0u64, |mask, (i, item)| {
                mask | if item.required { 1u64 << i } else { 0 }
            })
        } else {
            0
        };
        let mut dependency_masks = vec![0u64; n];
        let mut impossible = 0u64;
        if let Some(dependencies) = dependencies {
            let indexes: BTreeMap<_, _> = items.iter().enumerate()
                .map(|(i, item)| (item.key.as_str(), i)).collect();
            for (trigger, dependents) in dependencies {
                let Some(&i) = indexes.get(trigger.as_str()) else { continue };
                for dependent in dependents {
                    if let Some(&j) = indexes.get(dependent.as_str()) {
                        dependency_masks[i] |= 1u64 << j;
                    } else {
                        impossible |= 1u64 << i;
                    }
                }
            }
        }
        let can_close = |state: FixedObjectObligations| {
            (if unordered { state.seen & required == required }
                else { next_required[state.cursor] == n })
                && (!any_required || state.groups & 1 != 0)
                && (!exclusive_required || state.groups & 2 != 0)
                && state.seen & impossible == 0
                && (dependencies.is_none() || dependency_masks.iter().enumerate().all(|(i, &mask)| {
                    state.seen & (1u64 << i) == 0 || state.seen & mask == mask
                }))
        };
        // Preserve the abstract symbol numbering used by dense templates.
        if !sparse {
            for pair in symbols {
                builder.add_symbol(pair[0].clone());
                builder.add_symbol(pair[1].clone());
                builder.add_symbol(separator.clone());
            }
        }
        let initial = FixedObjectObligations { cursor: 0, seen: 0, groups: 0, phase: FixedObjectPhase::First };
        let mut ids = HashMap::from([(initial, builder.start_state())]);
        let mut pending = VecDeque::from([initial]);
        let mut continuations = HashMap::<(usize, u32), u32>::new();
        let tail_symbols = tail.map(Self::split_object_pair_symbols).transpose()?;
        let tail_states = if let Some(pair) = &tail_symbols {
            let after = builder.add_state();
            let select = builder.add_state();
            builder.set_accepting(after);
            builder.add_transition(after, separator.clone(), select);
            Self::add_expr_nfa_symbol_path(builder, select, pair.to_vec(), after);
            Some((select, after))
        } else { None };
        while let Some(state) = pending.pop_front() {
            let id = ids[&state];
            let close = can_close(state);
            let mut intern = |builder: &mut ExprNfaBuilder, next| {
                *ids.entry(next).or_insert_with(|| {
                    pending.push_back(next);
                    builder.add_state()
                })
            };
            if state.phase != FixedObjectPhase::Select && close {
                builder.set_accepting(id);
            }
            if state.phase == FixedObjectPhase::After {
                if state.cursor < n || unordered && state.seen.count_ones() < n as u32 || close && tail_states.is_some() {
                    let next = intern(builder, FixedObjectObligations { phase: FixedObjectPhase::Select, ..state });
                    builder.add_transition(id, separator.clone(), next);
                }
                continue;
            }
            if close && (!sparse || state.cursor == n) {
                if let Some((select, _)) = tail_states {
                    builder.add_epsilon(id, select);
                }
            }
            if sparse && state.cursor < n && !items[state.cursor].required {
                let next = intern(builder, FixedObjectObligations { cursor: state.cursor + 1, ..state });
                builder.add_epsilon(id, next);
            }
            let end = if unordered { n } else if sparse {
                (state.cursor + 1).min(n)
            } else {
                (next_required[state.cursor] + 1).min(n)
            };
            for i in state.cursor..end {
                let item = &items[i];
                if unordered && state.seen & (1u64 << i) != 0
                    || item.exclusive_group && state.groups & 2 != 0
                {
                    continue;
                }
                let next = intern(builder, FixedObjectObligations {
                    cursor: if unordered { 0 } else { i + 1 },
                    seen: if track_seen { state.seen | (1u64 << i) } else { 0 },
                    groups: state.groups | u8::from(any_required && item.satisfies_any_group)
                        | (u8::from(item.exclusive_group) << 1),
                    phase: FixedObjectPhase::After,
                });
                if sparse {
                    Self::add_expr_nfa_symbol_path(builder, id, symbols[i].to_vec(), next);
                } else {
                    // All sources selecting this property with the same
                    // remaining obligations can share its value continuation.
                    let key_state = *continuations.entry((i, next)).or_insert_with(|| {
                        let key_state = builder.add_state();
                        builder.add_transition(key_state, symbols[i][1].clone(), next);
                        key_state
                    });
                    builder.add_transition(id, symbols[i][0].clone(), key_state);
                }
            }
        }
        Ok(())
    }
    fn lower_fixed_object_plan(
        &mut self,
        items: &[ObjectItem],
        tail_pair: Option<GrammarExpr>,
        any_required: bool,
        exclusive_required: bool,
        dependencies: Option<&BTreeMap<String, BTreeSet<String>>>,
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
        let grouped = any_required || exclusive_required || items.iter().any(|item| item.exclusive_group);
        let ordinary = !grouped && dependencies.is_none();
        let sparse_direct_nfa = ordinary && self.config.sparse_large_optional_objects
            && items.len() - required_count >= SPARSE_FIXED_OBJECT_MIN_OPTIONAL_PROPERTIES;
        let mode = if !grouped && tail_pair.is_none() && !sparse_direct_nfa
            && !self.llguidance_compat_enabled()
            && (2..=EXACT_UNORDERED_SMALL_OBJECT_MAX_ITEMS).contains(&items.len())
        {
            FixedObjectSelection::Unordered
        } else if grouped || dependencies.is_some() || tail_pair.is_some() || sparse_direct_nfa {
            FixedObjectSelection::OrderedSparse
        } else {
            FixedObjectSelection::OrderedDense
        };
        let template_symbols = (ordinary && mode != FixedObjectSelection::OrderedSparse).then(|| {
            Self::fixed_object_template_symbols(
                items,
                &item_symbols,
                self.item_separator_expr(),
            )
        });
        if let Some((template_key, symbols)) = &template_symbols
            && let Some(mut template_nfa) = self.fixed_object_nfa_templates.get(template_key).cloned()
        {
            let rule_name = self.fresh_rule_name("json_closed_object_body");
            template_nfa.symbols = symbols.clone();
            let body = GrammarExpr::ExprNFA(Box::new(template_nfa));
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
        self.emit_fixed_object_plan(
            &mut builder, items, &item_symbols, tail_pair.as_ref(),
            any_required, exclusive_required, dependencies, mode,
        )?;
        let graph_build_ms = graph_build_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let determinize_minimize_started_at =
            profile_started_at.map(|_| std::time::Instant::now());
        let rule_name = self.fresh_rule_name("json_closed_object_body");
        let expr_nfa = if sparse_direct_nfa {
            builder.build().with_embedded_direct_nfa_emission()
        } else {
            builder.build().into_determinized_and_minimized()
        };
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
        Ok(self.lower_large_object_prefix(items, Some(tail_pair_expr)))
    }

    fn lower_required_prefix_large_optional_open_object_fused_prefix_chain(
        &mut self,
        items: &[ObjectItem],
        required_prefix_len: usize,
        tail_pair_expr: GrammarExpr,
    ) -> ImportResult<GrammarExpr> {
        debug_assert!(required_prefix_len > 0 && required_prefix_len <= items.len());
        Ok(self.lower_large_object_prefix(items, Some(tail_pair_expr)))
    }

    fn lower_large_closed_object_prefix_chain(&mut self, items: &[ObjectItem]) -> GrammarExpr {
        self.lower_large_object_prefix(items, None)
    }

    /// Prefix language recurrence: empty is legal exactly when all visited
    /// properties are optional; nonempty prefixes either append the next
    /// member (using its fused separator) or skip an optional member.
    /// This avoids enumerating every pair of optional-property positions.
    fn lower_large_object_prefix(
        &mut self,
        items: &[ObjectItem],
        tail: Option<GrammarExpr>,
    ) -> GrammarExpr {
        let mut empty = true;
        let mut prefix: Option<String> = None;
        let rule_prefix = if tail.is_some() { "json_open_object_prefix" } else { "json_closed_object_prefix" };
        for item in items {
            let mut alternatives = Vec::with_capacity(2);
            if empty { alternatives.push(item.pair.clone()); }
            if let Some(previous) = prefix {
                let append = if item.required { item.separator_pair.clone() }
                    else { GrammarExpr::Quantified(Box::new(item.separator_pair.clone()), Quantifier::Optional) };
                alternatives.push(seq(vec![r(&previous), append]));
            }
            let name = self.fresh_rule_name(rule_prefix);
            self.add_nonterminal_rule(&name, choice(alternatives));
            prefix = Some(name);
            empty &= !item.required;
        }
        let free = tail.map(|pair| {
            let name = self.fresh_rule_name("json_open_object_free_nonempty");
            self.add_nonterminal_rule(&name, seq(vec![
                pair.clone(),
                GrammarExpr::Quantified(Box::new(seq(vec![self.item_separator_expr(), pair])), Quantifier::ZeroPlus),
            ]));
            name
        });
        let mut alternatives = Vec::with_capacity(3);
        if empty {
            alternatives.push(GrammarExpr::Epsilon);
            if let Some(name) = &free { alternatives.push(r(name)); }
        }
        if let Some(name) = prefix {
            let mut body = vec![r(&name)];
            if let Some(free) = &free {
                body.push(GrammarExpr::Quantified(Box::new(seq(vec![self.item_separator_expr(), r(free)])), Quantifier::Optional));
            }
            alternatives.push(seq(body));
        }
        seq(vec![lit("{"), choice(alternatives), lit("}")])
    }

    fn lower_property_item(
        &mut self,
        property: &PropertySchema,
        pattern_properties: &[PatternPropertySchema],
        required: bool,
        satisfies_any_group: bool,
        exclusive_group: bool,
    ) -> ImportResult<ObjectItem> {
        let mut intersected_schema = None;
        for pattern_property in pattern_properties {
            if self.property_name_matches_pattern_cached(
                &pattern_property.pattern,
                &property.name,
            )? {
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
        let key = self.lower_literal_key_colon(&property.name);
        let separator_key = self.lower_literal_key_colon_with_prefix(b", ", &property.name);
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
            types.contains(&SchemaType::String)
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

    fn lower_object_property_value_schema(
        &mut self,
        schema: &Schema,
    ) -> ImportResult<GrammarExpr> {
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
        if matches!(resolved.kind, SchemaKind::Any) {
            return Ok(None);
        }
        let SchemaKind::Assertions(assertions) = &resolved.kind else {
            return Err(SchemaImportError::at(
                &resolved.location,
                "unsupported propertyNames: only boolean or inline/local-ref string pattern schemas are supported",
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
                "unsupported propertyNames: only boolean or inline/local-ref string pattern schemas are supported",
            ));
        }
        if let Some(types) = &assertions.types
            && !types.iter().all(|schema_type| *schema_type == SchemaType::String)
        {
            return Err(SchemaImportError::at(
                &resolved.location,
                "unsupported propertyNames: only boolean or inline/local-ref string pattern schemas are supported",
            ));
        }

        let Some(string) = &assertions.string else {
            return Ok(None);
        };
        if string.min_length != 0 || string.max_length.is_some() || string.format.is_some() {
            return Err(SchemaImportError::at(
                &resolved.location,
                "unsupported propertyNames: only boolean or inline/local-ref string pattern schemas are supported",
            ));
        }

        Ok(string.pattern.clone())
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
                if self.property_name_matches_pattern_cached(
                    &pattern_property.pattern,
                    required_name,
                )? {
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
    if let Some(types) = &assertions.types
        && !types.iter().all(|schema_type| *schema_type == SchemaType::Object) {
            return false;
        }

    let Some(object) = &assertions.object else {
        return false;
    };
    if !object.pattern_properties.is_empty() || object.properties.len() != 1 {
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

#[cfg(test)]
mod object_assertion_policy_tests {
    use super::*;
    use crate::json_schema::ast::{SchemaDocument, StringSchema};
    use crate::json_schema::config::JsonSchemaConfig;

    fn lower_schema(schema: Schema) -> ImportResult<GrammarExpr> {
        let document = SchemaDocument { root: schema, definitions: Vec::new(), ref_targets: Vec::new() };
        let grammar = crate::json_schema::lower::lower_document(&document, JsonSchemaConfig::default())?;
        let start = grammar.rules.iter().find(|rule| rule.name == grammar.start).unwrap();
        let GrammarExpr::Ref(root) = &start.expr else { panic!("expected root reference") };
        Ok(grammar.rules.iter().find(|rule| rule.name == *root).unwrap().expr.clone())
    }

    fn lower_object(object: &ObjectSchema) -> ImportResult<GrammarExpr> {
        lower_schema(Schema::assertions("#", SchemaAssertions {
            types: Some(vec![SchemaType::Object]),
            object: Some(object.clone()),
            ..SchemaAssertions::default()
        }))
    }

    fn closed_properties(count: usize) -> ObjectSchema {
        ObjectSchema {
            properties: (0..count).map(|i| PropertySchema {
                name: format!("p{i}"),
                schema: Schema::any(format!("#/properties/p{i}")),
            }).collect(),
            additional_properties: AdditionalProperties::Deny,
            ..ObjectSchema::default()
        }
    }

    fn name_pattern() -> Schema {
        Schema::assertions("#/propertyNames", SchemaAssertions {
            string: Some(StringSchema {
                pattern: Some("^allowed$".to_string()),
                ..StringSchema::default()
            }),
            ..SchemaAssertions::default()
        })
    }

    #[test]
    fn unsupported_fixed_property_counts_return_keyword_errors() {
        let mut object = closed_properties(4);
        object.max_properties = Some(2);
        let error = lower_object(&object).unwrap_err();
        assert!(error.message().contains("unsupported maxProperties"));
        object.max_properties = None;
        object.min_properties = 2;
        let error = lower_object(&object).unwrap_err();
        assert!(error.message().contains("unsupported minProperties"));
    }

    #[test]
    fn exact_and_redundant_fixed_property_counts_remain_supported() {
        let mut object = closed_properties(4);
        object.min_properties = 1;
        object.max_properties = Some(1);
        assert!(lower_object(&object).is_ok());
        object.max_properties = Some(4);
        assert!(lower_object(&object).is_ok());
        object.min_properties = 5;
        assert_eq!(lower_object(&object).unwrap(), never());
    }

    #[test]
    fn property_names_forbids_optional_names_without_rejecting_the_object() {
        let mut object = closed_properties(1);
        object.property_names = Some(name_pattern());
        assert!(lower_object(&object).is_ok());
        object.required.insert("p0".to_string());
        assert_eq!(lower_object(&object).unwrap(), never());
    }

    #[test]
    fn property_names_pattern_with_pattern_properties_is_explicitly_unsupported() {
        let mut object = ObjectSchema {
            property_names: Some(name_pattern()),
            pattern_properties: vec![PatternPropertySchema {
                pattern: "^p".to_string(),
                schema: Schema::any("#/patternProperties/^p"),
            }],
            ..ObjectSchema::default()
        };
        let error = lower_object(&object).unwrap_err();
        assert!(error.message().contains("#/propertyNames"));
        assert!(error.message().contains("unsupported propertyNames"));
        object.property_names = Some(Schema::any("#/propertyNames"));
        assert!(lower_object(&object).is_ok());
    }

    #[test]
    fn boolean_property_names_has_exact_empty_object_case() {
        let mut object = ObjectSchema {
            property_names: Some(Schema::never("#/propertyNames")),
            ..ObjectSchema::default()
        };
        assert_eq!(lower_object(&object).unwrap(), seq(vec![lit("{"), lit("}")]));
        object.min_properties = 1;
        assert_eq!(lower_object(&object).unwrap(), never());
    }

    #[test]
    fn dynamic_dependencies_cannot_bypass_the_fixed_property_check() {
        let mut object = ObjectSchema::default();
        object.property_dependencies.insert("a".to_string(), BTreeSet::from(["b".to_string()]));
        let error = lower_object(&object).unwrap_err();
        assert!(error.message().contains("unsupported property dependencies"));
        object.additional_properties = AdditionalProperties::Deny;
        // Neither trigger nor dependent can appear in a closed empty object.
        assert!(lower_object(&object).is_ok());
    }

    #[test]
    fn anyof_collectors_defer_when_object_assertions_need_enforcement() {
        let mut object = closed_properties(3);
        object.min_properties = 2;
        let branch = Schema::assertions("#/anyOf/0", SchemaAssertions {
            types: Some(vec![SchemaType::Object]),
            object: Some(object),
            ..SchemaAssertions::default()
        });
        let schema = Schema::assertions("#", SchemaAssertions {
            any_of: vec![branch, Schema::assertions("#/anyOf/1", SchemaAssertions {
                types: Some(vec![SchemaType::Object]),
                object: Some(closed_properties(1)),
                ..SchemaAssertions::default()
            })],
            ..SchemaAssertions::default()
        });
        assert!(lower_schema(schema).unwrap_err().message().contains("unsupported minProperties"));
    }

    #[test]
    fn unconstrained_object_shortcut_requires_no_name_or_dependency_assertions() {
        let mut object = ObjectSchema::default();
        assert!(is_unconstrained_open_object_schema(&object));
        object.property_names = Some(name_pattern());
        assert!(!is_unconstrained_open_object_schema(&object));
        object.property_names = None;
        object.property_dependencies.insert("a".to_string(), BTreeSet::from(["b".to_string()]));
        assert!(!is_unconstrained_open_object_schema(&object));
    }
}
