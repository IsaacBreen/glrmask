use std::collections::BTreeSet;

use regex::Regex;
use regex_syntax::hir::{Class, Hir, HirKind, Literal, Look, Repetition};
use regex_syntax::Parser;

use crate::import::ast::GrammarExpr;

use super::ast::StringSchema;
use super::error::{ImportResult, SchemaImportError};
use super::lower::{
    choice, lit, lit_bytes, never, r, seq, Lowerer,
    JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE,
    JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE, JSON_ADDITIONAL_KEY_COLON_SHARED_RULE,
    JSON_STRING_CHAR_RULE, JSON_STRING_RULE, MAX_SHARED_ADDITIONAL_EXCLUSION_KEYS,
};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_string(&mut self, schema: &StringSchema) -> ImportResult<GrammarExpr> {
        let should_terminalize_length = schema.max_length.is_none_or(|max_length| {
            !self.should_split_bounded_string(schema.min_length, max_length)
        });
        let has_recognized_format = schema.pattern.is_none()
            && recognized_string_format_body_regex(schema.format.as_deref()).is_some();
        if schema.pattern.is_some()
            || has_recognized_format
            || ((schema.min_length != 0 || schema.max_length.is_some()) && should_terminalize_length)
        {
            let expr = self.lower_constrained_string_terminal_expr(schema)?;
            let name = self.fresh_rule_name("json_string_constrained");
            self.add_terminal_rule(&name, expr);
            return Ok(r(&name));
        }
        self.lower_string_expr(schema)
    }

    pub(crate) fn lower_inline_bounded_array_string_item_expr(
        &mut self,
        schema: &super::ast::Schema,
    ) -> ImportResult<Option<GrammarExpr>> {
        let super::ast::SchemaKind::Assertions(assertions) = &schema.kind else {
            return Ok(None);
        };
        if assertions.enum_values.is_some()
            || assertions.const_value.is_some()
            || assertions.object.is_some()
            || assertions.array.is_some()
            || !assertions.any_of.is_empty()
            || !assertions.one_of.is_empty()
            || !assertions.all_of.is_empty()
        {
            return Ok(None);
        }
        let Some(string) = &assertions.string else {
            return Ok(None);
        };

        if let Some(pattern) = &string.pattern {
            if string.min_length != 0 || string.max_length.is_some() || string.format.is_some() {
                return Ok(None);
            }
            return Ok(Some(GrammarExpr::RawRegex(quoted_string_body_regex(
                &string_pattern_as_body_regex(pattern)?,
            ))));
        }

        if let Some(format_body_regex) = recognized_string_format_body_regex(string.format.as_deref()) {
            if string.min_length != 0 || string.max_length.is_some() {
                return Ok(None);
            }
            return Ok(Some(GrammarExpr::RawRegex(quoted_string_body_regex(
                format_body_regex,
            ))));
        }

        if string.max_length.is_some_and(|max| self.should_split_bounded_string(string.min_length, max)) {
            return Ok(None);
        }

        Ok(Some(self.lower_string_expr(string)?))
    }

    fn lower_constrained_string_terminal_expr(
        &mut self,
        schema: &StringSchema,
    ) -> ImportResult<GrammarExpr> {
        if let Some(pattern) = &schema.pattern {
            // NOTE: Pattern strength intentionally does NOT preserve min/max
            // length bounds here. Preserving those bounds alongside the
            // terminalized pattern can cause massive timeouts; this invariant is
            // intentional and this comment should NEVER EVER be removed.
            return Ok(GrammarExpr::RawRegex(quoted_string_body_regex(
                &string_pattern_as_body_regex(pattern)?,
            )));
        }

        let preserve_length_bounds = recognized_string_format_body_regex(schema.format.as_deref()).is_none();
        let body = if preserve_length_bounds {
            self.string_body_for_length(schema.min_length, schema.max_length)
        } else {
            // NOTE: Recognized format lowering intentionally does NOT preserve
            // sibling min/max length bounds inside the terminalized envelope.
            // Keeping a large bounded JSON string body alongside the format
            // intersection recreates the same timeout class we already avoid
            // for patterned strings. This invariant is intentional and this
            // comment should NEVER EVER be removed.
            self.string_body_for_length(0, None)
        };
        let mut expr = seq(vec![lit("\""), body, lit("\"")]);
        let mut constraints = Vec::new();

        if let Some(format_body_regex) = recognized_string_format_body_regex(schema.format.as_deref()) {
            constraints.push(quoted_string_body_regex(format_body_regex));
        }

        for (index, constraint) in constraints.into_iter().enumerate() {
            if index > 0 {
                let name = self.fresh_rule_name("json_string_constrained_part");
                self.add_terminal_rule(&name, expr);
                expr = r(&name);
            }
            expr = GrammarExpr::Intersect {
                expr: Box::new(expr),
                intersect: Box::new(GrammarExpr::RawRegex(constraint)),
            };
        }

        Ok(expr)
    }

    fn lower_string_expr(&mut self, schema: &StringSchema) -> ImportResult<GrammarExpr> {
        if schema.max_length.is_some_and(|max| max < schema.min_length) {
            return Ok(never());
        }

        if schema.min_length == 0
            && schema.max_length.is_none()
            && schema.pattern.is_none()
        {
            return Ok(r(JSON_STRING_RULE));
        }

        if schema.pattern.is_none()
            && let Some(max_length) = schema.max_length
            && self.should_split_bounded_string(schema.min_length, max_length)
        {
            return Ok(self.lower_split_bounded_string(schema.min_length, max_length));
        }

        let body = self.string_body_for_length(schema.min_length, schema.max_length);

        Ok(seq(vec![lit("\""), body, lit("\"")]))
    }

    fn should_split_bounded_string(&self, min: usize, max: usize) -> bool {
        if max <= self.config.terminalize_bounded_string_max.max(64) {
            return false;
        }
        let chunk = self.config.repeat_chunk_size.max(1);
        min > chunk || max > chunk
    }

    fn string_char_exact_ref(&mut self, count: usize) -> GrammarExpr {
        match count {
            0 => GrammarExpr::Epsilon,
            1 => r(JSON_STRING_CHAR_RULE),
            _ => {
                if let Some(rule_name) = self.shared_string_exact_rules.get(&count) {
                    return r(rule_name);
                }
                let rule_name = self.fresh_rule_name(&format!("json_string_char_exact_{count}"));
                self.add_terminal_rule(
                    &rule_name,
                    GrammarExpr::RepeatRange {
                        expr: Box::new(r(JSON_STRING_CHAR_RULE)),
                        min: count,
                        max: count,
                    },
                );
                self.shared_string_exact_rules.insert(count, rule_name.clone());
                r(&rule_name)
            }
        }
    }

    fn string_char_upto_ref(&mut self, max: usize) -> GrammarExpr {
        match max {
            0 => GrammarExpr::Epsilon,
            1 => GrammarExpr::Optional(Box::new(r(JSON_STRING_CHAR_RULE))),
            _ => {
                if let Some(rule_name) = self.shared_string_upto_rules.get(&max) {
                    return r(rule_name);
                }
                let rule_name = self.fresh_rule_name(&format!("json_string_char_upto_{max}"));
                self.add_terminal_rule(
                    &rule_name,
                    GrammarExpr::RepeatRange {
                        expr: Box::new(r(JSON_STRING_CHAR_RULE)),
                        min: 0,
                        max,
                    },
                );
                self.shared_string_upto_rules.insert(max, rule_name.clone());
                r(&rule_name)
            }
        }
    }

    fn string_char_upto_close_ref(&mut self, max: usize) -> GrammarExpr {
        if max == 0 {
            return lit("\"");
        }
        if let Some(rule_name) = self.shared_string_upto_close_rules.get(&max) {
            return r(rule_name);
        }
        let rule_name = self.fresh_rule_name(&format!("json_string_char_upto_close_{max}"));
        let upto = self.string_char_upto_ref(max);
        self.add_terminal_rule(&rule_name, seq(vec![upto, lit("\"")]));
        self.shared_string_upto_close_rules.insert(max, rule_name.clone());
        r(&rule_name)
    }

    fn string_char_exact_open_ref(&mut self, count: usize) -> GrammarExpr {
        if count == 0 {
            return lit("\"");
        }
        let rule_name = self.fresh_rule_name(&format!("json_string_char_exact_open_{count}"));
        let exact = self.string_char_exact_ref(count);
        self.add_terminal_rule(&rule_name, seq(vec![lit("\""), exact]));
        r(&rule_name)
    }

    fn string_char_upto_wrapped_ref(&mut self, max: usize) -> GrammarExpr {
        let rule_name = self.fresh_rule_name(&format!("json_string_char_upto_wrapped_{max}"));
        let upto = self.string_char_upto_ref(max);
        self.add_terminal_rule(&rule_name, seq(vec![lit("\""), upto, lit("\"")]));
        r(&rule_name)
    }

    fn split_string_exact_expr(&mut self, count: usize) -> GrammarExpr {
        let chunk = self.config.repeat_chunk_size.max(1);
        if count <= chunk {
            return self.string_char_exact_ref(count);
        }

        let full_chunks = count / chunk;
        let remainder = count % chunk;
        let mut parts = vec![GrammarExpr::RepeatRange {
            expr: Box::new(self.string_char_exact_ref(chunk)),
            min: full_chunks,
            max: full_chunks,
        }];
        if remainder > 0 {
            parts.push(self.string_char_exact_ref(remainder));
        }
        seq(parts)
    }

    fn split_string_upto_close_expr(&mut self, max: usize) -> GrammarExpr {
        let chunk = self.config.repeat_chunk_size.max(1);
        if max <= chunk {
            return self.string_char_upto_close_ref(max);
        }

        let full_chunks = max / chunk;
        let remainder = max % chunk;
        let exact_chunk = self.string_char_exact_ref(chunk);
        let mut alternatives = vec![seq(vec![
            GrammarExpr::RepeatRange {
                expr: Box::new(exact_chunk.clone()),
                min: 0,
                max: full_chunks.saturating_sub(1),
            },
            self.string_char_upto_close_ref(chunk - 1),
        ])];
        alternatives.push(seq(vec![
            GrammarExpr::RepeatRange {
                expr: Box::new(exact_chunk),
                min: full_chunks,
                max: full_chunks,
            },
            self.string_char_upto_close_ref(remainder),
        ]));
        choice(alternatives)
    }

    fn lower_split_bounded_string(&mut self, min: usize, max: usize) -> GrammarExpr {
        let chunk = self.config.repeat_chunk_size.max(1);
        let should_wrap = max >= 10_000 || (max / chunk) > 2;
        let expr = if should_wrap && min == 0 && max > chunk {
            let full_chunks = max / chunk;
            let remainder = max % chunk;
            let exact_chunk = self.string_char_exact_ref(chunk);
            let exact_open_chunk = self.string_char_exact_open_ref(chunk);
            let mut alternatives = vec![self.string_char_upto_wrapped_ref(chunk)];
            alternatives.push(seq(vec![
                exact_open_chunk.clone(),
                GrammarExpr::RepeatRange {
                    expr: Box::new(exact_chunk.clone()),
                    min: 0,
                    max: full_chunks.saturating_sub(2),
                },
                self.string_char_upto_close_ref(chunk),
            ]));
            if remainder > 0 {
                alternatives.push(seq(vec![
                    exact_open_chunk,
                    GrammarExpr::RepeatRange {
                        expr: Box::new(exact_chunk),
                        min: full_chunks.saturating_sub(1),
                        max: full_chunks.saturating_sub(1),
                    },
                    self.string_char_upto_close_ref(remainder),
                ]));
            }
            choice(alternatives)
        } else if min == max {
            seq(vec![lit("\""), self.split_string_exact_expr(min), lit("\"")])
        } else {
            seq(vec![
                lit("\""),
                self.split_string_exact_expr(min),
                self.split_string_upto_close_expr(max - min),
            ])
        };

        if !should_wrap {
            return expr;
        }

        let rule_name = self.fresh_rule_name("json_string_bounded_split");
        self.add_nonterminal_rule(&rule_name, expr);
        r(&rule_name)
    }

    pub(crate) fn lower_string_literal(&mut self, text: &str) -> GrammarExpr {
        let encoded = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string());
        let body_and_close = encoded
            .strip_prefix('"')
            .expect("serde_json string encoding starts with a quote");
        seq(vec![lit("\""), lit_bytes(body_and_close.as_bytes().to_vec())])
    }

    pub(crate) fn lower_literal_key_colon(&mut self, key: &str) -> GrammarExpr {
        self.lower_literal_key_colon_with_prefix(b"", key)
    }

    pub(crate) fn lower_literal_key_colon_with_prefix(
        &mut self,
        prefix: &[u8],
        key: &str,
    ) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        let mut bytes = Vec::with_capacity(prefix.len() + encoded.len() + 2);
        bytes.extend_from_slice(prefix);
        bytes.extend_from_slice(encoded.as_bytes());
        bytes.extend_from_slice(b": ");
        lit_bytes(bytes)
    }

    fn lower_pattern_key_colon_expr(&mut self, pattern: &str) -> ImportResult<GrammarExpr> {
        Ok(GrammarExpr::RawRegex(pattern_key_colon_regex(pattern)?))
    }

    fn pattern_key_colon_full_language(&mut self, pattern: &str) -> ImportResult<GrammarExpr> {
        let mut alternatives = vec![self.lower_pattern_key_colon_terminal(pattern)?];
        if let Some(overlap_literals) = self.shared_pattern_overlap_literal_rule(pattern)? {
            alternatives.push(overlap_literals);
        }
        Ok(choice(alternatives))
    }

    pub(crate) fn lower_pattern_key_colon_terminal(
        &mut self,
        pattern: &str,
    ) -> ImportResult<GrammarExpr> {
        if let Some(rule_name) = self.shared_ap_pattern_rules.get(pattern) {
            return Ok(r(rule_name));
        }
        let global_overlaps = self.pattern_overlapping_literal_keys(pattern)?;
        let expr = if global_overlaps.is_empty() {
            self.lower_pattern_key_colon_expr(pattern)?
        } else {
            let key_colon = GrammarExpr::RawRegex(pattern_key_colon_regex(pattern)?);
            let excluded = global_overlaps
                .iter()
                .map(|key| self.lower_literal_key_colon(key))
                .collect::<Vec<_>>();
            GrammarExpr::Exclude {
                expr: Box::new(key_colon),
                exclude: Box::new(choice(excluded)),
            }
        };
        let name = self.fresh_rule_name("json_pattern_key_colon");
        self.add_terminal_rule(&name, expr);
        self.shared_ap_pattern_rules.insert(pattern.to_string(), name.clone());
        Ok(r(&name))
    }

    fn pattern_overlapping_literal_keys(&mut self, pattern: &str) -> ImportResult<Vec<String>> {
        if let Some(overlaps) = self.shared_pattern_overlap_keys.get(pattern) {
            return Ok(overlaps.clone());
        }

        let mut overlaps = Vec::new();
        for key in &self.shared_ap_literal_keys {
            if property_name_matches_pattern(pattern, key)? {
                overlaps.push(key.clone());
            }
        }
        self.shared_pattern_overlap_keys
            .insert(pattern.to_string(), overlaps.clone());
        Ok(overlaps)
    }

    fn pattern_local_overlapping_literal_keys(
        &self,
        pattern: &str,
        fixed_keys: &BTreeSet<String>,
    ) -> ImportResult<Vec<String>> {
        let mut overlaps = Vec::new();
        for key in fixed_keys {
            if property_name_matches_pattern(pattern, key)? {
                overlaps.push(key.clone());
            }
        }
        Ok(overlaps)
    }

    fn shared_pattern_overlap_literal_rule(&mut self, pattern: &str) -> ImportResult<Option<GrammarExpr>> {
        let overlap_keys = self.pattern_overlapping_literal_keys(pattern)?;
        if overlap_keys.is_empty() {
            return Ok(None);
        }
        if let Some(rule_name) = self.shared_pattern_overlap_literal_rules.get(pattern) {
            return Ok(Some(r(rule_name)));
        }

        let name = self.fresh_rule_name("json_pattern_key_colon_overlap_literals");
        let expr = choice(
            overlap_keys
                .iter()
                .map(|key| self.lower_literal_key_colon(key))
                .collect::<Vec<_>>(),
        );
        self.add_terminal_rule(&name, expr);
        self.shared_pattern_overlap_literal_rules
            .insert(pattern.to_string(), name.clone());
        Ok(Some(r(&name)))
    }

    pub(crate) fn lower_pattern_key_colon_appearance(
        &mut self,
        pattern: &str,
        fixed_keys: &BTreeSet<String>,
    ) -> ImportResult<GrammarExpr> {
        let global_overlaps = self.pattern_overlapping_literal_keys(pattern)?;
        let local_overlaps = self.pattern_local_overlapping_literal_keys(pattern, fixed_keys)?;
        let cache_key = (pattern.to_string(), local_overlaps.clone());
        if let Some(rule_name) = self.shared_pattern_appearance_rules.get(&cache_key) {
            return Ok(r(rule_name));
        }

        let mut alternatives = vec![self.lower_pattern_key_colon_terminal(pattern)?];
        if let Some(overlap_container) = self.shared_pattern_overlap_literal_rule(pattern)? {
            if local_overlaps.is_empty() {
                alternatives.push(overlap_container);
            } else {
                let local_overlap_set = local_overlaps.iter().cloned().collect::<BTreeSet<_>>();
                let remaining = global_overlaps
                    .iter()
                    .filter(|key| !local_overlap_set.contains(*key))
                    .map(|key| self.lower_literal_key_colon(key))
                    .collect::<Vec<_>>();
                if !remaining.is_empty() {
                    alternatives.push(choice(remaining));
                }
            }
        }

        let name = self.fresh_rule_name("json_pattern_key_colon_appearance");
        self.add_nonterminal_rule(&name, choice(alternatives));
        self.shared_pattern_appearance_rules.insert(cache_key, name.clone());
        Ok(r(&name))
    }

    pub(crate) fn lower_additional_key_colon(
        &mut self,
        fixed_keys: &BTreeSet<String>,
        local_patterns: &[String],
    ) -> ImportResult<GrammarExpr> {
        if !self.use_shared_additional_key_colon() {
            return self.lower_additional_key_colon_expanded_addback(fixed_keys, local_patterns);
        }

        if self.shared_ap_patterns.is_empty() && local_patterns.is_empty() {
            return self.lower_additional_key_colon_literal_only(fixed_keys);
        }

        let mut alternatives = vec![self.shared_additional_key_colon_base()?];

        for key in self.shared_ap_literal_keys.clone() {
            if fixed_keys.contains(&key) {
                continue;
            }
            let mut covered_by_local_pattern = false;
            for pattern in local_patterns {
                if property_name_matches_pattern(pattern, &key)? {
                    covered_by_local_pattern = true;
                    break;
                }
            }
            if covered_by_local_pattern {
                continue;
            }
            alternatives.push(self.lower_literal_key_colon(&key));
        }

        for pattern in self.shared_ap_patterns.clone() {
            if local_patterns.iter().any(|local| local == &pattern) {
                continue;
            }
            alternatives.push(self.lower_pattern_key_colon_addback(&pattern, fixed_keys, local_patterns)?);
        }

        Ok(choice(alternatives))
    }

    fn use_shared_additional_key_colon(&self) -> bool {
        self.shared_ap_literal_keys.len() <= MAX_SHARED_ADDITIONAL_EXCLUSION_KEYS
    }

    fn lower_additional_key_colon_expanded_addback(
        &mut self,
        fixed_keys: &BTreeSet<String>,
        local_patterns: &[String],
    ) -> ImportResult<GrammarExpr> {
        let mut alternatives = vec![self.shared_additional_key_colon_base()?];

        for key in self.shared_ap_literal_keys.clone() {
            if fixed_keys.contains(&key) {
                continue;
            }
            let mut covered_by_local_pattern = false;
            for pattern in local_patterns {
                if property_name_matches_pattern(pattern, &key)? {
                    covered_by_local_pattern = true;
                    break;
                }
            }
            if covered_by_local_pattern {
                continue;
            }
            alternatives.push(self.lower_literal_key_colon(&key));
        }

        for pattern in self.shared_ap_patterns.clone() {
            if local_patterns.iter().any(|local| local == &pattern) {
                continue;
            }
            alternatives.push(self.lower_pattern_key_colon_addback(&pattern, fixed_keys, local_patterns)?);
        }

        Ok(choice(alternatives))
    }

    fn lower_additional_key_colon_literal_only(
        &mut self,
        fixed_keys: &BTreeSet<String>,
    ) -> ImportResult<GrammarExpr> {
        let base = self.shared_additional_key_colon_base()?;
        let excluded_addback = self.shared_additional_excluded_key_colon()?;
        let local_exclusions = fixed_keys
            .iter()
            .map(|key| self.lower_literal_key_colon(key))
            .collect::<Vec<_>>();

        let addback = if local_exclusions.is_empty() {
            excluded_addback
        } else {
            GrammarExpr::Exclude {
                expr: Box::new(excluded_addback),
                exclude: Box::new(choice(local_exclusions)),
            }
        };

        Ok(choice(vec![base, addback]))
    }

    fn shared_additional_excluded_key_colon(&mut self) -> ImportResult<GrammarExpr> {
        if let Some(rule_name) = &self.shared_ap_excluded_rule {
            return Ok(r(rule_name));
        }

        let mut excluded = Vec::new();
        for key in self.shared_ap_literal_keys.clone() {
            excluded.push(self.lower_literal_key_colon(&key));
        }
        for pattern in self.shared_ap_patterns.clone() {
            excluded.push(self.pattern_key_colon_full_language(&pattern)?);
        }

        let expr = choice(excluded);
        self.add_nonterminal_rule(JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE, expr.clone());
        self.add_internal_terminal_rule(JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE, expr);
        self.shared_ap_excluded_rule =
            Some(JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE.to_string());
        Ok(r(JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE))
    }

    fn shared_additional_key_colon_base(&mut self) -> ImportResult<GrammarExpr> {
        if let Some(rule_name) = &self.shared_ap_base_rule {
            return Ok(r(rule_name));
        }

        if self.shared_ap_patterns.is_empty() {
            self.shared_additional_excluded_key_colon()?;
            self.add_terminal_rule(
                JSON_ADDITIONAL_KEY_COLON_SHARED_RULE,
                GrammarExpr::Exclude {
                    expr: Box::new(seq(vec![r(JSON_STRING_RULE), self.key_separator_expr()])),
                    exclude: Box::new(r(JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE)),
                },
            );
            self.shared_ap_base_rule = Some(JSON_ADDITIONAL_KEY_COLON_SHARED_RULE.to_string());
            return Ok(r(JSON_ADDITIONAL_KEY_COLON_SHARED_RULE));
        }

        let mut excluded = self
            .shared_ap_literal_keys
            .clone()
            .into_iter()
            .map(|key| self.lower_literal_key_colon(&key))
            .collect::<Vec<_>>();
        for pattern in self.shared_ap_patterns.clone() {
            excluded.push(GrammarExpr::RawRegex(pattern_key_colon_regex(&pattern)?));
        }

        self.add_terminal_rule(
            JSON_ADDITIONAL_KEY_COLON_SHARED_RULE,
            GrammarExpr::Exclude {
                expr: Box::new(seq(vec![r(JSON_STRING_RULE), self.key_separator_expr()])),
                exclude: Box::new(choice(excluded)),
            },
        );
        self.shared_ap_base_rule = Some(JSON_ADDITIONAL_KEY_COLON_SHARED_RULE.to_string());
        Ok(r(JSON_ADDITIONAL_KEY_COLON_SHARED_RULE))
    }

    fn lower_pattern_key_colon_addback(
        &mut self,
        pattern: &str,
        fixed_keys: &BTreeSet<String>,
        local_patterns: &[String],
    ) -> ImportResult<GrammarExpr> {
        let mut pattern_expr = self.lower_pattern_key_colon_terminal(pattern)?;
        for local_pattern in local_patterns {
            pattern_expr = GrammarExpr::Exclude {
                expr: Box::new(pattern_expr),
                exclude: Box::new(self.pattern_key_colon_full_language(local_pattern)?),
            };
        }

        let mut alternatives = vec![pattern_expr];
        let overlap_keys = self.pattern_overlapping_literal_keys(pattern)?;
        let mut addback_literals = Vec::new();
        'keys: for key in &overlap_keys {
            if fixed_keys.contains(key) {
                continue;
            }
            for local_pattern in local_patterns {
                if property_name_matches_pattern(local_pattern, key)? {
                    continue 'keys;
                }
            }
            addback_literals.push(self.lower_literal_key_colon(key));
        }
        if !addback_literals.is_empty() {
            alternatives.push(choice(addback_literals));
        }

        if alternatives.len() == 1 {
            return Ok(alternatives.remove(0));
        }

        let name = self.fresh_rule_name("json_pattern_key_colon_addback");
        self.add_nonterminal_rule(&name, choice(alternatives));
        Ok(r(&name))
    }

    fn string_body_for_length(&self, min: usize, max: Option<usize>) -> GrammarExpr {
        let ch = r(JSON_STRING_CHAR_RULE);
        match (min, max) {
            (0, None) => GrammarExpr::Repeat(Box::new(ch)),
            (1, None) => GrammarExpr::RepeatOne(Box::new(ch)),
            (min, None) => seq(vec![
                self.repeat_exact_string_char(min),
                GrammarExpr::Repeat(Box::new(r(JSON_STRING_CHAR_RULE))),
            ]),
            (0, Some(0)) => GrammarExpr::Epsilon,
            (min, Some(max)) => GrammarExpr::RepeatRange { expr: Box::new(ch), min, max },
        }
    }

    fn repeat_exact_string_char(&self, count: usize) -> GrammarExpr {
        if count == 0 {
            return GrammarExpr::Epsilon;
        }
        let chunk = self.config.repeat_chunk_size.max(1);
        if count <= chunk {
            return GrammarExpr::RepeatRange {
                expr: Box::new(r(JSON_STRING_CHAR_RULE)),
                min: count,
                max: count,
            };
        }

        let mut parts = Vec::new();
        let mut remaining = count;
        while remaining > 0 {
            let take = remaining.min(chunk);
            parts.push(GrammarExpr::RepeatRange {
                expr: Box::new(r(JSON_STRING_CHAR_RULE)),
                min: take,
                max: take,
            });
            remaining -= take;
        }
        seq(parts)
    }
}

fn string_pattern_as_body_regex(pattern: &str) -> ImportResult<String> {
    let hir = Parser::new()
        .parse(pattern)
        .map_err(|error| SchemaImportError::new(format!("invalid string pattern {pattern:?}: {error}")))?;
    string_pattern_hir_as_body_regex(&hir)
}

fn string_pattern_hir_as_body_regex(hir: &Hir) -> ImportResult<String> {
    match hir.kind() {
        HirKind::Alternation(parts) => {
            let alternatives = parts
                .iter()
                .map(string_pattern_hir_as_body_regex)
                .collect::<ImportResult<Vec<_>>>()?;
            Ok(format!("(?:{})", alternatives.join("|")))
        }
        HirKind::Capture(capture) => {
            string_pattern_hir_as_body_regex(&capture.sub)
        }
        _ => string_pattern_branch_as_body_regex(hir.clone()),
    }
}

fn string_pattern_branch_as_body_regex(hir: Hir) -> ImportResult<String> {
    let (hir, anchored_start, anchored_end) = strip_outer_anchors(hir);
    let lowered = lower_decoded_regex_hir_to_json_body_regex(&hir)?;
    let string_char = json_string_body_char_regex();
    Ok(match (anchored_start, anchored_end) {
        (true, true) => lowered,
        (true, false) => format!("(?:{lowered})(?:{string_char})*"),
        (false, true) => format!("(?:{string_char})*(?:{lowered})"),
        (false, false) => format!("(?:{string_char})*(?:{lowered})(?:{string_char})*"),
    })
}

fn quoted_string_body_regex(body_regex: &str) -> String {
    format!(r#""(?:{body_regex})""#)
}

const DATE_FORMAT_BODY_REGEX: &str = r#"(?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)-02-29)"#;
const DATE_TIME_FORMAT_BODY_REGEX: &str = r#"(?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)-02-29)[Tt]([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\.[0-9]+)?([Zz]|[+-]([01][0-9]|2[0-3]):[0-5][0-9])"#;

fn recognized_string_format_body_regex(format: Option<&str>) -> Option<&'static str> {
    match format {
        Some("date-time") => Some(DATE_TIME_FORMAT_BODY_REGEX),
        Some("date") => Some(DATE_FORMAT_BODY_REGEX),
        Some("uuid") => Some(
            r#"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"#,
        ),
        Some("email") => Some(
            r#"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-][A-Za-z0-9!#$%&'*+/=?^_`{|}~-]*(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*|\[[A-Za-z0-9.\[\]:-]*)"#,
        ),
        Some("hostname") => Some(
            r#"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"#,
        ),
        Some("ipv4") => Some(
            r#"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?:\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}"#,
        ),
        Some("ipv6") => Some(
            r#"(?:[A-Fa-f0-9]{1,4}:){1,7}(?::[A-Fa-f0-9]{0,4}|[A-Fa-f0-9]{1,4})?|::[A-Fa-f0-9]{0,4}"#,
        ),
        Some("uri") => Some(
            r#"[A-Za-z][A-Za-z0-9+.-]*:(?://(?:[A-Za-z0-9._~!$&'()*+,;=:%-]*@)?(?:\[(?:[A-Fa-f0-9:.]*|[Vv][0-9A-Fa-f]+\.[A-Za-z0-9._~!$&'()*+,;=:-]+)\]|[A-Za-z0-9._~!$&'()*+,;=%-]*)(?::[0-9]*)?(?:[/?#](?:[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]|%[0-9A-Fa-f]{2})*)?|(?:[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]|%[0-9A-Fa-f]{2})*)"#,
        ),
        _ => None,
    }
}

fn pattern_key_colon_regex(pattern: &str) -> ImportResult<String> {
    let body = string_pattern_as_body_regex(pattern)?;
    Ok(format!(r#""{body}"(?:: )"#))
}

fn strip_outer_anchors(hir: Hir) -> (Hir, bool, bool) {
    let mut parts = match hir.kind() {
        HirKind::Concat(parts) => parts.clone(),
        HirKind::Look(look) if is_start_look(*look) => return (Hir::empty(), true, false),
        HirKind::Look(look) if is_end_look(*look) => return (Hir::empty(), false, true),
        _ => return (hir, false, false),
    };

    let anchored_start = parts
        .first()
        .is_some_and(|part| matches!(part.kind(), HirKind::Look(look) if is_start_look(*look)));
    if anchored_start {
        parts.remove(0);
    }
    let anchored_end = parts
        .last()
        .is_some_and(|part| matches!(part.kind(), HirKind::Look(look) if is_end_look(*look)));
    if anchored_end {
        parts.pop();
    }
    (Hir::concat(parts), anchored_start, anchored_end)
}

fn is_start_look(look: Look) -> bool {
    matches!(look, Look::Start | Look::StartLF | Look::StartCRLF)
}

fn is_end_look(look: Look) -> bool {
    matches!(look, Look::End | Look::EndLF | Look::EndCRLF)
}

fn lower_decoded_regex_hir_to_json_body_regex(hir: &Hir) -> ImportResult<String> {
    Ok(match hir.kind() {
        HirKind::Empty => String::new(),
        HirKind::Literal(Literal(bytes)) => {
            let literal = std::str::from_utf8(bytes).map_err(|error| {
                SchemaImportError::new(format!(
                    "string pattern literal is not valid UTF-8 after parsing: {error}"
                ))
            })?;
            literal
                .chars()
                .map(json_body_char_regex_for_decoded_char)
                .collect::<Vec<_>>()
                .join("")
        }
        HirKind::Class(class) => lower_decoded_class_to_json_body_regex(class),
        HirKind::Look(look) if is_start_look(*look) || is_end_look(*look) => String::new(),
        HirKind::Look(look) => {
            return Err(SchemaImportError::new(format!(
                "unsupported zero-width assertion in string pattern: {look:?}"
            )));
        }
        HirKind::Repetition(repetition) => lower_decoded_repetition_to_json_body_regex(repetition)?,
        HirKind::Capture(capture) => lower_decoded_regex_hir_to_json_body_regex(&capture.sub)?,
        HirKind::Concat(parts) => parts
            .iter()
            .map(lower_decoded_regex_hir_to_json_body_regex)
            .collect::<ImportResult<Vec<_>>>()?
            .join(""),
        HirKind::Alternation(parts) => {
            let alternatives = parts
                .iter()
                .map(lower_decoded_regex_hir_to_json_body_regex)
                .collect::<ImportResult<Vec<_>>>()?;
            format!("(?:{})", alternatives.join("|"))
        }
    })
}

fn lower_decoded_repetition_to_json_body_regex(repetition: &Repetition) -> ImportResult<String> {
    let sub = lower_decoded_regex_hir_to_json_body_regex(&repetition.sub)?;
    let atom = format!("(?:{sub})");
    Ok(match (repetition.min, repetition.max) {
        (0, None) => format!("{atom}*"),
        (1, None) => format!("{atom}+"),
        (0, Some(1)) => format!("{atom}?"),
        (min, Some(max)) if min == max => format!("{atom}{{{min}}}"),
        (min, Some(max)) => format!("{atom}{{{min},{max}}}"),
        (min, None) => format!("{atom}{{{min},}}"),
    })
}

fn lower_decoded_class_to_json_body_regex(class: &Class) -> String {
    if is_unicode_decimal_digit_class(class) {
        return "[0-9]".to_string();
    }
    if is_dot_like_unicode_class(class) {
        return json_string_body_dot_regex().to_string();
    }

    let mut raw_ranges = Vec::new();
    match class {
        Class::Unicode(class) => {
            for range in class.ranges() {
                if is_non_ascii_decimal_digit_range(range.start(), range.end()) {
                    continue;
                }
                push_safe_raw_char_ranges(range.start(), range.end(), &mut raw_ranges);
            }
        }
        Class::Bytes(class) => {
            for range in class.ranges() {
                let start = char::from(range.start());
                let end = char::from(range.end());
                push_safe_raw_char_ranges(start, end, &mut raw_ranges);
            }
        }
    }

    let mut alternatives = Vec::new();
    if !raw_ranges.is_empty() {
        alternatives.push(format!("[{}]", raw_ranges.join("")));
    }
    for ch in [
        '"', '\\', '\n', '\r', '\t', '\u{08}', '\u{0c}', '\u{85}', '\u{a0}', '\u{1680}',
        '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
        '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
        '\u{205f}', '\u{3000}',
    ] {
        if decoded_class_contains(class, ch) {
            alternatives.push(json_body_char_regex_for_decoded_char(ch));
        }
    }

    match alternatives.len() {
        0 => r"[^\s\S]".to_string(),
        1 => alternatives.remove(0),
        _ => format!("(?:{})", alternatives.join("|")),
    }
}

fn is_non_ascii_decimal_digit_range(start: char, end: char) -> bool {
    !start.is_ascii_digit()
        && start.is_numeric()
        && end.is_numeric()
}

fn is_unicode_decimal_digit_class(class: &Class) -> bool {
    let Class::Unicode(class) = class else {
        return false;
    };
    let ranges = class.ranges();
    if ranges.len() < 10 {
        return false;
    }
    let Some(first) = ranges.first() else {
        return false;
    };
    if first.start() != '0' || first.end() != '9' {
        return false;
    }
    ranges.iter().all(|range| {
        let start = u32::from(range.start());
        let end = u32::from(range.end());
        end >= start
            && end - start == 9
            && range.start().is_numeric()
            && range.end().is_numeric()
    })
}

fn is_dot_like_unicode_class(class: &Class) -> bool {
    let Class::Unicode(class) = class else {
        return false;
    };
    let ranges = class.ranges();
    ranges.len() == 2
        && ranges[0].start() == '\0'
        && ranges[0].end() == '\t'
        && ranges[1].start() == '\u{b}'
        && ranges[1].end() == '\u{10ffff}'
}

fn push_safe_raw_char_ranges(start: char, end: char, output: &mut Vec<String>) {
    let mut range_start = None;
    let mut previous = None;
    for codepoint in u32::from(start)..=u32::from(end) {
        let Some(ch) = char::from_u32(codepoint) else {
            continue;
        };
        if !is_safe_raw_json_string_char(ch) {
            if let (Some(first), Some(last)) = (range_start.take(), previous.take()) {
                output.push(regex_char_class_range(first, last));
            }
            continue;
        }
        if range_start.is_none() {
            range_start = Some(ch);
        }
        previous = Some(ch);
    }
    if let (Some(first), Some(last)) = (range_start, previous) {
        output.push(regex_char_class_range(first, last));
    }
}

fn decoded_class_contains(class: &Class, ch: char) -> bool {
    match class {
        Class::Unicode(class) => class
            .ranges()
            .iter()
            .any(|range| range.start() <= ch && ch <= range.end()),
        Class::Bytes(class) => ch.is_ascii()
            && class
                .ranges()
                .iter()
                .any(|range| range.start() <= ch as u8 && ch as u8 <= range.end()),
    }
}

fn regex_char_class_range(start: char, end: char) -> String {
    let start = escape_regex_class_char(start);
    let end = escape_regex_class_char(end);
    if start == end {
        start
    } else {
        format!("{start}-{end}")
    }
}

fn escape_regex_class_char(ch: char) -> String {
    match ch {
        '\\' => r"\\".to_string(),
        '-' => r"\-".to_string(),
        ']' => r"\]".to_string(),
        '^' => r"\^".to_string(),
        _ => regex::escape(&ch.to_string()),
    }
}

fn json_body_char_regex_for_decoded_char(ch: char) -> String {
    let decoded = ch.to_string();
    let encoded = serde_json::to_string(&decoded).unwrap_or_else(|_| "\"\"".to_string());
    let body = encoded
        .strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
        .unwrap_or("");
    regex::escape(body)
}

fn json_string_body_char_regex() -> &'static str {
    r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfnrt])"#
}

fn json_string_body_dot_regex() -> &'static str {
    r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bft])"#
}

fn is_safe_raw_json_string_char(ch: char) -> bool {
    ch.is_ascii() && !matches!(ch, '"' | '\\' | '\u{00}'..='\u{1f}' | '\u{7f}')
}

pub(crate) fn property_name_matches_pattern(pattern: &str, property_name: &str) -> ImportResult<bool> {
    let encoded = serde_json::to_string(property_name).unwrap_or_else(|_| "\"\"".to_string());
    let body = encoded.strip_prefix('"').and_then(|text| text.strip_suffix('"')).unwrap_or("");
    let body_regex = string_pattern_as_body_regex(pattern)?;
    let regex = Regex::new(&format!(r#"^(?:{})$"#, body_regex))
        .map_err(|error| SchemaImportError::new(format!("invalid patternProperties regex {pattern:?}: {error}")))?;
    Ok(regex.is_match(body))
}

pub(crate) fn string_value_satisfies_schema(
    value: &serde_json::Value,
    schema: &StringSchema,
) -> ImportResult<bool> {
    let Some(text) = value.as_str() else {
        return Ok(true);
    };
    if let Some(pattern) = &schema.pattern {
        return Regex::new(pattern)
            .map(|regex| regex.is_match(text))
            .map_err(|error| SchemaImportError::new(format!("invalid string pattern {pattern:?}: {error}")));
    }
    let length = text.chars().count();
    if length < schema.min_length || schema.max_length.is_some_and(|max| length > max) {
        return Ok(false);
    }
    if let Some(format_body_regex) = recognized_string_format_body_regex(schema.format.as_deref()) {
        let regex = Regex::new(&format!(r#"^(?:{format_body_regex})$"#)).map_err(|error| {
            SchemaImportError::new(format!(
                "invalid recognized string format regex {:?}: {error}",
                schema.format
            ))
        })?;
        return Ok(regex.is_match(text));
    }
    Ok(true)
}
