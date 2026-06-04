use std::collections::BTreeSet;

use regex::{Regex, escape as regex_escape};
use regex_syntax::hir::{Class, Hir, HirKind, Literal, Look, Repetition};
use regex_syntax::Parser;

use crate::import::ast::GrammarExpr;

use super::ast::StringSchema;
use super::error::{ImportResult, SchemaImportError};
use super::lower::{
    choice, lit, lit_bytes, never, r, seq, Lowerer,
    JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE,
    JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE, JSON_ADDITIONAL_KEY_COLON_SHARED_RULE,
    JSON_SEPARATOR_WS_REGEX, JSON_STRING_CHAR_RULE, JSON_STRING_RULE,
    MAX_SHARED_ADDITIONAL_EXCLUSION_KEYS,
};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_string(&mut self, schema: &StringSchema) -> ImportResult<GrammarExpr> {
        let should_terminalize_length = schema.max_length.is_none_or(|max_length| {
            !self.should_split_bounded_string(schema.min_length, max_length)
        });
        let has_recognized_format = schema.pattern.is_none()
            && recognized_string_format_body_regex_for_lowering(schema.format.as_deref()).is_some();
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
            || assertions.types.as_ref().is_some_and(|types| {
                !types
                    .iter()
                    .all(|schema_type| *schema_type == super::ast::SchemaType::String)
            })
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

        if let Some(format_body_regex) = recognized_string_format_body_regex_for_lowering(string.format.as_deref()) {
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
        if schema.pattern.is_none()
            && let Some(format_body_regex) =
                recognized_string_format_body_regex_for_lowering(schema.format.as_deref())
        {
            // Recognized JSON Schema formats are already emitted as JSON-string
            // body regexes. Do not build a generic JSON string terminal and then
            // intersect it with the format terminal: for trivial timestamp
            // schemas that intersection dominates compile time. The recognized
            // format bodies below contain only raw JSON-safe ASCII, so the
            // quoted terminal is already a valid JSON string spelling.
            //
            // As with the previous format-lowering policy, sibling min/max
            // bounds are intentionally not preserved here. Keeping them creates
            // the same timeout class as patterned-string length intersections.
            return Ok(GrammarExpr::RawRegex(quoted_string_body_regex(
                format_body_regex,
            )));
        }

        let mut expr = if let Some(pattern) = &schema.pattern {
            // NOTE: Pattern strength intentionally does NOT preserve sibling
            // minLength/maxLength inside terminalized string lowering.
            // Preserving those bounds with terminalized patterns causes severe
            // build-time blowups and timeouts. This is a deliberate importer
            // policy and this comment must NEVER EVER be removed under any
            // circumstances.
            GrammarExpr::RawRegex(quoted_string_body_regex(
                &string_pattern_as_body_regex(pattern)?,
            ))
        } else {
            seq(vec![
                lit("\""),
                self.string_body_for_length(schema.min_length, schema.max_length),
                lit("\""),
            ])
        };
        let mut constraints = Vec::new();

        if let Some(format_body_regex) = recognized_string_format_body_regex_for_lowering(schema.format.as_deref()) {
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
        let expr = self.split_bounded_string_terminal_expr(min, max);
        let chunk = self.config.repeat_chunk_size.max(1);
        let should_wrap = max >= 10_000 || (max / chunk) > 2;

        if !should_wrap {
            return expr;
        }

        let rule_name = self.fresh_rule_name("json_string_bounded_split");
        self.add_nonterminal_rule(&rule_name, expr);
        r(&rule_name)
    }

    fn split_bounded_string_terminal_expr(&mut self, min: usize, max: usize) -> GrammarExpr {
        let chunk = self.config.repeat_chunk_size.max(1);
        let should_wrap = max >= 10_000 || (max / chunk) > 2;
        if should_wrap && min == 0 && max > chunk {
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
        }
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
        if prefix == b", " {
            return GrammarExpr::RawRegex(format!(
                r#",{JSON_SEPARATOR_WS_REGEX}{}:{JSON_SEPARATOR_WS_REGEX}"#,
                regex_escape(&encoded)
            ));
        }

        let key = lit_bytes(encoded.as_bytes().to_vec());
        let key_colon = seq(vec![key, self.key_separator_expr()]);
        if prefix.is_empty() {
            key_colon
        } else {
            seq(vec![lit_bytes(prefix.to_vec()), key_colon])
        }
    }

    pub(crate) fn lower_literal_key_colon_with_prefix_and_suffix(
        &mut self,
        prefix: &[u8],
        key: &str,
        suffix: u8,
    ) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        let suffix_byte = suffix;
        let suffix = regex_escape(&String::from_utf8_lossy(&[suffix_byte]));
        if prefix == b", " {
            return GrammarExpr::RawRegex(format!(
                r#",{JSON_SEPARATOR_WS_REGEX}{}:{JSON_SEPARATOR_WS_REGEX}{}"#,
                regex_escape(&encoded),
                suffix
            ));
        }
        if prefix.is_empty() {
            return GrammarExpr::RawRegex(format!(
                r#"{}:{JSON_SEPARATOR_WS_REGEX}{}"#,
                regex_escape(&encoded),
                suffix
            ));
        }

        seq(vec![
            lit_bytes(prefix.to_vec()),
            self.lower_literal_key_colon_with_prefix_and_suffix(b"", key, suffix_byte),
        ])
    }

    pub(crate) fn lower_literal_key_colon_with_prefix_and_json_string(
        &mut self,
        prefix: &[u8],
        key: &str,
    ) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        let string_body = self.json_string_char_regex();
        if prefix == b", " {
            return GrammarExpr::RawRegex(format!(
                r#",{JSON_SEPARATOR_WS_REGEX}{}:{JSON_SEPARATOR_WS_REGEX}"(?:{})*""#,
                regex_escape(&encoded),
                string_body
            ));
        }
        if prefix.is_empty() {
            return GrammarExpr::RawRegex(format!(
                r#"{}:{JSON_SEPARATOR_WS_REGEX}"(?:{})*""#,
                regex_escape(&encoded),
                string_body
            ));
        }

        seq(vec![
            lit_bytes(prefix.to_vec()),
            self.lower_literal_key_colon_with_prefix_and_json_string(b"", key),
        ])
    }

    pub(crate) fn lower_literal_key_colon_with_prefix_and_literal_value(
        &mut self,
        prefix: &[u8],
        key: &str,
        value: &[u8],
    ) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        let value_regex = regex_escape(&String::from_utf8_lossy(value));
        if prefix == b", " {
            return GrammarExpr::RawRegex(format!(
                r#",{JSON_SEPARATOR_WS_REGEX}{}:{JSON_SEPARATOR_WS_REGEX}{}"#,
                regex_escape(&encoded),
                value_regex
            ));
        }
        if prefix.is_empty() {
            return GrammarExpr::RawRegex(format!(
                r#"{}:{JSON_SEPARATOR_WS_REGEX}{}"#,
                regex_escape(&encoded),
                value_regex
            ));
        }

        seq(vec![
            lit_bytes(prefix.to_vec()),
            self.lower_literal_key_colon_with_prefix_and_literal_value(b"", key, value),
        ])
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

    fn pattern_key_colon_shared_addback_alternatives(
        &mut self,
        pattern: &str,
    ) -> ImportResult<Vec<GrammarExpr>> {
        let global_overlaps = self.pattern_overlapping_literal_keys(pattern)?;
        let key_colon = GrammarExpr::RawRegex(pattern_key_colon_regex(pattern)?);
        let pattern_expr = if global_overlaps.is_empty() {
            key_colon
        } else {
            GrammarExpr::Exclude {
                expr: Box::new(key_colon),
                exclude: Box::new(choice(
                    global_overlaps
                        .iter()
                        .map(|key| self.lower_literal_key_colon(key))
                        .collect::<Vec<_>>(),
                )),
            }
        };

        Ok(vec![pattern_expr])
    }

    fn pattern_key_colon_local_exclusion_alternatives(
        &mut self,
        pattern: &str,
    ) -> ImportResult<Vec<GrammarExpr>> {
        let global_overlaps = self.pattern_overlapping_literal_keys(pattern)?;
        let mut alternatives = self.pattern_key_colon_shared_addback_alternatives(pattern)?;
        alternatives.extend(
            global_overlaps
                .iter()
                .map(|key| self.lower_literal_key_colon(key)),
        );
        Ok(alternatives)
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
            match property_name_matches_pattern(pattern, key) {
                Ok(true) => overlaps.push(key.clone()),
                Ok(false) => {}
                Err(error) if is_regex_compile_limit_error(&error) => {
                    // Broad build-parity fallback: if overlap checking would compile
                    // an oversized regex, skip the overlap optimization and keep lowering.
                    overlaps.clear();
                    break;
                }
                Err(error) => return Err(error),
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
            match property_name_matches_pattern(pattern, key) {
                Ok(true) => overlaps.push(key.clone()),
                Ok(false) => {}
                Err(error) if is_regex_compile_limit_error(&error) => {
                    // Same broad fallback as the shared overlap cache: unknown
                    // overlaps are treated as no optimization instead of a build error.
                    return Ok(Vec::new());
                }
                Err(error) => return Err(error),
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
        if super::share_additional_addback_choices_enabled() {
            return self.lower_additional_key_colon_shared(fixed_keys, local_patterns);
        }

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

    fn lower_additional_key_colon_shared(
        &mut self,
        fixed_keys: &BTreeSet<String>,
        local_patterns: &[String],
    ) -> ImportResult<GrammarExpr> {
        let cache_key = (
            fixed_keys.iter().cloned().collect::<Vec<_>>(),
            local_patterns.to_vec(),
        );
        if let Some(rule_name) = self.shared_additional_key_colon_local_rules.get(&cache_key) {
            return Ok(r(rule_name));
        }

        let base = self.shared_additional_key_colon_base()?;
        let excluded_addback = self.shared_additional_excluded_key_colon()?;
        let mut local_exclusions = fixed_keys
            .iter()
            .map(|key| self.lower_literal_key_colon(key))
            .collect::<Vec<_>>();
        for pattern in local_patterns {
            local_exclusions.extend(self.pattern_key_colon_local_exclusion_alternatives(pattern)?);
        }

        let addback = if local_exclusions.is_empty() {
            excluded_addback
        } else {
            GrammarExpr::Exclude {
                expr: Box::new(excluded_addback),
                exclude: Box::new(choice(local_exclusions)),
            }
        };

        let name = self.fresh_rule_name("json_additional_key_colon_local");
        self.add_nonterminal_rule(&name, choice(vec![base, addback]));
        self.shared_additional_key_colon_local_rules
            .insert(cache_key, name.clone());
        Ok(r(&name))
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

        if !super::share_additional_addback_choices_enabled() {
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
            return Ok(r(JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE));
        }

        let mut excluded = Vec::new();
        let mut terminal_excluded = Vec::new();
        for key in self.shared_ap_literal_keys.clone() {
            let literal = self.lower_literal_key_colon(&key);
            excluded.push(literal.clone());
            terminal_excluded.push(literal);
        }
        for pattern in self.shared_ap_patterns.clone() {
            excluded.extend(self.pattern_key_colon_shared_addback_alternatives(&pattern)?);
            terminal_excluded.push(GrammarExpr::RawRegex(pattern_key_colon_regex(&pattern)?));
        }

        let expr = choice(excluded);
        self.add_nonterminal_rule(JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE, expr.clone());
        self.add_internal_terminal_rule(
            JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE,
            choice(terminal_excluded),
        );
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
    let pattern = preprocess_ascii_shorthand(pattern);
    let hir = Parser::new()
        .parse(&pattern)
        .map_err(|error| SchemaImportError::new(format!("invalid string pattern {pattern:?}: {error}")))?;
    string_pattern_hir_as_body_regex(&hir)
}

fn preprocess_ascii_shorthand(pattern: &str) -> String {
    let mut lowered = String::with_capacity(pattern.len());
    let mut chars = pattern.chars().peekable();
    let mut in_class = false;

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('d') => {
                    if in_class {
                        lowered.push_str("0-9");
                    } else {
                        lowered.push_str("[0-9]");
                    }
                }
                Some('w') => {
                    if in_class {
                        lowered.push_str("A-Za-z0-9_");
                    } else {
                        lowered.push_str("[A-Za-z0-9_]");
                    }
                }
                Some(next) => {
                    lowered.push('\\');
                    lowered.push(next);
                }
                None => lowered.push('\\'),
            }
            continue;
        }

        match ch {
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            _ => {}
        }
        lowered.push(ch);
    }

    lowered
}

fn string_pattern_hir_as_body_regex(hir: &Hir) -> ImportResult<String> {
    match hir.kind() {
        HirKind::Alternation(parts) => {
            let alternatives = parts
                .iter()
                .cloned()
                .map(lower_string_pattern_branch_parts)
                .collect::<ImportResult<Vec<_>>>()?;
            if let Some((_, anchored_start, anchored_end)) = alternatives.first() {
                if alternatives
                    .iter()
                    .all(|(_, branch_start, branch_end)| branch_start == anchored_start && branch_end == anchored_end)
                {
                    let lowered = alternatives
                        .iter()
                        .map(|(lowered, _, _)| lowered.as_str())
                        .collect::<Vec<_>>();
                    let body = format!("(?:{})", lowered.join("|"));
                    return Ok(wrap_lowered_string_pattern_branch(
                        &body,
                        *anchored_start,
                        *anchored_end,
                    ));
                }
            }

            let wrapped = alternatives
                .iter()
                .map(|(lowered, anchored_start, anchored_end)| {
                    wrap_lowered_string_pattern_branch(lowered, *anchored_start, *anchored_end)
                })
                .collect::<Vec<_>>();
            Ok(format!("(?:{})", wrapped.join("|")))
        }
        HirKind::Capture(capture) => {
            string_pattern_hir_as_body_regex(&capture.sub)
        }
        _ => string_pattern_branch_as_body_regex(hir.clone()),
    }
}

fn string_pattern_branch_as_body_regex(hir: Hir) -> ImportResult<String> {
    let (lowered, anchored_start, anchored_end) = lower_string_pattern_branch_parts(hir)?;
    Ok(wrap_lowered_string_pattern_branch(
        &lowered,
        anchored_start,
        anchored_end,
    ))
}

fn lower_string_pattern_branch_parts(hir: Hir) -> ImportResult<(String, bool, bool)> {
    let (hir, anchored_start, anchored_end) = strip_outer_anchors(hir);
    let lowered = lower_decoded_regex_hir_to_json_body_regex(&hir)?;
    Ok((lowered, anchored_start, anchored_end))
}

fn wrap_lowered_string_pattern_branch(lowered: &str, anchored_start: bool, anchored_end: bool) -> String {
    let string_char = json_string_body_char_regex();
    match (anchored_start, anchored_end) {
        (true, true) => lowered.to_string(),
        (true, false) => format!("(?:{lowered})(?:{string_char})*"),
        (false, true) => format!("(?:{string_char})*(?:{lowered})"),
        (false, false) => format!("(?:{string_char})*(?:{lowered})(?:{string_char})*"),
    }
}

fn quoted_string_body_regex(body_regex: &str) -> String {
    format!(r#""(?:{body_regex})""#)
}

const DATE_FORMAT_BODY_REGEX: &str = r#"(?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)-02-29)"#;
const DATE_TIME_FORMAT_BODY_REGEX: &str = r#"(?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)-02-29)[Tt]([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\.[0-9]+)?([Zz]|[+-]([01][0-9]|2[0-3]):[0-5][0-9])"#;

// The strict calendar-valid date/date-time regexes above are useful for
// literal-value validation, but compiling them into the tokenizer is expensive
// enough that single-field timestamp schemas can spend seconds in build.  For
// constrained decoding we use a structurally valid RFC3339-ish envelope and
// leave exact month/day/leap-year filtering to string_value_satisfies_schema().
const DATE_FORMAT_LOWERING_BODY_REGEX: &str =
    r#"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"#;
const DATE_TIME_FORMAT_LOWERING_BODY_REGEX: &str = r#"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])[Tt](?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"#;

fn recognized_string_format_body_regex_for_lowering(format: Option<&str>) -> Option<&'static str> {
    match format {
        Some("date-time") => Some(DATE_TIME_FORMAT_LOWERING_BODY_REGEX),
        Some("date") => Some(DATE_FORMAT_LOWERING_BODY_REGEX),
        _ => recognized_string_format_body_regex(format),
    }
}

fn recognized_string_format_body_regex(format: Option<&str>) -> Option<&'static str> {
    match format {
        Some("date-time") => Some(DATE_TIME_FORMAT_BODY_REGEX),
        Some("date") => Some(DATE_FORMAT_BODY_REGEX),
        Some("uuid") => Some(
            r#"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"#,
        ),
        Some("email") => Some(
            r#"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-][A-Za-z0-9!#$%&'*+/=?^_`{|}~-]*(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?))*|\[(?:(?:[0-9]|[1-9][0-9]|25[0-5]|(?:2[0-4]|1[0-9])[0-9])\.){3}(?:[0-9]|[1-9][0-9]|25[0-5]|(?:2[0-4]|1[0-9])[0-9])\])"#,
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
    Ok(format!(r#""{body}":{JSON_SEPARATOR_WS_REGEX}"#))
}

fn strip_outer_captures(mut hir: Hir) -> Hir {
    loop {
        match hir.kind() {
            HirKind::Capture(capture) => hir = *capture.sub.clone(),
            _ => return hir,
        }
    }
}

fn strip_outer_start_anchor(hir: Hir) -> Option<Hir> {
    let hir = strip_outer_captures(hir);
    match hir.kind() {
        HirKind::Concat(parts)
            if parts
                .first()
                .is_some_and(|part| matches!(part.kind(), HirKind::Look(look) if is_start_look(*look))) =>
        {
            Some(Hir::concat(parts[1..].to_vec()))
        }
        HirKind::Look(look) if is_start_look(*look) => Some(Hir::empty()),
        _ => None,
    }
}

fn strip_outer_end_anchor(hir: Hir) -> Option<Hir> {
    let hir = strip_outer_captures(hir);
    match hir.kind() {
        HirKind::Concat(parts)
            if parts
                .last()
                .is_some_and(|part| matches!(part.kind(), HirKind::Look(look) if is_end_look(*look))) =>
        {
            Some(Hir::concat(parts[..parts.len() - 1].to_vec()))
        }
        HirKind::Look(look) if is_end_look(*look) => Some(Hir::empty()),
        _ => None,
    }
}

fn strip_outer_anchors(hir: Hir) -> (Hir, bool, bool) {
    let mut hir = strip_outer_captures(hir);
    let mut anchored_start = false;
    let mut anchored_end = false;

    if let Some(stripped) = strip_outer_start_anchor(hir.clone()) {
        hir = stripped;
        anchored_start = true;
    }
    if let Some(stripped) = strip_outer_end_anchor(hir.clone()) {
        hir = stripped;
        anchored_end = true;
    }

    hir = strip_outer_captures(hir);
    if let HirKind::Alternation(parts) = hir.kind() {
        let mut parts = parts.clone();

        if !anchored_start && parts.iter().all(|part| strip_outer_start_anchor(part.clone()).is_some()) {
            parts = parts
                .into_iter()
                .map(|part| strip_outer_start_anchor(part).expect("checked common start anchor"))
                .collect();
            anchored_start = true;
        }
        if !anchored_end && parts.iter().all(|part| strip_outer_end_anchor(part.clone()).is_some()) {
            parts = parts
                .into_iter()
                .map(|part| strip_outer_end_anchor(part).expect("checked common end anchor"))
                .collect();
            anchored_end = true;
        }

        hir = Hir::alternation(parts);
    }

    (hir, anchored_start, anchored_end)
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
        (min, Some(max)) if max <= 20 => {
            let alternatives = (min..=max)
                .map(|count| atom.repeat(count as usize))
                .collect::<Vec<_>>();
            format!("(?:{})", alternatives.join("|"))
        }
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
    let mut alternatives = Vec::new();
    match class {
        Class::Unicode(class) => {
            for range in class.ranges() {
                let start = range.start();
                let end = range.end();
                if start <= '\x7f' {
                    let ascii_end = std::cmp::min(end, '\x7f');
                    push_safe_raw_char_ranges(start, ascii_end, &mut raw_ranges);
                }
                if end >= '\u{80}' {
                    let non_ascii_start = std::cmp::max(start, '\u{80}');
                    alternatives.push(unicode_range_to_utf8_regex_string(non_ascii_start, end));
                }
            }
        }
        Class::Bytes(class) => {
            for range in class.ranges() {
                let start = range.start();
                let end = range.end();
                if start <= 127 {
                    let ascii_end = std::cmp::min(end, 127);
                    push_safe_raw_char_ranges(char::from(start), char::from(ascii_end), &mut raw_ranges);
                }
                if end >= 128 {
                    let non_ascii_start = std::cmp::max(start, 128);
                    if non_ascii_start == end {
                        alternatives.push(format!(r#"\x{:02x}"#, non_ascii_start));
                    } else {
                        alternatives.push(format!(r#"[\x{:02x}-\x{:02x}]"#, non_ascii_start, end));
                    }
                }
            }
        }
    }

    if !raw_ranges.is_empty() {
        alternatives.push(format!("[{}]", raw_ranges.join("")));
    }
    for ch in [
        '"', '/', '\\', '\n', '\r', '\t', '\u{08}', '\u{0c}', '\u{85}', '\u{a0}', '\u{1680}',
        '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
        '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
        '\u{205f}', '\u{3000}',
    ] {
        if decoded_class_contains(class, ch) {
            alternatives.push(json_body_char_regex_for_decoded_char(ch));
        }
    }
    if class_contains_general_non_ascii_non_whitespace(class) {
        alternatives.push(json_string_body_non_ascii_non_whitespace_regex().to_string());
    }

    match alternatives.len() {
        0 => r"[^\s\S]".to_string(),
        1 => alternatives.remove(0),
        _ => format!("(?:{})", alternatives.join("|")),
    }
}

fn utf8_sequence_to_regex_string(seq: &regex_syntax::utf8::Utf8Sequence) -> String {
    let mut parts = String::new();
    for range in seq.as_slice() {
        if range.start == range.end {
            parts.push_str(&format!(r#"\x{:02x}"#, range.start));
        } else {
            parts.push_str(&format!(r#"[\x{:02x}-\x{:02x}]"#, range.start, range.end));
        }
    }
    parts
}

fn unicode_range_to_utf8_regex_string(start: char, end: char) -> String {
    use regex_syntax::utf8::Utf8Sequences;
    let seqs: Vec<String> = Utf8Sequences::new(start, end)
        .map(|seq| utf8_sequence_to_regex_string(&seq))
        .collect();
    if seqs.len() == 1 {
        seqs.into_iter().next().unwrap()
    } else {
        format!("(?:{})", seqs.join("|"))
    }
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

fn class_contains_general_non_ascii_non_whitespace(class: &Class) -> bool {
    ['π', '中', '😀']
        .into_iter()
        .all(|ch| decoded_class_contains(class, ch))
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

pub(crate) const GLRMASK_LLGUIDANCE_COMPAT_ENV: &str = "GLRMASK_LLGUIDANCE_COMPAT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JsonStringCompatMode {
    JsonSchema,
    LlGuidanceNative,
}

fn llguidance_compat_enabled_from_env() -> bool {
    std::env::var_os(GLRMASK_LLGUIDANCE_COMPAT_ENV).is_some_and(|value| {
        let value = value.to_string_lossy();
        !value.is_empty() && value != "0"
    })
}

fn json_string_compat_mode() -> JsonStringCompatMode {
    if llguidance_compat_enabled_from_env() {
        JsonStringCompatMode::LlGuidanceNative
    } else {
        JsonStringCompatMode::JsonSchema
    }
}

fn json_body_char_regex_for_decoded_char(ch: char) -> String {
    json_body_char_regex_for_decoded_char_in_mode(ch, json_string_compat_mode())
}

fn json_body_char_regex_for_decoded_char_in_mode(ch: char, mode: JsonStringCompatMode) -> String {
    let decoded = ch.to_string();
    let encoded = serde_json::to_string(&decoded).unwrap_or_else(|_| "\"\"".to_string());
    let body = encoded
        .strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
        .unwrap_or("");
    let canonical = regex::escape(body);
    if ch == '/' && mode == JsonStringCompatMode::JsonSchema {
        format!(r#"(?:{}|\\/)"#, canonical)
    } else {
        canonical
    }
}

pub(crate) fn json_string_body_char_regex() -> &'static str {
    json_string_body_char_regex_in_mode(json_string_compat_mode())
}

fn json_string_body_char_regex_in_mode(mode: JsonStringCompatMode) -> &'static str {
    match mode {
        JsonStringCompatMode::JsonSchema => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["/\\bfnrt])"#
        }
        JsonStringCompatMode::LlGuidanceNative => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfnrt])"#
        }
    }
}

fn json_string_body_non_ascii_non_whitespace_regex() -> &'static str {
    r#"(?:\xC2[\x80-\x84\x86-\x9F\xA1-\xBF]|[\xC3-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|\xE1[\x80-\x99\x9B-\xBF][\x80-\xBF]|\xE1\x9A[\x81-\xBF]|\xE2\x80[\x8B-\xA7\xAA-\xAE\xB0-\xBF]|\xE2\x81[\x80-\x9E\xA0-\xBF]|\xE2[\x82-\xBF][\x80-\xBF]|\xE3\x80[\x81-\xBF]|\xE3[\x81-\xBF][\x80-\xBF]|[\xE4-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2})"#
}

fn json_string_body_dot_regex() -> &'static str {
    json_string_body_dot_regex_in_mode(json_string_compat_mode())
}

fn json_string_body_dot_regex_in_mode(mode: JsonStringCompatMode) -> &'static str {
    match mode {
        JsonStringCompatMode::JsonSchema => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["/\\bft])"#
        }
        JsonStringCompatMode::LlGuidanceNative => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bft])"#
        }
    }
}

fn is_safe_raw_json_string_char(ch: char) -> bool {
    ch.is_ascii() && !matches!(ch, '"' | '\\' | '\u{00}'..='\u{1f}' | '\u{7f}')
}

pub(crate) fn property_name_matches_pattern(pattern: &str, property_name: &str) -> ImportResult<bool> {
    let ascii_pattern = preprocess_ascii_shorthand(pattern);
    Regex::new(&ascii_pattern)
        .map(|regex| regex.is_match(property_name))
        .map_err(|error| SchemaImportError::new(format!("invalid patternProperties regex {pattern:?}: {error}")))
}

fn is_regex_compile_limit_error(error: &SchemaImportError) -> bool {
    error.message().contains("Compiled regex exceeds size limit")
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

#[cfg(test)]
mod tests {
    use regex::Regex;

    use super::{preprocess_ascii_shorthand, quoted_string_body_regex, string_pattern_as_body_regex};

    #[test]
    fn preprocess_ascii_shorthand_rewrites_generic_word_shorthand() {
        assert_eq!(preprocess_ascii_shorthand(r"^\w+$"), r"^[A-Za-z0-9_]+$");
        assert_eq!(preprocess_ascii_shorthand(r"^[\w.-]+$"), r"^[A-Za-z0-9_.-]+$");
    }

    #[test]
    fn preprocess_ascii_shorthand_preserves_escaped_word_shorthand() {
        assert_eq!(preprocess_ascii_shorthand(r"^\\w+$"), r"^\\w+$");
    }

    #[test]
    fn lowered_bounded_free_text_pattern_rejects_leading_space_slash() {
        let body = string_pattern_as_body_regex(r"^$|(^(?:\S+\s+){0,19}\S+$)").unwrap();
        let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

        assert!(regex.is_match(r#""REST API""#));
        assert!(!regex.is_match(r#"" /""#));
    }

    #[test]
    fn lowered_optional_decimal_pattern_rejects_backslash_digit_string() {
        let body = string_pattern_as_body_regex(r"^$|^\d{1,15}(?:\.\d{1,5})?$").unwrap();
        let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

        assert!(regex.is_match(r#""""#));
        assert!(regex.is_match(r#""123.45""#));
        assert!(!regex.is_match(r#""\\1""#));
    }
}
