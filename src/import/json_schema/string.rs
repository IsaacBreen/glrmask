use std::collections::BTreeSet;

use regex::Regex;

use crate::import::ast::GrammarExpr;

use super::ast::StringSchema;
use super::error::{ImportResult, SchemaImportError};
use super::formats::lookup_string_format;
use super::lower::{
    choice, lit, lit_bytes, never, r, seq, Lowerer,
    JSON_ADDITIONAL_KEY_COLON_SHARED_RULE, JSON_STRING_CHAR_RULE, JSON_STRING_RULE,
};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_string(&mut self, schema: &StringSchema) -> ImportResult<GrammarExpr> {
        if schema.max_length.is_some_and(|max| max < schema.min_length) {
            return Ok(never());
        }

        if schema.min_length == 0
            && schema.max_length.is_none()
            && schema.pattern.is_none()
            && schema.format.is_none()
        {
            return Ok(r(JSON_STRING_RULE));
        }

        if schema.pattern.is_none()
            && schema.format.is_none()
            && let Some(max_length) = schema.max_length
            && self.should_split_bounded_string(schema.min_length, max_length)
        {
            return Ok(self.lower_split_bounded_string(schema.min_length, max_length));
        }

        let mut body = self.string_body_for_length(schema.min_length, schema.max_length);
        if let Some(pattern) = &schema.pattern {
            body = GrammarExpr::Intersect {
                expr: Box::new(body),
                intersect: Box::new(GrammarExpr::RawRegex(string_pattern_as_body_regex(pattern))),
            };
        }
        if let Some(format) = &schema.format {
            if let Some(regex) = lookup_string_format(format) {
                body = GrammarExpr::Intersect {
                    expr: Box::new(body),
                    intersect: Box::new(GrammarExpr::RawRegex(regex.to_string())),
                };
            } else {
                return Err(SchemaImportError::new(format!("Unknown format: {format}")));
            }
        }

        Ok(seq(vec![lit("\""), body, lit("\"")]))
    }

    fn should_split_bounded_string(&self, min: usize, max: usize) -> bool {
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
            self.string_char_upto_close_ref(chunk),
        ])];
        if remainder > 0 {
            alternatives.push(seq(vec![
                GrammarExpr::RepeatRange {
                    expr: Box::new(exact_chunk),
                    min: full_chunks,
                    max: full_chunks,
                },
                self.string_char_upto_close_ref(remainder),
            ]));
        }
        choice(alternatives)
    }

    fn lower_split_bounded_string(&mut self, min: usize, max: usize) -> GrammarExpr {
        if min == max {
            return seq(vec![lit("\""), self.split_string_exact_expr(min), lit("\"")]);
        }
        seq(vec![
            lit("\""),
            self.split_string_exact_expr(min),
            self.split_string_upto_close_expr(max - min),
        ])
    }

    pub(crate) fn lower_string_literal(&mut self, text: &str) -> GrammarExpr {
        let encoded = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string());
        lit_bytes(encoded.into_bytes())
    }

    pub(crate) fn lower_literal_key_colon(&mut self, key: &str) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        seq(vec![lit_bytes(encoded.into_bytes()), self.key_separator_expr()])
    }

    pub(crate) fn lower_pattern_key_colon_terminal(
        &mut self,
        pattern: &str,
    ) -> ImportResult<GrammarExpr> {
        if let Some(rule_name) = self.shared_ap_pattern_rules.get(pattern) {
            return Ok(r(rule_name));
        }
        let key_colon = GrammarExpr::RawRegex(pattern_key_colon_regex(pattern));
        let global_overlaps = self.pattern_overlapping_literal_keys(pattern)?;
        let expr = if global_overlaps.is_empty() {
            key_colon
        } else {
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

    fn pattern_overlapping_literal_keys(&self, pattern: &str) -> ImportResult<Vec<String>> {
        let mut overlaps = Vec::new();
        for key in &self.shared_ap_literal_keys {
            if property_name_matches_pattern(pattern, key)? {
                overlaps.push(key.clone());
            }
        }
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
        self.add_nonterminal_rule(&name, expr);
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

    fn shared_additional_key_colon_base(&mut self) -> ImportResult<GrammarExpr> {
        if let Some(rule_name) = &self.shared_ap_base_rule {
            return Ok(r(rule_name));
        }

        let literal_keys = self.shared_ap_literal_keys.iter().cloned().collect::<Vec<_>>();
        let mut excluded = literal_keys
            .iter()
            .map(|key| self.lower_literal_key_colon(key))
            .collect::<Vec<_>>();
        for pattern in self.shared_ap_patterns.clone() {
            excluded.push(GrammarExpr::RawRegex(pattern_key_colon_regex(&pattern)));
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
        let mut alternatives = vec![self.lower_pattern_key_colon_terminal(pattern)?];
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

fn string_pattern_as_body_regex(pattern: &str) -> String {
    if let Some(stripped) = strip_simple_anchors(pattern) {
        rewrite_simple_json_string_body_pattern(stripped)
    } else if let Some(stripped) = pattern.strip_prefix('^') {
        format!("(?:{stripped}).*")
    } else if let Some(stripped) = pattern.strip_suffix('$') {
        format!(".*(?:{stripped})")
    } else {
        format!(".*({pattern}).*")
    }
}

fn pattern_key_colon_regex(pattern: &str) -> String {
    let string_char = r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\bfnrt])"#;
    let body = if let Some(stripped) = strip_simple_anchors(pattern) {
        if stripped == ".*" {
            format!("(?:{string_char})*")
        } else {
            rewrite_simple_json_string_body_pattern(stripped)
        }
    } else if let Some(stripped) = pattern.strip_prefix('^') {
        format!("(?:{stripped})(?:{string_char})*")
    } else if let Some(stripped) = pattern.strip_suffix('$') {
        format!("(?:{string_char})*(?:{stripped})")
    } else {
        format!("(?:{string_char})*(?:{pattern})(?:{string_char})*")
    };
    format!(r#""{body}"(?:: )"#)
}

fn strip_simple_anchors(pattern: &str) -> Option<&str> {
    let without_start = pattern.strip_prefix('^')?;
    without_start.strip_suffix('$')
}

fn rewrite_simple_json_string_body_pattern(pattern: &str) -> String {
    match pattern {
        "\"" => r#"\\\""#.to_string(),
        _ => pattern.to_string(),
    }
}

pub(crate) fn property_name_matches_pattern(pattern: &str, property_name: &str) -> ImportResult<bool> {
    let encoded = serde_json::to_string(property_name).unwrap_or_else(|_| "\"\"".to_string());
    let body = encoded.strip_prefix('"').and_then(|text| text.strip_suffix('"')).unwrap_or("");
    let regex = Regex::new(&format!(r#"^(?:{})$"#, string_pattern_as_body_regex(pattern)))
        .map_err(|error| SchemaImportError::new(format!("invalid patternProperties regex {pattern:?}: {error}")))?;
    Ok(regex.is_match(body))
}
