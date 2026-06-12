use std::collections::{BTreeMap, BTreeSet};

use regex::{Regex, escape as regex_escape};
use regex_syntax::hir::{Class, Hir, HirKind, Literal, Look, Repetition};
use regex_syntax::Parser;

use crate::import::ast::{GrammarExpr, Quantifier};

use super::ast::StringSchema;
use super::error::{ImportResult, SchemaImportError};
use super::lower::{
    choice, json_additional_key_string_rule, lit, lit_bytes, never, r, seq, Lowerer,
    JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_NT_RULE,
    JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED_RULE, JSON_ADDITIONAL_KEY_COLON_SHARED_RULE,
    JSON_KEY_SEPARATOR_RULE, JSON_SEPARATOR_WS_REGEX, JSON_STRING_CHAR_RULE,
    JSON_STRING_PATTERN_DOT_CHAR_RULE, JSON_STRING_RULE, MAX_SHARED_ADDITIONAL_EXCLUSION_KEYS,
};

fn encoded_json_key_regex(encoded: &str) -> String {
    // Keep literal property spelling exactly as serde_json emits it.
    // This matches llguidance's builder.string(json_dumps(name)) behavior.
    regex_escape(encoded)
}

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
            let name = self.fresh_rule_name("json_string_constrained");
            if schema.min_length == 0
                && schema.max_length.is_none()
                && let Some(pattern) = &schema.pattern
                && let Some(expr) = self.lower_unanchored_pattern_string_split_expr(pattern)?
            {
                self.add_nonterminal_rule(&name, expr);
                return Ok(r(&name));
            }
            let expr = self.lower_constrained_string_terminal_expr(schema)?;
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
            if self.llguidance_compat_enabled() {
                return Ok(None);
            }
            if string.min_length != 0 || string.max_length.is_some() || string.format.is_some() {
                return Ok(None);
            }
            return Ok(Some(GrammarExpr::RawRegex(quoted_string_body_regex(
                &string_pattern_as_body_regex(pattern, JsonStringContext::Value)?,
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

        let has_length_bounds = schema.min_length != 0 || schema.max_length.is_some();
        let mut expr = if let Some(pattern) = &schema.pattern {
            if !has_length_bounds
                && let Some(pattern_expr) = self.lower_string_pattern_expr(pattern)?
            {
                pattern_expr
            } else {
                GrammarExpr::RawRegex(quoted_string_body_regex(&string_pattern_as_body_regex(
                    pattern,
                    JsonStringContext::Value,
                )?))
            }
        } else {
            GrammarExpr::RawRegex(quoted_string_body_regex(&bounded_json_string_body_regex(
                &self.json_string_char_regex(),
                schema.min_length,
                schema.max_length,
            )))
        };
        let mut constraints = Vec::new();

        if schema.pattern.is_some() && has_length_bounds {
            constraints.push(quoted_string_body_regex(&bounded_json_string_body_regex(
                &self.json_string_char_regex(),
                schema.min_length,
                schema.max_length,
            )));
        }

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

    fn lower_string_pattern_expr(&mut self, pattern: &str) -> ImportResult<Option<GrammarExpr>> {
        let pattern = preprocess_ascii_shorthand(pattern);
        let hir = Parser::new()
            .parse(&pattern)
            .map_err(|error| SchemaImportError::new(format!("invalid string pattern {pattern:?}: {error}")))?;
        if matches!(json_string_compat_mode(), JsonStringCompatMode::LlGuidanceNative)
            && hir_contains_pattern_non_whitespace_class(&hir)
        {
            return Ok(None);
        }
        let Some(branches) = self.lower_string_pattern_hir_branch_expr_parts(hir)? else {
            return Ok(None);
        };
        Ok(Some(self.lower_string_pattern_split_expr_from_expr_branches(branches)))
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

    fn add_string_pattern_body_terminal(&mut self, body_regex: String) -> GrammarExpr {
        let name = self.fresh_rule_name("json_string_pattern_body");
        self.add_terminal_rule(&name, GrammarExpr::RawRegex(body_regex));
        r(&name)
    }

    fn lower_unanchored_pattern_string_split_expr(
        &mut self,
        pattern: &str,
    ) -> ImportResult<Option<GrammarExpr>> {
        let pattern = preprocess_ascii_shorthand(pattern);
        let hir = Parser::new()
            .parse(&pattern)
            .map_err(|error| SchemaImportError::new(format!("invalid string pattern {pattern:?}: {error}")))?;
        let branches = lower_string_pattern_hir_branch_parts(hir, JsonStringContext::Value)?;
        if branches.iter().all(|(_, anchored_start, _)| *anchored_start) {
            return Ok(None);
        }

        let alternatives = branches
            .into_iter()
            .map(|(lowered, anchored_start, anchored_end)| {
                if anchored_start {
                    self.add_string_pattern_anchored_start_terminal(lowered, anchored_end)
                } else {
                    match unanchored_pattern_split_mode() {
                        UnanchoredPatternSplitMode::ChunkedPrefixMiddle => {
                            let prefix_chunk = self.add_string_pattern_prefix_chunk_terminal();
                            let middle = self.add_string_pattern_middle_terminal(lowered);
                            let end = self.add_string_pattern_end_terminal(anchored_end);
                            seq(vec![
                                lit("\""),
                                GrammarExpr::Quantified(Box::new(prefix_chunk), Quantifier::ZeroPlus),
                                middle,
                                end,
                            ])
                        }
                        UnanchoredPatternSplitMode::OpenMiddle => {
                            let open_middle = self.add_string_pattern_open_middle_terminal(lowered);
                            let end = self.add_string_pattern_end_terminal(anchored_end);
                            seq(vec![open_middle, end])
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
        Ok(Some(choice(alternatives)))
    }

    fn lower_string_pattern_split_expr_from_expr_branches(
        &mut self,
        branches: Vec<(GrammarExpr, bool, bool)>,
    ) -> GrammarExpr {
        choice(branches
            .into_iter()
            .map(|(lowered, anchored_start, anchored_end)| {
                self.lower_string_pattern_split_expr_from_body_expr(lowered, anchored_start, anchored_end)
            })
            .collect())
    }

    fn add_string_pattern_open_middle_terminal(&mut self, body_regex: String) -> GrammarExpr {
        let body = self.add_string_pattern_body_terminal(body_regex);
        let name = self.fresh_rule_name("json_string_pattern_open_middle");
        self.add_terminal_rule(
            &name,
            seq(vec![
                lit("\""),
                GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::ZeroPlus),
                body,
            ]),
        );
        r(&name)
    }

    fn add_string_pattern_open_middle_expr_terminal(&mut self, body_expr: GrammarExpr) -> GrammarExpr {
        let body = self.add_string_pattern_body_expr_terminal(body_expr);
        let name = self.fresh_rule_name("json_string_pattern_open_middle");
        self.add_terminal_rule(
            &name,
            seq(vec![
                lit("\""),
                GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::ZeroPlus),
                body,
            ]),
        );
        r(&name)
    }

    fn add_string_pattern_prefix_chunk_terminal(&mut self) -> GrammarExpr {
        let chunk_size = unanchored_pattern_prefix_chunk_size();
        let name = self.fresh_rule_name("json_string_pattern_prefix_chunk");
        self.add_terminal_rule(
            &name,
            GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::Range(chunk_size, Some(chunk_size))),
        );
        r(&name)
    }

    fn add_string_pattern_middle_terminal(&mut self, body_regex: String) -> GrammarExpr {
        let chunk_size = unanchored_pattern_prefix_chunk_size();
        let body = self.add_string_pattern_body_terminal(body_regex);
        let name = self.fresh_rule_name("json_string_pattern_middle");
        self.add_terminal_rule(
            &name,
            seq(vec![
                GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::Range(0, Some(chunk_size.saturating_sub(1)))),
                body,
            ]),
        );
        r(&name)
    }

    fn add_string_pattern_middle_expr_terminal(&mut self, body_expr: GrammarExpr) -> GrammarExpr {
        let chunk_size = unanchored_pattern_prefix_chunk_size();
        let body = self.add_string_pattern_body_expr_terminal(body_expr);
        let name = self.fresh_rule_name("json_string_pattern_middle");
        self.add_terminal_rule(
            &name,
            seq(vec![
                GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::Range(0, Some(chunk_size.saturating_sub(1)))),
                body,
            ]),
        );
        r(&name)
    }

    fn add_string_pattern_end_terminal(&mut self, anchored_end: bool) -> GrammarExpr {
        let name = self.fresh_rule_name("json_string_pattern_end");
        let mut parts = Vec::new();
        if !anchored_end {
            parts.push(GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::ZeroPlus));
        }
        parts.push(lit("\""));
        self.add_terminal_rule(&name, seq(parts));
        r(&name)
    }

    fn add_string_pattern_anchored_start_expr_terminal(
        &mut self,
        body_expr: GrammarExpr,
        anchored_end: bool,
    ) -> GrammarExpr {
        let split_llguidance_expr = matches!(json_string_compat_mode(), JsonStringCompatMode::LlGuidanceNative)
            && anchored_end;
        let body = if split_llguidance_expr {
            body_expr
        } else {
            self.add_string_pattern_body_expr_terminal(body_expr)
        };
        let mut parts = vec![lit("\""), body];
        if !anchored_end {
            parts.push(GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::ZeroPlus));
        }
        parts.push(lit("\""));
        let expr = seq(parts);
        if split_llguidance_expr {
            return expr;
        }
        let name = self.fresh_rule_name("json_string_constrained_part");
        self.add_terminal_rule(&name, expr);
        r(&name)
    }

    fn lower_string_pattern_split_expr_from_body_expr(
        &mut self,
        body_expr: GrammarExpr,
        anchored_start: bool,
        anchored_end: bool,
    ) -> GrammarExpr {
        if anchored_start {
            return self.add_string_pattern_anchored_start_expr_terminal(body_expr, anchored_end);
        }
        match unanchored_pattern_split_mode() {
            UnanchoredPatternSplitMode::ChunkedPrefixMiddle => {
                let prefix_chunk = self.add_string_pattern_prefix_chunk_terminal();
                let middle = self.add_string_pattern_middle_expr_terminal(body_expr);
                let end = self.add_string_pattern_end_terminal(anchored_end);
                seq(vec![
                    lit("\""),
                    GrammarExpr::Quantified(Box::new(prefix_chunk), Quantifier::ZeroPlus),
                    middle,
                    end,
                ])
            }
            UnanchoredPatternSplitMode::OpenMiddle => {
                let open_middle = self.add_string_pattern_open_middle_expr_terminal(body_expr);
                let end = self.add_string_pattern_end_terminal(anchored_end);
                seq(vec![open_middle, end])
            }
        }
    }

    fn add_string_pattern_body_expr_terminal(&mut self, body_expr: GrammarExpr) -> GrammarExpr {
        let name = self.fresh_rule_name("json_string_pattern_body");
        self.add_terminal_rule(&name, body_expr);
        r(&name)
    }

    fn lower_string_pattern_hir_branch_expr_parts(
        &mut self,
        hir: Hir,
    ) -> ImportResult<Option<Vec<(GrammarExpr, bool, bool)>>> {
        match hir.kind() {
            HirKind::Alternation(parts) => {
                let mut lowered = Vec::with_capacity(parts.len());
                for part in parts.iter().cloned() {
                    let Some(branch) = self.lower_string_pattern_branch_expr_parts(part)? else {
                        return Ok(None);
                    };
                    lowered.push(branch);
                }
                Ok(Some(lowered))
            }
            HirKind::Capture(capture) => self.lower_string_pattern_hir_branch_expr_parts(*capture.sub.clone()),
            _ => self
                .lower_string_pattern_branch_expr_parts(hir)
                .map(|branch| branch.map(|branch| vec![branch])),
        }
    }

    fn lower_string_pattern_branch_expr_parts(
        &mut self,
        hir: Hir,
    ) -> ImportResult<Option<(GrammarExpr, bool, bool)>> {
        let (hir, anchored_start, anchored_end) = strip_outer_anchors(hir);
        let Some(lowered) = self.lower_decoded_regex_hir_to_json_body_expr_at_start(&hir, true)? else {
            return Ok(None);
        };
        Ok(Some((lowered, anchored_start, anchored_end)))
    }

    fn lower_decoded_regex_hir_to_json_body_expr(
        &mut self,
        hir: &Hir,
    ) -> ImportResult<Option<GrammarExpr>> {
        self.lower_decoded_regex_hir_to_json_body_expr_at_start(hir, true)
    }

    fn lower_decoded_regex_hir_to_json_body_expr_at_start(
        &mut self,
        hir: &Hir,
        at_start: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        Ok(match hir.kind() {
            HirKind::Empty => Some(GrammarExpr::Epsilon),
            HirKind::Literal(Literal(bytes)) => {
                let literal = std::str::from_utf8(bytes).map_err(|error| {
                    SchemaImportError::new(format!(
                        "string pattern literal is not valid UTF-8 after parsing: {error}"
                    ))
                })?;
                Some(seq(
                    literal
                        .chars()
                        .map(|ch| {
                            json_body_char_expr_for_pattern_literal_in_mode(
                                ch,
                                json_string_compat_mode(),
                                JsonStringContext::Value,
                                false,
                            )
                        })
                        .collect(),
                ))
            }
            HirKind::Class(class) => self.lower_decoded_class_to_json_body_expr(class, at_start),
            HirKind::Look(look) if is_start_look(*look) || is_end_look(*look) => Some(GrammarExpr::Epsilon),
            HirKind::Look(look) => {
                return Err(SchemaImportError::new(format!(
                    "unsupported zero-width assertion in string pattern: {look:?}"
                )));
            }
            HirKind::Repetition(repetition) => self.lower_decoded_repetition_to_json_body_expr_at_start(repetition, at_start)?,
            HirKind::Capture(capture) => self.lower_decoded_regex_hir_to_json_body_expr_at_start(&capture.sub, at_start)?,
            HirKind::Concat(parts) => {
                let mut current_at_start = at_start;
                let mut lowered = Vec::with_capacity(parts.len());
                for part in parts {
                    let Some(lowered_part) = self.lower_decoded_regex_hir_to_json_body_expr_at_start(part, current_at_start)? else {
                        return Ok(None);
                    };
                    lowered.push(lowered_part);
                    current_at_start = current_at_start && hir_can_match_empty(part);
                }
                Some(seq(lowered))
            }
            HirKind::Alternation(parts) => {
                let mut lowered = Vec::with_capacity(parts.len());
                for part in parts {
                    let Some(part) = self.lower_decoded_regex_hir_to_json_body_expr_at_start(part, at_start)? else {
                        return Ok(None);
                    };
                    lowered.push(part);
                }
                Some(choice(lowered))
            }
        })
    }

    fn lower_decoded_repetition_to_json_body_expr(
        &mut self,
        repetition: &Repetition,
    ) -> ImportResult<Option<GrammarExpr>> {
        self.lower_decoded_repetition_to_json_body_expr_at_start(repetition, true)
    }

    fn lower_decoded_repetition_to_json_body_expr_at_start(
        &mut self,
        repetition: &Repetition,
        at_start: bool,
    ) -> ImportResult<Option<GrammarExpr>> {
        let Some(sub) = self.lower_decoded_regex_hir_to_json_body_expr_at_start(&repetition.sub, at_start)? else {
            return Ok(None);
        };
        Ok(Some(match (repetition.min, repetition.max) {
            (0, Some(0)) => GrammarExpr::Epsilon,
            (0, None) => GrammarExpr::Quantified(Box::new(sub), Quantifier::ZeroPlus),
            (1, None) => GrammarExpr::Quantified(Box::new(sub), Quantifier::OnePlus),
            (0, Some(1)) => GrammarExpr::Quantified(Box::new(sub), Quantifier::Optional),
            (min, Some(max)) => GrammarExpr::Quantified(Box::new(sub), Quantifier::Range(min.try_into().unwrap(), Some(max.try_into().unwrap()))),
            (min, None) => seq(vec![
                GrammarExpr::Quantified(Box::new(sub.clone()), Quantifier::Range(min.try_into().unwrap(), Some(min.try_into().unwrap()))),
                GrammarExpr::Quantified(Box::new(sub), Quantifier::ZeroPlus),
            ]),
        }))
    }

    fn lower_decoded_class_to_json_body_expr(
        &mut self,
        class: &Class,
        at_start: bool,
    ) -> Option<GrammarExpr> {
        if is_unicode_decimal_digit_class(class) {
            return Some(self.string_pattern_digit_char_ref());
        }
        if is_unicode_non_decimal_digit_class(class) {
            return Some(GrammarExpr::Exclude {
                expr: Box::new(r(JSON_STRING_CHAR_RULE)),
                exclude: Box::new(self.string_pattern_digit_char_ref()),
            });
        }
        if is_unicode_pattern_whitespace_class(class) {
            return Some(self.string_pattern_whitespace_char_ref());
        }
        if is_unicode_pattern_non_whitespace_class(class) {
            return Some(self.string_pattern_non_whitespace_char_ref());
        }
        if matches!(json_string_compat_mode(), JsonStringCompatMode::LlGuidanceNative) {
            if is_dot_like_unicode_class(class) {
                return Some(self.string_pattern_dot_char_ref());
            }
            return lower_llguidance_value_pattern_class_expr(class, at_start);
        }
        None
    }

    fn string_pattern_digit_char_ref(&mut self) -> GrammarExpr {
        const NAME: &str = "JSON_STRING_PATTERN_DIGIT_CHAR";
        if !self.rules.iter().any(|rule| rule.name == NAME) {
            self.add_terminal_rule(NAME, GrammarExpr::RawRegex("[0-9]".to_string()));
        }
        r(NAME)
    }

    fn string_pattern_whitespace_char_ref(&mut self) -> GrammarExpr {
        const NAME: &str = "JSON_STRING_PATTERN_WHITESPACE_CHAR";
        if !self.rules.iter().any(|rule| rule.name == NAME) {
            let mut alternatives = PATTERN_WHITESPACE_CHARS
                .iter()
                .copied()
                .map(json_body_char_expr_for_decoded_char)
                .collect::<Vec<_>>();
            if matches!(json_string_compat_mode(), JsonStringCompatMode::LlGuidanceNative) {
                let mut escape = String::new();
                escape.push(char::from(92));
                escape.push_str("u000");
                alternatives.push(seq(vec![
                    lit(&escape),
                    GrammarExpr::CharClass {
                        def: "9-Da-d".to_string(),
                        negate: false,
                        utf8: true,
                    },
                ]));
            }
            self.add_terminal_rule(NAME, choice(alternatives));
        }
        r(NAME)
    }

    fn string_pattern_non_whitespace_char_ref(&mut self) -> GrammarExpr {
        const NAME: &str = "JSON_STRING_PATTERN_NON_WHITESPACE_CHAR";
        if !self.rules.iter().any(|rule| rule.name == NAME) {
            let whitespace = self.string_pattern_whitespace_char_ref();
            let rule_expr = GrammarExpr::Exclude {
                expr: Box::new(r(JSON_STRING_CHAR_RULE)),
                exclude: Box::new(whitespace),
            };
            self.add_terminal_rule(NAME, rule_expr);
        }
        r(NAME)
    }

    fn string_pattern_dot_char_ref(&mut self) -> GrammarExpr {
        if !self
            .rules
            .iter()
            .any(|rule| rule.name == JSON_STRING_PATTERN_DOT_CHAR_RULE)
        {
            self.add_terminal_rule(
                JSON_STRING_PATTERN_DOT_CHAR_RULE,
                GrammarExpr::RawRegex(
                    json_string_body_dot_regex_in_mode(
                        json_string_compat_mode(),
                        JsonStringContext::Value,
                    )
                    .to_string(),
                ),
            );
        }
        r(JSON_STRING_PATTERN_DOT_CHAR_RULE)
    }

    fn add_string_pattern_anchored_start_terminal(
        &mut self,
        body_regex: String,
        anchored_end: bool,
    ) -> GrammarExpr {
        let body = self.add_string_pattern_body_terminal(body_regex);
        let name = self.fresh_rule_name("json_string_constrained_part");
        let mut parts = vec![lit("\""), body];
        if !anchored_end {
            parts.push(GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::ZeroPlus));
        }
        parts.push(lit("\""));
        self.add_terminal_rule(&name, seq(parts));
        r(&name)
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
            _ => {
                if let Some(rule_name) = self.shared_string_exact_rules.get(&count) {
                    return r(rule_name);
                }
                let rule_name = self.fresh_rule_name(&format!("json_string_char_exact_{count}"));
                self.add_terminal_rule(
                    &rule_name,
                    GrammarExpr::Quantified(Box::new(r(JSON_STRING_CHAR_RULE)), Quantifier::Range(count, Some(count))),
                );
                self.shared_string_exact_rules.insert(count, rule_name.clone());
                r(&rule_name)
            }
        }
    }

    fn string_char_upto_ref(&mut self, max: usize) -> GrammarExpr {
        match max {
            0 => GrammarExpr::Epsilon,
            _ => {
                if let Some(rule_name) = self.shared_string_upto_rules.get(&max) {
                    return r(rule_name);
                }
                let rule_name = self.fresh_rule_name(&format!("json_string_char_upto_{max}"));
                self.add_terminal_rule(
                    &rule_name,
                    GrammarExpr::Quantified(
                        Box::new(r(JSON_STRING_CHAR_RULE)),
                        Quantifier::Range(0, Some(max)),
                    ),
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
        if let Some(rule_name) = self.shared_string_exact_open_rules.get(&count) {
            return r(rule_name);
        }
        let rule_name = self.fresh_rule_name(&format!("json_string_char_exact_open_{count}"));
        let exact = self.string_char_exact_ref(count);
        self.add_terminal_rule(&rule_name, seq(vec![lit("\""), exact]));
        self.shared_string_exact_open_rules
            .insert(count, rule_name.clone());
        r(&rule_name)
    }

    fn string_char_upto_wrapped_ref(&mut self, max: usize) -> GrammarExpr {
        if let Some(rule_name) = self.shared_string_upto_wrapped_rules.get(&max) {
            return r(rule_name);
        }
        let rule_name = self.fresh_rule_name(&format!("json_string_char_upto_wrapped_{max}"));
        let upto = self.string_char_upto_ref(max);
        self.add_terminal_rule(&rule_name, seq(vec![lit("\""), upto, lit("\"")]));
        self.shared_string_upto_wrapped_rules
            .insert(max, rule_name.clone());
        r(&rule_name)
    }

    fn split_string_exact_expr(&mut self, count: usize) -> GrammarExpr {
        let chunk = self.config.repeat_chunk_size.max(1);
        if count <= chunk {
            return self.string_char_exact_ref(count);
        }

        let full_chunks = count / chunk;
        let remainder = count % chunk;
        let mut parts = vec![GrammarExpr::Quantified(Box::new(self.string_char_exact_ref(chunk)), Quantifier::Range(full_chunks, Some(full_chunks)))];
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
            GrammarExpr::Quantified(Box::new(exact_chunk.clone()), Quantifier::Range(0, Some(full_chunks.saturating_sub(1)))),
            self.string_char_upto_close_ref(chunk - 1),
        ])];
        alternatives.push(seq(vec![
            GrammarExpr::Quantified(Box::new(exact_chunk), Quantifier::Range(full_chunks, Some(full_chunks))),
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
                GrammarExpr::Quantified(Box::new(exact_chunk.clone()), Quantifier::Range(0, Some(full_chunks.saturating_sub(2)))),
                self.string_char_upto_close_ref(chunk),
            ]));
            if remainder > 0 {
                alternatives.push(seq(vec![
                    exact_open_chunk,
                    GrammarExpr::Quantified(Box::new(exact_chunk), Quantifier::Range(full_chunks.saturating_sub(1), Some(full_chunks.saturating_sub(1)))),
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

    pub(crate) fn lower_literal_key_colon_with_prefix_and_string_schema(
        &mut self,
        prefix: &[u8],
        key: &str,
        schema: &StringSchema,
    ) -> ImportResult<GrammarExpr> {
        if schema.pattern.is_none()
            && recognized_string_format_body_regex_for_lowering(schema.format.as_deref()).is_none()
        {
            return Ok(seq(vec![
                self.lower_literal_key_colon_with_prefix(prefix, key),
                self.lower_string_expr(schema)?,
            ]));
        }

        if self.llguidance_compat_enabled() && schema.pattern.is_some() {
            return Ok(seq(vec![
                self.lower_literal_key_colon_with_prefix(prefix, key),
                self.lower_string_property_value_expr(schema)?,
            ]));
        }

        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        let mut literal_prefix = Vec::new();
        literal_prefix.extend_from_slice(prefix);
        literal_prefix.extend_from_slice(encoded.as_bytes());
        literal_prefix.extend_from_slice(b": ");
        let value = self.lower_string_property_value_expr(schema)?;
        let expr = prepend_literal_prefix_to_expr(literal_prefix, value);
        let name = self.fresh_rule_name("json_property_string_value");
        self.add_terminal_rule(&name, expr);
        Ok(r(&name))
    }

    fn lower_string_property_value_expr(
        &mut self,
        schema: &StringSchema,
    ) -> ImportResult<GrammarExpr> {
        if schema.pattern.is_none()
            && recognized_string_format_body_regex_for_lowering(schema.format.as_deref()).is_none()
        {
            return self.lower_string_expr(schema);
        }
        self.lower_constrained_string_terminal_expr(schema)
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

    fn lower_literal_key_colon_exact_with_prefix(
        &self,
        prefix: &[u8],
        key: &str,
    ) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        if matches!(json_string_compat_mode(), JsonStringCompatMode::LlGuidanceNative) {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(prefix);
            bytes.extend_from_slice(encoded.as_bytes());
            bytes.extend_from_slice(b": ");
            return lit_bytes(bytes);
        }
        let encoded_regex = encoded_json_key_regex(&encoded);
        if prefix == b", " {
            GrammarExpr::RawRegex(format!(
                r#",{JSON_SEPARATOR_WS_REGEX}{}:{JSON_SEPARATOR_WS_REGEX}"#,
                encoded_regex
            ))
        } else if prefix.is_empty() {
            GrammarExpr::RawRegex(format!(r#"{}:{JSON_SEPARATOR_WS_REGEX}"#, encoded_regex))
        } else {
            GrammarExpr::RawRegex(format!(
                r#"{}{}:{JSON_SEPARATOR_WS_REGEX}"#,
                regex_escape(&String::from_utf8_lossy(prefix)),
                encoded_regex
            ))
        }
    }

    pub(crate) fn lower_literal_key_colon_with_prefix(
        &mut self,
        prefix: &[u8],
        key: &str,
    ) -> GrammarExpr {
        let exact = self.lower_literal_key_colon_exact_with_prefix(prefix, key);

        exact
    }

    pub(crate) fn lower_literal_key_colon_with_prefix_and_suffix(
        &mut self,
        prefix: &[u8],
        key: &str,
        suffix: u8,
    ) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        let encoded_regex = encoded_json_key_regex(&encoded);
        let suffix_byte = suffix;
        let suffix = regex_escape(&String::from_utf8_lossy(&[suffix_byte]));
        let exact = if prefix == b", " {
            GrammarExpr::RawRegex(format!(
                r#",{JSON_SEPARATOR_WS_REGEX}{}:{JSON_SEPARATOR_WS_REGEX}{}"#,
                encoded_regex,
                suffix
            ))
        } else if prefix.is_empty() {
            GrammarExpr::RawRegex(format!(
                r#"{}:{JSON_SEPARATOR_WS_REGEX}{}"#,
                encoded_regex,
                suffix
            ))
        } else {
            seq(vec![
                lit_bytes(prefix.to_vec()),
                self.lower_literal_key_colon_with_prefix_and_suffix(b"", key, suffix_byte),
            ])
        };

        exact
    }

    pub(crate) fn lower_literal_key_colon_with_prefix_and_json_string(
        &mut self,
        prefix: &[u8],
        key: &str,
    ) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        let encoded_regex = encoded_json_key_regex(&encoded);
        let string_body = self.json_string_char_regex();
        let exact = if prefix == b", " {
            GrammarExpr::RawRegex(format!(
                r#",{JSON_SEPARATOR_WS_REGEX}{}:{JSON_SEPARATOR_WS_REGEX}"(?:{})*""#,
                encoded_regex,
                string_body
            ))
        } else if prefix.is_empty() {
            GrammarExpr::RawRegex(format!(
                r#"{}:{JSON_SEPARATOR_WS_REGEX}"(?:{})*""#,
                encoded_regex,
                string_body
            ))
        } else {
            seq(vec![
                lit_bytes(prefix.to_vec()),
                self.lower_literal_key_colon_with_prefix_and_json_string(b"", key),
            ])
        };

        exact
    }

    pub(crate) fn lower_literal_key_colon_with_prefix_and_literal_value(
        &mut self,
        prefix: &[u8],
        key: &str,
        value: &[u8],
    ) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        let encoded_regex = encoded_json_key_regex(&encoded);
        let value_regex = regex_escape(&String::from_utf8_lossy(value));
        let exact = if prefix == b", " {
            GrammarExpr::RawRegex(format!(
                r#",{JSON_SEPARATOR_WS_REGEX}{}:{JSON_SEPARATOR_WS_REGEX}{}"#,
                encoded_regex,
                value_regex
            ))
        } else if prefix.is_empty() {
            GrammarExpr::RawRegex(format!(
                r#"{}:{JSON_SEPARATOR_WS_REGEX}{}"#,
                encoded_regex,
                value_regex
            ))
        } else {
            seq(vec![
                lit_bytes(prefix.to_vec()),
                self.lower_literal_key_colon_with_prefix_and_literal_value(b"", key, value),
            ])
        };

        exact
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
        if matches!(json_string_compat_mode(), JsonStringCompatMode::LlGuidanceNative)
            && pattern_matches_any_key(pattern)
        {
            if let Some(rule_name) = self.shared_ap_pattern_rules.get(pattern) {
                return Ok(r(rule_name));
            }

            let global_overlaps = self.pattern_overlapping_literal_keys(pattern)?;
            let expr = if global_overlaps.is_empty() {
                seq(vec![r(json_additional_key_string_rule()), r(JSON_KEY_SEPARATOR_RULE)])
            } else {
                GrammarExpr::Exclude {
                    expr: Box::new(seq(vec![
                        r(json_additional_key_string_rule()),
                        r(JSON_KEY_SEPARATOR_RULE),
                    ])),
                    exclude: Box::new(choice(
                        global_overlaps
                            .iter()
                            .map(|key| self.lower_literal_key_colon_exact_with_prefix(b"", key))
                            .collect::<Vec<_>>(),
                    )),
                }
            };
            let name = self.fresh_rule_name("json_pattern_key_colon");
            self.add_nonterminal_rule(&name, expr);
            self.shared_ap_pattern_rules.insert(pattern.to_string(), name.clone());
            return Ok(r(&name));
        }

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
        if local_patterns.iter().any(|pattern| pattern_matches_any_key(pattern)) {
            return Ok(never());
        }

        if !fixed_keys.is_empty() {
            return self.lower_additional_key_colon_expanded_addback(fixed_keys, local_patterns);
        }

        if super::share_additional_addback_choices_enabled() && !self.llguidance_compat_enabled() {
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
        if !local_patterns.is_empty() {
            return self.lower_additional_key_colon_expanded_addback(fixed_keys, local_patterns);
        }

        for key in fixed_keys {
            if self.shared_ap_literal_keys.contains(key) {
                continue;
            }
            let mut covered_by_shared_pattern = false;
            for pattern in &self.shared_ap_patterns {
                match property_name_matches_pattern(pattern, key) {
                    Ok(true) => {
                        covered_by_shared_pattern = true;
                        break;
                    }
                    Ok(false) => {}
                    Err(error) if is_regex_compile_limit_error(&error) => {
                        covered_by_shared_pattern = true;
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
            if covered_by_shared_pattern {
                return self.lower_additional_key_colon_expanded_addback(fixed_keys, local_patterns);
            }
        }

        let cache_key = (
            fixed_keys.iter().cloned().collect::<Vec<_>>(),
            local_patterns.to_vec(),
        );
        if let Some(rule_name) = self.shared_additional_key_colon_local_rules.get(&cache_key) {
            return Ok(r(rule_name));
        }

        let base = self.shared_additional_key_colon_base()?;
        let excluded_addback = self.shared_additional_excluded_key_colon()?;
        let shared_literal_keys = self.shared_ap_literal_keys.clone();
        let mut local_exclusions = fixed_keys
            .iter()
            .filter(|key| shared_literal_keys.contains(*key))
            .map(|key| self.lower_literal_key_colon(key))
            .collect::<Vec<_>>();
        for pattern in local_patterns {
            local_exclusions.extend(self.pattern_key_colon_local_exclusion_alternatives(pattern)?);
        }
        let mut seen = std::collections::HashSet::new();
        local_exclusions.retain(|expr| seen.insert(expr.clone()));

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
        let shared_literal_keys = self.shared_ap_literal_keys.clone();
        let local_exclusions = fixed_keys
            .iter()
            .filter(|key| shared_literal_keys.contains(*key))
            .map(|key| self.lower_literal_key_colon(key))
            .collect::<Vec<_>>();

        if self.shared_ap_patterns.is_empty()
            && fixed_keys.len() >= self.shared_ap_literal_keys.len()
            && self.shared_ap_literal_keys.iter().all(|key| fixed_keys.contains(key))
        {
            return Ok(base);
        }

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
            terminal_excluded.push(self.lower_literal_key_colon_exact_with_prefix(b"", &key));
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
                    expr: Box::new(seq(vec![
                        r(json_additional_key_string_rule()),
                        self.key_separator_expr(),
                    ])),
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
                expr: Box::new(seq(vec![
                    r(json_additional_key_string_rule()),
                    self.key_separator_expr(),
                ])),
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

    fn string_body_for_length(&mut self, min: usize, max: Option<usize>) -> GrammarExpr {
        let ch = self.string_char_exact_ref(1);
        match (min, max) {
            (0, None) => GrammarExpr::Quantified(Box::new(ch), Quantifier::ZeroPlus),
            (1, None) => GrammarExpr::Quantified(Box::new(ch), Quantifier::OnePlus),
            (min, None) => seq(vec![
                self.repeat_exact_string_char(min),
                GrammarExpr::Quantified(Box::new(self.string_char_exact_ref(1)), Quantifier::ZeroPlus),
            ]),
            (0, Some(0)) => GrammarExpr::Epsilon,
            (0, Some(max)) => self.string_char_upto_ref(max),
            (min, Some(max)) if min == max => self.repeat_exact_string_char(min),
            (min, Some(max)) => seq(vec![
                self.repeat_exact_string_char(min),
                self.string_char_upto_ref(max - min),
            ]),
        }
    }

    fn repeat_exact_string_char(&mut self, count: usize) -> GrammarExpr {
        if count == 0 {
            return GrammarExpr::Epsilon;
        }
        let chunk = self.config.repeat_chunk_size.max(1);
        if count <= chunk {
            return self.string_char_exact_ref(count);
        }

        let mut parts = Vec::new();
        let mut remaining = count;
        while remaining > 0 {
            let take = remaining.min(chunk);
            parts.push(self.string_char_exact_ref(take));
            remaining -= take;
        }
        seq(parts)
    }
}

fn string_pattern_as_body_regex(pattern: &str, context: JsonStringContext) -> ImportResult<String> {
    let pattern = preprocess_ascii_shorthand(pattern);
    let hir = Parser::new()
        .parse(&pattern)
        .map_err(|error| SchemaImportError::new(format!("invalid string pattern {pattern:?}: {error}")))?;
    string_pattern_hir_as_body_regex(&hir, context)
}

pub(crate) fn preprocess_ascii_shorthand(pattern: &str) -> String {
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

fn pattern_matches_any_key(pattern: &str) -> bool {
    matches!(preprocess_ascii_shorthand(pattern).as_str(), ".*" | "^.*$")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnanchoredPatternSplitMode {
    ChunkedPrefixMiddle,
    OpenMiddle,
}

fn unanchored_pattern_split_mode() -> UnanchoredPatternSplitMode {
    static VALUE: std::sync::OnceLock<UnanchoredPatternSplitMode> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        if let Some(mode) = std::env::var("GLRMASK_JSON_SCHEMA_UNANCHORED_PATTERN_SPLIT_MODE")
            .ok()
            .and_then(|value| parse_unanchored_pattern_split_mode(&value))
        {
            return mode;
        }

        // Backwards-compatible spelling from the original two-mode experiment.
        // Preserve the original default: unset means legacy open-middle split.
        // Set this variable to 1/true/yes/on to force chunked prefix/middle.
        std::env::var("GLRMASK_JSON_SCHEMA_UNANCHORED_PATTERN_CHUNK_PREFIX_MIDDLE")
            .ok()
            .and_then(|value| parse_env_bool(&value))
            .map(|enabled| {
                if enabled {
                    UnanchoredPatternSplitMode::ChunkedPrefixMiddle
                } else {
                    UnanchoredPatternSplitMode::OpenMiddle
                }
            })
            .unwrap_or(UnanchoredPatternSplitMode::OpenMiddle)
    })
}

fn parse_unanchored_pattern_split_mode(value: &str) -> Option<UnanchoredPatternSplitMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "chunk" | "chunked" | "chunked_prefix_middle" | "chunked-prefix-middle" => {
            Some(UnanchoredPatternSplitMode::ChunkedPrefixMiddle)
        }
        "open_middle" | "open-middle" | "openmiddle" | "legacy" => {
            Some(UnanchoredPatternSplitMode::OpenMiddle)
        }
        _ => None,
    }
}

fn parse_env_bool(value: &str) -> Option<bool> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Some(false);
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn unanchored_pattern_prefix_chunk_size() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("GLRMASK_JSON_SCHEMA_UNANCHORED_PATTERN_PREFIX_CHUNK_SIZE")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(8)
    })
}

fn lower_string_pattern_hir_branch_parts(hir: Hir, context: JsonStringContext) -> ImportResult<Vec<(String, bool, bool)>> {
    match hir.kind() {
        HirKind::Alternation(parts) => parts
            .iter()
            .cloned()
            .map(|part| lower_string_pattern_branch_parts(part, context))
            .collect(),
        HirKind::Capture(capture) => lower_string_pattern_hir_branch_parts(*capture.sub.clone(), context),
        _ => lower_string_pattern_branch_parts(hir, context).map(|branch| vec![branch]),
    }
}

fn string_pattern_hir_as_body_regex(hir: &Hir, context: JsonStringContext) -> ImportResult<String> {
    match hir.kind() {
        HirKind::Alternation(parts) => {
            let alternatives = parts
                .iter()
                .cloned()
                .map(|part| lower_string_pattern_branch_parts(part, context))
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
                        context,
                    ));
                }
            }

            let wrapped = alternatives
                .iter()
                .map(|(lowered, anchored_start, anchored_end)| {
                    wrap_lowered_string_pattern_branch(lowered, *anchored_start, *anchored_end, context)
                })
                .collect::<Vec<_>>();
            Ok(format!("(?:{})", wrapped.join("|")))
        }
        HirKind::Capture(capture) => {
            string_pattern_hir_as_body_regex(&capture.sub, context)
        }
        _ => string_pattern_branch_as_body_regex(hir.clone(), context),
    }
}

fn string_pattern_branch_as_body_regex(hir: Hir, context: JsonStringContext) -> ImportResult<String> {
    let (lowered, anchored_start, anchored_end) = lower_string_pattern_branch_parts(hir, context)?;
    Ok(wrap_lowered_string_pattern_branch(
        &lowered,
        anchored_start,
        anchored_end,
        context,
    ))
}

fn lower_string_pattern_branch_parts(hir: Hir, context: JsonStringContext) -> ImportResult<(String, bool, bool)> {
    let (hir, anchored_start, anchored_end) = strip_outer_anchors(hir);
    let lowered = lower_decoded_regex_hir_to_json_body_regex(&hir, context)?;
    Ok((lowered, anchored_start, anchored_end))
}

fn wrap_lowered_string_pattern_branch(lowered: &str, anchored_start: bool, anchored_end: bool, context: JsonStringContext) -> String {
    let string_char = json_string_body_char_regex_in_mode(json_string_compat_mode(), context);
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

fn bounded_json_string_body_regex(string_char_regex: &str, min: usize, max: Option<usize>) -> String {
    let atom = format!("(?:{string_char_regex})");
    match (min, max) {
        (0, None) => format!("{atom}*"),
        (1, None) => format!("{atom}+"),
        (min, None) => format!("{atom}{{{min},}}"),
        (min, Some(max)) if min == max => format!("{atom}{{{min}}}"),
        (min, Some(max)) => format!("{atom}{{{min},{max}}}"),
    }
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
            // Deliberately smaller than llguidance's full RFC 3986 regex: keep the
            // tokenizer-state footprint low, but do not collapse path/query/fragment
            // into one repeated class.  In particular, '#' is only allowed once as the
            // fragment introducer, so prefixes like "https://##" stay aligned with
            // llguidance without importing its full IPv6/path machinery.
            r#"[A-Za-z][A-Za-z0-9+.-]*:(?://(?:(?:[A-Za-z0-9._~!$&'()*+,;=:-]|%[0-9A-Fa-f]{2})*@)?(?:\[(?:(?:[A-Fa-f0-9][A-Fa-f0-9:.]*|:[A-Fa-f0-9:.]*[A-Fa-f0-9][A-Fa-f0-9:.]*)|v[0-9A-Fa-f]+\.[A-Za-z0-9._~!$&'()*+,;=:-]+)\]|(?:[A-Za-z0-9._~!$&'()*+,;=-]|%[0-9A-Fa-f]{2})*)(?::[0-9]*)?(?:/(?:[A-Za-z0-9._~:@!$&'()*+,;=-]|%[0-9A-Fa-f]{2})*)*|(?:/(?:[A-Za-z0-9._~:@!$&'()*+,;=-]|%[0-9A-Fa-f]{2})*)?|(?:[A-Za-z0-9._~:@!$&'()*+,;=-]|%[0-9A-Fa-f]{2})+(?:/(?:[A-Za-z0-9._~:@!$&'()*+,;=-]|%[0-9A-Fa-f]{2})*)*)?(?:\?(?:[A-Za-z0-9._~:/?@!$&'()*+,;=-]|%[0-9A-Fa-f]{2})*)?(?:#(?:[A-Za-z0-9._~:/?@!$&'()*+,;=-]|%[0-9A-Fa-f]{2})*)?"#,
        ),
        _ => None,
    }
}

fn pattern_key_colon_regex(pattern: &str) -> ImportResult<String> {
    let context = JsonStringContext::KeyStrict;
    let body = string_pattern_as_body_regex(pattern, context)?;
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


fn hir_contains_pattern_non_whitespace_class(hir: &Hir) -> bool {
    match hir.kind() {
        HirKind::Class(class) => is_unicode_pattern_non_whitespace_class(class),
        HirKind::Repetition(repetition) => hir_contains_pattern_non_whitespace_class(&repetition.sub),
        HirKind::Capture(capture) => hir_contains_pattern_non_whitespace_class(&capture.sub),
        HirKind::Concat(parts) | HirKind::Alternation(parts) => {
            parts.iter().any(hir_contains_pattern_non_whitespace_class)
        }
        _ => false,
    }
}

fn hir_can_match_empty(hir: &Hir) -> bool {
    match hir.kind() {
        HirKind::Empty => true,
        HirKind::Literal(Literal(bytes)) => bytes.is_empty(),
        HirKind::Class(_) => false,
        HirKind::Look(_) => true,
        HirKind::Repetition(repetition) => repetition.min == 0 || hir_can_match_empty(&repetition.sub),
        HirKind::Capture(capture) => hir_can_match_empty(&capture.sub),
        HirKind::Concat(parts) => parts.iter().all(hir_can_match_empty),
        HirKind::Alternation(parts) => parts.iter().any(hir_can_match_empty),
    }
}

fn lower_decoded_regex_hir_to_json_body_regex(hir: &Hir, context: JsonStringContext) -> ImportResult<String> {
    lower_decoded_regex_hir_to_json_body_regex_at_start(hir, context, true)
}

fn lower_decoded_regex_hir_to_json_body_regex_at_start(
    hir: &Hir,
    context: JsonStringContext,
    at_start: bool,
) -> ImportResult<String> {
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
                .map(|ch| {
                    json_body_char_regex_for_pattern_literal_in_mode(
                        ch,
                        json_string_compat_mode(),
                        context,
                        false,
                    )
                })
                .collect::<Vec<_>>()
                .join("")
        }
        HirKind::Class(class) => lower_decoded_class_to_json_body_regex_at_start(class, context, at_start),
        HirKind::Look(look) if is_start_look(*look) || is_end_look(*look) => String::new(),
        HirKind::Look(look) => {
            return Err(SchemaImportError::new(format!(
                "unsupported zero-width assertion in string pattern: {look:?}"
            )));
        }
        HirKind::Repetition(repetition) => lower_decoded_repetition_to_json_body_regex(repetition, context, at_start)?,
        HirKind::Capture(capture) => lower_decoded_regex_hir_to_json_body_regex_at_start(&capture.sub, context, at_start)?,
        HirKind::Concat(parts) => {
            let mut current_at_start = at_start;
            let mut out = Vec::new();
            for part in parts {
                out.push(lower_decoded_regex_hir_to_json_body_regex_at_start(
                    part,
                    context,
                    current_at_start,
                )?);
                current_at_start = current_at_start && hir_can_match_empty(part);
            }
            out.join("")
        }
        HirKind::Alternation(parts) => {
            let alternatives = parts
                .iter()
                .map(|part| lower_decoded_regex_hir_to_json_body_regex_at_start(part, context, at_start))
                .collect::<ImportResult<Vec<_>>>()?;
            format!("(?:{})", alternatives.join("|"))
        }
    })
}

fn lower_decoded_repetition_to_json_body_regex(
    repetition: &Repetition,
    context: JsonStringContext,
    at_start: bool,
) -> ImportResult<String> {
    let sub = lower_decoded_regex_hir_to_json_body_regex_at_start(&repetition.sub, context, at_start)?;
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

fn lower_decoded_class_to_json_body_regex(class: &Class, context: JsonStringContext) -> String {
    lower_decoded_class_to_json_body_regex_at_start(class, context, true)
}

fn lower_decoded_class_to_json_body_regex_at_start(
    class: &Class,
    context: JsonStringContext,
    at_start: bool,
) -> String {
    let include_unicode_escapes = !matches!(
        (json_string_compat_mode(), context),
        (JsonStringCompatMode::LlGuidanceNative, JsonStringContext::Value)
            | (JsonStringCompatMode::LlGuidanceNative, JsonStringContext::KeyStrict)
    );
    lower_decoded_class_to_json_body_regex_with_unicode_escapes(
        class,
        context,
        include_unicode_escapes,
        true,
    )
}

fn lower_decoded_class_to_json_body_regex_without_unicode_escapes(
    class: &Class,
    context: JsonStringContext,
    at_start: bool,
) -> String {
    lower_decoded_class_to_json_body_regex_with_unicode_escapes(class, context, false, at_start)
}

fn lower_decoded_class_to_json_body_regex_with_unicode_escapes(
    class: &Class,
    context: JsonStringContext,
    include_unicode_escapes: bool,
    at_start: bool,
) -> String {
    if is_unicode_decimal_digit_class(class) {
        return ascii_json_body_class_regex("[0-9]", b'0'..=b'9', context);
    }
    let contains_control = class_contains_ascii_control(class);
    let add_ascii_unicode_escape_branch = should_add_ascii_json_unicode_escape_branch(context)
        || (matches!(json_string_compat_mode(), JsonStringCompatMode::LlGuidanceNative)
            && matches!(context, JsonStringContext::Value)
            && (class_contains_pattern_whitespace(class) || contains_control)
            && !class_is_ascii_hex_subset(class)
            && (!class_contains_url_punctuation(class) || contains_control));
    if is_dot_like_unicode_class(class) {
        return json_string_body_dot_regex_in_mode(json_string_compat_mode(), context).to_string();
    }

    let mut raw_ranges = Vec::new();
    let mut ascii_unicode_escape_codes = BTreeSet::new();
    let mut alternatives = Vec::new();
    match class {
        Class::Unicode(class) => {
            for range in class.ranges() {
                let start = range.start();
                let end = range.end();
                if start <= '\x7f' {
                    let ascii_end = std::cmp::min(end, '\x7f');
                    push_safe_raw_char_ranges(start, ascii_end, &mut raw_ranges);
                    if add_ascii_unicode_escape_branch {
                        collect_ascii_json_unicode_escape_codes(
                            start,
                            ascii_end,
                            &mut ascii_unicode_escape_codes,
                        );
                    }
                }
                if end >= '\u{80}' {
                    let non_ascii_start = std::cmp::max(start, '\u{80}');
                    alternatives.push(unicode_range_to_utf8_regex_string(non_ascii_start, end));
                }
                if include_unicode_escapes
                    && let Some(escaped_range) = unicode_range_to_json_unicode_escape_regex_string(start, end)
                {
                    alternatives.push(escaped_range);
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
                    if add_ascii_unicode_escape_branch {
                        collect_ascii_json_unicode_escape_codes(
                            char::from(start),
                            char::from(ascii_end),
                            &mut ascii_unicode_escape_codes,
                        );
                    }
                }
                if end >= 128 {
                    let non_ascii_start = std::cmp::max(start, 128);
                    if non_ascii_start == end {
                        alternatives.push(format!(r#"\x{:02x}"#, non_ascii_start));
                    } else {
                        alternatives.push(format!(r#"[\x{:02x}-\x{:02x}]"#, non_ascii_start, end));
                    }
                }
                if include_unicode_escapes
                    && let Some(escaped_range) = byte_range_to_json_unicode_escape_regex_string(start, end)
                {
                    alternatives.push(escaped_range);
                }
            }
        }
    }

    if !raw_ranges.is_empty() {
        alternatives.push(format!("[{}]", raw_ranges.join("")));
    }
    if let Some(escaped_ascii) = json_unicode_escape_regex_for_ascii_codes(&ascii_unicode_escape_codes) {
        alternatives.push(escaped_ascii);
    }
    for ch in [
        '"', '/', '\\', '\n', '\r', '\t', '\u{08}', '\u{0c}', '\u{85}', '\u{a0}', '\u{1680}',
        '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
        '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
        '\u{205f}', '\u{3000}',
    ] {
        if decoded_class_contains(class, ch) {
            alternatives.push(json_body_char_regex_for_decoded_char_in_mode(
                ch,
                json_string_compat_mode(),
                context,
            ));
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

fn lower_llguidance_value_pattern_class_expr(class: &Class, at_start: bool) -> Option<GrammarExpr> {
    let raw_regex = lower_decoded_class_to_json_body_regex_without_unicode_escapes(
        class,
        JsonStringContext::Value,
        at_start,
    );
    if raw_regex == r"[^\s\S]" {
        None
    } else {
        Some(GrammarExpr::RawRegex(raw_regex))
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

fn unicode_range_to_json_unicode_escape_regex_string(start: char, end: char) -> Option<String> {
    let start = u32::from(start);
    let end = u32::from(end).min(0xFFFF);
    if start > end {
        return None;
    }
    Some(json_unicode_escape_range_regex_string(start as u16, end as u16))
}

fn byte_range_to_json_unicode_escape_regex_string(start: u8, end: u8) -> Option<String> {
    if start > end {
        return None;
    }
    Some(json_unicode_escape_range_regex_string(start as u16, end as u16))
}

fn json_unicode_escape_range_regex_string(start: u16, end: u16) -> String {
    let mut patterns = Vec::new();
    push_hex_range_patterns(start, end, 0, &mut patterns);
    let prefixed = patterns
        .into_iter()
        .map(|pattern| format!(r#"\\u{}"#, pattern))
        .collect::<Vec<_>>();
    match prefixed.len() {
        0 => r#"[^\s\S]"#.to_string(),
        1 => prefixed.into_iter().next().unwrap(),
        _ => format!(r#"(?:{})"#, prefixed.join("|")),
    }
}

fn push_hex_range_patterns(start: u16, end: u16, digit_index: usize, output: &mut Vec<String>) {
    if start > end {
        return;
    }
    if digit_index == 4 {
        output.push(String::new());
        return;
    }

    let shift = 4 * (3 - digit_index);
    let low_digit = ((start >> shift) & 0xF) as u8;
    let high_digit = ((end >> shift) & 0xF) as u8;
    let suffix_mask = if shift == 0 { 0 } else { (1u16 << shift) - 1 };

    if low_digit == high_digit {
        let mut suffixes = Vec::new();
        push_hex_range_patterns(start, end, digit_index + 1, &mut suffixes);
        for suffix in suffixes {
            output.push(format!("{}{}", hex_digit_exact_regex(low_digit), suffix));
        }
        return;
    }

    let low_branch_end = start | suffix_mask;
    let mut low_suffixes = Vec::new();
    push_hex_range_patterns(start, low_branch_end, digit_index + 1, &mut low_suffixes);
    for suffix in low_suffixes {
        output.push(format!("{}{}", hex_digit_exact_regex(low_digit), suffix));
    }

    if low_digit + 1 < high_digit {
        output.push(format!(
            "{}{}",
            hex_digit_range_regex(low_digit + 1, high_digit - 1),
            any_hex_suffix_regex(3 - digit_index)
        ));
    }

    let high_branch_start = end & !suffix_mask;
    let mut high_suffixes = Vec::new();
    push_hex_range_patterns(high_branch_start, end, digit_index + 1, &mut high_suffixes);
    for suffix in high_suffixes {
        output.push(format!("{}{}", hex_digit_exact_regex(high_digit), suffix));
    }
}

fn any_hex_suffix_regex(digits: usize) -> String {
    match digits {
        0 => String::new(),
        1 => String::from("[0-9A-Fa-f]"),
        _ => format!("[0-9A-Fa-f]{{{digits}}}"),
    }
}

fn hex_digit_exact_regex(digit: u8) -> String {
    match digit {
        0..=9 => char::from(b'0' + digit).to_string(),
        10..=15 => {
            let upper = char::from(b'A' + (digit - 10));
            let lower = char::from(b'a' + (digit - 10));
            format!("[{}{lower}]", upper)
        }
        _ => unreachable!("hex digit out of range"),
    }
}

fn hex_digit_range_regex(start: u8, end: u8) -> String {
    let parts = (start..=end)
        .map(hex_digit_exact_regex)
        .collect::<Vec<_>>();
    match parts.len() {
        0 => String::new(),
        1 => parts.into_iter().next().unwrap(),
        _ => format!("(?:{})", parts.join("|")),
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

fn is_unicode_non_decimal_digit_class(class: &Class) -> bool {
    ['0', '1', '9'].into_iter().all(|ch| !decoded_class_contains(class, ch))
        && ['A', '_', '-', 'π', '中', '😀']
            .into_iter()
            .all(|ch| decoded_class_contains(class, ch))
}

const PATTERN_WHITESPACE_CHARS: &[char] = &[
    ' ', '\n', '\r', '\t', '\u{0c}', '\u{85}', '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}',
    '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}',
    '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}',
];

fn class_is_ascii_digit_only(class: &Class) -> bool {
    (0x30u8..=0x39u8).all(|byte| decoded_class_contains(class, char::from(byte)))
        && [0x2fu8, 0x3au8, b'A', b'a', b'_']
            .into_iter()
            .all(|byte| !decoded_class_contains(class, char::from(byte)))
}

fn class_is_ascii_hex_digit_only(class: &Class) -> bool {
    (0x30u8..=0x39u8).all(|byte| decoded_class_contains(class, char::from(byte)))
        && (b'A'..=b'F').all(|byte| decoded_class_contains(class, char::from(byte)))
        && (b'a'..=b'f').all(|byte| decoded_class_contains(class, char::from(byte)))
        && [0x2fu8, 0x3au8, b'G', b'g', b'_']
            .into_iter()
            .all(|byte| !decoded_class_contains(class, char::from(byte)))
}

fn class_is_ascii_hex_subset(class: &Class) -> bool {
    let mut any = false;
    for byte in 0u8..=0x7f {
        if decoded_class_contains(class, char::from(byte)) {
            any = true;
            if !byte.is_ascii_hexdigit() {
                return false;
            }
        }
    }
    any
}

fn class_contains_ascii_control(class: &Class) -> bool {
    (0u8..=0x1f).any(|byte| decoded_class_contains(class, char::from(byte)))
}

fn class_contains_url_punctuation(class: &Class) -> bool {
    [b'!', b'$', b'&', b'\'', b'(', b')', b'*', b'+', b',', b';', b'=', b'@', b'~']
        .into_iter()
        .any(|byte| decoded_class_contains(class, char::from(byte)))
}

fn class_contains_pattern_whitespace(class: &Class) -> bool {
    PATTERN_WHITESPACE_CHARS
        .iter()
        .copied()
        .any(|ch| decoded_class_contains(class, ch))
}

fn is_unicode_pattern_whitespace_class(class: &Class) -> bool {
    PATTERN_WHITESPACE_CHARS.iter().copied().all(|ch| decoded_class_contains(class, ch))
        && ['A', '0', '_', '-', '/', 'π', '中', '😀']
            .into_iter()
            .all(|ch| !decoded_class_contains(class, ch))
}

fn is_unicode_pattern_non_whitespace_class(class: &Class) -> bool {
    PATTERN_WHITESPACE_CHARS.iter().copied().all(|ch| !decoded_class_contains(class, ch))
        && ['A', '0', '_', '-', '/', 'π', '中', '😀']
            .into_iter()
            .all(|ch| decoded_class_contains(class, ch))
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

fn collect_ascii_json_unicode_escape_codes(start: char, end: char, output: &mut BTreeSet<u8>) {
    for codepoint in u32::from(start)..=u32::from(end) {
        if codepoint <= 0x7f {
            output.insert(codepoint as u8);
        }
    }
}

fn should_add_ascii_json_unicode_escape_branch(_context: JsonStringContext) -> bool {
    matches!(json_string_compat_mode(), JsonStringCompatMode::JsonSchema)
}

fn ascii_json_body_class_regex(
    raw_class_regex: &str,
    codes: std::ops::RangeInclusive<u8>,
    context: JsonStringContext,
) -> String {
    if !should_add_ascii_json_unicode_escape_branch(context) {
        return raw_class_regex.to_string();
    }
    let codes = codes.collect::<BTreeSet<_>>();
    let escaped_ascii = json_unicode_escape_regex_for_ascii_codes(&codes)
        .expect("non-empty ASCII range has a unicode escape branch");
    format!("(?:{raw_class_regex}|{escaped_ascii})")
}

fn json_unicode_escape_regex_for_ascii_codes(codes: &BTreeSet<u8>) -> Option<String> {
    if codes.is_empty() {
        return None;
    }

    let mut by_high_nibble: BTreeMap<u8, BTreeSet<u8>> = BTreeMap::new();
    for code in codes.iter().copied() {
        debug_assert!(code <= 0x7f);
        by_high_nibble.entry(code >> 4).or_default().insert(code & 0x0f);
    }

    let branches = by_high_nibble
        .into_iter()
        .map(|(high, lows)| {
            format!(
                "{}{}",
                ascii_hex_digit_regex_fragment(high),
                hex_low_nibble_set_regex_fragment(&lows)
            )
        })
        .collect::<Vec<_>>();

    let body = match branches.as_slice() {
        [single] => single.clone(),
        _ => format!("(?:{})", branches.join("|")),
    };
    Some(format!(r#"\\u00{body}"#))
}

fn ascii_hex_digit_regex_fragment(nibble: u8) -> String {
    match nibble {
        0..=9 => char::from(b'0' + nibble).to_string(),
        10..=15 => {
            let upper = char::from(b'A' + (nibble - 10));
            let lower = char::from(b'a' + (nibble - 10));
            format!("[{upper}{lower}]")
        }
        _ => unreachable!("hex nibble out of range: {nibble}"),
    }
}

fn hex_low_nibble_set_regex_fragment(lows: &BTreeSet<u8>) -> String {
    if lows.len() == 1 {
        return ascii_hex_digit_regex_fragment(*lows.iter().next().unwrap());
    }

    let mut parts = Vec::new();
    let mut range_start = None;
    let mut previous = None;
    for low in lows.iter().copied() {
        if let Some(prev) = previous {
            if low != prev + 1 {
                push_hex_nibble_class_range(range_start.take().unwrap(), prev, &mut parts);
                range_start = Some(low);
            }
        }
        if range_start.is_none() {
            range_start = Some(low);
        }
        previous = Some(low);
    }
    if let (Some(start), Some(end)) = (range_start, previous) {
        push_hex_nibble_class_range(start, end, &mut parts);
    }

    format!("[{}]", parts.join(""))
}

fn push_hex_nibble_class_range(start: u8, end: u8, output: &mut Vec<String>) {
    match (start, end) {
        (0..=9, 0..=9) if start == end => output.push(char::from(b'0' + start).to_string()),
        (0..=9, 0..=9) => output.push(format!("{}-{}", char::from(b'0' + start), char::from(b'0' + end))),
        (10..=15, 10..=15) if start == end => {
            let upper = char::from(b'A' + (start - 10));
            let lower = char::from(b'a' + (start - 10));
            output.push(format!("{upper}{lower}"));
        }
        (10..=15, 10..=15) => {
            let upper_start = char::from(b'A' + (start - 10));
            let upper_end = char::from(b'A' + (end - 10));
            let lower_start = char::from(b'a' + (start - 10));
            let lower_end = char::from(b'a' + (end - 10));
            output.push(format!("{upper_start}-{upper_end}{lower_start}-{lower_end}"));
        }
        (0..=9, 10..=15) => {
            push_hex_nibble_class_range(start, 9, output);
            push_hex_nibble_class_range(10, end, output);
        }
        _ => unreachable!("hex nibble range out of range: {start}..={end}"),
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

fn prepend_literal_prefix_to_expr(prefix: Vec<u8>, expr: GrammarExpr) -> GrammarExpr {
    if prefix.is_empty() {
        return expr;
    }
    match expr {
        GrammarExpr::Literal(mut bytes) => {
            let mut merged = prefix;
            merged.append(&mut bytes);
            lit_bytes(merged)
        }
        GrammarExpr::RawRegex(regex) => GrammarExpr::RawRegex(format!(
            "{}{}",
            regex::escape(&String::from_utf8_lossy(&prefix)),
            regex
        )),
        GrammarExpr::Sequence(mut parts) => {
            if let Some(GrammarExpr::Literal(bytes)) = parts.first_mut() {
                let mut merged = prefix;
                merged.append(bytes);
                *bytes = merged;
                return seq(parts);
            }
            let mut prefixed = Vec::with_capacity(parts.len() + 1);
            prefixed.push(lit_bytes(prefix));
            prefixed.extend(parts);
            seq(prefixed)
        }
        GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
            expr: Box::new(prepend_literal_prefix_to_expr(prefix.clone(), *expr)),
            intersect: Box::new(prepend_literal_prefix_to_expr(prefix, *intersect)),
        },
        other => seq(vec![lit_bytes(prefix), other]),
    }
}

pub(crate) const GLRMASK_LLGUIDANCE_COMPAT_ENV: &str = "GLRMASK_LLGUIDANCE_COMPAT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JsonStringCompatMode {
    JsonSchema,
    LlGuidanceNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JsonStringContext {
    Value,
    // Literal and pattern property key paths (llguidance json_dumps/json_quote behavior).
    KeyStrict,
    // Generic additional/unknown key path (llguidance CHAR_REGEX behavior).
    KeyAdditional,
}

fn llguidance_compat_enabled_from_env() -> bool {
    std::env::var_os(GLRMASK_LLGUIDANCE_COMPAT_ENV).is_some_and(|value| {
        let value = value.to_string_lossy();
        !value.is_empty() && value != "0"
    })
}

fn is_test_binary() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| {
                    let name_str = name.to_string_lossy().to_lowercase();
                    name_str.contains("test")
                        || name_str.contains("integration")
                        || name_str.contains("glrmask")
                })
        })
        .unwrap_or(false)
}

static INITIAL_COMPAT_MODE: std::sync::OnceLock<JsonStringCompatMode> = std::sync::OnceLock::new();

fn initial_compat_mode() -> JsonStringCompatMode {
    if is_test_binary() {
        JsonStringCompatMode::JsonSchema
    } else {
        *INITIAL_COMPAT_MODE.get_or_init(|| {
            if llguidance_compat_enabled_from_env() {
                JsonStringCompatMode::LlGuidanceNative
            } else {
                JsonStringCompatMode::JsonSchema
            }
        })
    }
}

std::thread_local! {
    pub(crate) static TEST_COMPAT_MODE: std::cell::Cell<JsonStringCompatMode> = std::cell::Cell::new(initial_compat_mode());
}

pub(crate) fn json_string_compat_mode() -> JsonStringCompatMode {
    TEST_COMPAT_MODE.with(|cell| cell.get())
}

fn json_body_char_regex_for_decoded_char(ch: char) -> String {
    json_body_char_regex_for_decoded_char_in_mode(ch, json_string_compat_mode(), JsonStringContext::Value)
}

fn json_body_char_expr_for_decoded_char(ch: char) -> GrammarExpr {
    GrammarExpr::RawRegex(json_body_char_regex_for_decoded_char(ch))
}

fn json_body_char_expr_for_pattern_literal(ch: char) -> GrammarExpr {
    json_body_char_expr_for_pattern_literal_in_mode(
        ch,
        json_string_compat_mode(),
        JsonStringContext::Value,
        false,
    )
}

fn json_body_char_expr_for_pattern_literal_in_mode(
    ch: char,
    mode: JsonStringCompatMode,
    context: JsonStringContext,
    at_start: bool,
) -> GrammarExpr {
    let canonical = GrammarExpr::RawRegex(json_body_char_regex_for_decoded_char_in_mode(
        ch,
        mode,
        context,
    ));
    if !should_allow_json_unicode_escape_for_pattern_literal(ch) {
        return canonical;
    }
    if mode == JsonStringCompatMode::LlGuidanceNative
        && !(matches!(context, JsonStringContext::Value) && at_start)
    {
        return canonical;
    }
    choice(vec![canonical, json_unicode_escape_for_char_expr(ch)])
}

fn json_body_char_regex_for_decoded_char_in_mode(
    ch: char,
    mode: JsonStringCompatMode,
    context: JsonStringContext,
) -> String {
    let decoded = ch.to_string();
    let encoded = serde_json::to_string(&decoded).unwrap_or_else(|_| "\"\"".to_string());
    let body = encoded
        .strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
        .unwrap_or("");
    let canonical = regex::escape(body);
    if ch == '/' {
        match (mode, context) {
            (JsonStringCompatMode::JsonSchema, _) => format!(r#"(?:{}|\\/)"#, canonical),
            (JsonStringCompatMode::LlGuidanceNative, _) => canonical,
        }
    } else {
        canonical
    }
}

fn json_body_char_regex_for_pattern_literal_in_mode(
    ch: char,
    mode: JsonStringCompatMode,
    context: JsonStringContext,
    at_start: bool,
) -> String {
    let canonical = json_body_char_regex_for_decoded_char_in_mode(ch, mode, context);
    if !should_allow_json_unicode_escape_for_pattern_literal(ch) {
        return canonical;
    }
    if mode == JsonStringCompatMode::LlGuidanceNative
        && !(matches!(context, JsonStringContext::Value) && at_start)
    {
        return canonical;
    }
    let escaped = json_unicode_escape_prefix_regex_for_char(ch);
    format!(r#"(?:{}|{})"#, canonical, escaped)
}

fn should_allow_json_unicode_escape_for_pattern_literal(ch: char) -> bool {
    u32::from(ch) <= 0xFFFF && (is_safe_raw_json_string_char(ch) || u32::from(ch) >= 0x80)
}

fn json_unicode_escape_for_char_regex(ch: char) -> String {
    let codepoint = u32::from(ch);
    format!(
        r#"\\u{}{}{}{}"#,
        hex_digit_exact_regex(((codepoint >> 12) & 0xF) as u8),
        hex_digit_exact_regex(((codepoint >> 8) & 0xF) as u8),
        hex_digit_exact_regex(((codepoint >> 4) & 0xF) as u8),
        hex_digit_exact_regex((codepoint & 0xF) as u8),
    )
}

fn json_unicode_escape_prefix_regex_for_char(ch: char) -> String {
    let codepoint = u32::from(ch);
    let d0 = hex_digit_exact_regex(((codepoint >> 12) & 0xF) as u8);
    let d1 = hex_digit_exact_regex(((codepoint >> 8) & 0xF) as u8);
    let d2 = hex_digit_exact_regex(((codepoint >> 4) & 0xF) as u8);
    let d3 = hex_digit_exact_regex((codepoint & 0xF) as u8);
    format!(r#"\\u(?:{d0}(?:{d1}(?:{d2}(?:{d3})?)?)?)?"#)
}

fn json_unicode_escape_for_char_expr(ch: char) -> GrammarExpr {
    let codepoint = u32::from(ch);
    seq(vec![
        lit("\\u"),
        GrammarExpr::RawRegex(hex_digit_exact_regex(((codepoint >> 12) & 0xF) as u8)),
        GrammarExpr::RawRegex(hex_digit_exact_regex(((codepoint >> 8) & 0xF) as u8)),
        GrammarExpr::RawRegex(hex_digit_exact_regex(((codepoint >> 4) & 0xF) as u8)),
        GrammarExpr::RawRegex(hex_digit_exact_regex((codepoint & 0xF) as u8)),
    ])
}

fn json_unicode_escape_regex_for_chars(chars: &[char]) -> String {
    let parts = chars
        .iter()
        .copied()
        .filter(|ch| u32::from(*ch) <= 0xFFFF)
        .map(json_unicode_escape_for_char_regex)
        .collect::<Vec<_>>();
    match parts.len() {
        0 => r#"[^\s\S]"#.to_string(),
        1 => parts.into_iter().next().unwrap(),
        _ => format!(r#"(?:{})"#, parts.join("|")),
    }
}

fn json_unicode_escape_regex_for_bmp_non_whitespace() -> String {
    let mut excluded = PATTERN_WHITESPACE_CHARS
        .iter()
        .copied()
        .filter_map(|ch| u16::try_from(u32::from(ch)).ok())
        .collect::<BTreeSet<_>>();
    for surrogate in 0xD800u16..=0xDFFFu16 {
        excluded.insert(surrogate);
    }

    let mut ranges = Vec::new();
    let mut range_start: Option<u16> = None;
    let mut previous: Option<u16> = None;
    for codepoint in 0u16..=0xFFFFu16 {
        if excluded.contains(&codepoint) {
            if let (Some(start), Some(end)) = (range_start.take(), previous.take()) {
                ranges.push(json_unicode_escape_range_regex_string(start, end));
            }
            continue;
        }
        if range_start.is_none() {
            range_start = Some(codepoint);
        }
        previous = Some(codepoint);
    }
    if let (Some(start), Some(end)) = (range_start, previous) {
        ranges.push(json_unicode_escape_range_regex_string(start, end));
    }

    match ranges.len() {
        0 => r#"[^\s\S]"#.to_string(),
        1 => ranges.into_iter().next().unwrap(),
        _ => format!(r#"(?:{})"#, ranges.join("|")),
    }
}

pub(crate) fn json_string_body_char_regex() -> &'static str {
    json_string_body_char_regex_in_mode(json_string_compat_mode(), JsonStringContext::Value)
}

pub(crate) fn json_string_body_char_regex_in_mode(
    mode: JsonStringCompatMode,
    context: JsonStringContext,
) -> &'static str {
    match (mode, context) {
        (JsonStringCompatMode::JsonSchema, _) => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["/\\bfnrt]|\\u[0-9A-Fa-f]{4})"#
        }
        (JsonStringCompatMode::LlGuidanceNative, JsonStringContext::KeyAdditional) => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["/\\bfnrt]|\\u(?:[0-9A-Fa-f]{0,3})?$)"#
        }
        (JsonStringCompatMode::LlGuidanceNative, JsonStringContext::KeyStrict) => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfnrt]|\\u00(?:[01][0-9A-Fa-f]|7[Ff]))"#
        }
        (JsonStringCompatMode::LlGuidanceNative, JsonStringContext::Value) => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfnrt]|\\u00(?:[01][0-9A-Fa-f]|7[Ff]))"#
        }
    }
}

fn json_string_body_non_ascii_non_whitespace_regex() -> &'static str {
    r#"(?:\xC2[\x80-\x84\x86-\x9F\xA1-\xBF]|[\xC3-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|\xE1[\x80-\x99\x9B-\xBF][\x80-\xBF]|\xE1\x9A[\x81-\xBF]|\xE2\x80[\x8B-\xA7\xAA-\xAE\xB0-\xBF]|\xE2\x81[\x80-\x9E\xA0-\xBF]|\xE2[\x82-\xBF][\x80-\xBF]|\xE3\x80[\x81-\xBF]|\xE3[\x81-\xBF][\x80-\xBF]|[\xE4-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2})"#
}

fn json_string_body_dot_regex() -> &'static str {
    json_string_body_dot_regex_in_mode(json_string_compat_mode(), JsonStringContext::Value)
}

pub(crate) fn json_string_body_dot_regex_in_mode(
    mode: JsonStringCompatMode,
    context: JsonStringContext,
) -> &'static str {
    match (mode, context) {
        (JsonStringCompatMode::JsonSchema, _) => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["/\\bfrt]|\\u(?:[1-9A-Fa-f][0-9A-Fa-f]{3}|0[1-9A-Fa-f][0-9A-Fa-f]{2}|00[1-9A-Fa-f][0-9A-Fa-f]|000[0-9B-Fb-f]))"#
        }
        (JsonStringCompatMode::LlGuidanceNative, JsonStringContext::KeyAdditional) => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["/\\bft]|\\u(?:[0-9A-Fa-f]{0,3})?$)"#
        }
        (JsonStringCompatMode::LlGuidanceNative, JsonStringContext::KeyStrict) => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfrt]|\\u00(?:0[0-9B-Fb-f]|1[0-9A-Fa-f]|7[Ff]))"#
        }
        (JsonStringCompatMode::LlGuidanceNative, JsonStringContext::Value) => {
            r#"(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfrt]|\\u00(?:0[0-9B-Fb-f]|1[0-9A-Fa-f]|7[Ff]))"#
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

    use super::{
        preprocess_ascii_shorthand, quoted_string_body_regex, string_pattern_as_body_regex,
        JsonStringCompatMode, JsonStringContext, TEST_COMPAT_MODE,
    };

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
        let body = string_pattern_as_body_regex(r"^$|(^(?:\S+\s+){0,19}\S+$)", JsonStringContext::Value).unwrap();
        let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

        assert!(regex.is_match(r#""REST API""#));
        assert!(!regex.is_match(r#"" /""#));
    }

    #[test]
    fn lowered_optional_decimal_pattern_rejects_backslash_digit_string() {
        let body = string_pattern_as_body_regex(r"^$|^\d{1,15}(?:\.\d{1,5})?$", JsonStringContext::Value).unwrap();
        let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

        assert!(regex.is_match(r#""""#));
        assert!(regex.is_match(r#""123.45""#));
        assert!(!regex.is_match(r#""\\1""#));
    }

    #[test]
    fn llguidance_value_pattern_terminal_regex_classes_add_ascii_unicode_escape_spellings() {
        TEST_COMPAT_MODE.with(|cell| cell.set(JsonStringCompatMode::LlGuidanceNative));

        let body = string_pattern_as_body_regex(r"^[0-9a-f]{8}$", JsonStringContext::Value).unwrap();
        assert!(body.contains(r"\u00"), "{body}");
        let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

        assert!(regex.is_match(r#""1234abcd""#));
        assert!(regex.is_match(r#""\u0031234abcd""#));
        assert!(!regex.is_match(r#""\uC1234abcd""#));
    }

    #[test]
    fn llguidance_value_pattern_whitespace_class_rejects_unicode_escape_spelling() {
        TEST_COMPAT_MODE.with(|cell| cell.set(JsonStringCompatMode::LlGuidanceNative));

        let body = string_pattern_as_body_regex(r"^\s$", JsonStringContext::Value).unwrap();
        let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

        assert!(regex.is_match(r#""\t""#));
        assert!(!regex.is_match(r#""\u0009""#));
    }

}
