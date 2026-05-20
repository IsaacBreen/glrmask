use std::collections::BTreeSet;

use crate::import::ast::GrammarExpr;

use super::ast::StringSchema;
use super::error::ImportResult;
use super::lower::{choice, lit, lit_bytes, never, r, seq, Lowerer, JSON_STRING_CHAR_RULE, JSON_STRING_RULE, KEY_VALUE_SEPARATOR};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_string(&mut self, schema: &StringSchema) -> ImportResult<GrammarExpr> {
        if schema.max_length.is_some_and(|max| max < schema.min_length) {
            return Ok(never());
        }

        if schema.min_length == 0 && schema.max_length.is_none() && schema.pattern.is_none() {
            return Ok(r(JSON_STRING_RULE));
        }

        let mut body = self.string_body_for_length(schema.min_length, schema.max_length);
        if let Some(pattern) = &schema.pattern {
            body = GrammarExpr::Intersect {
                expr: Box::new(body),
                intersect: Box::new(GrammarExpr::RawRegex(string_pattern_as_body_regex(pattern))),
            };
        }

        Ok(seq(vec![lit("\""), body, lit("\"")]))
    }

    pub(crate) fn lower_string_literal(&mut self, text: &str) -> GrammarExpr {
        let encoded = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string());
        lit_bytes(encoded.into_bytes())
    }

    pub(crate) fn lower_literal_key_colon(&mut self, key: &str) -> GrammarExpr {
        let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
        lit_bytes(format!("{encoded}{KEY_VALUE_SEPARATOR}").into_bytes())
    }

    pub(crate) fn lower_pattern_key_colon(&mut self, pattern: &str) -> ImportResult<GrammarExpr> {
        let key = self.lower_string(&StringSchema {
            min_length: 0,
            max_length: None,
            pattern: Some(pattern.to_string()),
            format: None,
        })?;
        Ok(seq(vec![key, lit(KEY_VALUE_SEPARATOR)]))
    }

    pub(crate) fn lower_additional_key_colon(&mut self, fixed_keys: &BTreeSet<String>) -> GrammarExpr {
        let key = if fixed_keys.is_empty() {
            r(JSON_STRING_RULE)
        } else {
            let excluded = fixed_keys
                .iter()
                .map(|key| {
                    let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
                    lit_bytes(encoded.into_bytes())
                })
                .collect::<Vec<_>>();
            GrammarExpr::Exclude {
                expr: Box::new(r(JSON_STRING_RULE)),
                exclude: Box::new(choice(excluded)),
            }
        };
        seq(vec![key, lit(KEY_VALUE_SEPARATOR)])
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
        stripped.to_string()
    } else {
        format!(".*({pattern}).*")
    }
}

fn strip_simple_anchors(pattern: &str) -> Option<&str> {
    let without_start = pattern.strip_prefix('^')?;
    without_start.strip_suffix('$')
}
