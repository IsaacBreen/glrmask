use std::collections::BTreeSet;

use crate::import::ast::GrammarExpr;

use super::ast::NumberSchema;
use super::error::{ImportResult, SchemaImportError};
use super::lower::{choice, lit_bytes, never, r, Lowerer, JSON_INTEGER_RULE, JSON_NUMBER_RULE};

const MAX_EXPLICIT_INTEGER_RANGE: i64 = 512;

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_number(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {
        if schema.integer {
            return self.lower_integer(schema);
        }
        if let Some(multiple) = schema.multiple_of {
            if let Some(regex) = power_of_ten_multiple_regex(multiple) {
                return Ok(GrammarExpr::RawRegex(regex));
            }
            if let Some(regex) = decimal_multiple_regex(multiple) {
                return Ok(GrammarExpr::RawRegex(regex));
            }
            return Err(SchemaImportError::new(
                format!(
                    "multipleOf={multiple} for non-integer numbers is unsupported in the simple importer"
                ),
            ));
        }
        Ok(r(JSON_NUMBER_RULE))
    }

    fn lower_integer(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {
        if let (Some(lower), Some(upper)) = (integer_lower_bound(schema), integer_upper_bound(schema)) {
            if lower > upper {
                return Ok(never());
            }
            if upper.saturating_sub(lower) <= MAX_EXPLICIT_INTEGER_RANGE {
                let alternatives = (lower..=upper)
                    .filter(|value| integer_satisfies_multiple(*value, schema.multiple_of))
                    .map(|value| lit_bytes(value.to_string().into_bytes()))
                    .collect::<Vec<_>>();
                return Ok(choice(alternatives));
            }
        }

        if let Some(multiple) = schema.multiple_of {
            if let Some(regex) = power_of_ten_multiple_regex(multiple) {
                return Ok(GrammarExpr::RawRegex(regex));
            }
            return Err(SchemaImportError::new(format!(
                "integer multipleOf={multiple} is unsupported without a small finite integer range"
            )));
        }

        Ok(r(JSON_INTEGER_RULE))
    }
}

fn integer_lower_bound(schema: &NumberSchema) -> Option<i64> {
    let value = schema.minimum?;
    if !value.is_finite() {
        return None;
    }
    let mut lower = value.ceil() as i64;
    if schema.exclusive_minimum && (lower as f64) <= value {
        lower += 1;
    }
    Some(lower)
}

fn integer_upper_bound(schema: &NumberSchema) -> Option<i64> {
    let value = schema.maximum?;
    if !value.is_finite() {
        return None;
    }
    let mut upper = value.floor() as i64;
    if schema.exclusive_maximum && (upper as f64) >= value {
        upper -= 1;
    }
    Some(upper)
}

fn integer_satisfies_multiple(value: i64, multiple: Option<f64>) -> bool {
    let Some(multiple) = multiple else {
        return true;
    };
    let quotient = (value as f64) / multiple;
    (quotient - quotient.round()).abs() < 1e-9
}

fn power_of_ten_multiple_regex(multiple: f64) -> Option<String> {
    if !multiple.is_finite() || multiple < 1.0 || multiple.fract() != 0.0 {
        return None;
    }
    let mut value = multiple as u64;
    if value == 1 {
        return Some(r#"-?(0|[1-9][0-9]*)"#.to_string());
    }

    let mut zeros = 0usize;
    while value > 1 && value % 10 == 0 {
        zeros += 1;
        value /= 10;
    }
    if value != 1 {
        return None;
    }

    Some(format!(r#"-?(0|[1-9][0-9]*{})"#, "0".repeat(zeros)))
}

fn decimal_multiple_regex(multiple: f64) -> Option<String> {
    let step = parse_decimal_step(multiple)?;
    let fraction = decimal_fraction_regex(&step)?;
    Some(format!(r#"-?(0|[1-9][0-9]*)(?:\.(?:{fraction}))?"#))
}

struct DecimalStep {
    numerator: u64,
    scale: u64,
    scale_digits: usize,
}

fn parse_decimal_step(multiple: f64) -> Option<DecimalStep> {
    if !multiple.is_finite() || multiple <= 0.0 || multiple.fract() == 0.0 {
        return None;
    }

    let text = format!("{multiple}");
    if text.contains(['e', 'E']) {
        return None;
    }

    let (integer_part, fractional_part) = text.split_once('.')?;
    if integer_part != "0" || fractional_part.is_empty() || !fractional_part.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }

    let scale_digits = fractional_part.len();
    let scale = 10u64.checked_pow(scale_digits as u32)?;
    let numerator = fractional_part.parse::<u64>().ok()?;
    if numerator == 0 || numerator >= scale {
        return None;
    }

    Some(DecimalStep {
        numerator,
        scale,
        scale_digits,
    })
}

fn decimal_fraction_regex(step: &DecimalStep) -> Option<String> {
    if step.scale % step.numerator != 0 {
        return None;
    }

    if step.numerator == 1 {
        if step.scale_digits == 1 {
            return Some(r#"[0-9]0*"#.to_string());
        }
        return Some(format!(r#"(?:[0-9]{{1,{}}}|[0-9]{{{}}}0*)"#, step.scale_digits - 1, step.scale_digits));
    }

    if step.scale_digits > 3 {
        return None;
    }

    let mut prefixes = BTreeSet::new();
    let mut value = 0u64;
    while value < step.scale {
        let full = format!("{:0width$}", value, width = step.scale_digits);
        let prefix = full.trim_end_matches('0');
        prefixes.insert(if prefix.is_empty() { "0".to_string() } else { prefix.to_string() });
        value = value.checked_add(step.numerator)?;
    }

    let parts = prefixes
        .into_iter()
        .map(|prefix| {
            if prefix == "0" {
                "0+".to_string()
            } else {
                format!("{prefix}0*")
            }
        })
        .collect::<Vec<_>>();
    Some(parts.join("|"))
}
