use std::collections::BTreeSet;

use crate::import::ast::GrammarExpr;
use crate::import::numeric_range::{rx_float_range, rx_int_range};

use super::ast::NumberSchema;
use super::error::{ImportResult, SchemaImportError};
use super::lower::{choice, lit_bytes, never, r, Lowerer, JSON_INTEGER_RULE, JSON_NUMBER_RULE};

const MAX_EXPLICIT_INTEGER_RANGE: i64 = 512;
const MAX_EXPLICIT_INTEGER_MULTIPLES: i64 = 2048;

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_number(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {
        if schema.integer {
            return self.lower_integer(schema);
        }

        let range_expr = if schema.minimum.is_some() || schema.maximum.is_some() {
            Some(GrammarExpr::RawRegex(
                rx_float_range(
                    schema.minimum,
                    schema.maximum,
                    !schema.exclusive_minimum,
                    !schema.exclusive_maximum,
                )
                .map_err(SchemaImportError::new)?,
            ))
        } else {
            None
        };

        let base_expr = if let Some(multiple) = schema.multiple_of {
            if let Some(regex) = power_of_ten_multiple_regex(multiple) {
                GrammarExpr::RawRegex(regex)
            } else if let Some(regex) = decimal_multiple_regex(multiple) {
                GrammarExpr::RawRegex(regex)
            } else {
                return Err(SchemaImportError::new(format!(
                    "multipleOf={multiple} for non-integer numbers is unsupported in the simple importer"
                )));
            }
        } else if let Some(range_expr) = range_expr.clone() {
            return Ok(range_expr);
        } else {
            r(JSON_NUMBER_RULE)
        };

        if let Some(range_expr) = range_expr {
            return Ok(GrammarExpr::Intersect {
                expr: Box::new(base_expr),
                intersect: Box::new(range_expr),
            });
        }

        Ok(base_expr)
    }

    fn lower_integer(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {
        let lower = integer_lower_bound(schema);
        let upper = integer_upper_bound(schema);
        if let (Some(lower), Some(upper)) = (lower, upper) {
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
            if let Some(expr) = bounded_integer_multiple_choice(lower, upper, schema.multiple_of) {
                return Ok(expr);
            }
        }
        if schema.multiple_of.is_none() && (lower.is_some() || upper.is_some()) {
            let regex = rx_int_range(lower, upper).map_err(SchemaImportError::new)?;
            return Ok(GrammarExpr::RawRegex(regex));
        }

        if let Some(multiple) = schema.multiple_of {
            if let Some(expr) = integer_multiple_expr(multiple) {
                if lower.is_some() || upper.is_some() {
                    let range_regex = rx_int_range(lower, upper).map_err(SchemaImportError::new)?;
                    return Ok(GrammarExpr::Intersect {
                        expr: Box::new(expr),
                        intersect: Box::new(GrammarExpr::RawRegex(range_regex)),
                    });
                }
                return Ok(expr);
            }
            if positive_integer_multiple_value(multiple).is_some() {
                if lower.is_some() || upper.is_some() {
                    let regex = rx_int_range(lower, upper).map_err(SchemaImportError::new)?;
                    return Ok(GrammarExpr::RawRegex(regex));
                }
                return Ok(r(JSON_INTEGER_RULE));
            }
            return Err(SchemaImportError::new(format!("integer multipleOf={multiple} is unsupported")));
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

fn bounded_integer_multiple_choice(
    lower: i64,
    upper: i64,
    multiple: Option<f64>,
) -> Option<GrammarExpr> {
    let multiple = positive_integer_multiple_i64(multiple?)?;
    let first = ceil_div_i64(lower, multiple).checked_mul(multiple)?;
    if first > upper {
        return Some(never());
    }
    let count = ((upper - first) / multiple) + 1;
    if count > MAX_EXPLICIT_INTEGER_MULTIPLES {
        return None;
    }
    let alternatives = (0..count)
        .map(|index| {
            let value = first + index * multiple;
            lit_bytes(value.to_string().into_bytes())
        })
        .collect::<Vec<_>>();
    Some(choice(alternatives))
}

fn ceil_div_i64(value: i64, divisor: i64) -> i64 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder > 0 { quotient + 1 } else { quotient }
}

fn integer_multiple_expr(multiple: f64) -> Option<GrammarExpr> {
    power_of_ten_multiple_regex(multiple).map(GrammarExpr::RawRegex)
}

fn positive_integer_multiple_value(multiple: f64) -> Option<u64> {
    if !multiple.is_finite() || multiple < 1.0 || multiple.fract() != 0.0 {
        return None;
    }
    let value = multiple as u64;
    if (value as f64) == multiple { Some(value) } else { None }
}

fn positive_integer_multiple_i64(multiple: f64) -> Option<i64> {
    let value = positive_integer_multiple_value(multiple)?;
    i64::try_from(value).ok()
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
