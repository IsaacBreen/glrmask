//! Numeric range regex generators for JSON schema constraints.
//!
//! Ported from llguidance's `parser/src/json/numeric.rs` (guidance-ai/llguidance).
//! Produces regex strings matching integers or floats within specified ranges.

type Result<T> = std::result::Result<T, String>;

fn mk_or(parts: Vec<String>) -> String {
    if parts.len() == 1 {
        parts[0].clone()
    } else {
        format!("({})", parts.join("|"))
    }
}

fn mk_or_opt(parts: Vec<String>) -> Option<String> {
    if parts.is_empty() {
        None
    } else {
        Some(mk_or(parts))
    }
}

fn num_digits(n: i64) -> usize {
    n.abs().to_string().len()
}

// ---------------------------------------------------------------------------
// Integer range
// ---------------------------------------------------------------------------

pub fn rx_int_range(left: Option<i64>, right: Option<i64>) -> Result<String> {
    match (left, right) {
        (None, None) => Ok("-?(0|[1-9][0-9]*)".to_string()),
        (Some(left), None) => {
            if left < 0 {
                Ok(mk_or(vec![
                    rx_int_range(Some(left), Some(-1))?,
                    rx_int_range(Some(0), None)?,
                ]))
            } else {
                let max_value: i64 = "9"
                    .repeat(num_digits(left))
                    .parse()
                    .map_err(|e| format!("Failed to parse max value for left {}: {}", left, e))?;
                Ok(mk_or(vec![
                    rx_int_range(Some(left), Some(max_value))?,
                    format!("[1-9][0-9]{{{},}}", num_digits(left)),
                ]))
            }
        }
        (None, Some(right)) => {
            if right >= 0 {
                Ok(mk_or(vec![
                    rx_int_range(Some(0), Some(right))?,
                    rx_int_range(None, Some(-1))?,
                ]))
            } else {
                Ok(format!("-{}", rx_int_range(Some(-right), None)?))
            }
        }
        (Some(left), Some(right)) => {
            if left > right {
                return Err(format!(
                    "Invalid range: left ({}) > right ({})",
                    left,
                    right
                ));
            }
            if left < 0 {
                if right < 0 {
                    Ok(format!("(-{})", rx_int_range(Some(-right), Some(-left))?))
                } else {
                    Ok(format!(
                        "(-{}|{})",
                        rx_int_range(Some(0), Some(-left))?,
                        rx_int_range(Some(0), Some(right))?
                    ))
                }
            } else if num_digits(left) == num_digits(right) {
                let l = left.to_string();
                let r = right.to_string();
                if left == right {
                    return Ok(format!("({l})"));
                }

                let lpref = &l[..l.len() - 1];
                let lx = &l[l.len() - 1..];
                let rpref = &r[..r.len() - 1];
                let rx = &r[r.len() - 1..];

                if lpref == rpref {
                    return Ok(format!("({lpref}[{lx}-{rx}])"));
                }

                let mut left_rec = lpref.parse::<i64>().unwrap_or(0);
                let mut right_rec = rpref.parse::<i64>().unwrap_or(0);

                let mut parts = Vec::new();

                if lx != "0" {
                    left_rec += 1;
                    parts.push(format!("{lpref}[{lx}-9]"));
                }

                if rx != "9" {
                    right_rec -= 1;
                    parts.push(format!("{rpref}[0-{rx}]"));
                }

                if left_rec <= right_rec {
                    let inner = rx_int_range(Some(left_rec), Some(right_rec))?;
                    parts.push(format!("{inner}[0-9]"));
                }

                Ok(mk_or(parts))
            } else {
                let break_point = 10_i64
                    .checked_pow(num_digits(left) as u32)
                    .ok_or_else(|| format!("Overflow when calculating break point"))?
                    - 1;
                Ok(mk_or(vec![
                    rx_int_range(Some(left), Some(break_point))?,
                    rx_int_range(Some(break_point + 1), Some(right))?,
                ]))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Float range helpers (lexicographic fractional part comparisons)
// ---------------------------------------------------------------------------

fn lexi_x_to_9(x: &str, incl: bool) -> Result<String> {
    if incl {
        if x.is_empty() {
            Ok("[0-9]*".to_string())
        } else if x.len() == 1 {
            Ok(format!("[{x}-9][0-9]*"))
        } else {
            let x0 = x.chars().next().unwrap().to_digit(10).unwrap();
            let x_rest = &x[1..];
            let mut parts = vec![format!(
                "{}{}",
                x.chars().next().unwrap(),
                lexi_x_to_9(x_rest, incl)?
            )];
            if x0 < 9 {
                parts.push(format!("[{}-9][0-9]*", x0 + 1));
            }
            Ok(mk_or(parts))
        }
    } else if x.is_empty() {
        Ok("[0-9]*[1-9]".to_string())
    } else {
        let x0 = x.chars().next().unwrap().to_digit(10).unwrap();
        let x_rest = &x[1..];
        let mut parts = vec![format!(
            "{}{}",
            x.chars().next().unwrap(),
            lexi_x_to_9(x_rest, incl)?
        )];
        if x0 < 9 {
            parts.push(format!("[{}-9][0-9]*", x0 + 1));
        }
        Ok(mk_or(parts))
    }
}

fn lexi_0_to_x(x: &str, incl: bool) -> Result<String> {
    if x.is_empty() {
        if incl {
            Ok("".to_string())
        } else {
            Err(format!("Inclusive flag must be true for an empty string"))
        }
    } else {
        let x0 = x.chars().next().unwrap().to_digit(10).unwrap();
        let x_rest = &x[1..];

        if !incl && x.len() == 1 {
            if x0 == 0 {
                return Err(format!(
                    "x0 must be greater than 0 for non-inclusive single character"
                ));
            }
            return Ok(format!("[0-{}][0-9]*", x0 - 1));
        }

        let mut parts = vec![format!(
            "{}{}",
            x.chars().next().unwrap(),
            lexi_0_to_x(x_rest, incl)?
        )];
        if x0 > 0 {
            parts.push(format!("[0-{}][0-9]*", x0 - 1));
        }
        Ok(mk_or(parts))
    }
}

fn lexi_range(ld: &str, rd: &str, ld_incl: bool, rd_incl: bool) -> Result<String> {
    if ld.len() != rd.len() {
        return Err(format!("ld and rd must have the same length"));
    }
    if ld == rd {
        if ld_incl && rd_incl {
            Ok(ld.to_string())
        } else {
            Err(format!(
                "Empty range when ld equals rd and not both inclusive"
            ))
        }
    } else {
        let l0 = ld.chars().next().unwrap().to_digit(10).unwrap();
        let r0 = rd.chars().next().unwrap().to_digit(10).unwrap();
        if l0 == r0 {
            let ld_rest = &ld[1..];
            let rd_rest = &rd[1..];
            Ok(format!(
                "{}{}",
                ld.chars().next().unwrap(),
                lexi_range(ld_rest, rd_rest, ld_incl, rd_incl)?
            ))
        } else {
            let ld_rest = ld[1..].trim_end_matches('0');
            let mut parts = vec![format!(
                "{}{}",
                ld.chars().next().unwrap(),
                lexi_x_to_9(ld_rest, ld_incl)?
            )];
            if l0 + 1 < r0 {
                parts.push(format!("[{}-{}][0-9]*", l0 + 1, r0 - 1));
            }
            let rd_rest = rd[1..].trim_end_matches('0');
            if !rd_rest.is_empty() || rd_incl {
                parts.push(format!(
                    "{}{}",
                    rd.chars().next().unwrap(),
                    lexi_0_to_x(rd_rest, rd_incl)?
                ));
            }
            Ok(mk_or(parts))
        }
    }
}

fn float_to_str(f: f64) -> String {
    format!("{f}")
}

/// Escape a float string for use in a regex (only `.` needs escaping).
fn escape_float_str(s: &str) -> String {
    s.replace('.', "\\.")
}

struct NonnegativeDecimalBounds {
    left_integer: i64,
    right_integer: i64,
    left_fraction: String,
    right_fraction: String,
}

fn decimal_fraction(rendered: &str) -> String {
    rendered.split('.').nth(1).unwrap_or("").to_string()
}

fn parse_decimal_integer(rendered: &str, label: &str) -> Result<i64> {
    rendered
        .split('.')
        .next()
        .unwrap_or("")
        .parse()
        .map_err(|e| format!("Failed to parse {label} integer part: {e}"))
}

fn nonnegative_decimal_bounds(left: f64, right: f64) -> Result<NonnegativeDecimalBounds> {
    if !left.is_finite() || !right.is_finite() {
        return Err("Infinite numbers not supported".to_string());
    }

    let left_text = float_to_str(left);
    let right_text = float_to_str(right);
    if left_text == right_text {
        return Err("Unexpected equality of left and right string representations".to_string());
    }

    Ok(NonnegativeDecimalBounds {
        left_integer: parse_decimal_integer(&left_text, "left")?,
        right_integer: parse_decimal_integer(&right_text, "right")?,
        left_fraction: decimal_fraction(&left_text),
        right_fraction: decimal_fraction(&right_text),
    })
}

fn pad_decimal_fractions(left_fraction: &mut String, right_fraction: &mut String) {
    while left_fraction.len() < right_fraction.len() {
        left_fraction.push('0');
    }
    while right_fraction.len() < left_fraction.len() {
        right_fraction.push('0');
    }
}

// ---------------------------------------------------------------------------
// Float range
// ---------------------------------------------------------------------------

pub fn rx_float_range(
    left: Option<f64>,
    right: Option<f64>,
    left_inclusive: bool,
    right_inclusive: bool,
) -> Result<String> {
    match (left, right) {
        (None, None) => Ok("-?(0|[1-9][0-9]*)(\\.[0-9]+)?".to_string()),
        (Some(left), None) => {
            if left < 0.0 {
                Ok(mk_or(vec![
                    rx_float_range(Some(left), Some(0.0), left_inclusive, false)?,
                    rx_float_range(Some(0.0), None, true, false)?,
                ]))
            } else {
                let left_int_part = left as i64;
                Ok(mk_or(vec![
                    rx_float_range(
                        Some(left),
                        Some(10f64.powi(num_digits(left_int_part) as i32)),
                        left_inclusive,
                        false,
                    )?,
                    format!("[1-9][0-9]{{{},}}(\\.[0-9]+)?", num_digits(left_int_part)),
                ]))
            }
        }
        (None, Some(right)) => {
            if right == 0.0 {
                let r = format!("-{}", rx_float_range(Some(0.0), None, false, false)?);
                if right_inclusive {
                    Ok(mk_or(vec![r, "0".to_string()]))
                } else {
                    Ok(r)
                }
            } else if right > 0.0 {
                Ok(mk_or(vec![
                    format!("-{}", rx_float_range(Some(0.0), None, false, false)?),
                    rx_float_range(Some(0.0), Some(right), true, right_inclusive)?,
                ]))
            } else {
                Ok(format!(
                    "-{}",
                    rx_float_range(Some(-right), None, right_inclusive, false)?
                ))
            }
        }
        (Some(left), Some(right)) => {
            if left > right {
                return Err(format!(
                    "Invalid range: left ({}) > right ({})",
                    left,
                    right
                ));
            }
            if left == right {
                if left_inclusive && right_inclusive {
                    Ok(format!("({})", escape_float_str(&float_to_str(left))))
                } else {
                    Err(format!(
                        "Empty range when left equals right and not both inclusive"
                    ))
                }
            } else if left < 0.0 {
                if right < 0.0 {
                    Ok(format!(
                        "(-{})",
                        rx_float_range(Some(-right), Some(-left), right_inclusive, left_inclusive)?
                    ))
                } else {
                    let mut parts = vec![];
                    let neg_part =
                        rx_float_range(Some(0.0), Some(-left), false, left_inclusive)?;
                    parts.push(format!("(-{neg_part})"));

                    if right > 0.0 || right_inclusive {
                        let pos_part =
                            rx_float_range(Some(0.0), Some(right), true, right_inclusive)?;
                        parts.push(pos_part);
                    }
                    Ok(mk_or(parts))
                }
            } else {
                nonneg_float_range(left, right, left_inclusive, right_inclusive)
            }
        }
    }
}

pub fn rx_noninteger_float_range(
    left: Option<f64>,
    right: Option<f64>,
    left_inclusive: bool,
    right_inclusive: bool,
) -> Result<Option<String>> {
    match (left, right) {
        (None, None) => Ok(Some("-?(0|[1-9][0-9]*)\\.[0-9]+".to_string())),
        (Some(left), None) => {
            if left < 0.0 {
                let mut parts = Vec::new();
                if let Some(part) = rx_noninteger_float_range(Some(left), Some(0.0), left_inclusive, false)? {
                    parts.push(part);
                }
                if let Some(part) = rx_noninteger_float_range(Some(0.0), None, true, false)? {
                    parts.push(part);
                }
                Ok(mk_or_opt(parts))
            } else {
                let left_int_part = left as i64;
                let mut parts = Vec::new();
                if let Some(part) = rx_noninteger_float_range(
                    Some(left),
                    Some(10f64.powi(num_digits(left_int_part) as i32)),
                    left_inclusive,
                    false,
                )? {
                    parts.push(part);
                }
                parts.push(format!("[1-9][0-9]{{{},}}\\.[0-9]+", num_digits(left_int_part)));
                Ok(Some(mk_or(parts)))
            }
        }
        (None, Some(right)) => {
            if right == 0.0 {
                let mut parts = Vec::new();
                if let Some(part) = rx_noninteger_float_range(Some(0.0), None, false, false)? {
                    parts.push(format!("-{part}"));
                }
                if right_inclusive {
                    parts.push("0\\.0+".to_string());
                }
                Ok(mk_or_opt(parts))
            } else if right > 0.0 {
                let mut parts = Vec::new();
                if let Some(part) = rx_noninteger_float_range(Some(0.0), None, false, false)? {
                    parts.push(format!("-{part}"));
                }
                if let Some(part) = rx_noninteger_float_range(Some(0.0), Some(right), true, right_inclusive)? {
                    parts.push(part);
                }
                Ok(mk_or_opt(parts))
            } else {
                Ok(rx_noninteger_float_range(Some(-right), None, right_inclusive, false)?
                    .map(|part| format!("-{part}")))
            }
        }
        (Some(left), Some(right)) => {
            if left > right {
                return Err(format!(
                    "Invalid range: left ({}) > right ({})",
                    left,
                    right
                ));
            }
            if left == right {
                if !left_inclusive || !right_inclusive {
                    return Ok(None);
                }

                let rendered = float_to_str(left);
                if rendered.contains('.') {
                    Ok(Some(format!("({})", escape_float_str(&rendered))))
                } else {
                    Ok(None)
                }
            } else if left < 0.0 {
                if right < 0.0 {
                    Ok(rx_noninteger_float_range(Some(-right), Some(-left), right_inclusive, left_inclusive)?
                        .map(|part| format!("(-{part})")))
                } else {
                    let mut parts = Vec::new();
                    if let Some(part) =
                        rx_noninteger_float_range(Some(0.0), Some(-left), false, left_inclusive)?
                    {
                        parts.push(format!("(-{part})"));
                    }

                    if right > 0.0 || right_inclusive {
                        if let Some(part) =
                            rx_noninteger_float_range(Some(0.0), Some(right), true, right_inclusive)?
                        {
                            parts.push(part);
                        }
                    }
                    Ok(mk_or_opt(parts))
                }
            } else {
                nonneg_float_range_no_ints(left, right, left_inclusive, right_inclusive)
            }
        }
    }
}

fn nonneg_float_range(
    left: f64,
    right: f64,
    left_inclusive: bool,
    right_inclusive: bool,
) -> Result<String> {
    let mut bounds = nonnegative_decimal_bounds(left, right)?;

    if bounds.left_integer == bounds.right_integer {
        pad_decimal_fractions(&mut bounds.left_fraction, &mut bounds.right_fraction);
        let suffix = format!(
            "\\.{}",
            lexi_range(
                &bounds.left_fraction,
                &bounds.right_fraction,
                left_inclusive,
                right_inclusive,
            )?
        );
        if bounds.left_fraction.parse::<i64>().unwrap_or(0) == 0 {
            Ok(format!("({}({suffix})?)", bounds.left_integer))
        } else {
            Ok(format!("({}{suffix})", bounds.left_integer))
        }
    } else {
        let mut parts = vec![];
        if !bounds.left_fraction.is_empty() || !left_inclusive {
            parts.push(format!(
                "({}\\.{})",
                bounds.left_integer,
                lexi_x_to_9(&bounds.left_fraction, left_inclusive)?
            ));
            bounds.left_integer += 1;
        }

        if bounds.right_integer > bounds.left_integer {
            let inner = rx_int_range(Some(bounds.left_integer), Some(bounds.right_integer - 1))?;
            parts.push(format!("({inner}(\\.[0-9]+)?)"));
        }

        if !bounds.right_fraction.is_empty() {
            parts.push(format!(
                "({}(\\.{})?)",
                bounds.right_integer,
                lexi_0_to_x(&bounds.right_fraction, right_inclusive)?
            ));
        } else if right_inclusive {
            parts.push(format!("{}(\\.0+)?", bounds.right_integer));
        }

        Ok(mk_or(parts))
    }
}

fn nonneg_float_range_no_ints(
    left: f64,
    right: f64,
    left_inclusive: bool,
    right_inclusive: bool,
) -> Result<Option<String>> {
    let mut bounds = nonnegative_decimal_bounds(left, right)?;

    if bounds.left_integer == bounds.right_integer {
        pad_decimal_fractions(&mut bounds.left_fraction, &mut bounds.right_fraction);
        let suffix = format!(
            "\\.{}",
            lexi_range(
                &bounds.left_fraction,
                &bounds.right_fraction,
                left_inclusive,
                right_inclusive,
            )?
        );
        return Ok(Some(format!("({}{suffix})", bounds.left_integer)));
    }

    let mut parts = vec![];
    if !bounds.left_fraction.is_empty() {
        parts.push(format!(
            "({}\\.{})",
            bounds.left_integer,
            lexi_x_to_9(&bounds.left_fraction, left_inclusive)?
        ));
        bounds.left_integer += 1;
    } else {
        parts.push(format!("({}\\.[0-9]+)", bounds.left_integer));
        bounds.left_integer += 1;
    }

    if bounds.right_integer > bounds.left_integer {
        let inner = rx_int_range(Some(bounds.left_integer), Some(bounds.right_integer - 1))?;
        parts.push(format!("({inner}\\.[0-9]+)"));
    }

    if !bounds.right_fraction.is_empty() {
        let right_frac = lexi_0_to_x(&bounds.right_fraction, right_inclusive)?;
        if !right_frac.is_empty() {
            parts.push(format!("({}\\.{})", bounds.right_integer, right_frac));
        }
    } else if right_inclusive {
        parts.push(format!("{}\\.0+", bounds.right_integer));
    }

    Ok(mk_or_opt(parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_range_0_1() {
        let r = rx_int_range(Some(0), Some(1)).unwrap();
        assert_eq!(r, "([0-1])");
    }

    #[test]
    fn test_int_range_0_255() {
        let r = rx_int_range(Some(0), Some(255)).unwrap();
        assert!(r.contains("[0-9]"));
    }

    #[test]
    fn test_float_range_0_1() {
        let r = rx_float_range(Some(0.0), Some(1.0), true, true).unwrap();
        assert!(r.contains("0"));
        assert!(r.contains("1"));
    }

    #[test]
    fn test_noninteger_float_range_0_1_requires_decimal_point() {
        let r = rx_noninteger_float_range(Some(0.0), Some(1.0), true, true)
            .unwrap()
            .unwrap();
        assert!(r.contains("\\."));
        assert!(!r.contains("(\\.[0-9]+)?"));
    }
}
