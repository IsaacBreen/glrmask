//! Numeric range regex generators for JSON schema constraints.
//!
//! Numeric-range parsing adapted from llguidance's JSON number parser.
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
    (!parts.is_empty()).then(|| mk_or(parts))
}

fn push_optional_part(parts: &mut Vec<String>, part: Option<String>) {
    if let Some(part) = part {
        parts.push(part);
    }
}

fn negative_regex(pattern: String) -> String {
    format!("-{pattern}")
}

/// A lexicographic interval of equal-width digit strings. Once both bounds
/// agree on a prefix, it is emitted once; divergent endpoints and the
/// unrestricted middle are disjoint. No numeric increment or power-of-ten
/// conversion is needed, even for the magnitude of i64::MIN.
fn digit_interval(left: &[u8], right: &[u8]) -> String {
    debug_assert_eq!(left.len(), right.len());
    debug_assert!(left <= right);
    if left.iter().all(|&digit| digit == b'0') && right.iter().all(|&digit| digit == b'9') {
        return any_digits(left.len());
    }
    let common = left.iter().zip(right).take_while(|(a, b)| a == b).count();
    let prefix = std::str::from_utf8(&left[..common]).expect("decimal digits");
    if common == left.len() { return prefix.to_owned(); }
    let lower = left[common];
    let upper = right[common];
    let rest = left.len() - common - 1;
    let mut alternatives = Vec::with_capacity(3);
    if rest == 0 {
        return format!("{prefix}[{}-{}]", lower as char, upper as char);
    }
    let low_tail = digit_interval(&left[common + 1..], &vec![b'9'; rest]);
    alternatives.push(format!("{}{low_tail}", lower as char));
    if lower + 1 < upper {
        alternatives.push(format!("[{}-{}]{}", (lower + 1) as char,
            (upper - 1) as char, any_digits(rest)));
    }
    let high_tail = digit_interval(&vec![b'0'; rest], &right[common + 1..]);
    alternatives.push(format!("{}{high_tail}", upper as char));
    format!("{prefix}{}", mk_or(alternatives))
}

fn any_digits(count: usize) -> String {
    match count {
        0 => String::new(),
        1 => "[0-9]".to_owned(),
        _ => format!("[0-9]{{{count}}}"),
    }
}

fn unsigned_range(lower: u64, upper: Option<u64>) -> String {
    let left = lower.to_string();
    let right = upper.map(|value| value.to_string());
    let last_width = right.as_ref().map_or(left.len(), String::len);
    let mut parts = Vec::new();
    for width in left.len()..=last_width {
        let low = if width == left.len() { left.clone() }
            else { format!("1{}", "0".repeat(width - 1)) };
        let high = right.as_ref().filter(|r| r.len() == width)
            .cloned().unwrap_or_else(|| "9".repeat(width));
        // Full canonical widths are one compact expression.
        let first_min = if width == 1 { b'0' } else { b'1' };
        if low.as_bytes()[0] == first_min
            && low.as_bytes()[1..].iter().all(|&b| b == b'0')
            && high.bytes().all(|b| b == b'9')
        {
            if width == 1 { parts.push("[0-9]".to_owned()); }
            else { parts.push(format!("[1-9]{}", any_digits(width - 1))); }
        } else {
            parts.push(digit_interval(low.as_bytes(), high.as_bytes()));
        }
    }
    if right.is_none() {
        parts.push(format!("[1-9][0-9]{{{},}}", left.len()));
    }
    mk_or(parts)
}

pub fn rx_int_range(left: Option<i64>, right: Option<i64>) -> Result<String> {
    if let (Some(left), Some(right)) = (left, right) && left > right {
        return Err(format!("Invalid range: left ({left}) > right ({right})"));
    }
    if left.is_none() && right.is_none() {
        return Ok("-?(0|[1-9][0-9]*)".to_string());
    }
    let mut parts = Vec::with_capacity(2);
    if left.is_none_or(|left| left < 0) {
        // Preserve the historical signed-zero spelling policy: a finite
        // interval crossing zero accepts -0; one-sided intervals use only
        // strictly negative magnitudes on the negative side.
        let lower_magnitude = match right {
            Some(right) if right < 0 => right.unsigned_abs(),
            Some(_) if left.is_some() => 0,
            _ => 1,
        };
        let upper_magnitude = left.map(i64::unsigned_abs);
        parts.push(format!("-(?:{})", unsigned_range(lower_magnitude, upper_magnitude)));
    }
    if right.is_none_or(|right| right >= 0) {
        parts.push(unsigned_range(left.unwrap_or(0).max(0) as u64, right.map(|right| right as u64)));
    }
    Ok(mk_or(parts))
}

// Float range helpers.

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
            Err("Inclusive flag must be true for an empty string".to_string())
        }
    } else {
        let x0 = x.chars().next().unwrap().to_digit(10).unwrap();
        let x_rest = &x[1..];

        if !incl && x.len() == 1 {
            if x0 == 0 {
                return Err("x0 must be greater than 0 for non-inclusive single character".to_string());
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
        return Err("ld and rd must have the same length".to_string());
    }
    if ld == rd {
        if ld_incl && rd_incl {
            Ok(ld.to_string())
        } else {
            Err("Empty range when ld equals rd and not both inclusive".to_string())
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

fn exact_float_regex(value: f64) -> String {
    format!("({})", escape_float_str(&float_to_str(value)))
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum FloatRangeMode {
    AllowIntegers,
    FractionOnly,
}

fn same_integer_float_part(
    bounds: &mut NonnegativeDecimalBounds,
    left_inclusive: bool,
    right_inclusive: bool,
    mode: FloatRangeMode,
) -> Result<String> {
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

    if mode == FloatRangeMode::AllowIntegers
        && left_inclusive
        && bounds.left_fraction.bytes().all(|digit| digit == b'0')
    {
        Ok(format!("({}({suffix})?)", bounds.left_integer))
    } else {
        Ok(format!("({}{suffix})", bounds.left_integer))
    }
}

fn lower_float_part(
    bounds: &mut NonnegativeDecimalBounds,
    left_inclusive: bool,
    mode: FloatRangeMode,
) -> Result<Option<String>> {
    match mode {
        FloatRangeMode::AllowIntegers => {
            if !bounds.left_fraction.is_empty() || !left_inclusive {
                let part = format!(
                    "({}\\.{})",
                    bounds.left_integer,
                    lexi_x_to_9(&bounds.left_fraction, left_inclusive)?
                );
                bounds.left_integer += 1;
                Ok(Some(part))
            } else {
                Ok(None)
            }
        }
        FloatRangeMode::FractionOnly => {
            let part = if !bounds.left_fraction.is_empty() {
                format!(
                    "({}\\.{})",
                    bounds.left_integer,
                    lexi_x_to_9(&bounds.left_fraction, left_inclusive)?
                )
            } else {
                format!("({}\\.{})", bounds.left_integer,
                    if left_inclusive { "[0-9]+".to_owned() } else { lexi_x_to_9("", false)? })
            };
            bounds.left_integer += 1;
            Ok(Some(part))
        }
    }
}

fn middle_float_part(bounds: &NonnegativeDecimalBounds, mode: FloatRangeMode) -> Result<Option<String>> {
    if bounds.right_integer <= bounds.left_integer {
        return Ok(None);
    }

    let inner = rx_int_range(Some(bounds.left_integer), Some(bounds.right_integer - 1))?;
    Ok(Some(match mode {
        FloatRangeMode::AllowIntegers => format!("({inner}(\\.[0-9]+)?)"),
        FloatRangeMode::FractionOnly => format!("({inner}\\.[0-9]+)"),
    }))
}

fn upper_float_part(
    bounds: &NonnegativeDecimalBounds,
    right_inclusive: bool,
    mode: FloatRangeMode,
) -> Result<Option<String>> {
    if !bounds.right_fraction.is_empty() {
        let right_fraction = lexi_0_to_x(&bounds.right_fraction, right_inclusive)?;
        if mode == FloatRangeMode::FractionOnly && right_fraction.is_empty() {
            return Ok(None);
        }

        return Ok(Some(match mode {
            FloatRangeMode::AllowIntegers => {
                format!("({}(\\.{})?)", bounds.right_integer, right_fraction)
            }
            FloatRangeMode::FractionOnly => {
                format!("({}\\.{})", bounds.right_integer, right_fraction)
            }
        }));
    }

    if !right_inclusive {
        return Ok(None);
    }

    Ok(Some(match mode {
        FloatRangeMode::AllowIntegers => format!("{}(\\.0+)?", bounds.right_integer),
        FloatRangeMode::FractionOnly => format!("{}\\.0+", bounds.right_integer),
    }))
}

fn collect_nonnegative_float_parts(
    bounds: &mut NonnegativeDecimalBounds,
    left_inclusive: bool,
    right_inclusive: bool,
    mode: FloatRangeMode,
) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    push_optional_part(&mut parts, lower_float_part(bounds, left_inclusive, mode)?);
    push_optional_part(&mut parts, middle_float_part(bounds, mode)?);
    push_optional_part(&mut parts, upper_float_part(bounds, right_inclusive, mode)?);
    Ok(parts)
}

// Float range. Both lexical modes use the same signed interval decomposition;
// the mode controls only whether an integer spelling is allowed at a leaf.

pub fn rx_float_range(
    left: Option<f64>, right: Option<f64>, left_inclusive: bool, right_inclusive: bool,
) -> Result<String> {
    float_range(left, right, left_inclusive, right_inclusive, FloatRangeMode::AllowIntegers)?
        .ok_or_else(|| "Empty numeric range".to_string())
}

pub fn rx_noninteger_float_range(
    left: Option<f64>, right: Option<f64>, left_inclusive: bool, right_inclusive: bool,
) -> Result<Option<String>> {
    float_range(left, right, left_inclusive, right_inclusive, FloatRangeMode::FractionOnly)
}

impl FloatRangeMode {
    fn full_fraction(self) -> &'static str {
        match self {
            Self::AllowIntegers => "(\\.[0-9]+)?",
            Self::FractionOnly => "\\.[0-9]+",
        }
    }

    fn zero_boundary(self) -> &'static str {
        match self {
            Self::AllowIntegers => "0",
            Self::FractionOnly => "0\\.0+",
        }
    }
}

fn float_range(
    left: Option<f64>, right: Option<f64>, left_inclusive: bool,
    right_inclusive: bool, mode: FloatRangeMode,
) -> Result<Option<String>> {
    if left.is_some_and(|v| !v.is_finite()) || right.is_some_and(|v| !v.is_finite()) {
        return Err("Infinite numbers not supported".to_string());
    }
    if let (Some(left), Some(right)) = (left, right) {
        if left > right { return Err(format!("Invalid range: left ({left}) > right ({right})")); }
        if left == right {
            return Ok((left_inclusive && right_inclusive
                && (mode == FloatRangeMode::AllowIntegers || float_to_str(left).contains('.')))
                .then(|| exact_float_regex(left)));
        }
    }
    if left.is_none() && right.is_none() {
        return Ok(Some(format!("-?(0|[1-9][0-9]*){}", mode.full_fraction())));
    }
    let mut parts = Vec::new();
    if left.is_none_or(|value| value < 0.0) {
        let (lower, inclusive) = match right {
            Some(value) if value < 0.0 => (-value, right_inclusive),
            _ => (0.0, false),
        };
        let upper = left.map(|value| -value);
        push_optional_part(&mut parts,
            nonnegative_range(lower, upper, inclusive, left_inclusive, mode)?
                .map(|part| format!("-(?:{part})")));
    }
    match right {
        Some(value) if value < 0.0 => {}
        Some(0.0) if left.is_none() => {
            if right_inclusive { parts.push(mode.zero_boundary().to_owned()); }
        }
        _ => {
            let (lower, inclusive) = match left {
                Some(value) if value >= 0.0 => (value, left_inclusive),
                _ => (0.0, true),
            };
            push_optional_part(&mut parts,
                nonnegative_range(lower, right, inclusive, right_inclusive, mode)?);
        }
    }
    Ok(mk_or_opt(parts))
}

fn nonnegative_range(
    left: f64, right: Option<f64>, left_inclusive: bool,
    right_inclusive: bool, mode: FloatRangeMode,
) -> Result<Option<String>> {
    if let Some(right) = right {
        if left == right {
            return Ok((left_inclusive && right_inclusive
                && (mode == FloatRangeMode::AllowIntegers || float_to_str(left).contains('.')))
                .then(|| exact_float_regex(left)));
        }
        let mut bounds = nonnegative_decimal_bounds(left, right)?;
        if bounds.left_integer == bounds.right_integer {
            return same_integer_float_part(&mut bounds, left_inclusive, right_inclusive, mode).map(Some);
        }
        return Ok(mk_or_opt(collect_nonnegative_float_parts(&mut bounds, left_inclusive, right_inclusive, mode)?));
    }
    // An unbounded endpoint does not require a fictitious finite power-of-ten
    // bound. Emit the lower integer's partial fractional band and then all
    // subsequent integer magnitudes with the requested fraction policy.
    let rendered = float_to_str(left);
    let integer = parse_decimal_integer(&rendered, "left")?;
    let fraction = decimal_fraction(&rendered);
    let mut parts = Vec::new();
    let mut first_full = integer as u64;
    if !fraction.is_empty() || !left_inclusive {
        let suffix = lexi_x_to_9(&fraction, left_inclusive)?;
        parts.push(format!("({integer}\\.{suffix})"));
        first_full += 1;
    }
    parts.push(format!("(?:{}){}", unsigned_range(first_full, None), mode.full_fraction()));
    Ok(mk_or_opt(parts))
}

#[cfg(test)]
mod range_tests {
    use super::*;

    fn accepts(pattern: &str, text: &str) -> bool {
        regex::Regex::new(&format!("^(?:{pattern})$")).unwrap().is_match(text)
    }

    #[test]
    fn integer_endpoints_do_not_overflow_signed_magnitudes() {
        let pattern = rx_int_range(Some(i64::MIN), Some(i64::MAX)).unwrap();
        assert!(accepts(&pattern, &i64::MIN.to_string()));
        assert!(accepts(&pattern, &i64::MAX.to_string()));
        assert!(!accepts(&pattern, "-9223372036854775809"));
        assert!(!accepts(&pattern, "9223372036854775808"));
        let pattern = rx_int_range(Some(i64::MAX), None).unwrap();
        assert!(accepts(&pattern, "9223372036854775808"));
        assert!(!accepts(&pattern, "9223372036854775806"));
    }

    #[test]
    fn interval_digit_covers_match_literal_arithmetic() {
        for (lo, hi) in [(-120, -7), (-13, 24), (0, 100), (8, 111), (120, 129)] {
            let pattern = rx_int_range(Some(lo), Some(hi)).unwrap();
            let regex = regex::Regex::new(&format!("^(?:{pattern})$")).unwrap();
            for value in -200..=200 {
                assert_eq!(regex.is_match(&value.to_string()), lo <= value && value <= hi);
            }
        }
    }

    #[test]
    fn exclusive_zero_fraction_does_not_accept_integer_alias() {
        let pattern = rx_float_range(Some(1.0), Some(1.5), false, true).unwrap();
        assert!(!accepts(&pattern, "1"));
        assert!(!accepts(&pattern, "1.0"));
        assert!(accepts(&pattern, "1.1"));
        let pattern = rx_noninteger_float_range(Some(0.0), None, false, false).unwrap().unwrap();
        assert!(!accepts(&pattern, "0.0"));
        assert!(accepts(&pattern, "0.1"));
    }
}
