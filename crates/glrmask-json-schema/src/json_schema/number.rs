use std::collections::BTreeSet;
use std::sync::Arc;

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::compile_expression_labeled_nfa;
use crate::grammar::expr_nfa::ExprNfaBuilder;
use crate::import::ast::GrammarExpr;
use crate::import::numeric_range::{rx_float_range, rx_int_range};

use super::ast::{NumberSchema, Schema, SchemaAssertions, SchemaDocument, SchemaKind, SchemaType};
use super::error::{ImportResult, SchemaImportError};
use super::lower::{choice, lit_bytes, never, r, Lowerer, JSON_INTEGER_RULE, JSON_NUMBER_RULE};

const MAX_EXPLICIT_INTEGER_RANGE: i64 = 512;
const MAX_EXPLICIT_INTEGER_MULTIPLES: i64 = 2048;
// A general divisor needs one remainder state per residue. Bound construction
// explicitly rather than accepting numbers which violate an unsupported
// multipleOf. This does not bound the number of input digits.
const MAX_INTEGER_REMAINDER_STATES: u64 = 4096;
pub(super) const JSON_INTEGER_ATOM_RULE_PREFIX: &str = "JSON_INTEGER_ATOM_";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SharedIntegerAtom {
    pub lower: Option<i64>,
    pub upper: Option<i64>,
    pub rule_name: String,
}

pub(super) fn collect_shared_integer_atoms(document: &SchemaDocument) -> Vec<SharedIntegerAtom> {
    let mut cuts = BTreeSet::<i64>::new();
    let mut representable = true;
    collect_plain_integer_range_cuts(&document.root, &mut cuts, &mut representable);
    for definition in &document.definitions {
        collect_plain_integer_range_cuts(&definition.schema, &mut cuts, &mut representable);
    }
    for target in &document.ref_targets {
        collect_plain_integer_range_cuts(&target.schema, &mut cuts, &mut representable);
    }
    // `Option<i64>` atoms use None for the two unbounded sides. A cut at
    // i64::MIN or immediately after i64::MAX cannot represent the outside atom
    // without extending the numeric-range machinery beyond i64. Fall back to
    // the previous per-range exact lowering for such documents.
    if !representable || cuts.is_empty() {
        return Vec::new();
    }
    let cuts = cuts.into_iter().collect::<Vec<_>>();
    let mut atoms = Vec::with_capacity(cuts.len() + 1);
    if cuts[0] != i64::MIN {
        atoms.push((None, Some(cuts[0] - 1)));
    }
    for window in cuts.windows(2) {
        let lower = window[0];
        let upper = window[1] - 1;
        if lower <= upper {
            atoms.push((Some(lower), Some(upper)));
        }
    }
    atoms.push((Some(*cuts.last().expect("non-empty integer partition cuts")), None));
    atoms.into_iter().enumerate().map(|(index, (lower, upper))| SharedIntegerAtom {
        lower, upper, rule_name: format!("{JSON_INTEGER_ATOM_RULE_PREFIX}{index}"),
    }).collect()
}

fn collect_plain_integer_range_cuts(
    schema: &Schema,
    cuts: &mut BTreeSet<i64>,
    representable: &mut bool,
) {
    for node in schema.descendants() {
        let SchemaKind::Assertions(assertions) = &node.kind else { continue; };
        if plain_integer_assertions(assertions)
            && let Some(number) = &assertions.number
            && integer_multiple_is_vacuous(number.multiple_of)
        {
            let (Ok(lower), Ok(upper)) = (integer_lower_bound(number), integer_upper_bound(number)) else {
                *representable = false;
                continue;
            };
            if lower == Some(i64::MIN) || upper == Some(i64::MAX) {
                *representable = false;
            } else {
                if let Some(lower) = lower { cuts.insert(lower); }
                if let Some(upper) = upper { cuts.insert(upper + 1); }
            }
        }
    }
}
fn plain_integer_assertions(assertions: &SchemaAssertions) -> bool {
    let Some(types) = &assertions.types else { return false; };
    types.contains(&SchemaType::Integer) && !types.contains(&SchemaType::Number)
}

impl<'a> Lowerer<'a> {
    pub fn lower_number(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {
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
            if let Some(regex) = power_of_ten_multiple_regex(multiple, false) {
                GrammarExpr::RawRegex(regex)
            } else if let Some(regex) = decimal_multiple_regex(multiple) {
                GrammarExpr::RawRegex(regex)
            } else {
                return Err(SchemaImportError::new(format!(
                    "multipleOf={multiple} isn't supported for non-integer numbers"
                )));
            }
        } else if let Some(range_expr) = range_expr.clone() {
            return Ok(range_expr);
        } else {
            r(JSON_NUMBER_RULE)
        };

        Ok(intersect_numeric_range(base_expr, range_expr))
    }

    fn ensure_shared_integer_atom_rules(&mut self) -> ImportResult<()> {
        if self.shared_integer_atom_rules_installed || self.shared_integer_atoms.is_empty() {
            return Ok(());
        }
        let atoms = self.shared_integer_atoms.clone();
        for atom in atoms {
            let regex = rx_int_range(atom.lower, atom.upper).map_err(SchemaImportError::new)?;
            self.add_terminal_rule(&atom.rule_name, GrammarExpr::RawRegex(regex));
        }
        self.shared_integer_atom_rules_installed = true;
        Ok(())
    }

    fn shared_integer_range_expr(
        &mut self,
        lower: Option<i64>,
        upper: Option<i64>,
    ) -> ImportResult<Option<GrammarExpr>> {
        if self.shared_integer_atoms.is_empty() {
            return Ok(None);
        }
        // Fail closed to the old exact lowering when this range was synthesized
        // after the document-global cut scan and therefore would split an atom.
        for atom in &self.shared_integer_atoms {
            let intersects = atom.upper.is_none_or(|a| lower.is_none_or(|l| a >= l))
                && atom.lower.is_none_or(|a| upper.is_none_or(|u| a <= u));
            if !intersects { continue; }
            let contained = lower.is_none_or(|l| atom.lower.is_some_and(|a| a >= l))
                && upper.is_none_or(|u| atom.upper.is_some_and(|a| a <= u));
            if !contained { return Ok(None); }
        }
        self.ensure_shared_integer_atom_rules()?;
        let refs = self.shared_integer_atoms.iter().filter(|atom| {
            lower.is_none_or(|l| atom.lower.is_some_and(|a| a >= l))
                && upper.is_none_or(|u| atom.upper.is_some_and(|a| a <= u))
        }).map(|atom| r(&atom.rule_name)).collect::<Vec<_>>();
        Ok(Some(choice(refs)))
    }

    fn lower_integer(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {
        let mut lower = integer_lower_bound(schema)?;
        let upper = integer_upper_bound(schema)?;
        if self.llguidance_compat_enabled()
            && schema.multiple_of.is_some()
            && lower.is_some_and(|value| value < 0)
            && upper.is_some_and(|value| value >= 0)
        {
            lower = Some(0);
        }
        if let (Some(lower), Some(upper)) = (lower, upper) && lower > upper {
            return Ok(never());
        }
        let divisor = IntegerDivisor::from_multiple(schema.multiple_of)
            .ok_or_else(|| SchemaImportError::new("integer multipleOf is not a representable positive decimal"))?;
        if divisor == IntegerDivisor::Exact(1) {
            if let Some(expr) = self.shared_integer_range_expr(lower, upper)? {
                return Ok(expr);
            }
            return Ok(integer_range_expr(lower, upper)?.unwrap_or_else(|| r(JSON_INTEGER_RULE)));
        }

        if let (Some(lower), Some(upper)) = (lower, upper) {
            if upper.saturating_sub(lower) <= MAX_EXPLICIT_INTEGER_RANGE {
                return Ok(choice((lower..=upper).filter(|&value| divisor.accepts(value))
                    .map(|value| lit_bytes(value.to_string().into_bytes())).collect()));
            }
            if let Some(expr) = divisor.bounded_choice(lower, upper) {
                return Ok(expr);
            }
        }

        let IntegerDivisor::Exact(divisor) = divisor else {
            return Err(SchemaImportError::new("integer multipleOf exceeds the representable divisor range"));
        };
        let base = if let Some(regex) = power_of_ten_divisor_regex(divisor, true) {
            GrammarExpr::RawRegex(regex)
        } else {
            integer_modulo_expr(divisor)?
        };
        Ok(intersect_numeric_range(base, integer_range_expr(lower, upper)?))
    }
}

fn integer_range_expr(lower: Option<i64>, upper: Option<i64>) -> ImportResult<Option<GrammarExpr>> {
    if lower.is_none() && upper.is_none() { return Ok(None); }
    Ok(Some(GrammarExpr::RawRegex(rx_int_range(lower, upper).map_err(SchemaImportError::new)?)))
}

fn intersect_numeric_range(base: GrammarExpr, range: Option<GrammarExpr>) -> GrammarExpr {
    match range {
        None => base,
        Some(range) => GrammarExpr::Intersect { expr: Box::new(base), intersect: Box::new(range) },
    }
}

fn integer_bound(bound: Option<f64>, exclusive: bool, lower: bool) -> ImportResult<Option<i64>> {
    let Some(value) = bound else { return Ok(None); };
    let rounded = if lower { value.ceil() } else { value.floor() };
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded >= i64::MAX as f64 {
        return Err(SchemaImportError::new("integer bound is outside the supported signed 64-bit range"));
    }
    let integer = rounded as i64;
    let integer = if exclusive && rounded == value {
        (if lower { integer.checked_add(1) } else { integer.checked_sub(1) })
            .ok_or_else(|| SchemaImportError::new("exclusive integer bound is outside the supported signed 64-bit range"))?
    } else { integer };
    Ok(Some(integer))
}

fn integer_lower_bound(schema: &NumberSchema) -> ImportResult<Option<i64>> {
    integer_bound(schema.minimum, schema.exclusive_minimum, true)
}

fn integer_upper_bound(schema: &NumberSchema) -> ImportResult<Option<i64>> {
    integer_bound(schema.maximum, schema.exclusive_maximum, false)
}
/// A finite decimal magnitude n * 10^e. Factors of ten are removed so
/// equality and divisibility do not depend on redundant decimal zeroes.
#[derive(Clone, Copy)]
struct DecimalMagnitude {
    digits: u128,
    exponent: i32,
}

impl DecimalMagnitude {
    fn new(mut digits: u128, mut exponent: i32) -> Self {
        if digits == 0 { return Self { digits, exponent: 0 }; }
        while digits.is_multiple_of(10) {
            digits /= 10;
            exponent += 1;
        }
        Self { digits, exponent }
    }

    fn parse(text: &str) -> Option<Self> {
        let text = text.trim_start_matches(['-', '+']);
        let (mantissa, exponent) = if let Some((mantissa, exponent)) = text.split_once(['e', 'E']) {
            (mantissa, exponent.parse::<i32>().ok()?)
        } else { (text, 0) };
        let mut digits = 0u128;
        let mut fraction = 0i32;
        let mut trailing_zeroes = 0u32;
        let mut after_point = false;
        for byte in mantissa.bytes() {
            if byte == b'.' && !after_point { after_point = true; continue; }
            if !byte.is_ascii_digit() { return None; }
            if after_point { fraction = fraction.checked_add(1)?; }
            if byte == b'0' {
                if digits != 0 { trailing_zeroes = trailing_zeroes.checked_add(1)?; }
                continue;
            }
            if digits != 0 {
                digits = digits.checked_mul(10u128.checked_pow(trailing_zeroes.checked_add(1)?)?)?;
            }
            digits = digits.checked_add(u128::from(byte - b'0'))?;
            trailing_zeroes = 0;
        }
        Some(Self::new(digits, exponent.checked_sub(fraction)?.checked_add(i32::try_from(trailing_zeroes).ok()?)?))
    }

    fn from_float(value: f64) -> Option<Self> {
        if !value.is_finite() { return None; }
        let magnitude = value.abs();
        // Integral binary floats may print a nearby rounded decimal rather
        // than the exact represented integer. Preserve the latter.
        if magnitude.fract() == 0.0 && magnitude < u128::MAX as f64 {
            Some(Self::new(magnitude as u128, 0))
        } else {
            Self::parse(&magnitude.to_string())
        }
    }

    fn from_number(value: &serde_json::Number) -> Option<Self> {
        if let Some(value) = value.as_u64() { return Some(Self::new(u128::from(value), 0)); }
        if let Some(value) = value.as_i64() { return Some(Self::new(u128::from(value.unsigned_abs()), 0)); }
        Self::from_float(value.as_f64()?)
    }

    fn divides(self, value: Self) -> bool {
        if self.digits == 0 { return false; }
        if value.digits == 0 { return true; }
        let gcd = gcd_u128(self.digits, value.digits);
        let denominator = self.digits / gcd;
        let mut numerator = value.digits / gcd;
        let exponent = value.exponent - self.exponent;
        if exponent >= 0 {
            cancel_decimal_denominator(denominator, exponent as u32) == 1
        } else {
            if denominator != 1 { return false; }
            for _ in 0..exponent.unsigned_abs() {
                if !numerator.is_multiple_of(10) { return false; }
                numerator /= 10;
            }
            true
        }
    }
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 { (a, b) = (b, a % b); }
    a
}

fn cancel_decimal_denominator(mut numerator: u128, scale_digits: u32) -> u128 {
    // gcd(n, 10^k) requires only the two prime factors of ten; the
    // denominator itself need not fit in a machine integer.
    for prime in [2, 5] {
        for _ in 0..scale_digits {
            if !numerator.is_multiple_of(prime) { break; }
            numerator /= prime;
        }
    }
    numerator
}

pub(super) fn number_is_multiple(value: &serde_json::Number, multiple: f64) -> bool {
    if !multiple.is_finite() || multiple <= 0.0 { return false; }
    DecimalMagnitude::from_float(multiple).zip(DecimalMagnitude::from_number(value))
        .is_some_and(|(divisor, value)| divisor.divides(value))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IntegerDivisor {
    Exact(u64),
    LargerThanU64,
}

impl IntegerDivisor {
    fn from_multiple(multiple: Option<f64>) -> Option<Self> {
        let Some(multiple) = multiple else { return Some(Self::Exact(1)); };
        if !multiple.is_finite() || multiple <= 0.0 { return None; }
        let decimal = DecimalMagnitude::from_float(multiple)?;
        let divisor = if decimal.exponent < 0 {
            Some(cancel_decimal_denominator(decimal.digits, decimal.exponent.unsigned_abs()))
        } else {
            10u128.checked_pow(decimal.exponent as u32).and_then(|scale| decimal.digits.checked_mul(scale))
        };
        Some(divisor.and_then(|value| u64::try_from(value).ok())
            .map_or(Self::LargerThanU64, Self::Exact))
    }

    fn accepts(self, value: i64) -> bool {
        match self {
            Self::Exact(divisor) => value.unsigned_abs().is_multiple_of(divisor),
            Self::LargerThanU64 => value == 0,
        }
    }

    fn bounded_choice(self, lower: i64, upper: i64) -> Option<GrammarExpr> {
        let divisor = match self {
            Self::LargerThanU64 => return Some(if lower <= 0 && upper >= 0 {
                lit_bytes(b"0".to_vec())
            } else { never() }),
            Self::Exact(divisor) => i128::from(divisor),
        };
        let lower = i128::from(lower);
        let upper = i128::from(upper);
        let first = (lower / divisor + i128::from(lower % divisor > 0)) * divisor;
        if first > upper { return Some(never()); }
        let count = (upper - first) / divisor + 1;
        if count > i128::from(MAX_EXPLICIT_INTEGER_MULTIPLES) { return None; }
        Some(choice((0..count).map(|index| {
            lit_bytes((first + index * divisor).to_string().into_bytes())
        }).collect()))
    }
}

fn integer_multiple_is_vacuous(multiple: Option<f64>) -> bool {
    IntegerDivisor::from_multiple(multiple) == Some(IntegerDivisor::Exact(1))
}

#[cfg(test)]
fn integer_satisfies_multiple(value: i64, multiple: Option<f64>) -> bool {
    IntegerDivisor::from_multiple(multiple).is_some_and(|divisor| divisor.accepts(value))
}

#[cfg(test)]
fn bounded_integer_multiple_choice(lower: i64, upper: i64, multiple: Option<f64>) -> Option<GrammarExpr> {
    IntegerDivisor::from_multiple(multiple)?.bounded_choice(lower, upper)
}

/// Exact signed canonical-decimal integers divisible by `divisor`.
///
/// After a nonzero first digit, state r represents the absolute value of the
/// consumed decimal prefix modulo divisor. Appending d updates r to
/// (10*r+d) mod divisor, so precisely remainder zero is accepting. The sign
/// does not affect divisibility; separate start/sign/zero states reject a
/// bare minus, plus signs, leading zeroes and non-digit spellings. The usual
/// integer lexical policy is retained (no decimal/exponent aliases).
fn integer_modulo_expr(divisor: u64) -> ImportResult<GrammarExpr> {
    if divisor == 0 || divisor > MAX_INTEGER_REMAINDER_STATES {
        return Err(SchemaImportError::new(format!(
            "integer multipleOf={divisor} exceeds the exact remainder-state budget of {MAX_INTEGER_REMAINDER_STATES}; supply finite bounds or a supported compact divisor"
        )));
    }
    let mut graph = ExprNfaBuilder::new();
    let start = graph.start_state();
    let after_minus = graph.add_state();
    let zero = graph.add_state();
    let remainders = (0..divisor).map(|_| graph.add_state()).collect::<Vec<_>>();
    let digits = (b'0'..=b'9')
        .map(|byte| graph.add_symbol(lit_bytes(vec![byte])))
        .collect::<Vec<_>>();
    graph.add_transition(start, lit_bytes(vec![b'-']), after_minus);
    graph.set_accepting(zero);
    graph.set_accepting(remainders[0]);
    for source in [start, after_minus] {
        graph.add_labeled_transition(source, digits[0], zero);
        for digit in 1..=9u64 {
            graph.add_labeled_transition(
                source,
                digits[digit as usize],
                remainders[(digit % divisor) as usize],
            );
        }
    }
    for remainder in 0..divisor {
        for digit in 0..=9u64 {
            let target = (10 * remainder + digit) % divisor;
            graph.add_labeled_transition(
                remainders[remainder as usize],
                digits[digit as usize],
                remainders[target as usize],
            );
        }
    }
    let (nfa, symbols) = graph.into_nfa_and_symbols();
    let symbols = symbols.into_iter().map(|symbol| match symbol {
        GrammarExpr::Literal(bytes) => Expr::U8Seq(bytes),
        _ => unreachable!("the remainder automaton uses literal-byte labels only"),
    }).collect::<Vec<_>>();
    let dfa = compile_expression_labeled_nfa(&nfa, &symbols)
        .map_err(SchemaImportError::new)?;
    Ok(GrammarExpr::LexerDfa(Arc::new(dfa)))
}

fn positive_integer_multiple_value(multiple: f64) -> Option<u64> {
    if !multiple.is_finite()
        || multiple < 1.0
        || multiple >= u64::MAX as f64
        || multiple.fract() != 0.0
    {
        return None;
    }
    let value = multiple as u64;
    if (value as f64) == multiple { Some(value) } else { None }
}

fn power_of_ten_multiple_regex(multiple: f64, allow_sign: bool) -> Option<String> {
    power_of_ten_divisor_regex(positive_integer_multiple_value(multiple)?, allow_sign)
}

fn power_of_ten_divisor_regex(mut value: u64, allow_sign: bool) -> Option<String> {
    if value == 0 { return None; }
    let sign = if allow_sign { "-?" } else { "" };
    let mut zeros = 0usize;
    while value > 1 && value.is_multiple_of(10) {
        zeros += 1;
        value /= 10;
    }
    if value != 1 { return None; }
    Some(format!(r#"{sign}(0|[1-9][0-9]*{})"#, "0".repeat(zeros)))
}
fn decimal_multiple_regex(multiple: f64) -> Option<String> {
    let step = parse_decimal_step(multiple)?;
    let fraction = decimal_fraction_regex(&step)?;
    // Local llguidance-parity compromise for simple decimal multiples:
    // llguidance-native rejects a signed numeric start for `multipleOf: 0.01`
    // in mask sweeps, while zero and integer spellings are valid. Keep this
    // compact language non-negative, but do not tighten to a fixed fractional
    // scale: redundant trailing zeros are normal JSON number spellings of the
    // same decimal value and should not be rejected without direct evidence
    // from llguidance/derivre.
    Some(format!(r#"(?:0|[1-9][0-9]*)(?:\.(?:{fraction}))?"#))
}

struct DecimalStep {
    numerator: u64,
    scale: u64,
    scale_digits: usize,
}

fn parse_decimal_step(multiple: f64) -> Option<DecimalStep> {
    if !multiple.is_finite() || multiple <= 0.0 || multiple >= 1.0 { return None; }
    let decimal = DecimalMagnitude::from_float(multiple)?;
    let scale_digits = decimal.exponent.unsigned_abs() as usize;
    let scale = 10u64.checked_pow(u32::try_from(scale_digits).ok()?)?;
    let numerator = u64::try_from(decimal.digits).ok()?;
    (numerator != 0 && numerator < scale).then_some(DecimalStep { numerator, scale, scale_digits })
}

fn decimal_fraction_regex(step: &DecimalStep) -> Option<String> {
    if !step.scale.is_multiple_of(step.numerator) {
        return None;
    }

    if step.numerator == 1 {
        if step.scale_digits == 1 {
            return Some(r#"[0-9]"#.to_string());
        }
        return Some(format!(r#"[0-9]{{1,{}}}"#, step.scale_digits));
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
        .map(|prefix| decimal_fraction_prefix_regex(&prefix, step.scale_digits))
        .collect::<Vec<_>>();
    Some(parts.join("|"))
}

fn decimal_fraction_prefix_regex(prefix: &str, scale_digits: usize) -> String {
    let extra_zeros = scale_digits.saturating_sub(prefix.len());
    if extra_zeros == 0 {
        return prefix.to_string();
    }
    if prefix == "0" {
        return format!("0{{1,{scale_digits}}}");
    }
    format!("{prefix}0{{0,{extra_zeros}}}")
}

#[cfg(test)]
mod integer_modulo_tests {
    use super::*;
    use crate::json_schema::config::JsonSchemaConfig;
    use crate::json_schema::load::load_document;
    use crate::json_schema::lower::lower_document;

    #[test]
    fn integer_modulo_unbounded_constraint_must_not_be_dropped() {
        let document = load_document(&serde_json::json!({"type": "integer"})).unwrap();
        let unrestricted = lower_document(&document, JsonSchemaConfig::default()).unwrap();
        for divisor in [3.0, 7.0, 12.0, 37.0] {
            let document = load_document(&serde_json::json!({"type": "integer", "multipleOf": divisor})).unwrap();
            let constrained = lower_document(&document, JsonSchemaConfig::default()).unwrap();
            assert_ne!(constrained.rules, unrestricted.rules, "multipleOf={divisor} became unconstrained integers");
        }
    }

    #[test]
    fn integer_modulo_unsupported_large_divisor_must_fail_closed() {
        let document = load_document(&serde_json::json!({"type": "integer", "multipleOf": 1_000_003})).unwrap();
        assert!(lower_document(&document, JsonSchemaConfig::default()).is_err(),
                "an unsupported divisor must not silently discard divisibility");
    }

    fn accepts(dfa: &crate::automata::lexer::DFA, bytes: &[u8]) -> bool {
        let mut state = 0;
        for &byte in bytes {
            let Some(next) = dfa.step(state, byte) else { return false; };
            state = next;
        }
        dfa.finalizers(state).contains(0)
    }

    #[test]
    fn integer_modulo_dfa_matches_arithmetic_exhaustively() {
        for divisor in (1..=64u64).chain([127, 257, 4096]) {
            let GrammarExpr::LexerDfa(dfa) = integer_modulo_expr(divisor).unwrap() else { unreachable!() };
            for value in -4096i64..=4096 {
                assert_eq!(accepts(&dfa, value.to_string().as_bytes()), value % divisor as i64 == 0,
                           "value={value}, divisor={divisor}");
            }
            for bytes in [b"".as_slice(), b"-", b"+0", b"00", b"-00", b"01", b"1.0", b"1e3", b" 0", b"0 ", b"1x"] {
                assert!(!accepts(&dfa, bytes), "invalid integer spelling {bytes:?}");
            }
            assert!(accepts(&dfa, b"-0"));
        }
    }

    #[test]
    fn integer_modulo_has_no_input_digit_horizon() {
        for divisor in [3u64, 7, 12, 37, 257, 4096] {
            let GrammarExpr::LexerDfa(dfa) = integer_modulo_expr(divisor).unwrap() else { unreachable!() };
            let mut bytes = Vec::new();
            let mut remainder = 0u64;
            for i in 0..2048 {
                let digit = if i == 0 { 1 } else { ((i * 7 + 3) % 10) as u64 };
                bytes.push(b'0' + digit as u8);
                remainder = (10 * remainder + digit) % divisor;
                assert_eq!(accepts(&dfa, &bytes), remainder == 0);
            }
            let mut exact_multiple = divisor.to_string().into_bytes();
            exact_multiple.extend(std::iter::repeat_n(b'0', 2048));
            assert!(accepts(&dfa, &exact_multiple));
        }
    }

    #[test]
    fn integer_modulo_finite_filter_is_exact_above_float_precision() {
        for value in [i64::MIN, i64::MIN + 1, -9_007_199_254_740_993, 9_007_199_254_740_993, i64::MAX - 1, i64::MAX] {
            for divisor in [2i64, 3, 7, 12, 37] {
                assert_eq!(integer_satisfies_multiple(value, Some(divisor as f64)), value % divisor == 0,
                           "value={value} divisor={divisor}");
            }
        }
    }

    #[test]
    fn integer_modulo_full_signed_range_does_not_overflow_enumeration() {
        assert!(bounded_integer_multiple_choice(i64::MIN, i64::MAX, Some(3.0)).is_none(),
                "too many multiples must fall through to the DFA, not become an empty language");
        let expr = bounded_integer_multiple_choice(i64::MIN, i64::MAX, Some((1u64 << 62) as f64)).unwrap();
        let expected = choice([i64::MIN, -(1i64 << 62), 0, 1i64 << 62]
            .into_iter().map(|value| lit_bytes(value.to_string().into_bytes())).collect());
        assert_eq!(expr, expected);
    }

    #[test]
    fn decimal_multiple_reduces_to_exact_integer_divisor() {
        for (multiple, divisor) in [(0.03, 3i64), (1.5, 3), (2.5, 5), (0.01, 1)] {
            for value in -100..=100 {
                assert_eq!(integer_satisfies_multiple(value, Some(multiple)), value % divisor == 0);
            }
        }
        let value = serde_json::Number::from_f64(0.3).unwrap();
        assert!(number_is_multiple(&value, 0.1));
        assert!(!number_is_multiple(&value, 0.2));
    }

    #[test]
    fn integer_modulo_large_divisor_conversion_is_not_saturating() {
        let two_to_63 = (1u64 << 63) as f64;
        let two_to_64 = u64::MAX as f64;
        assert_eq!(positive_integer_multiple_value(two_to_63), Some(1u64 << 63));
        assert!(positive_integer_multiple_value(two_to_64).is_none());
        assert!(!integer_satisfies_multiple(i64::MIN, Some(two_to_64)));
        assert!(!integer_satisfies_multiple(i64::MAX, Some(two_to_64)));
        assert!(integer_satisfies_multiple(0, Some(two_to_64)));
        assert!(integer_satisfies_multiple(i64::MIN, Some(two_to_63)));
        assert_eq!(bounded_integer_multiple_choice(i64::MIN, i64::MAX, Some(two_to_63)),
                   Some(choice(vec![lit_bytes(i64::MIN.to_string().into_bytes()), lit_bytes(b"0".to_vec())])));
    }
}
