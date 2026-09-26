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
    let SchemaKind::Assertions(assertions) = &schema.kind else { return; };
    if plain_integer_assertions(assertions) {
        if let Some(number) = &assertions.number
            && integer_multiple_is_vacuous(number.multiple_of)
        {
            let lower = integer_lower_bound(number);
            let upper = integer_upper_bound(number);
            if lower == Some(i64::MIN) || upper == Some(i64::MAX) {
                *representable = false;
            } else {
                if let Some(lower) = lower { cuts.insert(lower); }
                if let Some(upper) = upper { cuts.insert(upper + 1); }
            }
        }
    }
    if let Some(object) = &assertions.object {
        for property in &object.properties {
            collect_plain_integer_range_cuts(&property.schema, cuts, representable);
        }
        for property in &object.pattern_properties {
            collect_plain_integer_range_cuts(&property.schema, cuts, representable);
        }
        if let super::ast::AdditionalProperties::Schema(additional) = &object.additional_properties {
            collect_plain_integer_range_cuts(additional, cuts, representable);
        }
        if let Some(property_names) = &object.property_names {
            collect_plain_integer_range_cuts(property_names, cuts, representable);
        }
    }
    if let Some(array) = &assertions.array {
        collect_plain_integer_range_cuts(&array.items, cuts, representable);
        for item in &array.prefix_items {
            collect_plain_integer_range_cuts(item, cuts, representable);
        }
    }
    for branch in assertions.any_of.iter().chain(&assertions.one_of).chain(&assertions.all_of) {
        collect_plain_integer_range_cuts(branch, cuts, representable);
    }
    if let Some(not) = &assertions.not {
        collect_plain_integer_range_cuts(not, cuts, representable);
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
        let mut lower = integer_lower_bound(schema);
        let upper = integer_upper_bound(schema);
        if self.llguidance_compat_enabled()
            && schema.multiple_of.is_some()
            && lower.is_some_and(|value| value < 0)
            && upper.is_some_and(|value| value >= 0)
        {
            lower = Some(0);
        }
        if let (Some(lower), Some(upper)) = (lower, upper)
            && lower > upper
        {
            return Ok(never());
        }
        // `multipleOf: 1` is vacuous for an integer schema. Normalize it to
        // the ordinary integer-range representation after applying the optional
        // llguidance compatibility bound adjustment above. This avoids expanding
        // a finite [L,U] interval into one parser terminal per integer solely
        // because the schema redundantly states `multipleOf: 1`.
        if integer_multiple_is_vacuous(schema.multiple_of) {
            if let Some(expr) = self.shared_integer_range_expr(lower, upper)? {
                return Ok(expr);
            }
            if lower.is_some() || upper.is_some() {
                let regex = rx_int_range(lower, upper).map_err(SchemaImportError::new)?;
                return Ok(GrammarExpr::RawRegex(regex));
            }
            return Ok(r(JSON_INTEGER_RULE));
        }

        if let (Some(lower), Some(upper)) = (lower, upper) {
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
            if let Some(divisor) = positive_integer_multiple_value(multiple) {
                let expr = integer_modulo_expr(divisor)?;
                if lower.is_some() || upper.is_some() {
                    let regex = rx_int_range(lower, upper).map_err(SchemaImportError::new)?;
                    return Ok(GrammarExpr::Intersect {
                        expr: Box::new(expr),
                        intersect: Box::new(GrammarExpr::RawRegex(regex)),
                    });
                }
                return Ok(expr);
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

#[inline]
fn integer_multiple_is_vacuous(multiple: Option<f64>) -> bool {
    multiple.is_none_or(|multiple| multiple == 1.0)
}

fn integer_satisfies_multiple(value: i64, multiple: Option<f64>) -> bool {
    let Some(multiple) = multiple else {
        return true;
    };
    if !multiple.is_finite() || multiple <= 0.0 {
        return false;
    }
    if multiple.fract() == 0.0 {
        // Every i64 magnitude is below 2^64. Avoid both a saturating float
        // conversion at that boundary and loss of integer precision above2^53.
        if multiple >= u64::MAX as f64 {
            return value == 0;
        }
        return value.unsigned_abs() % (multiple as u64) == 0;
    }
    let quotient = (value as f64) / multiple;
    (quotient - quotient.round()).abs() < 1e-9
}

fn bounded_integer_multiple_choice(
    lower: i64,
    upper: i64,
    multiple: Option<f64>,
) -> Option<GrammarExpr> {
    let multiple = i128::from(positive_integer_multiple_value(multiple?)?);
    let lower = i128::from(lower);
    let upper = i128::from(upper);
    let quotient = lower / multiple;
    let first = (quotient + i128::from(lower % multiple > 0)) * multiple;
    if first > upper {
        return Some(never());
    }
    let count = ((upper - first) / multiple) + 1;
    if count > i128::from(MAX_EXPLICIT_INTEGER_MULTIPLES) {
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
    power_of_ten_multiple_regex(multiple, true).map(GrammarExpr::RawRegex)
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

fn positive_integer_multiple_i64(multiple: f64) -> Option<i64> {
    let value = positive_integer_multiple_value(multiple)?;
    i64::try_from(value).ok()
}

fn power_of_ten_multiple_regex(multiple: f64, allow_sign: bool) -> Option<String> {
    if !multiple.is_finite() || multiple < 1.0 || multiple.fract() != 0.0 {
        return None;
    }
    let mut value = multiple as u64;
    let sign = if allow_sign { "-?" } else { "" };
    if value == 1 {
        return Some(format!(r#"{sign}(0|[1-9][0-9]*)"#));
    }

    let mut zeros = 0usize;
    while value > 1 && value % 10 == 0 {
        zeros += 1;
        value /= 10;
    }
    if value != 1 {
        return None;
    }

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
