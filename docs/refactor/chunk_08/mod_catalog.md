# Number lowerer function catalog

Functions that map numeric schemas to JSON number regexes or literal choices.

| Line | Symbol | Priority | Publication action |
|---:|---|---:|---|
| 10 | `const MAX_EXPLICIT_INTEGER_RANGE: i64 = 512;` | 3 | keep in current phase, add exactness comment if semantic |
| 11 | `const MAX_EXPLICIT_INTEGER_MULTIPLES: i64 = 2048;` | 3 | keep in current phase, add exactness comment if semantic |
| 13 | `impl<'a> Lowerer<'a> {` | 1 | high-prominence boundary; document before modifying |
| 14 | `pub(crate) fn lower_number(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {` | 1 | high-prominence boundary; document before modifying |
| 59 | `fn lower_integer(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {` | 1 | high-prominence boundary; document before modifying |
| 107 | `fn integer_lower_bound(schema: &NumberSchema) -> Option<i64> {` | 1 | high-prominence boundary; document before modifying |
| 119 | `fn integer_upper_bound(schema: &NumberSchema) -> Option<i64> {` | 1 | high-prominence boundary; document before modifying |
| 131 | `fn integer_satisfies_multiple(value: i64, multiple: Option<f64>) -> bool {` | 3 | keep in current phase, add exactness comment if semantic |
| 139 | `fn bounded_integer_multiple_choice(` | 3 | keep in current phase, add exactness comment if semantic |
| 162 | `fn ceil_div_i64(value: i64, divisor: i64) -> i64 {` | 3 | keep in current phase, add exactness comment if semantic |
| 168 | `fn integer_multiple_expr(multiple: f64) -> Option<GrammarExpr> {` | 3 | keep in current phase, add exactness comment if semantic |
| 172 | `fn positive_integer_multiple_value(multiple: f64) -> Option<u64> {` | 3 | keep in current phase, add exactness comment if semantic |
| 180 | `fn positive_integer_multiple_i64(multiple: f64) -> Option<i64> {` | 3 | keep in current phase, add exactness comment if semantic |
| 185 | `fn power_of_ten_multiple_regex(multiple: f64) -> Option<String> {` | 2 | hotspot helper; split or proof-comment in follow-up |
| 206 | `fn decimal_multiple_regex(multiple: f64) -> Option<String> {` | 2 | hotspot helper; split or proof-comment in follow-up |
| 212 | `struct DecimalStep {` | 3 | keep in current phase, add exactness comment if semantic |
| 218 | `fn parse_decimal_step(multiple: f64) -> Option<DecimalStep> {` | 3 | keep in current phase, add exactness comment if semantic |
| 247 | `fn decimal_fraction_regex(step: &DecimalStep) -> Option<String> {` | 2 | hotspot helper; split or proof-comment in follow-up |
