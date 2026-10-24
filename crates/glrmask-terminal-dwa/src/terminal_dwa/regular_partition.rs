use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use regex_syntax::utf8::Utf8Sequences;

use crate::automata::lexer::compile::{build_regex, compile_terminal_expr_dfa};
use crate::automata::lexer::tokenizer::{Lexer, Tokenizer};
use crate::automata::lexer::DFA;
use crate::automata::regex::Expr;
use crate::ds::u8set::U8Set;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CharTypeRegularPartitionKey {
    pub(crate) p0_overflow_threshold: Option<usize>,
    pub(crate) p1_overflow_threshold: Option<usize>,
    pub(crate) p2_overflow_threshold: Option<usize>,
    pub(crate) p4_overflow_threshold: Option<usize>,
    pub(crate) structural_boundary_enabled: bool,
}

#[derive(Debug)]
pub(crate) struct RegularPartitionCell {
    index: usize,
    label: String,
}

impl RegularPartitionCell {
    pub(crate) fn index(&self) -> usize { self.index }
    pub(crate) fn label(&self) -> &str { &self.label }
}

#[derive(Debug)]
struct RegularPartitionRule {
    target_partition: usize,
    expression: Expr,
}

#[derive(Debug, Clone, Copy)]
enum RegularPartitionDefinition {
    CharType { structural_boundary_enabled: bool },
}

#[derive(Debug, Clone, Copy)]
enum RegularPartitionOutput {
    FirstMatchTarget,
}

/// A vocabulary partition whose cells are actual regular languages.
///
/// Unlike a callback `bytes -> partition_id`, the exact language object that
/// induces vocabulary membership remains available to containment proofs.
#[derive(Debug)]
pub(crate) struct RegularPartitionSpec {
    name: &'static str,
    cells: Vec<RegularPartitionCell>,
    definition: RegularPartitionDefinition,
    output: RegularPartitionOutput,
    rules: OnceLock<Vec<RegularPartitionRule>>,
    automaton: OnceLock<Arc<Tokenizer>>,
}

#[derive(Clone, Copy)]
pub(crate) struct RegularPartitionCellRef<'a> {
    pub(crate) spec: &'a RegularPartitionSpec,
    pub(crate) partition: usize,
    pub(crate) base_partition: usize,
    pub(crate) min_len: usize,
    pub(crate) max_len: Option<usize>,
}

impl RegularPartitionSpec {
    pub(crate) fn name(&self) -> &'static str { self.name }
    pub(crate) fn cells(&self) -> &[RegularPartitionCell] { &self.cells }
    pub(crate) fn cell(&self, index: usize) -> Option<&RegularPartitionCell> { self.cells.get(index) }

    fn rules(&self) -> &[RegularPartitionRule] {
        self.rules.get_or_init(|| match self.definition {
            RegularPartitionDefinition::CharType { structural_boundary_enabled } => {
                build_char_type_regular_rules(structural_boundary_enabled)
            }
        })
    }

    pub(crate) fn automaton(&self) -> &Tokenizer {
        self.automaton.get_or_init(|| {
            let started = Instant::now();
            let rules = self.rules();
            let expressions = rules.iter().map(|rule| rule.expression.clone()).collect::<Vec<_>>();
            let expressions_for_tokenizer: Arc<[Expr]> = Arc::from(expressions.clone().into_boxed_slice());
            let automaton = Arc::new(
                build_regex(&expressions)
                    .into_tokenizer(rules.len() as u32, Some(expressions_for_tokenizer)),
            );
            if std::env::var_os("GLRMASK_PROFILE_REGULAR_PARTITION_BUILD").is_some() {
                eprintln!(
                    "[glrmask/profile][regular_partition_automaton] spec={} states={} compile_ms={:.3}",
                    self.name,
                    automaton.num_states(),
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            automaton
        })
    }

    /// Output of the exact regular partition transducer after consuming a
    /// complete string. Rule order is semantic priority, mirroring an ordered
    /// regex decision list. The final fallback rule is `.*`, so every byte
    /// string is covered.
    pub(crate) fn partition_for_state(&self, state: u32) -> Option<usize> {
        let rules = self.rules();
        match self.output {
            RegularPartitionOutput::FirstMatchTarget => self
                .automaton()
                .matched_terminals_iter(state)
                .min()
                .map(|rule| rules[rule as usize].target_partition),
        }
    }

    pub(crate) fn state_accepts_cell(&self, state: u32, partition: usize) -> bool {
        self.partition_for_state(state) == Some(partition)
    }

    /// Exact standalone regular language for one output cell.
    ///
    /// This is derived mechanically from the same ordered rules / membership
    /// bits that define vocabulary membership. Containment code should use
    /// this language rather than approximating a finite token bucket.
    pub(crate) fn cell_expression(&self, partition: usize) -> Expr {
        match self.output {
            RegularPartitionOutput::FirstMatchTarget => {
                let mut earlier = Vec::<Expr>::new();
                let mut pieces = Vec::<Expr>::new();
                for rule in self.rules() {
                    if rule.target_partition == partition {
                        let piece = if earlier.is_empty() {
                            rule.expression.clone()
                        } else {
                            exclude(rule.expression.clone(), choice(earlier.clone()))
                        };
                        pieces.push(piece);
                    }
                    earlier.push(rule.expression.clone());
                }
                choice(pieces)
            }
        }
    }

    pub(crate) fn partition_index(&self, bytes: &[u8]) -> usize {
        let automaton = self.automaton();
        let mut state = automaton.start_state();
        for &byte in bytes {
            let Some(next) = automaton.step(state, byte) else {
                panic!("regular partition {} lost coverage for bytes={bytes:?}", self.name);
            };
            state = next;
        }
        self.partition_for_state(state).unwrap_or_else(|| {
            panic!("regular partition {} does not cover bytes={bytes:?}", self.name)
        })
    }
}

impl RegularPartitionCellRef<'_> {
    pub(crate) fn exact_expression(&self) -> Expr {
        let base = self.spec.cell_expression(self.base_partition);
        if self.min_len == 0 && self.max_len.is_none() {
            base
        } else {
            intersect(base, byte_len(self.min_len, self.max_len))
        }
    }

    pub(crate) fn accepts_state_at_len(&self, state: u32, len: usize) -> bool {
        self.spec.partition_for_state(state) == Some(self.base_partition)
            && len >= self.min_len
            && self.max_len.is_none_or(|max| len <= max)
    }

    /// Finite length-state transition for product containment. For finite
    /// upper bounds, no accepted string can exist beyond `max`, so `None`
    /// prunes that branch. For unbounded cells, lengths saturate at `min`.
    pub(crate) fn advance_len_state(&self, len: usize) -> Option<usize> {
        if let Some(max) = self.max_len {
            (len < max).then_some(len + 1)
        } else {
            Some((len + 1).min(self.min_len))
        }
    }
}

fn byte_class(set: U8Set) -> Expr { Expr::U8Class(set) }
fn literal(bytes: &[u8]) -> Expr { Expr::U8Seq(bytes.to_vec()) }
fn seq(parts: Vec<Expr>) -> Expr { Expr::Seq(parts) }
fn choice(parts: Vec<Expr>) -> Expr {
    match parts.len() {
        0 => Expr::U8Class(U8Set::empty()),
        1 => parts.into_iter().next().unwrap(),
        _ => Expr::Choice(parts),
    }
}
fn repeat(expr: Expr, min: usize, max: Option<usize>) -> Expr {
    Expr::Repeat { expr: Box::new(expr), min, max }
}
fn intersect(expr: Expr, other: Expr) -> Expr {
    Expr::Intersect { expr: Box::new(expr), intersect: Box::new(other) }
}
fn exclude(expr: Expr, other: Expr) -> Expr {
    Expr::Exclude { expr: Box::new(expr), exclude: Box::new(other) }
}
fn shared(expr: Expr) -> Expr { Expr::Shared(Arc::new(expr)) }
fn dfa_expr(dfa: DFA) -> Expr { Expr::Dfa(Arc::new(dfa)) }
fn any_byte() -> Expr { byte_class(!U8Set::empty()) }
fn any_bytes() -> Expr { repeat(any_byte(), 0, None) }
fn byte_len(min: usize, max: Option<usize>) -> Expr { repeat(any_byte(), min, max) }

fn utf8_scalar_range_expr(start: char, end: char) -> Expr {
    choice(
        Utf8Sequences::new(start, end)
            .map(|sequence| {
                seq(sequence.as_slice().iter().map(|range| {
                    byte_class(U8Set::from_range(range.start, range.end))
                }).collect())
            })
            .collect(),
    )
}

fn unicode_scalar_predicate_expr(predicate: fn(char) -> bool) -> Expr {
    let mut choices = Vec::new();
    let mut range_start = None::<char>;
    let mut previous = None::<char>;
    for codepoint in 0..=0x10FFFFu32 {
        let Some(character) = char::from_u32(codepoint) else { continue; };
        if predicate(character) {
            if range_start.is_none() { range_start = Some(character); }
            previous = Some(character);
        } else if let (Some(start), Some(end)) = (range_start.take(), previous.take()) {
            choices.push(utf8_scalar_range_expr(start, end));
        }
    }
    if let (Some(start), Some(end)) = (range_start, previous) {
        choices.push(utf8_scalar_range_expr(start, end));
    }
    choice(choices)
}

fn is_word_scalar(character: char) -> bool { character.is_alphanumeric() || character == '_' }
fn is_alpha_scalar(character: char) -> bool { character.is_alphabetic() || character == '_' }

fn json_literal_collision_expr() -> Expr {
    let ascii_alnum = byte_class(U8Set::from_predicate(|byte| byte.is_ascii_alphanumeric()));
    let mut alternatives = Vec::new();
    for literal_bytes in [b"true".as_slice(), b"false".as_slice(), b"null".as_slice()] {
        for prefix_len in 1..literal_bytes.len() {
            alternatives.push(literal(&literal_bytes[..prefix_len]));
        }
        alternatives.push(seq(vec![literal(literal_bytes), repeat(ascii_alnum.clone(), 0, None)]));
    }
    choice(alternatives)
}

fn build_char_type_raw_languages(structural_boundary_enabled: bool) -> CharTypeRawLanguages {
    let profile = std::env::var_os("GLRMASK_PROFILE_REGULAR_PARTITION_BUILD").is_some();
    let started = Instant::now();
    let any = shared(any_bytes());
    let any_scalar = dfa_expr(compile_terminal_expr_dfa(&utf8_scalar_range_expr('\0', '\u{10ffff}')));
    let any_scalar_ms = started.elapsed().as_secs_f64() * 1000.0;
    let valid_utf8 = shared(repeat(any_scalar.clone(), 0, None));
    let word_started = Instant::now();
    let word_scalar = dfa_expr(compile_terminal_expr_dfa(&unicode_scalar_predicate_expr(is_word_scalar)));
    let word_ms = word_started.elapsed().as_secs_f64() * 1000.0;
    let alpha_started = Instant::now();
    let alpha_scalar = dfa_expr(compile_terminal_expr_dfa(&unicode_scalar_predicate_expr(is_alpha_scalar)));
    let alpha_ms = alpha_started.elapsed().as_secs_f64() * 1000.0;
    let word_star = || repeat(word_scalar.clone(), 0, None);
    let word_plus = || repeat(word_scalar.clone(), 1, None);
    let optional_leading_space = || repeat(literal(b" "), 0, Some(1));
    let ascii_alpha = byte_class(U8Set::from_predicate(|byte| byte.is_ascii_alphabetic() || byte == b'_'));

    let word_with_ascii_alpha = seq(vec![word_star(), ascii_alpha.clone(), word_star()]);
    let word_with_alpha = seq(vec![word_star(), alpha_scalar.clone(), word_star()]);
    let p2_raw = seq(vec![optional_leading_space(), word_with_ascii_alpha.clone()]);
    // P2 precedes P4, and P4 precedes P3 in the ordered rule list below. The
    // raw languages can therefore stay simple; priority supplies the exact
    // set difference without constructing complement products.
    let p4_raw = seq(vec![optional_leading_space(), word_with_alpha]);
    let p3_raw = seq(vec![optional_leading_space(), word_plus()]);

    let structural_boundary = if structural_boundary_enabled {
        let collision = shared(json_literal_collision_expr());
        choice(vec![
            seq(vec![literal(b" "), collision.clone()]),
            seq(vec![literal(b"["), collision]),
            literal(b" -"),
        ])
    } else { choice(Vec::new()) };
    let quoted_identifier_boundary = if structural_boundary_enabled {
        seq(vec![literal(b"\""), ascii_alpha.clone(), any.clone()])
    } else { choice(Vec::new()) };
    let sign_mixed = seq(vec![
        repeat(literal(b" "), 0, Some(1)),
        byte_class(U8Set::from_bytes(b"+-")),
    ]);

    // Exact language corresponding to the old non-alphanumeric branch,
    // including its byte-level invalid-UTF-8 fallback.
    let nonword_scalar = shared(exclude(any_scalar.clone(), word_scalar.clone()));
    let valid_nonword = repeat(nonword_scalar, 1, None);
    let non_ascii_word_byte = byte_class(U8Set::from_predicate(|byte| !byte.is_ascii_alphanumeric() && byte != b'_'));
    let invalid_nonword = intersect(repeat(non_ascii_word_byte, 1, None), exclude(any.clone(), valid_utf8));
    let nonalnum = shared(choice(vec![valid_nonword, invalid_nonword]));

    let json_structural = U8Set::from_bytes(b"\":[]{},");
    let no_json_structural = repeat(byte_class(!json_structural), 1, None);
    let repeated_aux = choice(b"\n:{ ,".iter().map(|&byte| repeat(literal(&[byte]), 2, None)).collect());
    let auxiliary = shared(choice(vec![byte_len(1, Some(1)), no_json_structural, repeated_aux]));
    let p5_raw = choice(vec![
        Expr::Epsilon,
        intersect(nonalnum.clone(), intersect(auxiliary.clone(), byte_len(1, Some(8)))),
    ]);
    let p6_raw = intersect(nonalnum.clone(), intersect(auxiliary.clone(), byte_len(9, None)));
    if profile {
        eprintln!(
            "[glrmask/profile][regular_partition_unicode_atoms] any_scalar_ms={any_scalar_ms:.3} word_ms={word_ms:.3} alpha_ms={alpha_ms:.3} total_base_ms={:.3}",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    CharTypeRawLanguages {
        p8: quoted_identifier_boundary,
        p7: structural_boundary,
        p1_sign: sign_mixed,
        p2: p2_raw,
        p4: p4_raw,
        p3: p3_raw,
        p5: p5_raw,
        p6: p6_raw,
        p0: nonalnum,
        any,
    }
}

struct CharTypeRawLanguages {
    p8: Expr,
    p7: Expr,
    p1_sign: Expr,
    p2: Expr,
    p4: Expr,
    p3: Expr,
    p5: Expr,
    p6: Expr,
    p0: Expr,
    any: Expr,
}

fn build_char_type_regular_rules(structural_boundary_enabled: bool) -> Vec<RegularPartitionRule> {
    let profile = std::env::var_os("GLRMASK_PROFILE_REGULAR_PARTITION_BUILD").is_some();
    let started = Instant::now();
    let raw = build_char_type_raw_languages(structural_boundary_enabled);
    let rules = vec![
        RegularPartitionRule { target_partition: 8, expression: raw.p8 },
        RegularPartitionRule { target_partition: 7, expression: raw.p7 },
        RegularPartitionRule { target_partition: 1, expression: raw.p1_sign },
        RegularPartitionRule { target_partition: 2, expression: raw.p2 },
        RegularPartitionRule { target_partition: 4, expression: raw.p4 },
        RegularPartitionRule { target_partition: 3, expression: raw.p3 },
        RegularPartitionRule { target_partition: 5, expression: raw.p5 },
        RegularPartitionRule { target_partition: 6, expression: raw.p6 },
        RegularPartitionRule { target_partition: 0, expression: raw.p0 },
        RegularPartitionRule { target_partition: 1, expression: raw.any },
    ];
    if profile {
        eprintln!(
            "[glrmask/profile][regular_partition_rules] spec=char_type_regular rules={} total_ms={:.3}",
            rules.len(),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    rules
}


fn build_char_type_regular_partition(key: CharTypeRegularPartitionKey) -> RegularPartitionSpec {
    debug_assert!(key.p0_overflow_threshold.is_none());
    debug_assert!(key.p1_overflow_threshold.is_none());
    debug_assert!(key.p2_overflow_threshold.is_none());
    debug_assert!(key.p4_overflow_threshold.is_none());
    let cells = (0..9)
        .map(|index| RegularPartitionCell { index, label: format!("p{index}") })
        .collect();
    RegularPartitionSpec {
        name: "char_type_regular",
        cells,
        definition: RegularPartitionDefinition::CharType {
            structural_boundary_enabled: key.structural_boundary_enabled,
        },
        output: RegularPartitionOutput::FirstMatchTarget,
        rules: OnceLock::new(),
        automaton: OnceLock::new(),
    }
}


pub(crate) fn plain_partition_cell_ref<'a>(
    spec: &'a RegularPartitionSpec,
    partition: usize,
) -> RegularPartitionCellRef<'a> {
    RegularPartitionCellRef {
        spec,
        partition,
        base_partition: partition,
        min_len: 0,
        max_len: None,
    }
}

pub(crate) fn char_type_regular_partition(key: CharTypeRegularPartitionKey) -> Arc<RegularPartitionSpec> {
    static CACHE: OnceLock<Mutex<HashMap<CharTypeRegularPartitionKey, Arc<RegularPartitionSpec>>>> = OnceLock::new();
    let key = CharTypeRegularPartitionKey {
        p0_overflow_threshold: None,
        p1_overflow_threshold: None,
        p2_overflow_threshold: None,
        p4_overflow_threshold: None,
        structural_boundary_enabled: key.structural_boundary_enabled,
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(existing) = cache.lock().unwrap().get(&key).cloned() { return existing; }
    let built = Arc::new(build_char_type_regular_partition(key));
    let mut cache = cache.lock().unwrap();
    Arc::clone(cache.entry(key).or_insert(built))
}

pub(crate) fn char_type_partition_count(key: CharTypeRegularPartitionKey) -> usize {
    if key.p0_overflow_threshold.is_some() { 13 }
    else if key.p4_overflow_threshold.is_some() { 12 }
    else if key.p1_overflow_threshold.is_some() { 11 }
    else if key.p2_overflow_threshold.is_some() { 10 }
    else { 9 }
}

pub(crate) fn char_type_final_partition(
    base_partition: usize,
    len: usize,
    key: CharTypeRegularPartitionKey,
) -> usize {
    if base_partition == 0 && key.p0_overflow_threshold.is_some_and(|threshold| len > threshold) {
        12
    } else if base_partition == 1 && key.p1_overflow_threshold.is_some_and(|threshold| len > threshold) {
        10
    } else if base_partition == 2 && key.p2_overflow_threshold.is_some_and(|threshold| len > threshold) {
        9
    } else if base_partition == 4 && key.p4_overflow_threshold.is_some_and(|threshold| len > threshold) {
        11
    } else {
        base_partition
    }
}

pub(crate) fn char_type_partition_cell_ref<'a>(
    spec: &'a RegularPartitionSpec,
    key: CharTypeRegularPartitionKey,
    partition: usize,
) -> RegularPartitionCellRef<'a> {
    let (base_partition, min_len, max_len) = match partition {
        9 => (2, key.p2_overflow_threshold.map_or(usize::MAX, |threshold| threshold + 1), None),
        10 => (1, key.p1_overflow_threshold.map_or(usize::MAX, |threshold| threshold + 1), None),
        11 => (4, key.p4_overflow_threshold.map_or(usize::MAX, |threshold| threshold + 1), None),
        12 => (0, key.p0_overflow_threshold.map_or(usize::MAX, |threshold| threshold + 1), None),
        0 => (0, 0, key.p0_overflow_threshold),
        1 => (1, 0, key.p1_overflow_threshold),
        2 => (2, 0, key.p2_overflow_threshold),
        4 => (4, 0, key.p4_overflow_threshold),
        base => (base, 0, None),
    };
    RegularPartitionCellRef {
        spec,
        partition,
        base_partition,
        min_len,
        max_len,
    }
}
