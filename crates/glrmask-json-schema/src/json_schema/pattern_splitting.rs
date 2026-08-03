//! Exact importer-level decomposition for expensive anchored JSON string patterns.
//!
//! This module is deliberately restricted to JSON Schema `pattern` values. It
//! analyzes regex HIR before terminal compilation and only admits fully anchored
//! patterns containing one structurally unique, large bounded repetition in a
//! branch. Profitability is heuristic; language preservation is algebraic.

use regex_syntax::hir::{Hir, HirKind, Literal};
use regex_syntax::Parser;

use super::error::{ImportResult, SchemaImportError};
use super::string::strip_outer_anchors;

const MIN_REPEAT: usize = 24;
const MIN_COMPLEXITY_SCORE: usize = 256;
const MIN_BLOCK: usize = 8;
const MAX_GROUPS: usize = 32;

#[derive(Clone, Debug)]
pub(super) struct ComplexPatternPlan {
    pub(super) branches: Vec<ComplexPatternBranch>,
}

#[derive(Clone, Debug)]
pub(super) enum ComplexPatternBranch {
    Passthrough(Hir),
    Counted(CountedPatternBranch),
}

#[derive(Clone, Debug)]
pub(super) struct CountedPatternBranch {
    pub(super) prefix: Hir,
    pub(super) body: Hir,
    pub(super) suffix: Hir,
    pub(super) max_repeat: usize,
    pub(super) block: usize,
}

impl CountedPatternBranch {
    pub(super) fn count_intervals(&self) -> Vec<(usize, usize)> {
        let q = self.max_repeat / self.block;
        let r = self.max_repeat % self.block;
        let mut intervals = vec![(0, self.block - 1)];
        for group in 1..q {
            intervals.push((group * self.block, group * self.block + self.block - 1));
        }
        intervals.push((q * self.block, q * self.block + r));
        intervals
    }

    pub(super) fn count_cover_is_exact(&self) -> bool {
        let intervals = self.count_intervals();
        let mut expected = 0usize;
        for (index, (start, end)) in intervals.iter().copied().enumerate() {
            if start != expected || start > end || end > self.max_repeat {
                return false;
            }
            if end == self.max_repeat {
                return index + 1 == intervals.len();
            }
            let Some(next) = end.checked_add(1) else {
                return false;
            };
            expected = next;
        }
        false
    }
}

#[derive(Clone, Debug)]
struct RepeatContext {
    prefix: Vec<Hir>,
    body: Hir,
    max_repeat: usize,
    suffix: Vec<Hir>,
}

#[derive(Clone, Debug)]
enum ExtractResult {
    None,
    One(RepeatContext),
    Multiple,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CharBounds {
    min: usize,
    max: Option<usize>,
}

/// Analyze a JSON Schema pattern for importer-level decomposition.
///
/// `preserved_min_length` and `preserved_max_length` are the sibling length
/// constraints that the existing importer policy would actually enforce. The
/// optimization is admitted only when the anchored pattern itself implies those
/// constraints, so replacing the monolithic pattern terminal cannot discard an
/// effective intersection.
pub(super) fn analyze_complex_anchored_pattern(
    pattern: &str,
    preserved_min_length: usize,
    preserved_max_length: Option<usize>,
) -> ImportResult<Option<ComplexPatternPlan>> {
    let hir = Parser::new().parse(pattern).map_err(|error| {
        SchemaImportError::new(format!("invalid string pattern {pattern:?}: {error}"))
    })?;
    let branches = top_level_branches(hir);
    let mut analyzed = Vec::with_capacity(branches.len());
    let mut split_count = 0usize;
    let mut whole_min = usize::MAX;
    let mut whole_max = Some(0usize);

    for branch in branches {
        let (body, anchored_start, anchored_end) = strip_outer_anchors(branch);
        if !anchored_start || !anchored_end {
            return Ok(None);
        }
        let Some(bounds) = decoded_char_bounds(&body) else {
            return Ok(None);
        };
        whole_min = whole_min.min(bounds.min);
        whole_max = option_max(whole_max, bounds.max);

        match extract_unique_repeat(&body) {
            ExtractResult::One(context) => {
                let block = choose_block(context.max_repeat);
                let branch = CountedPatternBranch {
                    prefix: Hir::concat(context.prefix),
                    body: context.body,
                    suffix: Hir::concat(context.suffix),
                    max_repeat: context.max_repeat,
                    block,
                };
                if !branch.count_cover_is_exact() {
                    return Ok(None);
                }
                split_count += 1;
                analyzed.push(ComplexPatternBranch::Counted(branch));
            }
            ExtractResult::None => analyzed.push(ComplexPatternBranch::Passthrough(body)),
            ExtractResult::Multiple => return Ok(None),
        }
    }

    if split_count == 0 {
        return Ok(None);
    }
    if whole_min == usize::MAX {
        whole_min = 0;
    }
    if whole_min < preserved_min_length {
        return Ok(None);
    }
    if let Some(max) = preserved_max_length
        && whole_max.is_none_or(|pattern_max| pattern_max > max)
    {
        return Ok(None);
    }

    Ok(Some(ComplexPatternPlan { branches: analyzed }))
}

fn top_level_branches(hir: Hir) -> Vec<Hir> {
    match hir.kind() {
        HirKind::Capture(capture) => top_level_branches(*capture.sub.clone()),
        HirKind::Alternation(parts) => parts.clone(),
        _ => vec![hir],
    }
}

fn extract_unique_repeat(hir: &Hir) -> ExtractResult {
    match hir.kind() {
        HirKind::Capture(capture) => extract_unique_repeat(&capture.sub),
        HirKind::Repetition(repetition)
            if repetition.min == 0
                && repetition.max.is_some_and(|max| max as usize >= MIN_REPEAT)
                && decoded_char_bounds(&repetition.sub).is_some_and(|bounds| bounds.min > 0)
                && repetition_is_complex(repetition.max.unwrap() as usize, &repetition.sub) =>
        {
            ExtractResult::One(RepeatContext {
                prefix: Vec::new(),
                body: (*repetition.sub).clone(),
                max_repeat: repetition.max.unwrap() as usize,
                suffix: Vec::new(),
            })
        }
        HirKind::Concat(parts) => {
            let mut found: Option<(usize, RepeatContext)> = None;
            for (index, part) in parts.iter().enumerate() {
                match extract_unique_repeat(part) {
                    ExtractResult::None => {}
                    ExtractResult::Multiple => return ExtractResult::Multiple,
                    ExtractResult::One(context) => {
                        if found.is_some() {
                            return ExtractResult::Multiple;
                        }
                        found = Some((index, context));
                    }
                }
            }
            let Some((index, mut context)) = found else {
                return ExtractResult::None;
            };
            let mut prefix = parts[..index].to_vec();
            prefix.append(&mut context.prefix);
            let mut suffix = context.suffix;
            suffix.extend_from_slice(&parts[index + 1..]);
            context.prefix = prefix;
            context.suffix = suffix;
            ExtractResult::One(context)
        }
        // Nested alternations are deliberately left monolithic. Top-level
        // alternatives are handled independently by `analyze_complex_anchored_pattern`.
        HirKind::Alternation(_)
        | HirKind::Empty
        | HirKind::Literal(_)
        | HirKind::Class(_)
        | HirKind::Look(_)
        | HirKind::Repetition(_) => ExtractResult::None,
    }
}

fn repetition_is_complex(max_repeat: usize, body: &Hir) -> bool {
    let Some(bounds) = decoded_char_bounds(body) else {
        return false;
    };
    if bounds.max == Some(bounds.min) {
        return false;
    }
    max_repeat.saturating_mul(hir_node_count(body)) >= MIN_COMPLEXITY_SCORE
}

fn choose_block(max_repeat: usize) -> usize {
    max_repeat.div_ceil(MAX_GROUPS).max(MIN_BLOCK).min(max_repeat)
}

fn hir_node_count(hir: &Hir) -> usize {
    match hir.kind() {
        HirKind::Empty | HirKind::Literal(_) | HirKind::Class(_) | HirKind::Look(_) => 1,
        HirKind::Capture(capture) => 1usize.saturating_add(hir_node_count(&capture.sub)),
        HirKind::Repetition(repetition) => {
            1usize.saturating_add(hir_node_count(&repetition.sub))
        }
        HirKind::Concat(parts) | HirKind::Alternation(parts) => parts.iter().fold(1usize, |n, part| {
            n.saturating_add(hir_node_count(part))
        }),
    }
}

fn decoded_char_bounds(hir: &Hir) -> Option<CharBounds> {
    match hir.kind() {
        HirKind::Empty | HirKind::Look(_) => Some(CharBounds { min: 0, max: Some(0) }),
        HirKind::Literal(Literal(bytes)) => {
            let chars = std::str::from_utf8(bytes).ok()?.chars().count();
            Some(CharBounds { min: chars, max: Some(chars) })
        }
        HirKind::Class(_) => Some(CharBounds { min: 1, max: Some(1) }),
        HirKind::Capture(capture) => decoded_char_bounds(&capture.sub),
        HirKind::Concat(parts) => {
            let mut min = 0usize;
            let mut max = Some(0usize);
            for part in parts {
                let bounds = decoded_char_bounds(part)?;
                min = min.saturating_add(bounds.min);
                max = option_sum(max, bounds.max);
            }
            Some(CharBounds { min, max })
        }
        HirKind::Alternation(parts) => {
            let mut min = usize::MAX;
            let mut max = Some(0usize);
            for part in parts {
                let bounds = decoded_char_bounds(part)?;
                min = min.min(bounds.min);
                max = option_max(max, bounds.max);
            }
            Some(CharBounds {
                min: if min == usize::MAX { 0 } else { min },
                max,
            })
        }
        HirKind::Repetition(repetition) => {
            let bounds = decoded_char_bounds(&repetition.sub)?;
            let min_reps = repetition.min as usize;
            let min = bounds.min.saturating_mul(min_reps);
            let max = match repetition.max {
                Some(max_reps) => bounds
                    .max
                    .map(|body_max| body_max.saturating_mul(max_reps as usize)),
                None if bounds.max == Some(0) => Some(0),
                None => None,
            };
            Some(CharBounds { min, max })
        }
    }
}

fn option_sum(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    Some(left?.saturating_add(right?))
}

fn option_max(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    Some(left?.max(right?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_word_count_pattern_is_selected() {
        let plan = analyze_complex_anchored_pattern(
            r"^(?:\S+\s+){0,99}\S+$",
            1,
            None,
        )
        .unwrap()
        .expect("complex anchored pattern should split");
        assert_eq!(plan.branches.len(), 1);
        let ComplexPatternBranch::Counted(branch) = &plan.branches[0] else {
            panic!("expected counted branch");
        };
        assert_eq!(branch.max_repeat, 99);
        assert_eq!(branch.block, 8);
        assert!(branch.count_cover_is_exact());
    }

    #[test]
    fn simple_bounded_class_pattern_stays_monolithic() {
        assert!(
            analyze_complex_anchored_pattern(r"^[a-z]{0,100}$", 0, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn effective_length_constraint_must_be_implied() {
        assert!(
            analyze_complex_anchored_pattern(r"^(?:a+){0,99}$", 1, None)
                .unwrap()
                .is_none()
        );
        assert!(
            analyze_complex_anchored_pattern(r"^(?:a+b+){0,99}a+$", 1, None)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn count_intervals_are_exact_over_parameter_grid() {
        for max_repeat in MIN_REPEAT..=400 {
            let branch = CountedPatternBranch {
                prefix: Hir::empty(),
                body: Hir::literal(b"a".to_vec()),
                suffix: Hir::empty(),
                max_repeat,
                block: choose_block(max_repeat),
            };
            assert!(branch.count_cover_is_exact(), "max_repeat={max_repeat}");
        }
    }
}
