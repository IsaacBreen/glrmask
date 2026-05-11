pub mod candidate;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameData {
    pub version: u32,
    pub source: String,
    pub buf_words: usize,
    pub maps: Vec<Mapping>,
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mapping {
    pub id: u32,
    pub problem: String,
    pub internal_to_original: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub map_id: u32,
    pub problem: String,
    pub example_index: u32,
    pub step: u32,
    pub internal_ids: Vec<u32>,
    pub expected_sparse_words: Vec<[u32; 2]>,
}

pub trait Candidate {
    type Prepared;

    fn name() -> &'static str;
    fn prepare(mapping: &Mapping, buf_words: usize) -> Self::Prepared;
    fn fill(prepared: &Self::Prepared, internal_ids: &[u32], out: &mut [u32]);
}

#[derive(Debug, Clone)]
pub struct EvalSummary {
    pub candidate: &'static str,
    pub cases: usize,
    pub repetitions: usize,
    pub total_calls: usize,
    pub max_ns: u128,
    pub max_case_index: usize,
    pub max_repetition: usize,
    pub stabilized_max_ns: u128,
    pub stabilized_max_case_index: usize,
    pub mean_ns: f64,
    pub p50_ns: u128,
    pub p95_ns: u128,
    pub p99_ns: u128,
}

pub fn load_game_data(path: impl AsRef<Path>) -> Result<GameData, Box<dyn std::error::Error>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader: Box<dyn Read> = if path.extension().is_some_and(|ext| ext == "gz") {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn evaluate<C: Candidate>(
    data: &GameData,
    repetitions: usize,
) -> Result<EvalSummary, String> {
    let mut prepared_by_map: Vec<Option<C::Prepared>> =
        (0..data.maps.len()).map(|_| None).collect();
    for mapping in &data.maps {
        let idx = mapping.id as usize;
        if idx >= prepared_by_map.len() {
            return Err(format!("mapping id {} is out of range", mapping.id));
        }
        prepared_by_map[idx] = Some(C::prepare(mapping, data.buf_words));
    }

    let mut out = vec![0u32; data.buf_words];
    let mut timings = Vec::with_capacity(data.cases.len().saturating_mul(repetitions));
    let mut max_timing = Duration::ZERO;
    let mut max_case_index = 0usize;
    let mut max_repetition = 0usize;
    let mut case_min_timings = vec![Duration::MAX; data.cases.len()];

    for repetition in 0..repetitions {
        for (case_index, case) in data.cases.iter().enumerate() {
            let prepared = prepared_by_map
                .get(case.map_id as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| format!("missing mapping {}", case.map_id))?;

            out.fill(0);
            let started = Instant::now();
            C::fill(prepared, &case.internal_ids, &mut out);
            let elapsed = started.elapsed();

            verify_output(case, &out)?;
            if elapsed > max_timing {
                max_timing = elapsed;
                max_case_index = case_index;
                max_repetition = repetition;
            }
            if elapsed < case_min_timings[case_index] {
                case_min_timings[case_index] = elapsed;
            }
            timings.push(elapsed);
        }
    }

    let (stabilized_max_case_index, stabilized_max) = case_min_timings
        .iter()
        .enumerate()
        .filter(|(_, timing)| **timing != Duration::MAX)
        .max_by_key(|(_, timing)| **timing)
        .map(|(idx, timing)| (idx, *timing))
        .unwrap_or((0, Duration::ZERO));

    Ok(summarize(
        C::name(),
        data.cases.len(),
        repetitions,
        timings,
        max_case_index,
        max_repetition,
        stabilized_max_case_index,
        stabilized_max,
    ))
}

fn verify_output(case: &Case, out: &[u32]) -> Result<(), String> {
    let mut expected = vec![0u32; out.len()];
    for &[word_idx, mask] in &case.expected_sparse_words {
        let idx = word_idx as usize;
        if idx >= expected.len() {
            return Err(format!(
                "case {}:{} expected word {} outside output length {}",
                case.problem,
                case.step,
                word_idx,
                expected.len()
            ));
        }
        expected[idx] = mask;
    }
    if expected == out {
        return Ok(());
    }

    for (idx, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
        if got != want {
            return Err(format!(
                "case {} example {} step {} word {} mismatch: got {got:#010x}, want {want:#010x}",
                case.problem, case.example_index, case.step, idx
            ));
        }
    }
    Err("output length mismatch".to_string())
}

fn summarize(
    candidate: &'static str,
    cases: usize,
    repetitions: usize,
    mut timings: Vec<Duration>,
    max_case_index: usize,
    max_repetition: usize,
    stabilized_max_case_index: usize,
    stabilized_max: Duration,
) -> EvalSummary {
    timings.sort_unstable();
    let total_calls = timings.len();
    let total_ns: u128 = timings.iter().map(Duration::as_nanos).sum();
    let at = |pct: f64| -> u128 {
        if timings.is_empty() {
            return 0;
        }
        let idx = ((timings.len() - 1) as f64 * pct).round() as usize;
        timings[idx].as_nanos()
    };

    EvalSummary {
        candidate,
        cases,
        repetitions,
        total_calls,
        max_ns: timings.last().map(Duration::as_nanos).unwrap_or(0),
        max_case_index,
        max_repetition,
        stabilized_max_ns: stabilized_max.as_nanos(),
        stabilized_max_case_index,
        mean_ns: if total_calls == 0 {
            0.0
        } else {
            total_ns as f64 / total_calls as f64
        },
        p50_ns: at(0.50),
        p95_ns: at(0.95),
        p99_ns: at(0.99),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{
        BaselineCandidate, ComplementCandidate, CopyFirstGroupRunCandidate, GlrMaskLikeCandidate,
        ParallelComplementCandidate,
    };

    fn tiny_data() -> GameData {
        GameData {
            version: 1,
            source: "unit".to_string(),
            buf_words: 3,
            maps: vec![Mapping {
                id: 0,
                problem: "tiny".to_string(),
                internal_to_original: vec![
                    vec![0, 1, 32],
                    vec![2, 63],
                    vec![64],
                    vec![95, 96],
                ],
            }],
            cases: vec![
                Case {
                    map_id: 0,
                    problem: "tiny".to_string(),
                    example_index: 0,
                    step: 0,
                    internal_ids: vec![0, 2],
                    expected_sparse_words: vec![[0, 0b11], [1, 0b1], [2, 0b1]],
                },
                Case {
                    map_id: 0,
                    problem: "tiny".to_string(),
                    example_index: 0,
                    step: 1,
                    internal_ids: vec![1, 3],
                    expected_sparse_words: vec![
                        [0, 0b100],
                        [1, 1u32 << 31],
                        [2, 1u32 << 31],
                    ],
                },
            ],
        }
    }

    #[test]
    fn baseline_expands_internal_ids_to_original_bitset() {
        let summary = evaluate::<BaselineCandidate>(&tiny_data(), 3).expect("baseline verifies");
        assert_eq!(summary.cases, 2);
        assert_eq!(summary.repetitions, 3);
        assert_eq!(summary.total_calls, 6);
    }

    #[test]
    fn glrmask_like_candidate_expands_internal_ids_to_original_bitset() {
        let summary =
            evaluate::<GlrMaskLikeCandidate>(&tiny_data(), 3).expect("optimized verifies");
        assert_eq!(summary.cases, 2);
        assert_eq!(summary.repetitions, 3);
        assert_eq!(summary.total_calls, 6);
    }

    #[test]
    fn copy_first_candidate_expands_internal_ids_to_original_bitset() {
        let summary =
            evaluate::<CopyFirstGroupRunCandidate>(&tiny_data(), 3).expect("copy-first verifies");
        assert_eq!(summary.cases, 2);
        assert_eq!(summary.repetitions, 3);
        assert_eq!(summary.total_calls, 6);
    }

    #[test]
    fn complement_candidate_expands_internal_ids_to_original_bitset() {
        let summary = evaluate::<ComplementCandidate>(&tiny_data(), 3).expect("complement verifies");
        assert_eq!(summary.cases, 2);
        assert_eq!(summary.repetitions, 3);
        assert_eq!(summary.total_calls, 6);
    }

    #[test]
    fn parallel_complement_candidate_expands_internal_ids_to_original_bitset() {
        let summary = evaluate::<ParallelComplementCandidate>(&tiny_data(), 3)
            .expect("parallel complement verifies");
        assert_eq!(summary.cases, 2);
        assert_eq!(summary.repetitions, 3);
        assert_eq!(summary.total_calls, 6);
    }

    struct EmptyCandidate;

    impl Candidate for EmptyCandidate {
        type Prepared = ();

        fn name() -> &'static str {
            "empty"
        }

        fn prepare(_mapping: &Mapping, _buf_words: usize) -> Self::Prepared {}

        fn fill(_prepared: &Self::Prepared, _internal_ids: &[u32], _out: &mut [u32]) {}
    }

    #[test]
    fn evaluator_rejects_wrong_output() {
        let err = evaluate::<EmptyCandidate>(&tiny_data(), 1).expect_err("empty candidate is wrong");
        assert!(err.contains("mismatch"), "{err}");
    }

    #[test]
    fn evaluator_rejects_missing_mapping() {
        let mut data = tiny_data();
        data.cases[0].map_id = 9;
        let err = evaluate::<BaselineCandidate>(&data, 1).expect_err("map id is invalid");
        assert!(err.contains("missing mapping"), "{err}");
    }
}
