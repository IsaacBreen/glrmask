use range_set_blaze::RangeSetBlaze;
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::datastructures::abstract_weight::{current_num_tsids, normalize_num_tsids, WeightBackend};

const PROFILE_PRINT_EVERY_SECS: u64 = 5;
const PROFILE_PRINT_EVERY_CALLS: u64 = 20_000;
const PROFILE_MAX_SAMPLES: usize = 4096;
const DIFFERENCE_EXPAND_THRESHOLD: usize = 128;

/// Global collection of weights for analysis (protected by mutex for thread safety)
static WEIGHT_DUMP: OnceLock<Mutex<WeightDumpState>> = OnceLock::new();
static PROFILE_ACTIVE: OnceLock<AtomicBool> = OnceLock::new();

struct WeightDumpEntry {
    label: String,
    data: serde_json::Value,
    backtrace: String,
}

struct WeightDumpState {
    weights: Vec<WeightDumpEntry>,
    max_weights: usize,
}

fn dump_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("DUMP_FACTORIZED_WEIGHTS")
            .map(|v| v.eq_ignore_ascii_case("1") || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn dump_min_pairs() -> usize {
    static MIN_PAIRS: OnceLock<usize> = OnceLock::new();
    *MIN_PAIRS.get_or_init(|| {
        std::env::var("FACTORIZED_WEIGHT_DUMP_THRESHOLD")
            .or_else(|_| std::env::var("DUMP_FACTORIZED_WEIGHTS_MIN_PAIRS"))
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10)
    })
}

fn dump_path() -> String {
    static PATH: OnceLock<String> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("FACTORIZED_WEIGHT_DUMP_FILE")
            .or_else(|_| std::env::var("DUMP_FACTORIZED_WEIGHTS_FILE"))
            .unwrap_or_else(|_| ".cache/factorized_weights_dump.json".to_string())
    })
    .clone()
}

fn dump_flush_every() -> usize {
    static FLUSH_EVERY: OnceLock<usize> = OnceLock::new();
    *FLUSH_EVERY.get_or_init(|| {
        std::env::var("DUMP_FACTORIZED_WEIGHTS_FLUSH_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

fn get_weight_dump() -> &'static Mutex<WeightDumpState> {
    WEIGHT_DUMP.get_or_init(|| {
        let max_weights = std::env::var("DUMP_FACTORIZED_WEIGHTS_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);
        Mutex::new(WeightDumpState {
            weights: Vec::new(),
            max_weights,
        })
    })
}

/// Record a weight for later analysis
pub fn record_weight_for_dump(label: &str, weight: &FactorizedWeight) {
    if !dump_enabled() {
        return;
    }
    let min_pairs = dump_min_pairs();
    if weight.pairs.len() < min_pairs {
        // Only record "interesting" weights with many pairs
        return;
    }
    let state = get_weight_dump();
    let flush_every = dump_flush_every();
    let mut should_flush = false;
    if let Ok(mut guard) = state.lock() {
        if guard.weights.len() < guard.max_weights {
            let backtrace = Backtrace::capture().to_string();
            guard.weights.push(WeightDumpEntry {
                label: label.to_string(),
                data: weight.to_json_value(),
                backtrace,
            });
            let len = guard.weights.len();
            if len == guard.max_weights {
                should_flush = true;
            }
            if flush_every > 0 && len % flush_every == 0 {
                should_flush = true;
            }
        }
    }
    if should_flush {
        let path = dump_path();
        let _ = flush_weight_dump(&path);
    }
}

/// Write all recorded weights to a file
pub fn flush_weight_dump(path: &str) -> std::io::Result<()> {
    let state = get_weight_dump();
    if let Ok(guard) = state.lock() {
        if guard.weights.is_empty() {
            eprintln!("[DUMP] No weights collected (min threshold: 10 pairs)");
            return Ok(());
        }
        eprintln!("[DUMP] Writing {} weights to {}", guard.weights.len(), path);
        let json = serde_json::json!({
            "weights": guard
                .weights
                .iter()
                .map(|entry| {
                    serde_json::json!({
                        "label": entry.label,
                        "data": entry.data,
                        "backtrace": entry.backtrace,
                    })
                })
                .collect::<Vec<_>>(),
        });
        std::fs::write(path, serde_json::to_string_pretty(&json).unwrap())?;
    }
    Ok(())
}

#[derive(Copy, Clone, Debug)]
enum OpKind {
    Intersect,
    Union,
    Difference,
    NormalizePairs,
    FromRsb,
    ExpandToRsb,
}

#[derive(Clone, Debug)]
struct RangeSetKey(RangeSetBlaze<usize>);

impl PartialEq for RangeSetKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for RangeSetKey {}

impl Hash for RangeSetKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_rangeset(&self.0, state);
    }
}

impl OpKind {
    fn idx(self) -> usize {
        match self {
            OpKind::Intersect => 0,
            OpKind::Union => 1,
            OpKind::Difference => 2,
            OpKind::NormalizePairs => 3,
            OpKind::FromRsb => 4,
            OpKind::ExpandToRsb => 5,
        }
    }

    fn name(self) -> &'static str {
        match self {
            OpKind::Intersect => "intersect",
            OpKind::Union => "union",
            OpKind::Difference => "difference",
            OpKind::NormalizePairs => "normalize_pairs",
            OpKind::FromRsb => "from_rsb",
            OpKind::ExpandToRsb => "expand_to_rsb",
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct OpSample {
    duration_ns: u64,
    in_pairs: usize,
    out_pairs: usize,
    in_ranges: usize,
    out_ranges: usize,
}

#[derive(Default, Debug)]
struct OpStats {
    calls: u64,
    total_ns: u128,
    samples: Vec<OpSample>,
}

impl OpStats {
    fn record(&mut self, sample: OpSample) {
        self.calls = self.calls.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(sample.duration_ns as u128);
        if self.samples.len() < PROFILE_MAX_SAMPLES {
            self.samples.push(sample);
        } else {
            let idx = (self.calls as usize) % PROFILE_MAX_SAMPLES;
            self.samples[idx] = sample;
        }
    }

    fn clear_window(&mut self) {
        self.calls = 0;
        self.total_ns = 0;
        self.samples.clear();
    }
}

struct FactorizedWeightStats {
    ops: [OpStats; 6],
    last_print: Instant,
}

impl FactorizedWeightStats {
    fn new() -> Self {
        Self {
            ops: std::array::from_fn(|_| OpStats::default()),
            last_print: Instant::now(),
        }
    }

    fn record(&mut self, op: OpKind, sample: OpSample) {
        self.ops[op.idx()].record(sample);
        self.maybe_report();
    }

    fn maybe_report(&mut self) {
        let total_calls: u64 = self.ops.iter().map(|op| op.calls).sum();
        if total_calls == 0 {
            return;
        }
        let elapsed = self.last_print.elapsed();
        if elapsed.as_secs() < PROFILE_PRINT_EVERY_SECS && total_calls < PROFILE_PRINT_EVERY_CALLS {
            return;
        }
        self.print(false);
    }

    fn print(&mut self, final_summary: bool) {
        let total_calls: u64 = self.ops.iter().map(|op| op.calls).sum();
        if total_calls == 0 {
            return;
        }
        let elapsed = self.last_print.elapsed();
        crate::debug!(
            2,
            "FactorizedWeight profiling{}: {} calls in {:.2?}",
            if final_summary { " (final)" } else { "" },
            total_calls,
            elapsed
        );

        for (idx, op) in [
            OpKind::Intersect,
            OpKind::Union,
            OpKind::Difference,
            OpKind::NormalizePairs,
            OpKind::FromRsb,
            OpKind::ExpandToRsb,
        ]
        .iter()
        .enumerate()
        {
            let stats = &self.ops[idx];
            if stats.calls == 0 {
                continue;
            }
            let summary = summarize_samples(&stats.samples);
            let total_ms = (stats.total_ns as f64) / 1_000_000.0;
            if let Some(summary) = summary {
                crate::debug!(
                    2,
                    "  {:>16}: calls={}, total={:.2}ms, time_us p50/p99/max={}/{}/{}, pairs in p50/p99/max={}/{}/{}, out p50/p99/max={}/{}/{}, ranges in p50/p99/max={}/{}/{}, out p50/p99/max={}/{}/{},",
                    op.name(),
                    stats.calls,
                    total_ms,
                    summary.time_p50_us,
                    summary.time_p99_us,
                    summary.time_p100_us,
                    summary.in_pairs_p50,
                    summary.in_pairs_p99,
                    summary.in_pairs_p100,
                    summary.out_pairs_p50,
                    summary.out_pairs_p99,
                    summary.out_pairs_p100,
                    summary.in_ranges_p50,
                    summary.in_ranges_p99,
                    summary.in_ranges_p100,
                    summary.out_ranges_p50,
                    summary.out_ranges_p99,
                    summary.out_ranges_p100,
                );
            } else {
                crate::debug!(
                    2,
                    "  {:>16}: calls={}, total={:.2}ms",
                    op.name(),
                    stats.calls,
                    total_ms
                );
            }
        }

        for op in &mut self.ops {
            op.clear_window();
        }
        self.last_print = Instant::now();
    }

    fn reset_window(&mut self) {
        for op in &mut self.ops {
            op.clear_window();
        }
        self.last_print = Instant::now();
    }
}

impl Drop for FactorizedWeightStats {
    fn drop(&mut self) {
        if profiling_enabled() {
            self.print(true);
        }
    }
}

thread_local! {
    static FACTORIZED_WEIGHT_STATS: RefCell<FactorizedWeightStats> = RefCell::new(FactorizedWeightStats::new());
}

#[derive(Debug)]
struct Summary {
    time_p50_us: u64,
    time_p99_us: u64,
    time_p100_us: u64,
    in_pairs_p50: u64,
    in_pairs_p99: u64,
    in_pairs_p100: u64,
    out_pairs_p50: u64,
    out_pairs_p99: u64,
    out_pairs_p100: u64,
    in_ranges_p50: u64,
    in_ranges_p99: u64,
    in_ranges_p100: u64,
    out_ranges_p50: u64,
    out_ranges_p99: u64,
    out_ranges_p100: u64,
}

fn summarize_samples(samples: &[OpSample]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }

    let mut times: Vec<u64> = samples.iter().map(|s| s.duration_ns / 1_000).collect();
    let mut in_pairs: Vec<u64> = samples.iter().map(|s| s.in_pairs as u64).collect();
    let mut out_pairs: Vec<u64> = samples.iter().map(|s| s.out_pairs as u64).collect();
    let mut in_ranges: Vec<u64> = samples.iter().map(|s| s.in_ranges as u64).collect();
    let mut out_ranges: Vec<u64> = samples.iter().map(|s| s.out_ranges as u64).collect();

    Some(Summary {
        time_p50_us: percentile(&mut times, 0.50),
        time_p99_us: percentile(&mut times, 0.99),
        time_p100_us: *times.iter().max().unwrap_or(&0),
        in_pairs_p50: percentile(&mut in_pairs, 0.50),
        in_pairs_p99: percentile(&mut in_pairs, 0.99),
        in_pairs_p100: *in_pairs.iter().max().unwrap_or(&0),
        out_pairs_p50: percentile(&mut out_pairs, 0.50),
        out_pairs_p99: percentile(&mut out_pairs, 0.99),
        out_pairs_p100: *out_pairs.iter().max().unwrap_or(&0),
        in_ranges_p50: percentile(&mut in_ranges, 0.50),
        in_ranges_p99: percentile(&mut in_ranges, 0.99),
        in_ranges_p100: *in_ranges.iter().max().unwrap_or(&0),
        out_ranges_p50: percentile(&mut out_ranges, 0.50),
        out_ranges_p99: percentile(&mut out_ranges, 0.99),
        out_ranges_p100: *out_ranges.iter().max().unwrap_or(&0),
    })
}

fn percentile(values: &mut [u64], pct: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let idx = ((values.len() - 1) as f64 * pct).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if let Ok(val) = std::env::var("PROFILE_FACTORIZED_WEIGHT") {
            let v = val.to_ascii_lowercase();
            return matches!(v.as_str(), "1" | "true" | "yes" | "y" | "on");
        }
        std::env::var("ABSTRACT_WEIGHT_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("factorized"))
            .unwrap_or(false)
    })
}

fn minimize_only_enabled() -> bool {
    std::env::var("PROFILE_FACTORIZED_WEIGHT_MINIMIZE_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn profiling_active() -> bool {
    PROFILE_ACTIVE
    .get_or_init(|| AtomicBool::new(!minimize_only_enabled()))
        .load(Ordering::Relaxed)
}

pub fn set_factorized_weight_profile_active(active: bool) {
    PROFILE_ACTIVE
        .get_or_init(|| AtomicBool::new(true))
        .store(active, Ordering::Relaxed);
}

pub fn reset_factorized_weight_profile() {
    if !profiling_enabled() {
        return;
    }
    FACTORIZED_WEIGHT_STATS.with(|stats| stats.borrow_mut().reset_window());
}

pub fn flush_factorized_weight_profile(label: &str) {
    if !profiling_enabled() {
        return;
    }
    crate::debug!(2, "FactorizedWeight profiling window: {}", label);
    FACTORIZED_WEIGHT_STATS.with(|stats| stats.borrow_mut().print(true));
}

fn record_profile(
    op: OpKind,
    start: Instant,
    in_pairs: usize,
    in_ranges: usize,
    out_pairs: usize,
    out_ranges: usize,
) {
    if !profiling_enabled() || !profiling_active() {
        return;
    }
    let elapsed_ns = start.elapsed().as_nanos();
    let duration_ns = if elapsed_ns > u64::MAX as u128 {
        u64::MAX
    } else {
        elapsed_ns as u64
    };
    let sample = OpSample {
        duration_ns,
        in_pairs,
        out_pairs,
        in_ranges,
        out_ranges,
    };
    FACTORIZED_WEIGHT_STATS.with(|stats| stats.borrow_mut().record(op, sample));
}

fn pairs_ranges_len(pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)]) -> usize {
    pairs
        .iter()
        .map(|(tsid_set, token_set)| tsid_set.ranges_len() + token_set.ranges_len())
        .sum()
}

/// Factorized weight representation as a union of (tsid_set × token_set) pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorizedWeight {
    pub(crate) pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>,
    num_tsids: usize,
    disjoint_tsids: bool,
}

impl FactorizedWeight {
    pub(crate) fn new(num_tsids: usize) -> Self {
        Self {
            pairs: Vec::new(),
            num_tsids: normalize_num_tsids(num_tsids),
            disjoint_tsids: true,
        }
    }
    
    /// Create a factorized weight from pairs directly.
    pub fn from_pairs(pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>, num_tsids: usize) -> Self {
        let mut fw = Self {
            pairs,
            num_tsids: normalize_num_tsids(num_tsids),
            disjoint_tsids: true,
        };
        fw.normalize_pairs();
        fw
    }

    pub(crate) fn num_tsids(&self) -> usize {
        normalize_num_tsids(self.num_tsids)
    }

    pub fn pairs(&self) -> &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)] {
        &self.pairs
    }

    /// Serialize to a JSON-compatible format for analysis.
    /// Returns a JSON object with:
    /// - num_tsids: the total number of terminal signature IDs
    /// - disjoint_tsids: whether the tsid sets are disjoint
    /// - pairs: array of {tsid_ranges: [[start, end], ...], token_ranges: [[start, end], ...]}
    pub fn to_json_value(&self) -> serde_json::Value {
        let pairs_json: Vec<serde_json::Value> = self.pairs.iter().map(|(tsid_set, token_set)| {
            let tsid_ranges: Vec<Vec<usize>> = tsid_set.ranges().map(|r| vec![*r.start(), *r.end()]).collect();
            let token_ranges: Vec<Vec<usize>> = token_set.ranges().map(|r| vec![*r.start(), *r.end()]).collect();
            serde_json::json!({
                "tsid_ranges": tsid_ranges,
                "token_ranges": token_ranges,
                "tsid_count": tsid_set.len(),
                "token_count": token_set.len(),
            })
        }).collect();
        
        serde_json::json!({
            "num_tsids": self.num_tsids,
            "disjoint_tsids": self.disjoint_tsids,
            "num_pairs": self.pairs.len(),
            "total_ranges": pairs_ranges_len(&self.pairs),
            "pairs": pairs_json,
        })
    }

    fn add_pair(&mut self, tsid_set: RangeSetBlaze<usize>, token_set: RangeSetBlaze<usize>) {
        if tsid_set.is_empty() || token_set.is_empty() {
            return;
        }
        for (existing_tsids, existing_tokens) in &mut self.pairs {
            if *existing_tsids == tsid_set {
                *existing_tokens |= &token_set;
                return;
            }
        }
        self.pairs.push((tsid_set, token_set));
    }

    /// Normalize pairs to find a more compact representation.
    ///
    /// This applies iterative merging plus two greedy re-factorizations:
    /// 1. Merge pairs with identical tsid_sets (union their token_sets)
    /// 2. Merge pairs with identical token_sets (union their tsid_sets)
    /// 3. Rebuild by grouping tokens by their combined tsid_set
    /// 4. Rebuild by grouping tsids by their combined token_set
    /// 5. Pick the smallest representation
    fn normalize_pairs(&mut self) {
        let profile = profiling_enabled();
        let in_pairs = if profile { self.pairs.len() } else { 0 };
        let in_ranges = if profile { pairs_ranges_len(&self.pairs) } else { 0 };
        let start = if profile { Some(Instant::now()) } else { None };

        if self.pairs.is_empty() {
            self.disjoint_tsids = true;
            if let Some(start) = start {
                record_profile(OpKind::NormalizePairs, start, in_pairs, in_ranges, 0, 0);
            }
            return;
        }

        let mut pairs = std::mem::take(&mut self.pairs);
        pairs.retain(|(tsid_set, token_set)| !tsid_set.is_empty() && !token_set.is_empty());

        let mut best = Self::merge_identical_pairs(pairs);
        if best.len() > 500 {
            let mut tsid_size_dist: BTreeMap<usize, usize> = BTreeMap::new();
            let mut token_size_dist: BTreeMap<usize, usize> = BTreeMap::new();
            for (tsid_set, token_set) in &best {
                let tsid_len = usize::try_from(tsid_set.len()).unwrap_or(usize::MAX);
                let token_len = usize::try_from(token_set.len()).unwrap_or(usize::MAX);
                *tsid_size_dist.entry(tsid_len).or_insert(0) += 1;
                *token_size_dist.entry(token_len).or_insert(0) += 1;
            }

            let sample_size = 50.min(best.len());
            let mut tsid_overlap_counts: Vec<usize> = Vec::with_capacity(sample_size);
            let mut token_overlap_counts: Vec<usize> = Vec::with_capacity(sample_size);
            for i in 0..sample_size {
                let tsid_i = &best[i].0;
                let token_i = &best[i].1;
                let mut tsid_overlap = 0usize;
                let mut token_overlap = 0usize;
                for (j, (tsid_j, token_j)) in best.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    if !(tsid_i & tsid_j).is_empty() {
                        tsid_overlap += 1;
                    }
                    if !(token_i & token_j).is_empty() {
                        token_overlap += 1;
                    }
                }
                tsid_overlap_counts.push(tsid_overlap);
                token_overlap_counts.push(token_overlap);
            }

            let tsid_overlap_sum: usize = tsid_overlap_counts.iter().sum();
            let token_overlap_sum: usize = token_overlap_counts.iter().sum();
            let tsid_overlap_min = tsid_overlap_counts.iter().min().copied().unwrap_or(0);
            let tsid_overlap_max = tsid_overlap_counts.iter().max().copied().unwrap_or(0);
            let token_overlap_min = token_overlap_counts.iter().min().copied().unwrap_or(0);
            let token_overlap_max = token_overlap_counts.iter().max().copied().unwrap_or(0);
            let tsid_overlap_avg = tsid_overlap_sum as f64 / sample_size as f64;
            let token_overlap_avg = token_overlap_sum as f64 / sample_size as f64;

            crate::debug!(
                3,
                "normalize_pairs pair_count={} tsid_size_dist={:?} token_size_dist={:?}",
                best.len(),
                tsid_size_dist,
                token_size_dist,
            );
            crate::debug!(
                3,
                "normalize_pairs tsid_overlap_counts(first {}): min={} avg={:.1} max={} counts={:?}",
                sample_size,
                tsid_overlap_min,
                tsid_overlap_avg,
                tsid_overlap_max,
                tsid_overlap_counts,
            );
            crate::debug!(
                3,
                "normalize_pairs token_overlap_counts(first {}): min={} avg={:.1} max={} counts={:?}",
                sample_size,
                token_overlap_min,
                token_overlap_avg,
                token_overlap_max,
                token_overlap_counts,
            );
        }
        if best.len() <= 1 {
            self.pairs = best;
            self.disjoint_tsids = Self::compute_disjoint_tsids(&self.pairs);
            if let Some(start) = start {
                record_profile(
                    OpKind::NormalizePairs,
                    start,
                    in_pairs,
                    in_ranges,
                    self.pairs.len(),
                    pairs_ranges_len(&self.pairs),
                );
            }
            return;
        }

        let mut candidates: Vec<Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>> = Vec::new();
        let best_len = best.len();
        let num_tsids = self.num_tsids();

        // Always try both normalizations for larger pair counts.
        if best_len > 50 {
            candidates.push(Self::normalize_by_tokens(&best));
            candidates.push(Self::normalize_by_tsids(&best, num_tsids));
        } else {
            if let Some(max_token) = Self::max_token_in_pairs(&best) {
                let token_bound = max_token.saturating_add(1);
                if token_bound < best_len {
                    candidates.push(Self::normalize_by_tokens(&best));
                }
            }

            if num_tsids < best_len {
                candidates.push(Self::normalize_by_tsids(&best, num_tsids));
            }
        }

        if !candidates.is_empty() {
            for candidate in candidates {
                if candidate.is_empty() {
                    continue;
                }
                let candidate = Self::merge_identical_pairs(candidate);
                if Self::is_better_candidate(&candidate, &best) {
                    best = candidate;
                }
            }
        }

        self.pairs = best;
        self.disjoint_tsids = Self::compute_disjoint_tsids(&self.pairs);

        // Record weights with many pairs for analysis - these are the "stuck" ones
        if self.pairs.len() >= 100 {
            record_weight_for_dump("normalize_pairs_large", self);
        }

        if let Some(start) = start {
            record_profile(
                OpKind::NormalizePairs,
                start,
                in_pairs,
                in_ranges,
                self.pairs.len(),
                pairs_ranges_len(&self.pairs),
            );
        }
    }

    fn compute_disjoint_tsids(
        pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
    ) -> bool {
        if pairs.len() <= 1 {
            return true;
        }
        let mut union = RangeSetBlaze::new();
        let mut total: u128 = 0;
        for (tsid_set, _) in pairs {
            total = total.saturating_add(tsid_set.len());
            union |= tsid_set;
            if union.len() < total {
                return false;
            }
        }
        union.len() == total
    }

    fn difference_disjoint(&self, other: &Self) -> Self {
        let mut other_by_tsid: HashMap<usize, RangeSetBlaze<usize>> = HashMap::new();
        for (tsid_set, token_set) in &other.pairs {
            for range in tsid_set.ranges() {
                let start = *range.start();
                let end = *range.end();
                for tsid in start..=end {
                    other_by_tsid.insert(tsid, token_set.clone());
                }
            }
        }

        let mut out = FactorizedWeight::new(self.num_tsids());
        for (tsid_set, token_set) in &self.pairs {
            if tsid_set.len() == 1 {
                if let Some(range) = tsid_set.ranges().next() {
                    let tsid = *range.start();
                    if let Some(other_tokens) = other_by_tsid.get(&tsid) {
                        let token_diff = token_set - other_tokens;
                        if !token_diff.is_empty() {
                            out.add_pair(tsid_set.clone(), token_diff);
                        }
                    } else {
                        out.add_pair(tsid_set.clone(), token_set.clone());
                    }
                }
                continue;
            }

            let mut by_tokens: HashMap<RangeSetKey, RangeSetBlaze<usize>> = HashMap::new();
            for range in tsid_set.ranges() {
                let start = *range.start();
                let end = *range.end();
                for tsid in start..=end {
                    let token_diff = match other_by_tsid.get(&tsid) {
                        Some(other_tokens) => token_set - other_tokens,
                        None => token_set.clone(),
                    };
                    if token_diff.is_empty() {
                        continue;
                    }
                    let entry = by_tokens
                        .entry(RangeSetKey(token_diff))
                        .or_insert_with(RangeSetBlaze::new);
                    *entry |= &RangeSetBlaze::from_iter([tsid..=tsid]);
                }
            }
            for (token_key, tsids) in by_tokens {
                out.add_pair(tsids, token_key.0);
            }
        }

        out.normalize_pairs();
        out
    }

    fn merge_identical_pairs(
        mut pairs: Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)>,
    ) -> Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> {
        loop {
            let before_count = pairs.len();

            // First pass: merge by identical tsid_set
            let mut by_tsids: HashMap<RangeSetKey, RangeSetBlaze<usize>> = HashMap::with_capacity(pairs.len());
            for (tsid_set, token_set) in pairs {
                if tsid_set.is_empty() || token_set.is_empty() {
                    continue;
                }
                by_tsids
                    .entry(RangeSetKey(tsid_set))
                    .and_modify(|existing_tokens| *existing_tokens |= &token_set)
                    .or_insert(token_set);
            }

            // Second pass: merge by identical token_set
            let mut by_tokens: HashMap<RangeSetKey, RangeSetBlaze<usize>> = HashMap::with_capacity(by_tsids.len());
            for (tsid_key, token_set) in by_tsids {
                let tsid_set = tsid_key.0;
                by_tokens
                    .entry(RangeSetKey(token_set))
                    .and_modify(|existing_tsids| *existing_tsids |= &tsid_set)
                    .or_insert(tsid_set);
            }

            pairs = by_tokens
                .into_iter()
                .map(|(token_key, tsid_set)| (tsid_set, token_key.0))
                .collect();

            if pairs.len() >= before_count {
                break;
            }
        }

        pairs
    }

    fn max_token_in_pairs(
        pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
    ) -> Option<usize> {
        pairs
            .iter()
            .filter_map(|(_, token_set)| token_set.ranges().last().map(|r| *r.end()))
            .max()
    }

    fn normalize_by_tokens(
        pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
    ) -> Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> {
        let max_token = pairs
            .iter()
            .filter_map(|(_, token_set)| token_set.ranges().last().map(|r| *r.end()))
            .max();
        let Some(max_token) = max_token else {
            return Vec::new();
        };

        let mut token_tsids: Vec<RangeSetBlaze<usize>> = vec![RangeSetBlaze::new(); max_token + 1];

        for (tsid_set, token_set) in pairs {
            for token_range in token_set.ranges() {
                for token in *token_range.start()..=*token_range.end() {
                    if token_tsids[token].is_empty() {
                        token_tsids[token] = tsid_set.clone();
                    } else {
                        token_tsids[token] |= tsid_set;
                    }
                }
            }
        }

        let mut grouped: HashMap<RangeSetKey, RangeSetBlaze<usize>> = HashMap::new();
        for (token, tsid_set) in token_tsids.into_iter().enumerate() {
            if tsid_set.is_empty() {
                continue;
            }
            let token_set = RangeSetBlaze::from_iter([token..=token]);
            grouped
                .entry(RangeSetKey(tsid_set))
                .and_modify(|existing_tokens| *existing_tokens |= &token_set)
                .or_insert(token_set);
        }

        grouped
            .into_iter()
            .map(|(tsid_key, token_set)| (tsid_key.0, token_set))
            .collect()
    }

    fn normalize_by_tsids(
        pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
        num_tsids: usize,
    ) -> Vec<(RangeSetBlaze<usize>, RangeSetBlaze<usize>)> {
        let num_tsids = normalize_num_tsids(num_tsids);
        let mut tsid_tokens: Vec<RangeSetBlaze<usize>> = vec![RangeSetBlaze::new(); num_tsids];

        for (tsid_set, token_set) in pairs {
            for tsid_range in tsid_set.ranges() {
                for tsid in *tsid_range.start()..=*tsid_range.end() {
                    if tsid_tokens[tsid].is_empty() {
                        tsid_tokens[tsid] = token_set.clone();
                    } else {
                        tsid_tokens[tsid] |= token_set;
                    }
                }
            }
        }

        let mut grouped: HashMap<RangeSetKey, RangeSetBlaze<usize>> = HashMap::new();
        for (tsid, token_set) in tsid_tokens.into_iter().enumerate() {
            if token_set.is_empty() {
                continue;
            }
            let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
            grouped
                .entry(RangeSetKey(token_set))
                .and_modify(|existing_tsids| *existing_tsids |= &tsid_set)
                .or_insert(tsid_set);
        }

        grouped
            .into_iter()
            .map(|(token_key, tsid_set)| (tsid_set, token_key.0))
            .collect()
    }

    fn is_better_candidate(
        candidate: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
        best: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
    ) -> bool {
        let candidate_cost = Self::candidate_cost(candidate);
        let best_cost = Self::candidate_cost(best);
        candidate_cost < best_cost
    }

    fn candidate_cost(
        pairs: &[(RangeSetBlaze<usize>, RangeSetBlaze<usize>)],
    ) -> (usize, usize, u128) {
        let total_ranges: usize = pairs
            .iter()
            .map(|(tsid_set, token_set)| tsid_set.ranges_len() + token_set.ranges_len())
            .sum();
        let total_items: u128 = pairs
            .iter()
            .map(|(tsid_set, token_set)| {
                let tsid_len = tsid_set.len() as u128;
                let token_len = token_set.len() as u128;
                tsid_len.saturating_mul(token_len)
            })
            .sum();
        (pairs.len(), total_ranges, total_items)
    }

    pub(crate) fn from_position_with_num_tsids(pos: usize, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
        let token_set = RangeSetBlaze::from_iter([token..=token]);
        let mut weight = Self {
            pairs: vec![(tsid_set, token_set)],
            num_tsids,
            disjoint_tsids: true,
        };
        weight.normalize_pairs();
        weight
    }

    pub(crate) fn all_with_max_position(max_position: usize, num_tsids: usize) -> Self {
        let num_tsids = normalize_num_tsids(num_tsids);
        if max_position == 0 {
            return Self::from_position_with_num_tsids(0, num_tsids);
        }

        let full_tsids = RangeSetBlaze::from_iter([0..=num_tsids - 1]);
        let full_tokens = max_position / num_tsids;
        let last_tsid = max_position % num_tsids;

        let mut weight = Self::new(num_tsids);
        if last_tsid == num_tsids - 1 {
            let token_set = RangeSetBlaze::from_iter([0..=full_tokens]);
            weight.add_pair(full_tsids, token_set);
        } else {
            if full_tokens > 0 {
                let token_set = RangeSetBlaze::from_iter([0..=full_tokens - 1]);
                weight.add_pair(full_tsids.clone(), token_set);
            }
            let token_set = RangeSetBlaze::from_iter([full_tokens..=full_tokens]);
            let tsid_set = RangeSetBlaze::from_iter([0..=last_tsid]);
            weight.add_pair(tsid_set, token_set);
        }
        weight.normalize_pairs();
        weight
    }

    pub(crate) fn from_rsb_with_num_tsids(rsb: &RangeSetBlaze<usize>, num_tsids: usize) -> Self {
        let profile = profiling_enabled();
        let in_pairs = 0usize;
        let in_ranges = if profile { rsb.ranges_len() } else { 0 };
        let start = if profile { Some(Instant::now()) } else { None };

        let num_tsids = normalize_num_tsids(num_tsids);
        if rsb.is_empty() {
            let empty = Self::new(num_tsids);
            if let Some(start) = start {
                record_profile(
                    OpKind::FromRsb,
                    start,
                    in_pairs,
                    in_ranges,
                    empty.pairs.len(),
                    pairs_ranges_len(&empty.pairs),
                );
            }
            return empty;
        }

        let ranges_len = rsb.ranges_len();
        if rsb.len() == 1 {
            if let Some(pos) = rsb.ranges().next().map(|r| *r.start()) {
                let token = pos / num_tsids;
                let tsid = pos % num_tsids;
                let weight = Self {
                    pairs: vec![(
                        RangeSetBlaze::from_iter([tsid..=tsid]),
                        RangeSetBlaze::from_iter([token..=token]),
                    )],
                    num_tsids,
                    disjoint_tsids: true,
                };
                if let Some(start) = start {
                    record_profile(
                        OpKind::FromRsb,
                        start,
                        in_pairs,
                        in_ranges,
                        weight.pairs.len(),
                        pairs_ranges_len(&weight.pairs),
                    );
                }
                return weight;
            }
        }

        if ranges_len == 1 {
            if let Some(range) = rsb.ranges().next() {
                let range_start = *range.start();
                let range_end = *range.end();
                let start_token = range_start / num_tsids;
                let end_token = range_end / num_tsids;
                let start_tsid = range_start % num_tsids;
                let end_tsid = range_end % num_tsids;

                let full_tsid_set = RangeSetBlaze::from_iter([0..=num_tsids - 1]);
                let mut pairs = Vec::new();
                if start_token == end_token {
                    pairs.push((
                        RangeSetBlaze::from_iter([start_tsid..=end_tsid]),
                        RangeSetBlaze::from_iter([start_token..=start_token]),
                    ));
                } else {
                    pairs.push((
                        RangeSetBlaze::from_iter([start_tsid..=num_tsids - 1]),
                        RangeSetBlaze::from_iter([start_token..=start_token]),
                    ));

                    if start_token + 1 <= end_token.saturating_sub(1) {
                        pairs.push((
                            full_tsid_set.clone(),
                            RangeSetBlaze::from_iter([start_token + 1..=end_token - 1]),
                        ));
                    }

                    pairs.push((
                        RangeSetBlaze::from_iter([0..=end_tsid]),
                        RangeSetBlaze::from_iter([end_token..=end_token]),
                    ));
                }

                let weight = Self {
                    disjoint_tsids: Self::compute_disjoint_tsids(&pairs),
                    pairs,
                    num_tsids,
                };
                if let Some(start) = start {
                    record_profile(
                        OpKind::FromRsb,
                        start,
                        in_pairs,
                        in_ranges,
                        weight.pairs.len(),
                        pairs_ranges_len(&weight.pairs),
                    );
                }
                return weight;
            }
        }

        if ranges_len <= 5 {
            let mut ranges = rsb.ranges();
            if let Some(first_range) = ranges.next() {
                let first_start = *first_range.start();
                let first_end = *first_range.end();
                let first_token = first_start / num_tsids;
                let first_end_token = first_end / num_tsids;
                let mut all_same_token = first_token == first_end_token;
                let mut tsid_set = if all_same_token {
                    let start_tsid = first_start % num_tsids;
                    let end_tsid = first_end % num_tsids;
                    RangeSetBlaze::from_iter([start_tsid..=end_tsid])
                } else {
                    RangeSetBlaze::new()
                };

                for range in ranges {
                    if !all_same_token {
                        break;
                    }
                    let start = *range.start();
                    let end = *range.end();
                    let start_token = start / num_tsids;
                    let end_token = end / num_tsids;
                    if start_token != first_token || end_token != first_token {
                        all_same_token = false;
                        break;
                    }
                    let start_tsid = start % num_tsids;
                    let end_tsid = end % num_tsids;
                    tsid_set |= &RangeSetBlaze::from_iter([start_tsid..=end_tsid]);
                }

                if all_same_token {
                    let weight = Self {
                        pairs: vec![(
                            tsid_set,
                            RangeSetBlaze::from_iter([first_token..=first_token]),
                        )],
                        num_tsids,
                        disjoint_tsids: true,
                    };
                    if let Some(start) = start {
                        record_profile(
                            OpKind::FromRsb,
                            start,
                            in_pairs,
                            in_ranges,
                            weight.pairs.len(),
                            pairs_ranges_len(&weight.pairs),
                        );
                    }
                    return weight;
                }
            }
        }

        let mut token_to_tsids: BTreeMap<usize, RangeSetBlaze<usize>> = BTreeMap::new();
        let full_tsid_set = RangeSetBlaze::from_iter([0..=num_tsids - 1]);

        for range in rsb.ranges() {
            let start = *range.start();
            let end = *range.end();
            let start_token = start / num_tsids;
            let end_token = end / num_tsids;
            let start_tsid = start % num_tsids;
            let end_tsid = end % num_tsids;

            if start_token == end_token {
                let entry = token_to_tsids.entry(start_token).or_insert_with(RangeSetBlaze::new);
                *entry |= &RangeSetBlaze::from_iter([start_tsid..=end_tsid]);
                continue;
            }

            let entry = token_to_tsids.entry(start_token).or_insert_with(RangeSetBlaze::new);
            *entry |= &RangeSetBlaze::from_iter([start_tsid..=num_tsids - 1]);

            if start_token + 1 <= end_token.saturating_sub(1) {
                for token in (start_token + 1)..=end_token - 1 {
                    let entry = token_to_tsids.entry(token).or_insert_with(RangeSetBlaze::new);
                    *entry |= &full_tsid_set;
                }
            }

            let entry = token_to_tsids.entry(end_token).or_insert_with(RangeSetBlaze::new);
            *entry |= &RangeSetBlaze::from_iter([0..=end_tsid]);
        }

        let mut weight = Self::new(num_tsids);
        for (token, tsid_set) in token_to_tsids {
            let token_set = RangeSetBlaze::from_iter([token..=token]);
            weight.add_pair(tsid_set, token_set);
        }
        weight.normalize_pairs();
        if let Some(start) = start {
            record_profile(
                OpKind::FromRsb,
                start,
                in_pairs,
                in_ranges,
                weight.pairs.len(),
                pairs_ranges_len(&weight.pairs),
            );
        }
        weight
    }

    pub fn expand_to_rsb(&self) -> RangeSetBlaze<usize> {
        if std::env::var("ALLOW_FACTORIZED_EXPANSION").is_err() {
            panic!(
                "Unexpected factorized weight expansion at: FactorizedWeight::expand_to_rsb(). Set ALLOW_FACTORIZED_EXPANSION=1 to allow."
            );
        }
        self.expand_to_rsb_internal()
    }

    pub(crate) fn expand_to_rsb_unchecked(&self) -> RangeSetBlaze<usize> {
        self.expand_to_rsb_internal()
    }

    fn expand_to_rsb_internal(&self) -> RangeSetBlaze<usize> {
        let profile = profiling_enabled();
        let in_pairs = if profile { self.pairs.len() } else { 0 };
        let in_ranges = if profile { pairs_ranges_len(&self.pairs) } else { 0 };
        let start = if profile { Some(Instant::now()) } else { None };

        if self.pairs.is_empty() {
            let empty = RangeSetBlaze::new();
            if let Some(start) = start {
                record_profile(OpKind::ExpandToRsb, start, in_pairs, in_ranges, 0, empty.ranges_len());
            }
            return empty;
        }
        let num_tsids = self.num_tsids();
        let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();

        for (tsid_set, token_set) in &self.pairs {
            for token_range in token_set.ranges() {
                let token_start = *token_range.start();
                let token_end = *token_range.end();
                for tsid_range in tsid_set.ranges() {
                    let tsid_start = *tsid_range.start();
                    let tsid_end = *tsid_range.end();
                    for token in token_start..=token_end {
                        let base = token.saturating_mul(num_tsids);
                        ranges.push(base.saturating_add(tsid_start)..=base.saturating_add(tsid_end));
                    }
                }
            }
        }

        let rsb = RangeSetBlaze::from_iter(ranges);
        if let Some(start) = start {
            record_profile(OpKind::ExpandToRsb, start, in_pairs, in_ranges, 0, rsb.ranges_len());
        }
        rsb
    }
}

fn hash_rangeset<H: Hasher>(rsb: &RangeSetBlaze<usize>, state: &mut H) {
    for range in rsb.ranges() {
        range.start().hash(state);
        range.end().hash(state);
    }
}

impl Hash for FactorizedWeight {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.num_tsids.hash(state);
        self.disjoint_tsids.hash(state);
        self.pairs.len().hash(state);
        for (tsid_set, token_set) in &self.pairs {
            hash_rangeset(tsid_set, state);
            hash_rangeset(token_set, state);
        }
    }
}

impl WeightBackend for FactorizedWeight {
    fn empty() -> Self {
        FactorizedWeight::new(current_num_tsids())
    }

    fn all(max_position: usize) -> Self {
        FactorizedWeight::all_with_max_position(max_position, current_num_tsids())
    }

    fn from_position(pos: usize) -> Self {
        FactorizedWeight::from_position_with_num_tsids(pos, current_num_tsids())
    }

    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        let rsb = RangeSetBlaze::from_iter(ranges);
        FactorizedWeight::from_rsb_with_num_tsids(&rsb, current_num_tsids())
    }

    fn is_empty(&self) -> bool {
        self.pairs.is_empty() || self.pairs.iter().all(|(a, b)| a.is_empty() || b.is_empty())
    }

    fn len(&self) -> usize {
        let mut total: u128 = 0;
        for (tsid_set, token_set) in &self.pairs {
            let pair_count = tsid_set.len().saturating_mul(token_set.len());
            total = total.saturating_add(pair_count);
        }
        if total > usize::MAX as u128 {
            usize::MAX
        } else {
            total as usize
        }
    }

    fn contains(&self, pos: usize) -> bool {
        if self.pairs.is_empty() {
            return false;
        }
        let num_tsids = self.num_tsids();
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        self.pairs.iter().any(|(tsid_set, token_set)| {
            tsid_set.contains(tsid) && token_set.contains(token)
        })
    }

    fn ranges_len(&self) -> usize {
        self.pairs
            .iter()
            .map(|(tsid_set, token_set)| tsid_set.ranges_len() + token_set.ranges_len())
            .sum()
    }

    fn insert(&mut self, pos: usize) {
        let num_tsids = self.num_tsids();
        let token = pos / num_tsids;
        let tsid = pos % num_tsids;
        let tsid_set = RangeSetBlaze::from_iter([tsid..=tsid]);
        let token_set = RangeSetBlaze::from_iter([token..=token]);
        self.add_pair(tsid_set, token_set);
        self.normalize_pairs();
    }

    fn intersect(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        let profile = profiling_enabled();
        let in_pairs = if profile { self.pairs.len().saturating_add(other.pairs.len()) } else { 0 };
        let in_ranges = if profile {
            pairs_ranges_len(&self.pairs).saturating_add(pairs_ranges_len(&other.pairs))
        } else {
            0
        };
        let start = if profile { Some(Instant::now()) } else { None };

        let mut out = FactorizedWeight::new(self.num_tsids());
        for (tsid_a, token_a) in &self.pairs {
            for (tsid_b, token_b) in &other.pairs {
                let tsid_inter = tsid_a & tsid_b;
                let token_inter = token_a & token_b;
                if !tsid_inter.is_empty() && !token_inter.is_empty() {
                    out.add_pair(tsid_inter, token_inter);
                }
            }
        }
        out.normalize_pairs();
        
        // Record for analysis if dumping is enabled
        if dump_enabled() && out.pairs.len() >= 10 {
            record_weight_for_dump("intersect_result", &out);
        }
        
        if let Some(start) = start {
            record_profile(
                OpKind::Intersect,
                start,
                in_pairs,
                in_ranges,
                out.pairs.len(),
                pairs_ranges_len(&out.pairs),
            );
        }
        out
    }

    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }

    fn union(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        let profile = profiling_enabled();
        let in_pairs = if profile { self.pairs.len().saturating_add(other.pairs.len()) } else { 0 };
        let in_ranges = if profile {
            pairs_ranges_len(&self.pairs).saturating_add(pairs_ranges_len(&other.pairs))
        } else {
            0
        };
        let start = if profile { Some(Instant::now()) } else { None };

        let mut out = self.clone();
        for (tsid_set, token_set) in &other.pairs {
            out.add_pair(tsid_set.clone(), token_set.clone());
        }
        out.normalize_pairs();
        
        // Record for analysis if dumping is enabled
        if dump_enabled() && out.pairs.len() >= 10 {
            record_weight_for_dump("union_result", &out);
        }
        
        if let Some(start) = start {
            record_profile(
                OpKind::Union,
                start,
                in_pairs,
                in_ranges,
                out.pairs.len(),
                pairs_ranges_len(&out.pairs),
            );
        }
        out
    }

    fn union_assign(&mut self, other: &Self) {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        let profile = profiling_enabled();
        let in_pairs = if profile { self.pairs.len().saturating_add(other.pairs.len()) } else { 0 };
        let in_ranges = if profile {
            pairs_ranges_len(&self.pairs).saturating_add(pairs_ranges_len(&other.pairs))
        } else {
            0
        };
        let start = if profile { Some(Instant::now()) } else { None };

        for (tsid_set, token_set) in &other.pairs {
            self.add_pair(tsid_set.clone(), token_set.clone());
        }
        self.normalize_pairs();
        if let Some(start) = start {
            record_profile(
                OpKind::Union,
                start,
                in_pairs,
                in_ranges,
                self.pairs.len(),
                pairs_ranges_len(&self.pairs),
            );
        }
    }

    fn difference(&self, other: &Self) -> Self {
        assert_eq!(self.num_tsids(), other.num_tsids(), "FactorizedWeight num_tsids mismatch");
        let profile = profiling_enabled();
        let in_pairs = if profile { self.pairs.len().saturating_add(other.pairs.len()) } else { 0 };
        let in_ranges = if profile {
            pairs_ranges_len(&self.pairs).saturating_add(pairs_ranges_len(&other.pairs))
        } else {
            0
        };
        let start = if profile { Some(Instant::now()) } else { None };

        if self.is_empty() {
            let empty = FactorizedWeight::new(self.num_tsids());
            if let Some(start) = start {
                record_profile(
                    OpKind::Difference,
                    start,
                    in_pairs,
                    in_ranges,
                    empty.pairs.len(),
                    pairs_ranges_len(&empty.pairs),
                );
            }
            return empty;
        }
        if other.is_empty() {
            let out = self.clone();
            if let Some(start) = start {
                record_profile(
                    OpKind::Difference,
                    start,
                    in_pairs,
                    in_ranges,
                    out.pairs.len(),
                    pairs_ranges_len(&out.pairs),
                );
            }
            return out;
        }

        if self.disjoint_tsids && other.disjoint_tsids {
            let out = self.difference_disjoint(other);
            if let Some(start) = start {
                record_profile(
                    OpKind::Difference,
                    start,
                    in_pairs,
                    in_ranges,
                    out.pairs.len(),
                    pairs_ranges_len(&out.pairs),
                );
            }
            return out;
        }

        if self.pairs.len() > DIFFERENCE_EXPAND_THRESHOLD && other.pairs.len() > DIFFERENCE_EXPAND_THRESHOLD {
            let self_rsb = self.expand_to_rsb_internal();
            let other_rsb = other.expand_to_rsb_internal();
            let result_rsb = &self_rsb - &other_rsb;
            let out = FactorizedWeight::from_rsb_with_num_tsids(&result_rsb, self.num_tsids());
            if let Some(start) = start {
                record_profile(
                    OpKind::Difference,
                    start,
                    in_pairs,
                    in_ranges,
                    out.pairs.len(),
                    pairs_ranges_len(&out.pairs),
                );
            }
            return out;
        }

        let mut out = FactorizedWeight::new(self.num_tsids());
        for (tsid_set, token_set) in &self.pairs {
            let mut remainders = vec![(tsid_set.clone(), token_set.clone())];
            for (other_tsids, other_tokens) in &other.pairs {
                if remainders.is_empty() {
                    break;
                }
                let mut next = Vec::new();
                for (rem_tsids, rem_tokens) in remainders {
                    let tsid_inter = &rem_tsids & other_tsids;
                    let token_inter = &rem_tokens & other_tokens;
                    if tsid_inter.is_empty() || token_inter.is_empty() {
                        next.push((rem_tsids, rem_tokens));
                        continue;
                    }

                    let tsid_diff = &rem_tsids - other_tsids;
                    if !tsid_diff.is_empty() {
                        next.push((tsid_diff, rem_tokens.clone()));
                    }

                    let token_diff = &rem_tokens - other_tokens;
                    if !token_diff.is_empty() && !tsid_inter.is_empty() {
                        next.push((tsid_inter, token_diff));
                    }
                }
                remainders = next;
            }

            for (rem_tsids, rem_tokens) in remainders {
                out.add_pair(rem_tsids, rem_tokens);
            }
        }

        out.normalize_pairs();
        if let Some(start) = start {
            record_profile(
                OpKind::Difference,
                start,
                in_pairs,
                in_ranges,
                out.pairs.len(),
                pairs_ranges_len(&out.pairs),
            );
        }
        out
    }

    fn complement(&self, max_position: usize) -> Self {
        let all = FactorizedWeight::all_with_max_position(max_position, self.num_tsids());
        all.difference(self)
    }

    fn min_item(&self) -> Option<usize> {
        let num_tsids = self.num_tsids();
        self.pairs
            .iter()
            .filter_map(|(tsid_set, token_set)| {
                let min_token = token_set.ranges().next().map(|r| *r.start())?;
                let min_tsid = tsid_set.ranges().next().map(|r| *r.start())?;
                Some(min_token.saturating_mul(num_tsids).saturating_add(min_tsid))
            })
            .min()
    }

    fn max_item(&self) -> Option<usize> {
        let num_tsids = self.num_tsids();
        self.pairs
            .iter()
            .filter_map(|(tsid_set, token_set)| {
                let max_token = token_set.ranges().last().map(|r| *r.end())?;
                let max_tsid = tsid_set.ranges().last().map(|r| *r.end())?;
                Some(max_token.saturating_mul(num_tsids).saturating_add(max_tsid))
            })
            .max()
    }

}
