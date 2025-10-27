use crate::trie3_opt::core::MiniTrie;
use std::collections::BTreeMap;
use std::fmt::Write;
use kdam::{tqdm, BarExt};
use crate::profiler::PROGRESS_BAR_ENABLED;

/// A helper for collecting numeric data and computing summary statistics.
#[derive(Debug, Clone, Default)]
pub struct NumericStats {
    values: Vec<f64>,
}

impl NumericStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_samples<T: Into<f64> + Copy>(samples: &[T]) -> Self {
        Self {
            values: samples.iter().map(|&v| v.into()).collect(),
        }
    }

    pub fn push<T: Into<f64>>(&mut self, value: T) {
        self.values.push(value.into());
    }

    pub fn to_pretty_string(&self) -> String {
        if self.values.is_empty() {
            return "{ count: 0 }".to_string();
        }

        let mut sorted_values = self.values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = self.values.len();
        let sum: f64 = self.values.iter().sum();
        let mean = sum / count as f64;

        let min = sorted_values[0];
        let max = sorted_values[count - 1];

        let median = if count % 2 == 1 {
            sorted_values[count / 2]
        } else {
            (sorted_values[count / 2 - 1] + sorted_values[count / 2]) / 2.0
        };

        let p25 = sorted_values[(count as f64 * 0.25).floor() as usize];
        let p75 = sorted_values[(count as f64 * 0.75).floor() as usize];
        let p95 = sorted_values[(count as f64 * 0.95).floor() as usize];

        let variance = self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let stdev = variance.sqrt();

        format!(
            "{{ count: {}, sum: {:.2}, mean: {:.2}, stdev: {:.2}, min: {:.2}, p25: {:.2}, median: {:.2}, p75: {:.2}, p95: {:.2}, max: {:.2} }}",
            count, sum, mean, stdev, min, p25, median, p75, p95, max
        )
    }
}

/// A trait for a modular metric that can be computed on a MiniTrie.
pub trait Metric {
    /// The name of the metric.
    fn name(&self) -> &'static str;
    /// Computes the metric and returns a formatted string.
    fn compute(&self, trie: &MiniTrie) -> String;
}

// --- Concrete Metric Implementations ---

pub struct NumNodesMetric;
impl Metric for NumNodesMetric {
    fn name(&self) -> &'static str { "num_nodes" }
    fn compute(&self, trie: &MiniTrie) -> String { trie.nodes.len().to_string() }
}

pub struct NumEdgesMetric;
impl Metric for NumEdgesMetric {
    fn name(&self) -> &'static str { "num_edges" }
    fn compute(&self, trie: &MiniTrie) -> String {
        trie.nodes.iter().map(|n| n.out_degree()).sum::<usize>().to_string()
    }
}

pub struct NumRootsMetric;
impl Metric for NumRootsMetric {
    fn name(&self) -> &'static str { "num_roots" }
    fn compute(&self, trie: &MiniTrie) -> String { trie.root_ids.len().to_string() }
}

pub struct NumEndNodesMetric;
impl Metric for NumEndNodesMetric {
    fn name(&self) -> &'static str { "num_end_nodes" }
    fn compute(&self, trie: &MiniTrie) -> String {
        trie.nodes.iter().filter(|n| n.end).count().to_string()
    }
}

pub struct NumReachableNodesMetric;
impl Metric for NumReachableNodesMetric {
    fn name(&self) -> &'static str { "num_reachable_nodes" }
    fn compute(&self, trie: &MiniTrie) -> String { trie.reachable_from_roots().len().to_string() }
}

pub struct NumProductiveNodesMetric;
impl Metric for NumProductiveNodesMetric {
    fn name(&self) -> &'static str { "num_productive_nodes" }
    fn compute(&self, trie: &MiniTrie) -> String { trie.can_reach_end().len().to_string() }
}

pub struct RootFanoutMetric;
impl Metric for RootFanoutMetric {
    fn name(&self) -> &'static str { "root_fanout" }
    fn compute(&self, trie: &MiniTrie) -> String {
        let fanouts: Vec<f64> = trie
            .nodes
            .iter()
            .filter(|n| trie.root_ids.contains(&n.id))
            .map(|n| n.out_degree() as f64)
            .collect();
        NumericStats::from_samples(&fanouts).to_pretty_string()
    }
}

pub struct NonRootFanoutMetric;
impl Metric for NonRootFanoutMetric {
    fn name(&self) -> &'static str { "non_root_fanout" }
    fn compute(&self, trie: &MiniTrie) -> String {
        let fanouts: Vec<f64> = trie
            .nodes
            .iter()
            .filter(|n| !trie.root_ids.contains(&n.id))
            .map(|n| n.out_degree() as f64)
            .collect();
        NumericStats::from_samples(&fanouts).to_pretty_string()
    }
}

pub struct EdgeOverlapMetric;
impl Metric for EdgeOverlapMetric {
    fn name(&self) -> &'static str { "edge_overlap" }
    fn compute(&self, trie: &MiniTrie) -> String {
        use crate::trie3_opt::core::{NodeId, SortedSet};
        use std::collections::BTreeMap;

        let mut scores: Vec<f64> = Vec::new();
        #[cfg(not(rustrover))]
        let it = tqdm!(
            trie.nodes.iter(),
            desc = "Metric: EdgeOverlap",
            total = trie.nodes.len(),
            disable = !PROGRESS_BAR_ENABLED,
            leave = false
        );
        #[cfg(rustrover)]
        let it = trie.nodes.iter();

        for node in it {
            let mut node_score = 0.0;

            // Group destination maps by pop
            let mut by_pop: BTreeMap<isize, Vec<&BTreeMap<NodeId, SortedSet>>> = BTreeMap::new();
            for (ek, dm) in &node.children {
                by_pop.entry(ek.pop).or_default().push(dm);
            }

            for (_pop, dms) in by_pop {
                if dms.len() < 2 {
                    continue;
                }
                for i in 0..dms.len() {
                    for j in (i + 1)..dms.len() {
                        let dm1 = dms[i];
                        let dm2 = dms[j];

                        let mut common_dests_with_overlap = 0;
                        for (d, sids1) in dm1 {
                            if let Some(sids2) = dm2.get(d) {
                                if !sids1.intersect(sids2).is_empty() {
                                    common_dests_with_overlap += 1;
                                }
                            }
                        }
                        node_score += common_dests_with_overlap as f64;
                    }
                }
            }
            if node_score > 0.0 {
                scores.push(node_score);
            }
        }
        NumericStats::from_samples(&scores).to_pretty_string()
    }
}

pub struct StateFanoutMetric;
impl Metric for StateFanoutMetric {
    fn name(&self) -> &'static str { "state_fanout" }
    fn compute(&self, trie: &MiniTrie) -> String {
        use crate::trie3_opt::core::{NodeId, SortedSet};
        use std::collections::BTreeMap;

        let mut stats = NumericStats::new();

        #[cfg(not(rustrover))]
        let it = tqdm!(
            trie.nodes.iter(),
            desc = "Metric: StateFanout",
            total = trie.nodes.len(),
            disable = !PROGRESS_BAR_ENABLED,
            leave = false
        );
        #[cfg(rustrover)]
        let it = trie.nodes.iter();

        for node in it {
            // For this node, build a map from (pop, state_id) -> Vec<(dest, tokens)>
            let mut fanout_map: BTreeMap<(isize, usize), Vec<(NodeId, SortedSet)>> = BTreeMap::new();

            for (edge_key, dest_map) in &node.children {
                let pop = edge_key.pop;
                let tokens = &edge_key.tokens;

                for (dest, state_set) in dest_map {
                    for state_id in state_set.iter() {
                        fanout_map
                            .entry((pop, state_id))
                            .or_default()
                            .push((*dest, tokens.clone()));
                    }
                }
            }

            // Collect fanout values for this node
            for fanout_vec in fanout_map.values() {
                stats.push(fanout_vec.len() as f64);
            }
        }

        stats.to_pretty_string()
    }
}

/// Instantiates and runs all standard metrics on a given trie.
pub fn run_all_metrics(trie: &MiniTrie) -> BTreeMap<String, String> {
    let metrics: Vec<Box<dyn Metric>> = vec![
        Box::new(NumNodesMetric),
        Box::new(NumEdgesMetric),
        Box::new(NumRootsMetric),
        Box::new(NumEndNodesMetric),
        Box::new(NumReachableNodesMetric),
        Box::new(NumProductiveNodesMetric),
        Box::new(RootFanoutMetric),
        Box::new(NonRootFanoutMetric),
        Box::new(EdgeOverlapMetric),
        Box::new(StateFanoutMetric),
    ];

    let mut results = BTreeMap::new();
    for metric in metrics {
        crate::debug!(3, "  Computing metric: {}", metric.name());
        results.insert(metric.name().to_string(), metric.compute(trie));
    }
    results
}

/// Formats a map of metrics into a pretty, indented string.
pub fn pretty_print_metrics_map(metrics: &BTreeMap<String, String>) -> String {
    let mut buf = String::new();
    buf.push_str("{\n");
    let mut first = true;
    for (k, v) in metrics {
        if !first {
            buf.push_str(",\n");
        }
        first = false;
        write!(buf, "  \"{}\": {}", k, v).unwrap();
    }
    buf.push_str("\n}");
    buf
}

