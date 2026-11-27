//! One signed DAG abstract interpretation. A domain may forget coefficient
//! correlations (Boolean top support) or retain them (weighted top support).
//! Neither abstraction treats a freshly exposed tail as a known LR predecessor.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum StackEffect {
    Push(u32),
    Read(u32),
    Forget,
}

impl StackEffect {
    fn from_label(label: i32, alphabet: u32) -> Option<Self> {
        if label == DEFAULT_LABEL {
            return Some(Self::Forget);
        }
        let push = is_negative_label(label);
        let symbol = if push { negative_to_positive_label(label) } else { label };
        if symbol < 0 || symbol as u32 >= alphabet {
            return None;
        }
        Some(if push { Self::Push(symbol as u32) } else { Self::Read(symbol as u32) })
    }
}

#[derive(Debug, Default)]
pub(super) struct TopSupportProfile {
    pub states: usize,
    pub live_states: usize,
    pub edges_before: usize,
    pub edges_after: usize,
    pub blocked_labels: usize,
    pub changed_edges: usize,
    pub top_classes: usize,
    pub domains: usize,
    pub stored_pairs: usize,
    pub work: usize,
}

pub(super) trait TopDomain {
    fn live(&self, input: u32) -> bool;
    fn final_weight(&mut self, input: u32, weight: u32) -> u32;
    fn epsilon(&mut self, input: u32, weight: u32) -> Option<(u32, u32)>;
    // Compute the reusable input-dependent part once per label, then apply
    // the individual coefficient and stack effect for every branch.
    fn prepare(&mut self, input: u32, effect: StackEffect) -> Option<u32>;
    fn branch(&mut self, prepared: u32, effect: StackEffect, weight: u32) -> Option<(u32, u32)>;
    fn join(&mut self, old: u32, incoming: u32) -> Option<u32>;
    fn finish_row(&mut self, work: usize) -> bool { let _ = work; true }
    fn finish_profile(&self, profile: &mut TopSupportProfile);
}

pub(super) fn restrict<D: TopDomain>(
    states: &mut [FastBoundaryNwaState],
    starts: &[u32],
    alphabet: u32,
    domain: &mut D,
) -> Option<TopSupportProfile> {
    let n = states.len();
    if n == 0 || n > 200_000 || alphabet == 0 || alphabet > 50_000
        || starts.iter().any(|&state| state as usize >= n)
    {
        return None;
    }
    let order = fast_boundary_topological_order(states)?;
    let mut inputs = vec![0u32; n];
    for &state in starts {
        inputs[state as usize] = 1;
    }
    let mut profile = TopSupportProfile { states: n, ..Default::default() };
    for state in order {
        let input = inputs[state as usize];
        profile.live_states += usize::from(domain.live(input));
        let row = &mut states[state as usize];
        row.final_weight = domain.final_weight(input, row.final_weight);
        let mut publish = |target: u32, weight: &mut u32, output: u32, new: u32, domain: &mut D| -> Option<()> {
            profile.edges_before += 1;
            profile.changed_edges += usize::from(*weight != new);
            *weight = new;
            if new != 0 {
                profile.edges_after += 1;
                let slot = inputs.get_mut(target as usize)?;
                *slot = TopDomain::join(domain, *slot, output)?;
            }
            Some(())
        };
        for (target, weight) in &mut row.epsilons {
            if *weight != 0 {
                let (output, new) = domain.epsilon(input, *weight)?;
                publish(*target, weight, output, new, domain)?;
            }
        }
        for (label, branches) in &mut row.transitions {
            let effect = StackEffect::from_label(*label, alphabet)?;
            let prepared = domain.prepare(input, effect)?;
            if prepared == 0 && !branches.is_empty() {
                profile.blocked_labels += 1;
            }
            for (target, weight) in branches {
                if *weight != 0 {
                    let (output, new) = domain.branch(prepared, effect, *weight)?;
                    publish(*target, weight, output, new, domain)?;
                }
            }
        }
        if profile.edges_before > 4_000_000
            || !domain.finish_row(row.epsilons.len() + row.transitions.len() + 1)
        {
            return None;
        }
    }
    domain.finish_profile(&mut profile);
    Some(profile)
}
