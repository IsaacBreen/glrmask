//! Checked boundary-analysis scope.
//!
//! Ordinary terminal-DWA construction observes the whole raw tokenizer state
//! domain. Composition boundary shards may start only in states owned by one
//! immediate component, while reset after a terminal commit still follows the
//! merged tokenizer. Keep those domains explicit so a quotient cannot seed a
//! representative owned by another component.

use std::sync::Arc;

use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::bitset::BitSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImmediateComponentId(pub u32);

/// Checked ownership of concrete terminal labels by the immediate components
/// of the link currently being compiled. A nested component may own several
/// concrete leaves; callers provide the leaf-to-immediate projection explicitly
/// so leaf identity is never confused with immediate-component identity.
#[derive(Debug, Clone)]
pub struct BoundaryOwnership {
    terminal_owner: Arc<[ImmediateComponentId]>,
    num_immediate_components: u32,
}

impl BoundaryOwnership {
    pub fn from_leaf_layout(
        terminal_offsets: &[u32],
        num_terminals: u32,
        leaf_to_immediate: &[ImmediateComponentId],
        num_immediate_components: u32,
    ) -> Result<Self, String> {
        if terminal_offsets.is_empty() || terminal_offsets[0] != 0 {
            return Err("boundary ownership terminal offsets must start at zero".to_owned());
        }
        if terminal_offsets.len() != leaf_to_immediate.len() {
            return Err(format!(
                "boundary ownership has {} terminal leaf offsets but {} leaf owners",
                terminal_offsets.len(),
                leaf_to_immediate.len(),
            ));
        }
        if num_immediate_components == 0 {
            return Err("boundary ownership has no immediate components".to_owned());
        }
        if terminal_offsets.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("boundary ownership terminal offsets are not strictly increasing".to_owned());
        }
        if terminal_offsets.last().copied().unwrap_or(0) >= num_terminals && num_terminals != 0 {
            return Err("boundary ownership final leaf starts outside terminal domain".to_owned());
        }
        if leaf_to_immediate
            .iter()
            .any(|owner| owner.0 >= num_immediate_components)
        {
            return Err("boundary ownership references an out-of-range immediate component".to_owned());
        }
        let mut terminal_owner = Vec::with_capacity(num_terminals as usize);
        for terminal in 0..num_terminals {
            let leaf = terminal_offsets
                .partition_point(|&offset| offset <= terminal)
                .saturating_sub(1);
            let owner = *leaf_to_immediate
                .get(leaf)
                .ok_or_else(|| format!("terminal {terminal} has no owning leaf"))?;
            terminal_owner.push(owner);
        }
        Ok(Self {
            terminal_owner: Arc::from(terminal_owner.into_boxed_slice()),
            num_immediate_components,
        })
    }

    pub fn flat(terminal_offsets: &[u32], num_terminals: u32) -> Result<Self, String> {
        let owners = (0..terminal_offsets.len())
            .map(|index| ImmediateComponentId(index as u32))
            .collect::<Vec<_>>();
        Self::from_leaf_layout(
            terminal_offsets,
            num_terminals,
            &owners,
            terminal_offsets.len() as u32,
        )
    }

    #[inline]
    pub fn owner_of_terminal(&self, terminal: u32) -> Option<ImmediateComponentId> {
        self.terminal_owner.get(terminal as usize).copied()
    }

    #[inline]
    pub fn num_terminals(&self) -> usize {
        self.terminal_owner.len()
    }

    #[inline]
    pub fn num_immediate_components(&self) -> u32 {
        self.num_immediate_components
    }
}

#[derive(Debug, Clone)]
pub struct InitialStateDomain {
    keep_raw: Arc<[bool]>,
    exact_singleton_map: ManyToOneIdMap,
}

impl InitialStateDomain {
    pub fn from_mask(num_raw_states: usize, keep_raw: Vec<bool>) -> Result<Self, String> {
        if keep_raw.len() != num_raw_states {
            return Err(format!(
                "boundary initial-state mask has length {}, expected {num_raw_states}",
                keep_raw.len(),
            ));
        }
        if !keep_raw.iter().any(|&keep| keep) {
            return Err("boundary initial-state domain is empty".to_owned());
        }

        let mut original_to_internal = vec![u32::MAX; num_raw_states];
        let mut representative_original_ids = Vec::new();
        for (raw, &keep) in keep_raw.iter().enumerate() {
            if !keep {
                continue;
            }
            let internal = representative_original_ids.len() as u32;
            original_to_internal[raw] = internal;
            representative_original_ids.push(raw as u32);
        }
        let exact_singleton_map =
            ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                original_to_internal,
                representative_original_ids,
            );
        Ok(Self {
            keep_raw: Arc::from(keep_raw.into_boxed_slice()),
            exact_singleton_map,
        })
    }

    #[inline]
    pub fn keep_raw(&self) -> &[bool] {
        &self.keep_raw
    }

    #[inline]
    pub fn exact_singleton_map(&self) -> &ManyToOneIdMap {
        &self.exact_singleton_map
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.exact_singleton_map.representative_original_ids.len()
    }

    #[inline]
    pub fn contains(&self, raw: u32) -> bool {
        self.keep_raw.get(raw as usize).copied().unwrap_or(false)
    }
}

#[derive(Debug, Clone)]
pub struct BoundaryAnalysisScope {
    initial_states: InitialStateDomain,
    reset_states: Arc<[u32]>,
    ownership: Arc<BoundaryOwnership>,
    start_component: ImmediateComponentId,
    require_crossing: bool,
    follow_transparent: Option<BitSet>,
}

impl BoundaryAnalysisScope {
    pub fn new(
        initial_states: InitialStateDomain,
        reset_states: Vec<u32>,
        ownership: Arc<BoundaryOwnership>,
        start_component: ImmediateComponentId,
        require_crossing: bool,
        follow_transparent: Option<BitSet>,
    ) -> Result<Self, String> {
        if start_component.0 >= ownership.num_immediate_components() {
            return Err(format!(
                "boundary start component {} is outside {} ownership blocks",
                start_component.0,
                ownership.num_immediate_components(),
            ));
        }
        let mut reset_states = reset_states;
        reset_states.sort_unstable();
        reset_states.dedup();
        if reset_states.is_empty() {
            return Err("boundary reset-state domain is empty".to_owned());
        }
        if reset_states
            .iter()
            .any(|&raw| raw as usize >= initial_states.keep_raw().len())
        {
            return Err("boundary reset state lies outside raw tokenizer domain".to_owned());
        }
        Ok(Self {
            initial_states,
            reset_states: Arc::from(reset_states.into_boxed_slice()),
            ownership,
            start_component,
            require_crossing,
            follow_transparent,
        })
    }

    #[inline]
    pub fn initial_states(&self) -> &InitialStateDomain {
        &self.initial_states
    }

    #[inline]
    pub fn reset_states(&self) -> &[u32] {
        &self.reset_states
    }

    #[inline]
    pub fn ownership(&self) -> &BoundaryOwnership {
        &self.ownership
    }

    #[inline]
    pub fn start_component(&self) -> ImmediateComponentId {
        self.start_component
    }

    #[inline]
    pub fn require_crossing(&self) -> bool {
        self.require_crossing
    }

    #[inline]
    pub fn follow_transparent(&self) -> Option<&BitSet> {
        self.follow_transparent.as_ref()
    }

    #[inline]
    pub fn owner_of_terminal(&self, terminal: u32) -> Option<ImmediateComponentId> {
        self.ownership.owner_of_terminal(terminal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_domain_builds_only_in_scope_singleton_classes() {
        let domain = InitialStateDomain::from_mask(
            6,
            vec![false, true, false, true, true, false],
        )
        .unwrap();
        assert_eq!(domain.len(), 3);
        assert_eq!(domain.exact_singleton_map().representative_original_ids, vec![1, 3, 4]);
        assert_eq!(
            domain.exact_singleton_map().original_to_internal,
            vec![u32::MAX, 0, u32::MAX, 1, 2, u32::MAX],
        );
    }

    #[test]
    fn nested_leaf_layout_maps_to_immediate_owner_blocks() {
        let ownership = BoundaryOwnership::from_leaf_layout(
            &[0, 2, 5, 7],
            9,
            &[
                ImmediateComponentId(0),
                ImmediateComponentId(0),
                ImmediateComponentId(1),
                ImmediateComponentId(1),
            ],
            2,
        )
        .unwrap();
        assert_eq!(ownership.owner_of_terminal(0), Some(ImmediateComponentId(0)));
        assert_eq!(ownership.owner_of_terminal(6), Some(ImmediateComponentId(1)));
        assert_eq!(ownership.owner_of_terminal(8), Some(ImmediateComponentId(1)));
    }
}
