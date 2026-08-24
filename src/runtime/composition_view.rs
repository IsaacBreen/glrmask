use std::sync::Arc;

use crate::automata::lexer::Lexer;
use super::Constraint;

const UNMAPPED_ID: u32 = u32::MAX;

/// Bidirectional projection between the virtual ID namespace of an immediate
/// composed constraint and one intact component constraint's private IDs.
///
/// Common component coordinates are contiguous ranges and therefore use an
/// affine offset. Dense storage is reserved for coordinates (currently LR
/// states) whose linker relation is not affine. `u32::MAX` denotes an outer or
/// local ID which is not owned by this view.
#[derive(Debug, Clone)]
pub(crate) enum VirtualIdMap {
    Offset {
        outer_base: u32,
        local_len: u32,
    },
    Dense {
        outer_to_local: Arc<[u32]>,
        local_to_outer: Arc<[u32]>,
    },
}

impl VirtualIdMap {
    #[inline]
    pub(crate) fn offset(outer_base: u32, local_len: u32) -> Result<Self, String> {
        outer_base.checked_add(local_len).ok_or_else(|| {
            format!(
                "affine component ID projection overflows u32 (outer_base={outer_base}, local_len={local_len})"
            )
        })?;
        Ok(Self::Offset {
            outer_base,
            local_len,
        })
    }

    /// Build a bidirectional dense map from an outer->local relation. Unmapped
    /// outer entries are allowed. A local ID may be owned by at most one outer
    /// ID; that is the property required by an intact component view.
    pub(crate) fn dense_from_outer_to_local(
        outer_to_local: Vec<u32>,
        local_len: u32,
    ) -> Result<Self, String> {
        let mut local_to_outer = vec![UNMAPPED_ID; local_len as usize];
        for (outer, &local) in outer_to_local.iter().enumerate() {
            if local == UNMAPPED_ID {
                continue;
            }
            let slot = local_to_outer.get_mut(local as usize).ok_or_else(|| {
                format!(
                    "component ID projection maps outer ID {outer} to out-of-range local ID {local} (local_len={local_len})"
                )
            })?;
            let outer = u32::try_from(outer)
                .map_err(|_| "component outer ID exceeds u32 coordinate".to_owned())?;
            if *slot != UNMAPPED_ID && *slot != outer {
                return Err(format!(
                    "component ID projection maps local ID {local} from multiple outer IDs ({}, {outer})",
                    *slot
                ));
            }
            *slot = outer;
        }
        Ok(Self::Dense {
            outer_to_local: outer_to_local.into(),
            local_to_outer: local_to_outer.into(),
        })
    }

    #[inline]
    pub(crate) fn to_local(&self, outer: u32) -> Option<u32> {
        match self {
            Self::Offset {
                outer_base,
                local_len,
            } => {
                let local = outer.checked_sub(*outer_base)?;
                (local < *local_len).then_some(local)
            }
            Self::Dense { outer_to_local, .. } => outer_to_local
                .get(outer as usize)
                .copied()
                .filter(|&local| local != UNMAPPED_ID),
        }
    }

    #[inline]
    pub(crate) fn to_outer(&self, local: u32) -> Option<u32> {
        match self {
            Self::Offset {
                outer_base,
                local_len,
            } => (local < *local_len)
                .then(|| outer_base.checked_add(local))
                .flatten(),
            Self::Dense { local_to_outer, .. } => local_to_outer
                .get(local as usize)
                .copied()
                .filter(|&outer| outer != UNMAPPED_ID),
        }
    }

    #[inline]
    pub(crate) fn outer_base(&self) -> Option<u32> {
        match self {
            Self::Offset { outer_base, .. } => Some(*outer_base),
            Self::Dense { .. } => None,
        }
    }

    #[inline]
    pub(crate) fn local_len(&self) -> u32 {
        match self {
            Self::Offset { local_len, .. } => *local_len,
            Self::Dense { local_to_outer, .. } => local_to_outer.len() as u32,
        }
    }

    #[inline]
    pub(crate) fn outer_domain_len(&self) -> usize {
        match self {
            Self::Offset {
                outer_base,
                local_len,
            } => (*outer_base + *local_len) as usize,
            Self::Dense { outer_to_local, .. } => outer_to_local.len(),
        }
    }
}

/// O(1) routing from an outer ID to one immediate child view.
///
/// This is deliberately a route, not a claim that the child exclusively owns
/// the ID. The segmented parser union uses it only after compile-time
/// certification has proved that at most one child root has a live transition
/// for each outer LR state.
#[derive(Debug, Clone, Default)]
pub(crate) struct DenseViewRouting {
    outer_to_view: Vec<u32>,
}

impl DenseViewRouting {
    pub(crate) fn new(outer_to_view: Vec<u32>, view_count: usize) -> Result<Self, String> {
        for (outer, &view) in outer_to_view.iter().enumerate() {
            if view != UNMAPPED_ID && view as usize >= view_count {
                return Err(format!(
                    "component route for outer ID {outer} selects missing view {view} (view_count={view_count})"
                ));
            }
        }
        Ok(Self { outer_to_view })
    }

    #[inline]
    pub(crate) fn route(&self, outer: u32) -> Option<usize> {
        self.outer_to_view
            .get(outer as usize)
            .copied()
            .filter(|&view| view != UNMAPPED_ID)
            .map(|view| view as usize)
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.outer_to_view.is_empty()
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.outer_to_view.clear();
    }
}

/// A leaf component as seen from the virtual namespace of an immediate
/// composition node. The compiled child itself is retained intact; only these
/// projections belong to the parent composition.
///
/// TSID and nonterminal projections are deliberately not represented yet: the
/// current segmented runtime evaluates parser-DWA weights in the child's own
/// TSID coordinate and does not route child GLR reductions through a provider.
/// They should be added when those runtime operations move behind this view,
/// rather than storing placeholder maps with unclear semantics.
#[derive(Debug, Clone)]
pub(crate) struct ConstraintView {
    pub(crate) constraint: Arc<Constraint>,
    pub(crate) terminal_ids: VirtualIdMap,
    pub(crate) lexer_states: VirtualIdMap,
    pub(crate) lrids: VirtualIdMap,
}

impl ConstraintView {
    pub(crate) fn segmented_leaf(
        constraint: Arc<Constraint>,
        terminal_outer_base: u32,
        lexer_outer_base: u32,
        outer_to_local_lrids: Vec<u32>,
    ) -> Result<Self, String> {
        let terminal_ids =
            VirtualIdMap::offset(terminal_outer_base, constraint.table.num_terminals)?;
        let lexer_states =
            VirtualIdMap::offset(lexer_outer_base, constraint.tokenizer.num_states())?;
        let lrids = VirtualIdMap::dense_from_outer_to_local(
            outer_to_local_lrids,
            constraint.table.num_states,
        )?;
        Ok(Self {
            constraint,
            terminal_ids,
            lexer_states,
            lrids,
        })
    }

    /// Project an outer raw lexer state into this child. The merged runtime's
    /// reset/commit-initial state is a shared dispatcher rather than part of
    /// any child's affine range, but semantically aliases every child's local
    /// tokenizer start state. Reverse projection remains the canonical affine
    /// component state, avoiding a false claim that this alias is bijective.
    #[inline]
    pub(crate) fn local_lexer_state(&self, outer: u32, outer_reset: u32) -> Option<u32> {
        if outer == outer_reset {
            Some(self.constraint.tokenizer.start_state())
        } else {
            self.lexer_states.to_local(outer)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_projection_is_bidirectional_and_bounded() {
        let map = VirtualIdMap::offset(10, 3).unwrap();
        assert_eq!(map.to_local(9), None);
        assert_eq!(map.to_local(10), Some(0));
        assert_eq!(map.to_local(12), Some(2));
        assert_eq!(map.to_local(13), None);
        assert_eq!(map.to_outer(0), Some(10));
        assert_eq!(map.to_outer(2), Some(12));
        assert_eq!(map.to_outer(3), None);
    }

    #[test]
    fn dense_projection_preserves_private_coordinate() {
        let map = VirtualIdMap::dense_from_outer_to_local(
            vec![u32::MAX, 2, 0, u32::MAX, 1],
            3,
        )
        .unwrap();
        assert_eq!(map.to_local(0), None);
        assert_eq!(map.to_local(1), Some(2));
        assert_eq!(map.to_local(2), Some(0));
        assert_eq!(map.to_local(4), Some(1));
        assert_eq!(map.to_outer(0), Some(2));
        assert_eq!(map.to_outer(1), Some(4));
        assert_eq!(map.to_outer(2), Some(1));
    }

    #[test]
    fn dense_projection_rejects_non_functional_reverse_relation() {
        let error = VirtualIdMap::dense_from_outer_to_local(vec![0, 0], 1).unwrap_err();
        assert!(error.contains("multiple outer IDs"));
    }

    #[test]
    fn dense_view_routing_distinguishes_missing_and_selected_routes() {
        let routing = DenseViewRouting::new(vec![u32::MAX, 1, 0], 2).unwrap();
        assert_eq!(routing.route(0), None);
        assert_eq!(routing.route(1), Some(1));
        assert_eq!(routing.route(2), Some(0));
        assert_eq!(routing.route(3), None);
    }
}
