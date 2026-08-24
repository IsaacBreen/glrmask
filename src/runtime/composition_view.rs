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
        local_to_outer_offsets: Arc<[u32]>,
        local_to_outers: Arc<[u32]>,
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
    /// outer entries are allowed. Composition may split one local LR state into
    /// several outer LR states, so the reverse relation is intentionally
    /// one-to-many.
    pub(crate) fn dense_from_outer_to_local(
        outer_to_local: Vec<u32>,
        local_len: u32,
    ) -> Result<Self, String> {
        let mut counts = vec![0u32; local_len as usize];
        for (outer, &local) in outer_to_local.iter().enumerate() {
            if local == UNMAPPED_ID {
                continue;
            }
            let count = counts.get_mut(local as usize).ok_or_else(|| {
                format!(
                    "component ID projection maps outer ID {outer} to out-of-range local ID {local} (local_len={local_len})"
                )
            })?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| "component reverse ID projection count overflows u32".to_owned())?;
        }
        let mut local_to_outer_offsets = Vec::with_capacity(local_len as usize + 1);
        local_to_outer_offsets.push(0u32);
        for count in counts {
            let next = local_to_outer_offsets
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(count)
                .ok_or_else(|| "component reverse ID projection offset overflows u32".to_owned())?;
            local_to_outer_offsets.push(next);
        }
        let mut cursors = local_to_outer_offsets[..local_len as usize].to_vec();
        let mut local_to_outers = vec![0u32; *local_to_outer_offsets.last().unwrap_or(&0) as usize];
        for (outer, &local) in outer_to_local.iter().enumerate() {
            if local == UNMAPPED_ID {
                continue;
            }
            let cursor = &mut cursors[local as usize];
            local_to_outers[*cursor as usize] = u32::try_from(outer)
                .map_err(|_| "component outer ID exceeds u32 coordinate".to_owned())?;
            *cursor += 1;
        }
        Ok(Self::Dense {
            outer_to_local: outer_to_local.into(),
            local_to_outer_offsets: local_to_outer_offsets.into(),
            local_to_outers: local_to_outers.into(),
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
    pub(crate) fn outer_ids_for_local(&self, local: u32) -> smallvec::SmallVec<[u32; 4]> {
        match self {
            Self::Offset {
                outer_base,
                local_len,
            } => (local < *local_len)
                .then(|| outer_base.checked_add(local))
                .flatten()
                .into_iter()
                .collect(),
            Self::Dense {
                local_to_outer_offsets,
                local_to_outers,
                ..
            } => {
                let index = local as usize;
                let Some((&start, &end)) = local_to_outer_offsets
                    .get(index)
                    .zip(local_to_outer_offsets.get(index + 1))
                else {
                    return smallvec::SmallVec::new();
                };
                local_to_outers[start as usize..end as usize]
                    .iter()
                    .copied()
                    .collect()
            }
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
            Self::Dense { local_to_outer_offsets, .. } => local_to_outer_offsets.len().saturating_sub(1) as u32,
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
        assert_eq!(map.outer_ids_for_local(0).as_slice(), &[10]);
        assert_eq!(map.outer_ids_for_local(2).as_slice(), &[12]);
        assert!(map.outer_ids_for_local(3).is_empty());
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
        assert_eq!(map.outer_ids_for_local(0).as_slice(), &[2]);
        assert_eq!(map.outer_ids_for_local(1).as_slice(), &[4]);
        assert_eq!(map.outer_ids_for_local(2).as_slice(), &[1]);
    }

    #[test]
    fn dense_projection_preserves_split_local_state() {
        let map = VirtualIdMap::dense_from_outer_to_local(vec![0, 0], 1).unwrap();
        assert_eq!(map.to_local(0), Some(0));
        assert_eq!(map.to_local(1), Some(0));
        assert_eq!(map.outer_ids_for_local(0).as_slice(), &[0u32, 1]);
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
