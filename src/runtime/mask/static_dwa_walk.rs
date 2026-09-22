//! Shared single-stack parser-DWA traversal.
//!
//! State-coordinate mapping and shard selection happen before entry. The row
//! provider and mask operations are monomorphized closures: the ordinary path
//! does not test a composition flag or call through a trait object per edge.
//! Weight representation is independent of traversal (packed/materialized
//! weights and the small u64 boundary representation use the same loop).

#[derive(Clone, Copy)]
pub(super) enum StackWalkEvent<W> {
    Final(W),
    Top(u32),
    Intersect(W),
}

/// Visit every accepting prefix of a top-first parser stack. `visit` returns
/// false when its path mask becomes empty or a caller's planning budget ends.
/// In either case already-emitted contributions remain valid. Row providers
/// own positive/domain/default-label lookup; the kernel never remaps IDs.
#[inline(always)]
pub(super) fn walk_single_stack<const VISIT_TOP: bool, W: Copy>(
    start_state: u32,
    top_first: &[u32],
    mut final_weight: impl FnMut(u32) -> Option<W>,
    mut transition: impl FnMut(u32, u32) -> Option<(u32, W)>,
    mut visit: impl FnMut(StackWalkEvent<W>) -> bool,
) {
    let mut state = start_state;
    let mut stack_index = 0;
    loop {
        if let Some(weight) = final_weight(state)
            && !visit(StackWalkEvent::Final(weight))
        {
            break;
        }
        let Some(&parser_state) = top_first.get(stack_index) else {
            break;
        };
        stack_index += 1;
        if VISIT_TOP && stack_index == 1 && !visit(StackWalkEvent::Top(parser_state)) {
            break;
        }
        let Some((target, weight)) = transition(state, parser_state) else {
            break;
        };
        if !visit(StackWalkEvent::Intersect(weight)) {
            break;
        }
        state = target;
    }
}

#[cfg(test)]
mod tests {
    use super::{StackWalkEvent, walk_single_stack};

    #[test]
    fn accepting_prefixes_survive_a_dead_edge_and_mapping_is_provider_owned() {
        let mut path = 0b111_u64;
        let mut accepted = 0;
        let mut tops = Vec::new();
        walk_single_stack::<true, _>(
            0,
            &[100, 101, 102],
            |state| [Some(0b001), Some(0b110), Some(0b100)].get(state as usize).copied().flatten(),
            |state, parser| match (state, parser.checked_sub(100)) {
                (0, Some(0)) => Some((1, 0b110)),
                (1, Some(1)) => Some((2, 0)),
                _ => panic!("must not traverse below the dead edge"),
            },
            |event| {
                match event {
                    StackWalkEvent::Final(weight) => accepted |= path & weight,
                    StackWalkEvent::Top(state) => tops.push(state),
                    StackWalkEvent::Intersect(weight) => {
                        path &= weight;
                        return path != 0;
                    }
                }
                true
            },
        );
        assert_eq!(accepted, 0b111);
        assert_eq!(tops, [100]);
    }

    #[test]
    fn boundary_specialization_accepts_empty_stack_without_top_dispatch() {
        let mut accepted = 0_u64;
        walk_single_stack::<false, _>(
            7, &[],
            |state| { assert_eq!(state, 7); Some(0b101) },
            |_, _| panic!("empty stack must not look up an edge"),
            |event| {
                match event {
                    StackWalkEvent::Final(weight) => accepted |= weight,
                    _ => panic!("no top/edge event for empty stack"),
                }
                true
            },
        );
        assert_eq!(accepted, 0b101);
    }
}
