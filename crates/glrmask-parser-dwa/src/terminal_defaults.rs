//! Least fixed point of terminal states after negative-label removal.
//! An eligible state becomes terminal only when every DEFAULT target is
//! terminal. Structural zero-weight branches still count as dependencies.
use std::collections::VecDeque;

/// The callback emits the complete dependency list for an eligible state.
/// Returning false rejects that state even if it emitted some dependencies.
pub(crate) fn close(
    terminal: &mut [bool],
    mut dependencies: impl FnMut(usize, &mut dyn FnMut(u32)) -> bool,
) {
    let n = terminal.len();
    let mut dependents = vec![Vec::<usize>::new(); n];
    let mut remaining = vec![usize::MAX; n];
    let mut ready = VecDeque::new();
    // A state already terminal at construction needs no wake-up: subsequent
    // dependency registration sees it immediately.
    for state in 0..n {
        if terminal[state] {
            continue;
        }
        let mut count = 0usize;
        let eligible = dependencies(state, &mut |target| {
            let target = target as usize;
            if target >= n {
                count += 1;
            } else if !terminal[target] {
                count += 1;
                dependents[target].push(state);
            }
        });
        if !eligible {
            continue;
        }
        remaining[state] = count;
        if count == 0 {
            terminal[state] = true;
            ready.push_back(state);
        }
    }
    while let Some(done) = ready.pop_front() {
        for &state in &dependents[done] {
            if terminal[state] || remaining[state] == usize::MAX {
                continue;
            }
            debug_assert!(remaining[state] > 0);
            remaining[state] -= 1;
            if remaining[state] == 0 {
                terminal[state] = true;
                ready.push_back(state);
            }
        }
    }
}
