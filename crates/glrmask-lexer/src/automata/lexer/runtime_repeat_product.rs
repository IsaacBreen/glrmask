//! Exact lazy runtime for the intersection of two large bounded repetitions.
//!
//! Each repetition body is required to have a deterministic, non-nullable,
//! prefix-free DFA.  A component residual is therefore exactly
//! `(completed_copies, body_state)`.  The product residual is the pair of
//! component residuals.  We intern product residuals only when runtime input
//! reaches them; the two regex bounds never determine an allocation size.

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::dfa::DFA;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

const VIRTUAL_STATE_LIMIT: u32 = 1 << 31;
const DEAD_TRANSITION: u32 = u32::MAX;

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct VirtualBoundedRepeatSpec {
    pub base_dfa: Arc<DFA>,
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct VirtualBinaryRepeatIntersectionDescriptor {
    pub left: VirtualBoundedRepeatSpec,
    pub right: VirtualBoundedRepeatSpec,
    pub byte_support: U8Set,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RepeatResidual {
    completed: u32,
    body_state: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ProductResidual {
    left: RepeatResidual,
    right: RepeatResidual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaskRepeatResidual {
    /// Remaining copies, truncated at `horizon + 1`; the top value means
    /// "farther than one vocabulary token can reach".
    remaining: u32,
    body_state: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaskProductResidual {
    left: MaskRepeatResidual,
    right: MaskRepeatResidual,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct VirtualBinaryRepeatIntersectionMaskProjection {
    runtime: Arc<VirtualBinaryRepeatIntersectionRuntime>,
    far: u32,
    mask_state_by_residual: Arc<FxHashMap<MaskProductResidual, u32>>,
}

impl VirtualBinaryRepeatIntersectionMaskProjection {
    #[inline]
    pub fn project(&self, full_state: u32) -> Option<u32> {
        if full_state < self.runtime.physical_state_count {
            return Some(full_state);
        }
        let store = self.runtime.store.lock().unwrap();
        let exact = self.runtime.residual_for_state_locked(&store, full_state)?;
        let projected = self.runtime.mask_residual(exact, self.far);
        self.mask_state_by_residual.get(&projected).copied()
    }
}

#[derive(Debug, Default)]
struct LazyProductStore {
    states: Vec<ProductResidual>,
    state_by_residual: FxHashMap<ProductResidual, u32>,
    transitions: Vec<SmallVec<[(u8, u32); 8]>>,
    root_transitions: SmallVec<[(u8, u32); 8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BodyProductEdge {
    target: u32,
    left_completed: u32,
    right_completed: u32,
}

#[derive(Debug)]
pub(super) struct VirtualBinaryRepeatIntersectionRuntime {
    left: VirtualBoundedRepeatSpec,
    right: VirtualBoundedRepeatSpec,
    byte_support: U8Set,
    terminal: TerminalID,
    physical_state_count: u32,
    root_state: u32,
    accepting: BitSet,
    live: BitSet,
    dead: BitSet,
    accepting_list: Box<[TerminalID]>,
    /// For each pair of body-DFA states, the Pareto-minimal numbers of
    /// additional completed copies on the left/right needed by a *nonempty*
    /// common byte path to return both bodies to their repeat boundary.
    ///
    /// This is the exact future-language certificate for the product.  Counts
    /// are independent of the declared repeat bounds and are finite because a
    /// resource-feasible path can have cycles removed without increasing
    /// either completed-copy count.
    future_requirements: Arc<[Box<[(u32, u32)]>]>,
    future_requirement_cap: u32,
    store: Mutex<LazyProductStore>,
}

impl VirtualBinaryRepeatIntersectionRuntime {
    pub(super) fn new(
        descriptor: VirtualBinaryRepeatIntersectionDescriptor,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
    ) -> Option<Self> {
        if descriptor.byte_support.is_empty()
            || terminal >= num_terminals
            || physical_state_count == 0
            || root_state >= physical_state_count
            || physical_state_count >= VIRTUAL_STATE_LIMIT
            || descriptor.left.min > descriptor.left.max
            || descriptor.right.min > descriptor.right.max
            || descriptor.left.base_dfa.num_states() == 0
            || descriptor.right.base_dfa.num_states() == 0
        {
            return None;
        }
        let (future_requirements, future_requirement_cap) =
            Self::compute_future_requirements(&descriptor.left, &descriptor.right)?;
        let mut accepting = BitSet::new(num_terminals as usize);
        accepting.set(terminal as usize);
        let live = accepting.clone();
        Some(Self {
            left: descriptor.left,
            right: descriptor.right,
            byte_support: descriptor.byte_support,
            terminal,
            physical_state_count,
            root_state,
            accepting,
            live,
            dead: BitSet::new(num_terminals as usize),
            accepting_list: vec![terminal].into_boxed_slice(),
            future_requirements,
            future_requirement_cap,
            store: Mutex::new(LazyProductStore::default()),
        })
    }

    #[inline]
    fn pair_index(&self, left_state: u32, right_state: u32) -> Option<usize> {
        (left_state as usize)
            .checked_mul(self.right.base_dfa.num_states())?
            .checked_add(right_state as usize)
    }

    fn body_step(spec: &VirtualBoundedRepeatSpec, state: u32, byte: u8) -> Option<(u32, u32)> {
        let target = spec.base_dfa.step(state, byte)?;
        if spec.base_dfa.finalizers(target).contains(0) {
            return Some((0, 1));
        }
        if !spec.base_dfa.possible_future_group_ids(target).contains(0) {
            return None;
        }
        Some((target, 0))
    }

    fn insert_pareto(front: &mut Vec<(u32, u32)>, candidate: (u32, u32)) -> bool {
        if front
            .iter()
            .any(|&(left, right)| left <= candidate.0 && right <= candidate.1)
        {
            return false;
        }
        front.retain(|&(left, right)| !(candidate.0 <= left && candidate.1 <= right));
        front.push(candidate);
        true
    }

    fn compute_future_requirements(
        left: &VirtualBoundedRepeatSpec,
        right: &VirtualBoundedRepeatSpec,
    ) -> Option<(Arc<[Box<[(u32, u32)]>]>, u32)> {
        let left_states = left.base_dfa.num_states();
        let right_states = right.base_dfa.num_states();
        let pair_count = left_states.checked_mul(right_states)?;
        if pair_count == 0 || pair_count > u32::MAX as usize {
            return None;
        }

        let mut edges = vec![Vec::<BodyProductEdge>::new(); pair_count];
        for left_state in 0..left_states as u32 {
            for right_state in 0..right_states as u32 {
                let source = left_state as usize * right_states + right_state as usize;
                for byte in 0u16..=255 {
                    let byte = byte as u8;
                    let Some((next_left, left_completed)) = Self::body_step(left, left_state, byte)
                    else {
                        continue;
                    };
                    let Some((next_right, right_completed)) =
                        Self::body_step(right, right_state, byte)
                    else {
                        continue;
                    };
                    let target = next_left as usize * right_states + next_right as usize;
                    let edge = BodyProductEdge {
                        target: target as u32,
                        left_completed,
                        right_completed,
                    };
                    if !edges[source].contains(&edge) {
                        edges[source].push(edge);
                    }
                }
            }
        }

        // Fixed point of Pareto-minimal resource costs for a nonempty path to
        // pair state (0,0). We deliberately do not seed (0,0) with cost (0,0):
        // tokenizer `possible_future` means that at least one more byte can be
        // consumed before the terminal matches again.
        let accept = 0u32;
        let mut requirements = vec![Vec::<(u32, u32)>::new(); pair_count];
        loop {
            let snapshot = requirements.clone();
            let mut changed = false;
            for source in 0..pair_count {
                for edge in &edges[source] {
                    if edge.target == accept {
                        changed |= Self::insert_pareto(
                            &mut requirements[source],
                            (edge.left_completed, edge.right_completed),
                        );
                    }
                    for &(left_cost, right_cost) in &snapshot[edge.target as usize] {
                        let Some(left_cost) = left_cost.checked_add(edge.left_completed) else {
                            return None;
                        };
                        let Some(right_cost) = right_cost.checked_add(edge.right_completed) else {
                            return None;
                        };
                        changed |= Self::insert_pareto(
                            &mut requirements[source],
                            (left_cost, right_cost),
                        );
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let cap = requirements
            .iter()
            .flat_map(|front| front.iter())
            .fold(0u32, |cap, &(left, right)| cap.max(left).max(right));
        let requirements = requirements
            .into_iter()
            .map(|mut front| {
                front.sort_unstable();
                front.into_boxed_slice()
            })
            .collect::<Vec<_>>();
        Some((Arc::from(requirements.into_boxed_slice()), cap))
    }

    fn residual_has_future(&self, residual: ProductResidual) -> bool {
        let Some(index) = self.pair_index(residual.left.body_state, residual.right.body_state)
        else {
            return false;
        };
        let left_remaining = self.left.max.saturating_sub(residual.left.completed);
        let right_remaining = self.right.max.saturating_sub(residual.right.completed);
        self.future_requirements[index]
            .iter()
            .any(|&(left, right)| left <= left_remaining && right <= right_remaining)
    }

    pub(super) fn root_has_future(&self) -> bool {
        self.residual_has_future(self.initial_residual())
    }

    #[inline]
    pub(super) fn terminal(&self) -> TerminalID {
        self.terminal
    }

    #[inline]
    pub(super) fn byte_support(&self) -> U8Set {
        self.byte_support
    }

    #[inline]
    pub(super) fn physical_state_count(&self) -> u32 {
        self.physical_state_count
    }

    #[inline]
    pub(super) fn root_state(&self) -> u32 {
        self.root_state
    }

    #[inline]
    fn initial_residual(&self) -> ProductResidual {
        ProductResidual {
            left: RepeatResidual {
                completed: 0,
                body_state: 0,
            },
            right: RepeatResidual {
                completed: 0,
                body_state: 0,
            },
        }
    }

    fn residual_for_state_locked(
        &self,
        store: &LazyProductStore,
        state: u32,
    ) -> Option<ProductResidual> {
        if state == self.root_state {
            return Some(self.initial_residual());
        }
        let index = state.checked_sub(self.physical_state_count)? as usize;
        store.states.get(index).copied()
    }

    fn intern_locked(
        &self,
        store: &mut LazyProductStore,
        residual: ProductResidual,
    ) -> Option<u32> {
        if residual == self.initial_residual() {
            return Some(self.root_state);
        }
        if let Some(&state) = store.state_by_residual.get(&residual) {
            return Some(state);
        }
        let index = u32::try_from(store.states.len()).ok()?;
        let state = self.physical_state_count.checked_add(index)?;
        if state >= VIRTUAL_STATE_LIMIT {
            return None;
        }
        store.states.push(residual);
        store.transitions.push(SmallVec::new());
        store.state_by_residual.insert(residual, state);
        Some(state)
    }

    fn step_component(
        spec: &VirtualBoundedRepeatSpec,
        residual: RepeatResidual,
        byte: u8,
    ) -> Option<RepeatResidual> {
        // `completed` counts finished copies. A partial next copy can only
        // exist while completed < max.
        if residual.completed >= spec.max {
            return None;
        }
        let target = spec.base_dfa.step(residual.body_state, byte)?;
        if spec.base_dfa.finalizers(target).contains(0) {
            return Some(RepeatResidual {
                completed: residual.completed.checked_add(1)?,
                body_state: 0,
            });
        }
        if !spec.base_dfa.possible_future_group_ids(target).contains(0) {
            return None;
        }
        Some(RepeatResidual {
            completed: residual.completed,
            body_state: target,
        })
    }

    fn step_residual(&self, residual: ProductResidual, byte: u8) -> Option<ProductResidual> {
        Some(ProductResidual {
            left: Self::step_component(&self.left, residual.left, byte)?,
            right: Self::step_component(&self.right, residual.right, byte)?,
        })
    }

    #[inline]
    fn mask_component(
        spec: &VirtualBoundedRepeatSpec,
        residual: RepeatResidual,
        far: u32,
    ) -> MaskRepeatResidual {
        MaskRepeatResidual {
            remaining: spec.max.saturating_sub(residual.completed).min(far),
            body_state: residual.body_state,
        }
    }

    #[inline]
    fn mask_residual(&self, residual: ProductResidual, far: u32) -> MaskProductResidual {
        MaskProductResidual {
            left: Self::mask_component(&self.left, residual.left, far),
            right: Self::mask_component(&self.right, residual.right, far),
        }
    }

    fn step_mask_component(
        spec: &VirtualBoundedRepeatSpec,
        residual: MaskRepeatResidual,
        far: u32,
        byte: u8,
    ) -> Option<MaskRepeatResidual> {
        if residual.remaining == 0 {
            return None;
        }
        let target = spec.base_dfa.step(residual.body_state, byte)?;
        if spec.base_dfa.finalizers(target).contains(0) {
            let remaining = if residual.remaining == far {
                far
            } else {
                residual.remaining.checked_sub(1)?
            };
            return Some(MaskRepeatResidual {
                remaining,
                body_state: 0,
            });
        }
        if !spec.base_dfa.possible_future_group_ids(target).contains(0) {
            return None;
        }
        Some(MaskRepeatResidual {
            remaining: residual.remaining,
            body_state: target,
        })
    }

    fn step_mask_residual(
        &self,
        residual: MaskProductResidual,
        far: u32,
        byte: u8,
    ) -> Option<MaskProductResidual> {
        Some(MaskProductResidual {
            left: Self::step_mask_component(&self.left, residual.left, far, byte)?,
            right: Self::step_mask_component(&self.right, residual.right, far, byte)?,
        })
    }

    fn mask_accepting(residual: MaskProductResidual) -> bool {
        residual.left.body_state == 0 && residual.right.body_state == 0
    }

    /// Build the finite observation DFA used for one model-token walk.
    ///
    /// `remaining = horizon + 1` is a mathematically exact finite-horizon
    /// abstraction: a token of at most `horizon` bytes cannot finish enough
    /// non-nullable repetitions to reach the upper bound from that class.
    pub(super) fn build_mask_projection(
        self: &Arc<Self>,
        horizon: usize,
        mut dfa: DFA,
        num_terminals: u32,
    ) -> Option<(DFA, VirtualBinaryRepeatIntersectionMaskProjection)> {
        let horizon = u32::try_from(horizon).ok()?;
        // `future_requirement_cap` bounds a cycle-free witness for every live
        // body-state pair. Keep another `horizon` copies exact so a model token
        // can consume at most `horizon` non-nullable repetitions and still end
        // in a state whose unbounded future observation is exact. The top value
        // is the merged deep-interior class and self-loops on copy completion.
        let far = self.future_requirement_cap.checked_add(horizon)?;
        let physical_state_count = self.physical_state_count;
        if dfa.num_states() as u32 != physical_state_count || self.left.min != 0 || self.right.min != 0 {
            return None;
        }

        // Every exact committed residual must have a projection, including
        // upper-tail states that are not reachable from the reset-side `far`
        // class. Seed every remaining-count pair at a copy boundary; BFS then
        // adds exactly the partial-body residuals reachable from those seeds.
        let mut state_by_residual = FxHashMap::<MaskProductResidual, u32>::default();
        let mut queue = VecDeque::<MaskProductResidual>::new();
        for left_remaining in 0..=far {
            for right_remaining in 0..=far {
                let residual = MaskProductResidual {
                    left: MaskRepeatResidual {
                        remaining: left_remaining,
                        body_state: 0,
                    },
                    right: MaskRepeatResidual {
                        remaining: right_remaining,
                        body_state: 0,
                    },
                };
                let state = dfa.add_state();
                state_by_residual.insert(residual, state);
                queue.push_back(residual);
            }
        }

        while let Some(residual) = queue.pop_front() {
            let state = state_by_residual[&residual];
            let mut finalizers = BitSet::new(num_terminals as usize);
            if Self::mask_accepting(residual) {
                finalizers.set(self.terminal as usize);
            }
            dfa.overwrite_state_metadata(
                state,
                finalizers,
                BitSet::new(num_terminals as usize),
            );

            for byte in self.byte_support.iter() {
                let Some(next) = self.step_mask_residual(residual, far, byte) else {
                    continue;
                };
                let target = if let Some(&target) = state_by_residual.get(&next) {
                    target
                } else {
                    let target = dfa.add_state();
                    state_by_residual.insert(next, target);
                    queue.push_back(next);
                    target
                };
                dfa.add_transition(state, byte, target);
            }
        }

        // The physical proxy root is the special drained zero-byte state. Its
        // first byte must enter the abstract residual that results from the
        // exact all-zero counter pair, but the root itself stays non-final.
        let exact_root = self.initial_residual();
        for byte in self.byte_support.iter() {
            let Some(next) = self.step_residual(exact_root, byte) else {
                continue;
            };
            let next = self.mask_residual(next, far);
            let target = *state_by_residual.get(&next)?;
            dfa.add_transition(self.root_state, byte, target);
        }

        // Exact graph reachability on the finite quotient supplies future bits;
        // do not approximate intersection liveness from the two components.
        dfa.recompute_possible_futures();

        Some((
            dfa,
            VirtualBinaryRepeatIntersectionMaskProjection {
                runtime: Arc::clone(self),
                far,
                mask_state_by_residual: Arc::new(state_by_residual),
            },
        ))
    }

    pub(super) fn handles_state(&self, state: u32) -> bool {
        if state == self.root_state {
            return true;
        }
        if state < self.physical_state_count {
            return false;
        }
        let store = self.store.lock().unwrap();
        (state - self.physical_state_count) < store.states.len() as u32
    }

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        let mut store = self.store.lock().unwrap();
        let residual = self.residual_for_state_locked(&store, state)?;
        let cached = if state == self.root_state {
            store
                .root_transitions
                .iter()
                .find_map(|&(cached_byte, target)| (cached_byte == byte).then_some(target))
        } else {
            let index = (state - self.physical_state_count) as usize;
            store.transitions[index]
                .iter()
                .find_map(|&(cached_byte, target)| (cached_byte == byte).then_some(target))
        };
        if let Some(target) = cached {
            return (target != DEAD_TRANSITION).then_some(target);
        }

        let target = self
            .step_residual(residual, byte)
            .and_then(|target| self.intern_locked(&mut store, target))
            .unwrap_or(DEAD_TRANSITION);
        if state == self.root_state {
            store.root_transitions.push((byte, target));
        } else {
            let index = (state - self.physical_state_count) as usize;
            store.transitions[index].push((byte, target));
        }
        (target != DEAD_TRANSITION).then_some(target)
    }

    fn component_accepting(spec: &VirtualBoundedRepeatSpec, residual: RepeatResidual) -> bool {
        residual.body_state == 0 && residual.completed >= spec.min
    }

    fn observation(&self, state: u32) -> Option<(bool, bool)> {
        let store = self.store.lock().unwrap();
        let residual = self.residual_for_state_locked(&store, state)?;
        // The physical proxy is the drained zero-byte configuration. Even if
        // both repetitions are nullable through min=0, it must not report a
        // terminal match before input has been consumed.
        let accepting = state != self.root_state
            && Self::component_accepting(&self.left, residual.left)
            && Self::component_accepting(&self.right, residual.right);
        let live = self.residual_has_future(residual);
        Some((accepting, live))
    }

    pub(super) fn finalizers(&self, state: u32) -> Option<&BitSet> {
        let (accepting, _) = self.observation(state)?;
        Some(if accepting { &self.accepting } else { &self.dead })
    }

    pub(super) fn finalizer_list(&self, state: u32) -> Option<&[TerminalID]> {
        let (accepting, _) = self.observation(state)?;
        Some(if accepting {
            self.accepting_list.as_ref()
        } else {
            &[]
        })
    }

    pub(super) fn futures(&self, state: u32) -> Option<&BitSet> {
        let (_, live) = self.observation(state)?;
        Some(if live { &self.live } else { &self.dead })
    }

    pub(super) fn transitions(&self, state: u32) -> Option<Vec<(u8, u32)>> {
        if !self.handles_state(state) {
            return None;
        }
        let mut transitions = Vec::new();
        for byte in self.byte_support.iter() {
            if let Some(target) = self.step(state, byte) {
                transitions.push((byte, target));
            }
        }
        Some(transitions)
    }

    pub(super) fn interned_state_count(&self) -> usize {
        self.store.lock().unwrap().states.len()
    }
}
