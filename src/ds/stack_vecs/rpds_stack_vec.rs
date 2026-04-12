use std::hash::{Hash, Hasher};
use super::stack_vec::StackVec;

/// Stack vector backed by `rpds::Stack<T>` (persistent linked list).
/// O(1) clone/push/pop, O(n) access/iterate.
/// `try_push` always succeeds.
///
/// Note: rpds::Stack stores elements in LIFO order (top first).
/// Our convention is bottom-first, so we reverse on conversion.
#[derive(Clone, Debug)]
pub struct RpdsStackVec<T>(rpds::Stack<T>);

impl<T: Clone + Eq + Hash> RpdsStackVec<T> {
    /// Iterate from bottom to top. O(n) — collects and reverses.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        // rpds::Stack iterates top-to-bottom; we need bottom-to-top.
        // Collect references, reverse, return.
        let items: Vec<&T> = self.0.iter().collect();
        items.into_iter().rev()
    }
}

impl<T: PartialEq> PartialEq for RpdsStackVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.0.size() != other.0.size() {
            return false;
        }
        self.0.iter().eq(other.0.iter())
    }
}

impl<T: Eq> Eq for RpdsStackVec<T> {}

impl<T: Hash> Hash for RpdsStackVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.size().hash(state);
        for item in self.0.iter() {
            item.hash(state);
        }
    }
}

impl<T> Default for RpdsStackVec<T> {
    fn default() -> Self {
        Self(rpds::Stack::new())
    }
}

impl<T: Clone + Eq + Hash + std::fmt::Debug> StackVec<T> for RpdsStackVec<T>
{
    fn unit(val: T) -> Self {
        Self(rpds::Stack::new().push(val))
    }

    fn from_vec(v: Vec<T>) -> Self {
        // Push each element; rpds::Stack pushes to front, so push in order
        // to get bottom-first (first pushed = deepest = last in rpds iteration).
        let mut stack = rpds::Stack::new();
        for item in v {
            stack = stack.push(item);
        }
        Self(stack)
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.size()
    }

    #[inline]
    fn last(&self) -> Option<&T> {
        // "Last" = top of stack = rpds front
        self.0.peek()
    }

    fn take(&self, n: usize) -> Self {
        let n = n.min(self.0.size());
        if n == self.0.size() {
            return self.clone();
        }
        // Collect bottom-to-top, take first n, rebuild
        let items: Vec<T> = {
            let refs: Vec<&T> = self.0.iter().collect();
            refs.into_iter().rev().take(n).cloned().collect()
        };
        Self::from_vec(items)
    }

    fn truncate(&mut self, new_len: usize) {
        let new_len = new_len.min(self.0.size());
        let pops = self.0.size() - new_len;
        for _ in 0..pops {
            self.0 = self.0.pop().unwrap_or_else(rpds::Stack::new);
        }
    }

    #[inline]
    fn try_push(&mut self, val: T) -> bool {
        self.0 = self.0.push(val);
        true
    }

    fn append(&self, other: &Self) -> Self {
        if other.0.is_empty() {
            return self.clone();
        }
        // Collect self bottom-to-top, then other bottom-to-top, rebuild
        let mut items: Vec<T> = {
            let refs: Vec<&T> = self.0.iter().collect();
            refs.into_iter().rev().cloned().collect()
        };
        let other_items: Vec<&T> = other.0.iter().collect();
        items.extend(other_items.into_iter().rev().cloned());
        Self::from_vec(items)
    }

    fn to_vec(&self) -> Vec<T> {
        let refs: Vec<&T> = self.0.iter().collect();
        refs.into_iter().rev().cloned().collect()
    }
}
