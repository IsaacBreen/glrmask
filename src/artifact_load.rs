//! Shared, conservative admission budget for native artifact decoding.
use crate::ErrorExt as _;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

const MAX_BYTES: usize = 1 << 30;
const MAX_DEPTH: usize = 64;

#[derive(Clone)]
pub(crate) struct LoadBudget {
    remaining: Arc<AtomicUsize>,
    depth: usize,
}

impl LoadBudget {
    pub(crate) fn new(input_len: usize) -> Self {
        let limit = input_len.saturating_mul(64).saturating_add(64 << 20).min(MAX_BYTES);
        Self { remaining: Arc::new(AtomicUsize::new(limit)), depth: 0 }
    }

    pub(crate) fn charge(&self, bytes: usize) -> Result<(), String> {
        self.remaining.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| left.checked_sub(bytes))
            .map(|_| ()).map_err(|_| "artifact decoded-size/work budget exceeded".to_owned())
    }

    pub(crate) fn child(&self) -> crate::Result<Self> {
        if self.depth >= MAX_DEPTH {
            return Err(crate::Error::Serialization("artifact nesting is too deep".into()));
        }
        Ok(Self { remaining: Arc::clone(&self.remaining), depth: self.depth + 1 })
    }

    pub(crate) fn charge_items<T>(&self, count: usize) -> Result<(), String> {
        self.charge(count.checked_mul(std::mem::size_of::<T>().max(1))
            .ok_or_else(|| "artifact allocation size overflow".to_owned())?)
    }
}

pub(crate) fn reserve<T>(count: usize) -> Result<Vec<T>, String> {
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(|_| "artifact allocation failed".to_owned())?;
    Ok(result)
}

pub(crate) fn copy_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut result = reserve(bytes.len())?;
    result.extend_from_slice(bytes);
    Ok(result)
}

/// Last-resort containment of recoverable decoder invariant failures.
pub(crate) fn guard<T>(work: impl FnOnce() -> crate::Result<T>) -> crate::Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
        .map_err(|_| crate::Error::Serialization("invalid artifact caused a decoder failure".into()))?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn artifact_budget_is_shared_by_children_and_parallel_decoders() {
        let root = LoadBudget::new(0);
        let child = root.child().unwrap();
        child.charge(32 << 20).unwrap();
        rayon::join(|| root.charge(16 << 20).unwrap(), || child.charge(16 << 20).unwrap());
        assert!(root.charge(1).is_err());
        assert!(child.charge_items::<u64>(usize::MAX).is_err());
        let mut branch = root;
        for _ in 0..MAX_DEPTH { branch = branch.child().unwrap(); }
        assert!(branch.child().is_err());
    }
}
