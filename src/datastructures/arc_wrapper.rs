use std::sync::{Arc, Weak};
use std::ops::Deref;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added

/// A wrapper around `Arc<T>` that implements `PartialEq`, `Eq`, `PartialOrd`, `Ord`,
/// and `Hash` based on the pointer value of the `Arc`.
/// This allows `Arc<T>` instances to be used in collections like `BTreeSet` or
/// `HashMap` where identity is determined by the `Arc`'s pointer, not its content.
/// It also dereferences to the underlying `Arc<T>`.
#[derive(Clone)]
pub struct ArcPtrWrapper<T>(Arc<T>);

// ArcPtrWrapper serialization:
// Serializing based on pointer identity is not meaningful for JSON.
// We will serialize the *content* of the Arc.
// Deserialization will create a new Arc, so pointer identity will not be preserved.
// This is a fundamental limitation when serializing pointer-based identity wrappers.
impl<T: JSONConvertible> JSONConvertible for ArcPtrWrapper<T> {
    fn to_json(&self) -> JSONNode {
        self.0.as_ref().to_json() // Serialize the content
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        T::from_json(node).map(|content| ArcPtrWrapper(Arc::new(content)))
    }
}

/// A wrapper around `Weak<T>` that implements `PartialEq`, `Eq`, `PartialOrd`, `Ord`,
/// and `Hash` based on the pointer value of the `Weak`.
/// This allows `Weak<T>` instances to be used as map/set keys where identity is
/// determined by the allocation pointer, not its content.
///
/// Note:
/// - `Weak<T>` may fail to upgrade; code using this wrapper should be prepared
///   to skip dangling weak references.
#[derive(Clone)]
pub struct WeakPtrWrapper<T>(Weak<T>);

impl<T> WeakPtrWrapper<T> {
    /// Creates a new `WeakPtrWrapper` from a `Weak<T>`.
    pub fn new(weak: Weak<T>) -> Self {
        WeakPtrWrapper(weak)
    }

    /// Returns a reference to the inner `Weak<T>`.
    pub fn as_weak(&self) -> &Weak<T> {
        &self.0
    }

    /// Attempts to upgrade the weak reference to an `Arc<T>`.
    pub fn upgrade(&self) -> Option<Arc<T>> {
        self.0.upgrade()
    }

    /// Consumes the wrapper and returns the inner `Weak<T>`.
    pub fn into_weak(self) -> Weak<T> {
        self.0
    }
}

impl<T> Deref for WeakPtrWrapper<T> {
    type Target = Weak<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> PartialEq for WeakPtrWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        Weak::as_ptr(&self.0) == Weak::as_ptr(&other.0)
    }
}

impl<T> Eq for WeakPtrWrapper<T> {}

impl<T> Hash for WeakPtrWrapper<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Weak::as_ptr(&self.0) as usize).hash(state);
    }
}

impl<T> fmt::Debug for WeakPtrWrapper<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("WeakPtrWrapper")
         .field(&(Weak::as_ptr(&self.0) as *const ()))
         .finish()
    }
}

impl<T> ArcPtrWrapper<T> {
    /// Creates a new `ArcPtrWrapper` from an `Arc<T>`.
    pub fn new(arc: Arc<T>) -> Self {
        ArcPtrWrapper(arc)
    }

    /// Returns a reference to the inner `Arc<T>`.
    pub fn as_arc(&self) -> &Arc<T> {
        &self.0
    }

    /// Consumes the wrapper and returns the inner `Arc<T>`.
    pub fn into_arc(self) -> Arc<T> {
        self.0
    }
}

impl<T> Deref for ArcPtrWrapper<T> {
    type Target = Arc<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> PartialEq for ArcPtrWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for ArcPtrWrapper<T> {}

impl<T> Hash for ArcPtrWrapper<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl<T> fmt::Debug for ArcPtrWrapper<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ArcPtrWrapper")
         .field(&(Arc::as_ptr(&self.0) as *const ())) // Print the pointer
         .finish()
    }
}

impl <T> PartialOrd for ArcPtrWrapper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ArcPtrWrapper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        Arc::as_ptr(&self.0).cmp(&Arc::as_ptr(&other.0))
    }
}