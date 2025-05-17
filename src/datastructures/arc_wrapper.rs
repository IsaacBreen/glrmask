use std::sync::{Arc, Mutex};
use std::ops::Deref;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;

use crate::json_serialization::{JSONNode, JSONConvertible}; // Add this line


/// A wrapper around `Arc<T>` that implements `PartialEq`, `Eq`, `PartialOrd`, `Ord`,
/// and `Hash` based on the pointer value of the `Arc`.
/// This allows `Arc<T>` instances to be used in collections like `BTreeSet` or
/// `HashMap` where identity is determined by the `Arc`'s pointer, not its content.
/// It also dereferences to the underlying `Arc<T>`.
pub struct ArcPtrWrapper<T>(Arc<T>);

// Implement JSONConvertible for ArcPtrWrapper<T> where T is JSONConvertible
// Note: This implementation assumes T itself is JSONConvertible.
// If T is, for example, Mutex<U>, then Arc<Mutex<U>> needs to be JSONConvertible.
// This specific wrapper is used for Arc<Mutex<Trie>>, and we have
// JSONConvertible implemented for Arc<Mutex<Trie>> in json_serialization.rs.
impl<T: JSONConvertible> JSONConvertible for ArcPtrWrapper<T> {
    fn to_json(&self) -> JSONNode {
        self.0.to_json() // Delegates to Arc<T>'s to_json
    }

    fn from_json(node: &JSONNode) -> Result<Self, String> {
        Arc::<T>::from_json(node).map(ArcPtrWrapper::new)
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

impl<T> Clone for ArcPtrWrapper<T> {
    fn clone(&self) -> Self {
        ArcPtrWrapper(self.0.clone())
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

impl<T> PartialOrd for ArcPtrWrapper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ArcPtrWrapper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (Arc::as_ptr(&self.0) as usize).cmp(&(Arc::as_ptr(&other.0) as usize))
    }
}

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
