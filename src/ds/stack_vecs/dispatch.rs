use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use super::arc_array_vec::ArcArrayVec;
use super::stack_vec::StackVec;
use super::vec_stack_vec::VecStackVec;

/// Which StackVec variant to use, selected once at startup via `STACKVEC` env var.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    ArcArray,
    Vec,
}

static VARIANT: OnceLock<Variant> = OnceLock::new();

fn selected_variant() -> Variant {
    *VARIANT.get_or_init(|| {
        match std::env::var("STACKVEC").as_deref() {
            Ok("arc") | Ok("arc_array") => Variant::ArcArray,
            Ok("vec") => Variant::Vec,
            _ => Variant::Vec, // default
        }
    })
}

/// Generates the DynStackVec enum and all method dispatch boilerplate.
///
/// Each variant entry: (VariantName, InnerType, iter_kind)
///   iter_kind = slice | im
macro_rules! define_dyn_stack_vec {
    (
        $(
            $variant:ident($inner:ty) => $iter_kind:ident
        ),+ $(,)?
    ) => {
        /// Enum-dispatched StackVec that selects implementation via `STACKVEC` env var.
        #[derive(Clone)]
        pub enum DynStackVec<T: Clone> {
            $( $variant($inner), )+
        }

        impl<T: Clone + Eq + Hash> DynStackVec<T> {
            #[inline]
            pub fn unit(val: T) -> Self {
                match selected_variant() {
                    $( Variant::$variant => Self::$variant(<$inner as StackVec<T>>::unit(val)), )+
                }
            }

            pub fn from_vec(v: Vec<T>) -> Self {
                match selected_variant() {
                    $( Variant::$variant => Self::$variant(<$inner as StackVec<T>>::from_vec(v)), )+
                }
            }

            #[inline]
            pub fn len(&self) -> usize {
                match self { $( Self::$variant(v) => v.len(), )+ }
            }

            #[inline]
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            #[inline]
            pub fn last(&self) -> Option<&T> {
                match self { $( Self::$variant(v) => v.last(), )+ }
            }

            #[inline]
            pub fn take(&self, n: usize) -> Self {
                match self { $( Self::$variant(v) => Self::$variant(v.take(n)), )+ }
            }

            #[inline]
            pub fn truncate(&mut self, new_len: usize) {
                match self { $( Self::$variant(v) => v.truncate(new_len), )+ }
            }

            #[inline]
            pub fn try_push(&mut self, val: T) -> bool {
                match self { $( Self::$variant(v) => v.try_push(val), )+ }
            }

            #[inline]
            pub fn try_harder_push(&mut self, val: T) -> bool {
                match self { $( Self::$variant(v) => v.try_harder_push(val), )+ }
            }

            pub fn append(&self, other: &Self) -> Self {
                match (self, other) {
                    $( (Self::$variant(a), Self::$variant(b)) => Self::$variant(a.append(b)), )+
                    _ => panic!("DynStackVec: variant mismatch in append"),
                }
            }

            pub fn try_append(&self, other: &Self) -> Option<Self> {
                match (self, other) {
                    $( (Self::$variant(a), Self::$variant(b)) => a.try_append(b).map(Self::$variant), )+
                    _ => panic!("DynStackVec: variant mismatch in try_append"),
                }
            }

            #[inline]
            pub fn capacity(&self) -> usize {
                match self { $( Self::$variant(v) => v.capacity(), )+ }
            }

            pub fn to_vec(&self) -> Vec<T> {
                match self { $( Self::$variant(v) => v.to_vec(), )+ }
            }

            /// Iterate from bottom to top.
            pub fn iter(&self) -> DynIter<'_, T> {
                match self {
                    $( Self::$variant(v) => define_dyn_stack_vec!(@iter $iter_kind v), )+
                }
            }
        }

        impl<T: PartialEq + Clone> PartialEq for DynStackVec<T> {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    $( (Self::$variant(a), Self::$variant(b)) => a == b, )+
                    _ => false,
                }
            }
        }

        impl<T: Eq + Clone> Eq for DynStackVec<T> {}

        impl<T: Hash + Clone> Hash for DynStackVec<T> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                std::mem::discriminant(self).hash(state);
                match self { $( Self::$variant(v) => v.hash(state), )+ }
            }
        }

        impl<T: std::fmt::Debug + Clone> std::fmt::Debug for DynStackVec<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $( Self::$variant(v) => write!(f, concat!("DynSV::", stringify!($variant), "({v:?})"), v = v), )+
                }
            }
        }

        impl<T: Clone + Eq + Hash> Default for DynStackVec<T> {
            fn default() -> Self {
                Self::from_vec(Vec::new())
            }
        }
    };

    // Helper: generate DynIter from iter kind
    (@iter slice $v:ident) => { DynIter::Slice($v.iter()) };
    (@iter im $v:ident) => { DynIter::Im($v.iter()) };
    (@iter vec_refs $v:ident) => { DynIter::VecRefs($v.iter()) };
}

define_dyn_stack_vec! {
    ArcArray(ArcArrayVec<T>)        => slice,
    Vec(VecStackVec<T>)             => slice,
}

/// Iterator enum for DynStackVec. Supports DoubleEndedIterator.
pub enum DynIter<'a, T> {
    Slice(std::slice::Iter<'a, T>),
}

impl<'a, T: Clone> Iterator for DynIter<'a, T> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        match self {
            Self::Slice(it) => it.next(),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Slice(it) => it.size_hint(),
        }
    }
}

impl<'a, T: Clone> DoubleEndedIterator for DynIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        match self {
            Self::Slice(it) => it.next_back(),
        }
    }
}

impl<'a, T: Clone> ExactSizeIterator for DynIter<'a, T> {}
