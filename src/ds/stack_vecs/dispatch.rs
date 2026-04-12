use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use super::arc_array_vec::ArcArrayVec;
use super::array_stack_vec::{ArrayStackVec128, ArrayStackVec64};
use super::im_stack_vec::ImStackVec;
use super::seg_vec::SegVec;
use super::vec_stack_vec::VecStackVec;
use super::small_stack_vec::{SmallStackVec64, SmallStackVec128};

// Import StackVec trait so trait methods are available on types that lack inherent equivalents.
use super::stack_vec::StackVec;

/// Which StackVec variant to use, selected once at startup via `STACKVEC` env var.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    ArcArray,
    Array64,
    Array128,
    Im,
    Seg,
    Vec,
    Small64,
    Small128,
}

static VARIANT: OnceLock<Variant> = OnceLock::new();

fn selected_variant() -> Variant {
    *VARIANT.get_or_init(|| {
        match std::env::var("STACKVEC").as_deref() {
            Ok("arc") | Ok("arc_array") => Variant::ArcArray,
            Ok("array64") => Variant::Array64,
            Ok("array128") | Ok("array") => Variant::Array128,
            Ok("im") | Ok("im_vector") => Variant::Im,
            Ok("seg") | Ok("seg_vec") => Variant::Seg,
            Ok("vec") => Variant::Vec,
            Ok("small64") | Ok("small") => Variant::Small64,
            Ok("small128") => Variant::Small128,
            _ => Variant::ArcArray, // default
        }
    })
}

/// Enum-dispatched StackVec that selects implementation via `STACKVEC` env var.
///
/// Set `STACKVEC` environment variable before first use:
///   `arc` / `arc_array` — ArcArrayVec (default)
///   `array64` — ArrayStackVec<64>
///   `array128` / `array` — ArrayStackVec<128>
///   `im` / `im_vector` — im::Vector
///   `seg` / `seg_vec` — SegVec (Arc<[T]> view)
///   `vec` — plain Vec
///   `small64` / `small` — SmallVec<64>
///   `small128` — SmallVec<128>
#[derive(Clone)]
pub enum DynStackVec<T: Clone> {
    ArcArray(ArcArrayVec<T>),
    Array64(ArrayStackVec64<T>),
    Array128(ArrayStackVec128<T>),
    Im(ImStackVec<T>),
    Seg(SegVec<T>),
    Vec(VecStackVec<T>),
    Small64(SmallStackVec64<T>),
    Small128(SmallStackVec128<T>),
}

macro_rules! dispatch {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match $self {
            DynStackVec::ArcArray(v) => v.$method($($args),*),
            DynStackVec::Array64(v) => v.$method($($args),*),
            DynStackVec::Array128(v) => v.$method($($args),*),
            DynStackVec::Im(v) => v.$method($($args),*),
            DynStackVec::Seg(v) => v.$method($($args),*),
            DynStackVec::Vec(v) => v.$method($($args),*),
            DynStackVec::Small64(v) => v.$method($($args),*),
            DynStackVec::Small128(v) => v.$method($($args),*),
        }
    };
}

macro_rules! dispatch_wrap {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match $self {
            DynStackVec::ArcArray(v) => DynStackVec::ArcArray(v.$method($($args),*)),
            DynStackVec::Array64(v) => DynStackVec::Array64(v.$method($($args),*)),
            DynStackVec::Array128(v) => DynStackVec::Array128(v.$method($($args),*)),
            DynStackVec::Im(v) => DynStackVec::Im(v.$method($($args),*)),
            DynStackVec::Seg(v) => DynStackVec::Seg(v.$method($($args),*)),
            DynStackVec::Vec(v) => DynStackVec::Vec(v.$method($($args),*)),
            DynStackVec::Small64(v) => DynStackVec::Small64(v.$method($($args),*)),
            DynStackVec::Small128(v) => DynStackVec::Small128(v.$method($($args),*)),
        }
    };
}

impl<T: Clone + Eq + Hash> DynStackVec<T> {
    #[inline]
    pub fn unit(val: T) -> Self {
        match selected_variant() {
            Variant::ArcArray => Self::ArcArray(ArcArrayVec::unit(val)),
            Variant::Array64 => Self::Array64(ArrayStackVec64::unit(val)),
            Variant::Array128 => Self::Array128(ArrayStackVec128::unit(val)),
            Variant::Im => Self::Im(ImStackVec::unit(val)),
            Variant::Seg => Self::Seg(SegVec::unit(val)),
            Variant::Vec => Self::Vec(VecStackVec::unit(val)),
            Variant::Small64 => Self::Small64(SmallStackVec64::unit(val)),
            Variant::Small128 => Self::Small128(SmallStackVec128::unit(val)),
        }
    }

    pub fn from_vec(v: Vec<T>) -> Self {
        match selected_variant() {
            Variant::ArcArray => Self::ArcArray(ArcArrayVec::from_vec(v)),
            Variant::Array64 => Self::Array64(ArrayStackVec64::from_vec(v)),
            Variant::Array128 => Self::Array128(ArrayStackVec128::from_vec(v)),
            Variant::Im => Self::Im(ImStackVec::from_vec(v)),
            Variant::Seg => Self::Seg(SegVec::from_vec(v)),
            Variant::Vec => Self::Vec(VecStackVec::from_vec(v)),
            Variant::Small64 => Self::Small64(SmallStackVec64::from_vec(v)),
            Variant::Small128 => Self::Small128(SmallStackVec128::from_vec(v)),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        dispatch!(self, len)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn last(&self) -> Option<&T> {
        dispatch!(self, last)
    }

    #[inline]
    pub fn take(&self, n: usize) -> Self {
        dispatch_wrap!(self, take, n)
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        match self {
            Self::ArcArray(v) => v.truncate(new_len),
            Self::Array64(v) => v.truncate(new_len),
            Self::Array128(v) => v.truncate(new_len),
            Self::Im(v) => v.truncate(new_len),
            Self::Seg(v) => v.truncate(new_len),
            Self::Vec(v) => v.truncate(new_len),
            Self::Small64(v) => v.truncate(new_len),
            Self::Small128(v) => v.truncate(new_len),
        }
    }

    #[inline]
    pub fn try_push(&mut self, val: T) -> bool {
        match self {
            Self::ArcArray(v) => v.try_push(val),
            Self::Array64(v) => v.try_push(val),
            Self::Array128(v) => v.try_push(val),
            Self::Im(v) => v.try_push(val),
            Self::Seg(_v) => { let _ = val; false }, // SegVec can't push in-place
            Self::Vec(v) => v.try_push(val),
            Self::Small64(v) => v.try_push(val),
            Self::Small128(v) => v.try_push(val),
        }
    }

    pub fn append(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::ArcArray(a), Self::ArcArray(b)) => Self::ArcArray(a.append(b)),
            (Self::Array64(a), Self::Array64(b)) => Self::Array64(a.append(b)),
            (Self::Array128(a), Self::Array128(b)) => Self::Array128(a.append(b)),
            (Self::Im(a), Self::Im(b)) => Self::Im(a.append(b)),
            (Self::Seg(a), Self::Seg(b)) => Self::Seg(a.append(b)),
            (Self::Vec(a), Self::Vec(b)) => Self::Vec(a.append(b)),
            (Self::Small64(a), Self::Small64(b)) => Self::Small64(a.append(b)),
            (Self::Small128(a), Self::Small128(b)) => Self::Small128(a.append(b)),
            _ => panic!("DynStackVec: variant mismatch in append"),
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        dispatch!(self, to_vec)
    }

    /// Iterate from bottom to top.
    pub fn iter(&self) -> DynIter<'_, T> {
        match self {
            Self::ArcArray(v) => DynIter::Slice(v.iter()),
            Self::Array64(v) => DynIter::Slice(v.iter()),
            Self::Array128(v) => DynIter::Slice(v.iter()),
            Self::Im(v) => DynIter::Im(v.iter()),
            Self::Seg(v) => DynIter::Slice(v.iter()),
            Self::Vec(v) => DynIter::Slice(v.iter()),
            Self::Small64(v) => DynIter::Slice(v.iter()),
            Self::Small128(v) => DynIter::Slice(v.iter()),
        }
    }
}

/// Iterator enum for DynStackVec. Supports DoubleEndedIterator.
pub enum DynIter<'a, T> {
    Slice(std::slice::Iter<'a, T>),
    Im(im::vector::Iter<'a, T>),
}

impl<'a, T: Clone> Iterator for DynIter<'a, T> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        match self {
            Self::Slice(it) => it.next(),
            Self::Im(it) => it.next(),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Slice(it) => it.size_hint(),
            Self::Im(it) => it.size_hint(),
        }
    }
}

impl<'a, T: Clone> DoubleEndedIterator for DynIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        match self {
            Self::Slice(it) => it.next_back(),
            Self::Im(it) => it.next_back(),
        }
    }
}

impl<'a, T: Clone> ExactSizeIterator for DynIter<'a, T> {}

impl<T: PartialEq + Clone> PartialEq for DynStackVec<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ArcArray(a), Self::ArcArray(b)) => a == b,
            (Self::Array64(a), Self::Array64(b)) => a == b,
            (Self::Array128(a), Self::Array128(b)) => a == b,
            (Self::Im(a), Self::Im(b)) => a == b,
            (Self::Seg(a), Self::Seg(b)) => a == b,
            (Self::Vec(a), Self::Vec(b)) => a == b,
            (Self::Small64(a), Self::Small64(b)) => a == b,
            (Self::Small128(a), Self::Small128(b)) => a == b,
            _ => false,
        }
    }
}

impl<T: Eq + Clone> Eq for DynStackVec<T> {}

impl<T: Hash + Clone> Hash for DynStackVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash discriminant + inner
        std::mem::discriminant(self).hash(state);
        match self {
            Self::ArcArray(v) => v.hash(state),
            Self::Array64(v) => v.hash(state),
            Self::Array128(v) => v.hash(state),
            Self::Im(v) => v.hash(state),
            Self::Seg(v) => v.hash(state),
            Self::Vec(v) => v.hash(state),
            Self::Small64(v) => v.hash(state),
            Self::Small128(v) => v.hash(state),
        }
    }
}

impl<T: std::fmt::Debug + Clone> std::fmt::Debug for DynStackVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArcArray(v) => write!(f, "DynSV::Arc({v:?})"),
            Self::Array64(v) => write!(f, "DynSV::Arr64({v:?})"),
            Self::Array128(v) => write!(f, "DynSV::Arr128({v:?})"),
            Self::Im(v) => write!(f, "DynSV::Im({v:?})"),
            Self::Seg(v) => write!(f, "DynSV::Seg({v:?})"),
            Self::Vec(v) => write!(f, "DynSV::Vec({v:?})"),
            Self::Small64(v) => write!(f, "DynSV::Sm64({v:?})"),
            Self::Small128(v) => write!(f, "DynSV::Sm128({v:?})"),
        }
    }
}

impl<T: Clone + Eq + Hash> Default for DynStackVec<T> {
    fn default() -> Self {
        Self::from_vec(Vec::new())
    }
}
