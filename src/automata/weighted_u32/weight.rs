//! Thin compatibility layer over the relocated 2D range-set backend.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub use range_set_blaze::RangeSetBlaze;

pub use crate::ds::rangeset2d::{
    RangeSet2D,
    bare,
    vec_btmap_rsb,
    vec_rsb,
};

pub type Weight = crate::ds::rangeset2d::RangeSet2D;
