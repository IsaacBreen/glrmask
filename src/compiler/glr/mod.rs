//! GLR parser and table generation.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod analysis;
pub mod labels;
pub mod parser;
pub mod table;

pub use analysis as grammar;
