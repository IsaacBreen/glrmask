//! Template-related compiler stages.
//!
//! These stages split parser-side template analysis from parser-side template
//! compilation so orchestration can expose the intended pipeline shape.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

pub mod characterize;
pub mod compile;
