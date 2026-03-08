
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

// SEP1_MAP: This submodule is the direct glrmask counterpart to sep1's `glr/` tree, but reduced to the pieces the current compiler pipeline still uses.

pub mod analysis;
pub mod labels;
pub mod parser;
pub mod table;

pub use analysis as grammar;

#[cfg(test)]
mod test_glr_parser;
