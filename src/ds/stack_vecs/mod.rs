mod stack_vec;
pub mod arc_array_vec;
pub mod vec_stack_vec;
pub mod dispatch;

#[cfg(feature = "stackvec-experiments")]
pub mod array_stack_vec;
#[cfg(feature = "stackvec-experiments")]
pub mod im_stack_vec;
#[cfg(feature = "stackvec-experiments")]
pub mod seg_vec;
#[cfg(feature = "stackvec-experiments")]
pub mod small_stack_vec;
#[cfg(feature = "stackvec-experiments")]
pub mod rpds_stack_vec;

#[allow(unused_imports)]
pub use stack_vec::StackVec;
