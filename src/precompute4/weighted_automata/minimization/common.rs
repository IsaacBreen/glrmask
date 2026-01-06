//! Common types shared between DWA and NWA minimization.

pub const MAX_OPTIMIZE_ITERATIONS: usize = 1000;

/// Partition for state minimization.
#[derive(Clone, Debug)]
pub struct Partition {
    pub class_of: Vec<usize>,
    pub num_classes: usize,
}

impl Partition {
    pub fn new(num_states: usize) -> Self {
        Partition {
            class_of: vec![0; num_states],
            num_classes: 1,
        }
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}
