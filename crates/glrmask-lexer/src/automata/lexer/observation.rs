//! Shared exact byte-action layouts for compiler observations.
use std::sync::Arc;
/// An exact byte action on the represented alphabet. Dense physical rows and
/// finite-vocabulary quotient rows differ only in their column map. The rows
/// always exist; derived transposes must never stand in for missing topology.
#[derive(Debug, Clone)]
pub struct ByteRows {
    pub values: Arc<[u32]>,
    pub byte_to_column: [u8; 256],
    pub width: usize,
}

pub fn column_representatives(classes: &[u8; 256]) -> Vec<u8> {
    let width = classes.iter().copied().max().unwrap() as usize + 1;
    let mut representatives = vec![0; width];
    for byte in (0..=255u8).rev() {
        representatives[classes[byte as usize] as usize] = byte;
    }
    representatives
}

/// Give every relevant byte a column, reserving zero for the unused domain.
/// A wider alphabet can use the dense representation without expansion.
pub fn conservative_byte_classes(relevant: &[bool; 256]) -> Option<[u8; 256]> {
    if relevant.iter().filter(|&&active| active).count() >= 255 {
        return None;
    }
    let mut classes = [0; 256];
    let mut next = 1u8;
    for (byte, &active) in relevant.iter().enumerate() {
        if active {
            classes[byte] = next;
            next += 1;
        }
    }
    Some(classes)
}

impl ByteRows {
    pub fn dense(values: Arc<[u32]>) -> Self {
        Self::new(values, std::array::from_fn(|byte| byte as u8))
    }

    pub fn new(values: Arc<[u32]>, byte_to_column: [u8; 256]) -> Self {
        let width = byte_to_column.iter().copied().max().unwrap() as usize + 1;
        assert_eq!(values.len() % width, 0, "invalid byte rows");
        Self { values, byte_to_column, width }
    }

    pub fn num_states(&self) -> usize {
        self.values.len() / self.width
    }

    #[inline]
    pub fn row(&self, state: usize) -> &[u32] {
        let base = state * self.width;
        &self.values[base..base + self.width]
    }

    #[inline]
    pub fn trans(&self, state: usize, byte: usize) -> u32 {
        self.values[state * self.width + self.byte_to_column[byte] as usize]
    }

    pub fn same_topology(&self, other: &Self) -> bool {
        self.byte_to_column == other.byte_to_column
            && (Arc::ptr_eq(&self.values, &other.values) || self.values == other.values)
    }

    /// Materialize an alternative exact layout, or borrow the existing row
    /// storage by Arc when the requested coordinate is already present.
    pub fn repack(&self, classes: &[u8; 256], transposed: bool) -> Arc<[u32]> {
        if !transposed && self.byte_to_column == *classes {
            return Arc::clone(&self.values);
        }
        let representatives = column_representatives(classes);
        let width = representatives.len();
        let states = self.num_states();
        let mut values = vec![u32::MAX; states * width];
        for state in 0..states {
            let row = self.row(state);
            for (class, &byte) in representatives.iter().enumerate() {
                let index = if transposed { class * states + state } else { state * width + class };
                values[index] = row[self.byte_to_column[byte as usize] as usize];
            }
        }
        Arc::from(values)
    }
}

