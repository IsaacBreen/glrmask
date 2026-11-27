//! Exact physical-liveness memoization, local to one vocabulary walk.
//! Each parser node owns only the row it actually queries.

const SLOTS: usize = 16;
const PROMOTE_COLLISIONS: u16 = 64;
const MAX_DENSE_BYTES: usize = 64 * 1024;

pub(super) struct PhysicalBoundaryPolicy {
    pub(super) width: usize,
    pub(super) compact: bool,
    promoted_bytes: usize,
}

impl PhysicalBoundaryPolicy {
    pub(super) fn new(width: usize, compact: bool) -> Self {
        Self {
            width,
            compact: compact && width > std::mem::size_of::<CompactRow>(),
            promoted_bytes: 0,
        }
    }
}

pub(super) struct CompactRow {
    cells: [u64; SLOTS],
    collisions: u16,
}

/// Compact exact tags may evict a result, never alias it. Promotion retains
/// all non-evicted cells and shares a fixed additional dense-memory budget.
pub(super) enum PhysicalBoundaryRow {
    Compact(Box<CompactRow>),
    Dense(Box<[u8]>),
}

impl PhysicalBoundaryRow {
    pub(super) fn new(policy: &PhysicalBoundaryPolicy) -> Self {
        if policy.compact {
            Self::Compact(Box::new(CompactRow { cells: [0; SLOTS], collisions: 0 }))
        } else {
            Self::Dense(vec![0; policy.width].into_boxed_slice())
        }
    }

    #[inline(always)]
    pub(super) fn get(&self, lexer: u32) -> u8 {
        match self {
            Self::Dense(row) => row.get(lexer as usize).copied().unwrap_or(0),
            Self::Compact(row) => {
                let cell = row.cells[lexer as usize & (SLOTS - 1)];
                if cell >> 2 == u64::from(lexer) { (cell & 3) as u8 } else { 0 }
            }
        }
    }

    #[inline(always)]
    pub(super) fn insert(&mut self, lexer: u32, value: u8, policy: &mut PhysicalBoundaryPolicy) {
        debug_assert!(value == 1 || value == 2);
        match self {
            Self::Dense(row) => {
                if let Some(cell) = row.get_mut(lexer as usize) { *cell = value; }
            }
            Self::Compact(row) => {
                let slot = lexer as usize & (SLOTS - 1);
                let previous = row.cells[slot];
                if previous & 3 != 0 && previous >> 2 != u64::from(lexer) {
                    row.collisions = row.collisions.saturating_add(1);
                }
                row.cells[slot] = (u64::from(lexer) << 2) | u64::from(value);
                if row.collisions >= PROMOTE_COLLISIONS
                    && policy.width != 0 && (lexer as usize) < policy.width
                    && policy.width <= MAX_DENSE_BYTES.saturating_sub(policy.promoted_bytes)
                {
                    let mut dense = vec![0; policy.width].into_boxed_slice();
                    for &cell in &row.cells {
                        let retained = (cell & 3) as u8;
                        if retained != 0
                            && let Some(target) = dense.get_mut((cell >> 2) as usize)
                        {
                            *target = retained;
                        }
                    }
                    policy.promoted_bytes += policy.width;
                    *self = Self::Dense(dense);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_tags_and_disjoint_rows_preserve_answers() {
        let mut policy = PhysicalBoundaryPolicy::new(257, true);
        let mut a = PhysicalBoundaryRow::new(&policy);
        let mut b = PhysicalBoundaryRow::new(&policy);
        a.insert(0, 2, &mut policy);
        b.insert(0, 1, &mut policy);
        a.insert(16, 1, &mut policy);
        assert_eq!(a.get(0), 0);
        assert_eq!(a.get(16), 1);
        assert_eq!(b.get(0), 1);
        a.insert(u32::MAX, 2, &mut policy);
        assert_eq!(a.get(u32::MAX), 2);
        assert_eq!(a.get(15), 0);
    }

    #[test]
    fn promotion_retains_observations_and_shares_a_fixed_budget() {
        let mut policy = PhysicalBoundaryPolicy::new(1025, true);
        let mut rows = Vec::new();
        for _ in 0..100 {
            let mut row = PhysicalBoundaryRow::new(&policy);
            for lexer in 0..128 {
                row.insert(lexer, 1 + (lexer % 2) as u8, &mut policy);
            }
            rows.push(row);
        }
        assert_eq!(rows.iter().filter(|r| matches!(r, PhysicalBoundaryRow::Dense(_))).count(),
                   MAX_DENSE_BYTES / 1025);
        assert!(policy.promoted_bytes <= MAX_DENSE_BYTES);
        for row in rows {
            for lexer in 0..1025 {
                let cached = row.get(lexer);
                if cached != 0 { assert_eq!(cached, 1 + (lexer % 2) as u8); }
            }
        }
    }

    #[test]
    fn promotion_requires_collisions_and_a_bounded_domain() {
        for width in [MAX_DENSE_BYTES + 1, usize::MAX] {
            let mut policy = PhysicalBoundaryPolicy::new(width, true);
            let mut row = PhysicalBoundaryRow::new(&policy);
            for lexer in 0..256 { row.insert(lexer, 2, &mut policy); }
            assert!(matches!(row, PhysicalBoundaryRow::Compact(_)));
            assert_eq!(row.get(255), 2);
        }
        let mut policy = PhysicalBoundaryPolicy::new(257, true);
        let mut row = PhysicalBoundaryRow::new(&policy);
        for _ in 0..1000 { row.insert(7, 2, &mut policy); }
        assert!(matches!(row, PhysicalBoundaryRow::Compact(_)));
    }
}
