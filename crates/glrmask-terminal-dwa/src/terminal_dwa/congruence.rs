//! Sparse exact deterministic partition refinement shared by L1 and TI.

use std::collections::VecDeque;

/// Refine an exact partition by byte-labelled successor classes. When a
/// synthetic dead state exists, its implicit preimages are complements of
/// all other preimages. Never split on its incomplete explicit edge list.
pub(crate) fn refine_partition<'a>(
    mut classes: Vec<u32>,
    edges_from: impl Fn(usize) -> &'a [(u8, u32)],
    dead: Option<usize>,
) -> (Vec<u32>, Vec<u32>, usize) {
    let n = classes.len();
    let initial_count = classes.iter().copied().max().map_or(0, |c| c as usize + 1);
    let dead_class = dead.map(|s| classes[s] as usize);
    let mut counts = vec![0usize; initial_count];
    for &c in &classes { counts[c as usize] += 1; }
    let mut starts = Vec::with_capacity(initial_count);
    let mut ends = Vec::with_capacity(initial_count);
    let mut offset = 0;
    for count in counts {
        starts.push(offset);
        offset += count;
        ends.push(offset);
    }
    let mut next = starts.clone();
    let mut states = vec![0u32; n];
    let mut positions = vec![0usize; n];
    for (state, &class) in classes.iter().enumerate() {
        let pos = &mut next[class as usize];
        states[*pos] = state as u32;
        positions[state] = *pos;
        *pos += 1;
    }

    let mut reverse_offsets = vec![0usize; n + 1];
    for state in 0..n {
        for &(_, target) in edges_from(state) {
            reverse_offsets[target as usize + 1] += 1;
        }
    }
    for i in 0..n { reverse_offsets[i + 1] += reverse_offsets[i]; }
    let mut next_reverse = reverse_offsets[..n].to_vec();
    let mut reverse = vec![(0u8, 0u32); reverse_offsets[n]];
    for state in 0..n {
        for &(byte, target) in edges_from(state) {
            let slot = &mut next_reverse[target as usize];
            reverse[*slot] = (byte, state as u32);
            *slot += 1;
        }
    }

    let mut queue = VecDeque::new();
    let mut queued = vec![false; initial_count];
    for class in 0..initial_count {
        if Some(class) != dead_class {
            queue.push_back(class);
            queued[class] = true;
        }
    }
    let mut by_byte: [Vec<u32>; 256] = std::array::from_fn(|_| Vec::new());
    let mut touched_bytes = Vec::new();
    let mut marked_count = vec![0usize; initial_count];
    let mut touched = Vec::new();
    let mut pops = 0usize;
    while let Some(splitter) = queue.pop_front() {
        pops += 1;
        queued[splitter] = false;
        for &target in &states[starts[splitter]..ends[splitter]] {
            let target = target as usize;
            for &(byte, source) in &reverse[reverse_offsets[target]..reverse_offsets[target + 1]] {
                let bucket = &mut by_byte[byte as usize];
                if bucket.is_empty() { touched_bytes.push(byte); }
                bucket.push(source);
            }
        }
        for byte in touched_bytes.drain(..) {
            for source in by_byte[byte as usize].drain(..) {
                let source = source as usize;
                let class = classes[source] as usize;
                if marked_count[class] == 0 { touched.push(class); }
                let target_pos = ends[class] - 1 - marked_count[class];
                let source_pos = positions[source];
                let other = states[target_pos] as usize;
                states.swap(source_pos, target_pos);
                positions[source] = target_pos;
                positions[other] = source_pos;
                marked_count[class] += 1;
            }
            for class in touched.drain(..) {
                let marked = std::mem::take(&mut marked_count[class]);
                let start = starts[class];
                let end = ends[class];
                let unmarked = end - start - marked;
                if marked == 0 || unmarked == 0 { continue; }
                let split = end - marked;
                let was_queued = queued[class];
                let new_class = starts.len();
                // Keep synthetic dead in the old block (it cannot have an
                // explicit outgoing edge). Otherwise move the smaller half.
                let (new_start, new_end) = if Some(class) == dead_class || marked <= unmarked {
                    ends[class] = split;
                    (split, end)
                } else {
                    starts[class] = split;
                    (start, split)
                };
                starts.push(new_start);
                ends.push(new_end);
                marked_count.push(0);
                queued.push(true);
                for &state in &states[new_start..new_end] {
                    classes[state as usize] = new_class as u32;
                }
                // New block is the smaller half except when splitting dead;
                // there the complement argument requires the new nondead block.
                // If the old block was queued it remains queued independently.
                debug_assert!(was_queued || Some(class) == dead_class ||
                    new_end - new_start <= ends[class] - starts[class]);
                queue.push_back(new_class);
            }
        }
    }

    let mut canonical = vec![u32::MAX; starts.len()];
    let mut representatives = Vec::new();
    for (state, class) in classes.iter_mut().enumerate() {
        let id = &mut canonical[*class as usize];
        if *id == u32::MAX {
            *id = representatives.len() as u32;
            representatives.push(state as u32);
        }
        *class = *id;
    }
    (classes, representatives, pops)
}

