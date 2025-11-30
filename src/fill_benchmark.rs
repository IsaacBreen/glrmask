//! Benchmark different strategies for fill_internal_bv_to_original_i32.
//!
//! Run with: cargo test --release benchmark_fill_strategies -- --nocapture --ignored

use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

use crate::datastructures::hybrid_bitset::RangeSet;

type LLMTokenBV = RangeSet;

/// Benchmark context - holds precomputed data structures for different strategies.
pub struct FillBenchmark {
    /// Sparse matrix: internal_id -> [(word_idx, u64_word), ...]
    sparse_matrix_u64: Vec<Vec<(u16, u64)>>,

    /// Sparse matrix with u32 words
    sparse_matrix_u32: Vec<Vec<(u32, u32)>>,

    /// Dense matrix: flattened [internal_id * num_words + word_idx] = u64
    dense_matrix_u64: Vec<u64>,

    /// Original->internal mapping for scan-based strategy
    original_to_internal: Vec<u32>,

    /// Stats
    num_internal_tokens: usize,
    max_original_id: usize,
    num_u64_words: usize,
    num_i32_words: usize,
}

impl FillBenchmark {
    /// Build benchmark context from internal_to_original map
    pub fn new(
        internal_to_original: &BTreeMap<usize, LLMTokenBV>,
        max_original_llm_token_id: usize,
        internal_max_llm_token: usize,
    ) -> Self {
        let num_internal_tokens = internal_max_llm_token + 1;
        let num_u64_words = (max_original_llm_token_id / 64) + 1;
        let num_i32_words = (max_original_llm_token_id / 32) + 1;

        // Build sparse matrix u64
        let mut sparse_matrix_u64: Vec<Vec<(u16, u64)>> = vec![Vec::new(); num_internal_tokens];
        for (internal_id, original_bv) in internal_to_original.iter() {
            if *internal_id >= num_internal_tokens {
                continue;
            }
            let mut temp_row = BTreeMap::<u16, u64>::new();
            for original_id in original_bv.inner.iter() {
                if original_id > max_original_llm_token_id {
                    continue;
                }
                let word_idx = (original_id / 64) as u16;
                let bit_idx = original_id % 64;
                *temp_row.entry(word_idx).or_insert(0) |= 1u64 << bit_idx;
            }
            if !temp_row.is_empty() {
                sparse_matrix_u64[*internal_id] = temp_row.into_iter().collect();
            }
        }

        // Build sparse matrix u32
        let mut sparse_matrix_u32: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_internal_tokens];
        for (internal_id, original_bv) in internal_to_original.iter() {
            if *internal_id >= num_internal_tokens {
                continue;
            }
            let mut temp_row = BTreeMap::<u32, u32>::new();
            for original_id in original_bv.inner.iter() {
                if original_id > max_original_llm_token_id {
                    continue;
                }
                let word_idx = (original_id / 32) as u32;
                let bit_idx = original_id % 32;
                *temp_row.entry(word_idx).or_insert(0) |= 1u32 << bit_idx;
            }
            if !temp_row.is_empty() {
                sparse_matrix_u32[*internal_id] = temp_row.into_iter().collect();
            }
        }

        // Build dense matrix
        let mut dense_matrix_u64 = vec![0u64; num_internal_tokens * num_u64_words];
        for (internal_id, original_bv) in internal_to_original.iter() {
            if *internal_id >= num_internal_tokens {
                continue;
            }
            let row_start = *internal_id * num_u64_words;
            for original_id in original_bv.inner.iter() {
                if original_id > max_original_llm_token_id {
                    continue;
                }
                let word_idx = original_id / 64;
                let bit_idx = original_id % 64;
                dense_matrix_u64[row_start + word_idx] |= 1u64 << bit_idx;
            }
        }

        // Build original_to_internal for scan strategy
        let mut original_to_internal = vec![0u32; max_original_llm_token_id + 1];
        for (internal_id, original_bv) in internal_to_original.iter() {
            for original_id in original_bv.inner.iter() {
                if original_id <= max_original_llm_token_id {
                    original_to_internal[original_id] = *internal_id as u32;
                }
            }
        }

        Self {
            sparse_matrix_u64,
            sparse_matrix_u32,
            dense_matrix_u64,
            original_to_internal,
            num_internal_tokens,
            max_original_id: max_original_llm_token_id,
            num_u64_words,
            num_i32_words,
        }
    }

    /// Strategy 1: Current implementation - sparse u64, convert to i32 pairs
    pub fn strategy_sparse_u64_to_i32(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        out.fill(0);
        for internal_id in internal_bv.iter_up_to(self.num_internal_tokens - 1) {
            if let Some(sparse_row) = self.sparse_matrix_u64.get(internal_id) {
                for &(word_idx, word) in sparse_row {
                    let i32_base = (word_idx as usize) * 2;
                    if i32_base < out.len() {
                        out[i32_base] |= word as i32;
                    }
                    if i32_base + 1 < out.len() {
                        out[i32_base + 1] |= (word >> 32) as i32;
                    }
                }
            }
        }
    }

    /// Strategy 2: Sparse u32 - native i32 format, no conversion needed
    pub fn strategy_sparse_u32_native(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        out.fill(0);
        for internal_id in internal_bv.iter_up_to(self.num_internal_tokens - 1) {
            if let Some(sparse_row) = self.sparse_matrix_u32.get(internal_id) {
                for &(word_idx, word) in sparse_row {
                    if (word_idx as usize) < out.len() {
                        out[word_idx as usize] |= word as i32;
                    }
                }
            }
        }
    }

    /// Strategy 3: Dense matrix - direct u64 lookup
    pub fn strategy_dense_u64(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        // First pass to u64, then convert to i32
        let mut result_u64 = vec![0u64; self.num_u64_words];
        for internal_id in internal_bv.iter_up_to(self.num_internal_tokens - 1) {
            let row_start = internal_id * self.num_u64_words;
            let row_end = row_start + self.num_u64_words;
            if row_end <= self.dense_matrix_u64.len() {
                for (i, &word) in self.dense_matrix_u64[row_start..row_end].iter().enumerate() {
                    result_u64[i] |= word;
                }
            }
        }
        // Convert to i32
        out.fill(0);
        for (word_idx, &word) in result_u64.iter().enumerate() {
            let i32_base = word_idx * 2;
            if i32_base < out.len() {
                out[i32_base] = word as i32;
            }
            if i32_base + 1 < out.len() {
                out[i32_base + 1] = (word >> 32) as i32;
            }
        }
    }

    /// Strategy 4: Scan original_to_internal (good when internal_bv is dense)
    pub fn strategy_scan_o2i(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        // Build active set
        let mut active = vec![false; self.num_internal_tokens];
        for i in internal_bv.iter_up_to(self.num_internal_tokens - 1) {
            if i < active.len() {
                active[i] = true;
            }
        }

        out.fill(0);
        for original_id in 0..=self.max_original_id {
            let internal_id = self.original_to_internal[original_id] as usize;
            if active.get(internal_id).copied().unwrap_or(false) {
                let word_idx = original_id / 32;
                let bit_idx = original_id % 32;
                if word_idx < out.len() {
                    out[word_idx] |= 1i32 << bit_idx;
                }
            }
        }
    }

    /// Strategy 5: Parallel scan
    pub fn strategy_parallel_scan(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        // Build active set
        let mut active = vec![false; self.num_internal_tokens];
        for i in internal_bv.iter_up_to(self.num_internal_tokens - 1) {
            if i < active.len() {
                active[i] = true;
            }
        }

        let o2i = &self.original_to_internal;
        let max_orig = self.max_original_id;

        out.par_iter_mut().enumerate().for_each(|(word_idx, word)| {
            let base_id = word_idx * 32;
            let mut current_word = 0i32;
            for bit_idx in 0..32 {
                let original_id = base_id + bit_idx;
                if original_id > max_orig {
                    break;
                }
                let internal_id = o2i[original_id] as usize;
                if active.get(internal_id).copied().unwrap_or(false) {
                    current_word |= 1i32 << bit_idx;
                }
            }
            *word = current_word;
        });
    }

    /// Strategy 6: Sparse u64 with unsafe, no bounds checks
    pub fn strategy_sparse_u64_unsafe(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        out.fill(0);
        for internal_id in internal_bv.iter_up_to(self.num_internal_tokens - 1) {
            if internal_id < self.sparse_matrix_u64.len() {
                let sparse_row = unsafe { self.sparse_matrix_u64.get_unchecked(internal_id) };
                for &(word_idx, word) in sparse_row {
                    let i32_base = (word_idx as usize) * 2;
                    unsafe {
                        if i32_base < out.len() {
                            *out.get_unchecked_mut(i32_base) |= word as i32;
                        }
                        if i32_base + 1 < out.len() {
                            *out.get_unchecked_mut(i32_base + 1) |= (word >> 32) as i32;
                        }
                    }
                }
            }
        }
    }

    /// Strategy 7: Direct transmute to i32 slice (reinterpret u64 as [i32; 2])
    #[cfg(target_endian = "little")]
    pub fn strategy_sparse_u64_transmute(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        out.fill(0);
        for internal_id in internal_bv.iter_up_to(self.num_internal_tokens - 1) {
            if let Some(sparse_row) = self.sparse_matrix_u64.get(internal_id) {
                for &(word_idx, word) in sparse_row {
                    let i32_base = (word_idx as usize) * 2;
                    // On little-endian, u64 bytes are: [lo_i32 bytes][hi_i32 bytes]
                    let i32_pair: [i32; 2] = unsafe { std::mem::transmute(word) };
                    if i32_base < out.len() {
                        out[i32_base] |= i32_pair[0];
                    }
                    if i32_base + 1 < out.len() {
                        out[i32_base + 1] |= i32_pair[1];
                    }
                }
            }
        }
    }

    /// Strategy 8: Collect internal IDs first, then iterate (better cache locality)
    pub fn strategy_collect_then_fill(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        out.fill(0);
        let internal_ids: Vec<usize> = internal_bv
            .iter_up_to(self.num_internal_tokens - 1)
            .collect();
        for &internal_id in &internal_ids {
            if let Some(sparse_row) = self.sparse_matrix_u64.get(internal_id) {
                for &(word_idx, word) in sparse_row {
                    let i32_base = (word_idx as usize) * 2;
                    if i32_base < out.len() {
                        out[i32_base] |= word as i32;
                    }
                    if i32_base + 1 < out.len() {
                        out[i32_base + 1] |= (word >> 32) as i32;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;

    fn create_test_benchmark() -> FillBenchmark {
        // Create a test scenario similar to JS grammar
        let mut internal_to_original: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();

        // Simulate ~200 internal tokens mapping to ~50K original tokens
        let num_internal = 200;
        let max_original = 50256;

        for i in 0..num_internal {
            let mut bv = RangeSet::zeros();
            // Each internal token maps to ~250 original tokens on average
            let base = i * 250;
            for j in 0..250 {
                let orig = (base + j) % max_original;
                bv.insert(orig);
            }
            internal_to_original.insert(i, bv);
        }

        FillBenchmark::new(&internal_to_original, max_original, num_internal - 1)
    }

    #[test]
    #[ignore]
    fn benchmark_fill_strategies() {
        let bench = create_test_benchmark();

        // Create internal bitvector with ~100 active internal tokens
        let mut internal_bv = RangeSet::zeros();
        for i in 0..100 {
            internal_bv.insert(i * 2); // Every other token
        }

        let num_iterations = 10000;
        let out_size = bench.num_i32_words;
        let mut out = vec![0i32; out_size];

        println!("\nBenchmarking fill_internal_bv_to_original_i32 strategies");
        println!("  Internal tokens: {}", bench.num_internal_tokens);
        println!("  Max original ID: {}", bench.max_original_id);
        println!("  Output buffer: {} i32 words", out_size);
        println!("  Active internal tokens in test: {}", internal_bv.len());
        println!("  Iterations: {}\n", num_iterations);

        // Warm up
        for _ in 0..100 {
            bench.strategy_sparse_u64_to_i32(&internal_bv, &mut out);
        }

        // Strategy 1: Current implementation
        let start = Instant::now();
        for _ in 0..num_iterations {
            bench.strategy_sparse_u64_to_i32(&internal_bv, &mut out);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_nanos() as f64 / num_iterations as f64 / 1000.0;
        println!("Strategy 1 (sparse u64 -> i32):   {:.2}µs/iter", per_iter);
        let expected_out = out.clone();

        // Strategy 2: Sparse u32 native
        let start = Instant::now();
        for _ in 0..num_iterations {
            bench.strategy_sparse_u32_native(&internal_bv, &mut out);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_nanos() as f64 / num_iterations as f64 / 1000.0;
        let matches = out == expected_out;
        println!(
            "Strategy 2 (sparse u32 native):   {:.2}µs/iter (match: {})",
            per_iter, matches
        );

        // Strategy 3: Dense u64
        let start = Instant::now();
        for _ in 0..num_iterations {
            bench.strategy_dense_u64(&internal_bv, &mut out);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_nanos() as f64 / num_iterations as f64 / 1000.0;
        let matches = out == expected_out;
        println!(
            "Strategy 3 (dense u64):           {:.2}µs/iter (match: {})",
            per_iter, matches
        );

        // Strategy 4: Scan o2i
        let start = Instant::now();
        for _ in 0..num_iterations {
            bench.strategy_scan_o2i(&internal_bv, &mut out);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_nanos() as f64 / num_iterations as f64 / 1000.0;
        let matches = out == expected_out;
        println!(
            "Strategy 4 (scan o2i):            {:.2}µs/iter (match: {})",
            per_iter, matches
        );

        // Strategy 5: Parallel scan
        let start = Instant::now();
        for _ in 0..num_iterations {
            bench.strategy_parallel_scan(&internal_bv, &mut out);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_nanos() as f64 / num_iterations as f64 / 1000.0;
        let matches = out == expected_out;
        println!(
            "Strategy 5 (parallel scan):       {:.2}µs/iter (match: {})",
            per_iter, matches
        );

        // Strategy 6: Sparse u64 unsafe
        let start = Instant::now();
        for _ in 0..num_iterations {
            bench.strategy_sparse_u64_unsafe(&internal_bv, &mut out);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_nanos() as f64 / num_iterations as f64 / 1000.0;
        let matches = out == expected_out;
        println!(
            "Strategy 6 (sparse u64 unsafe):   {:.2}µs/iter (match: {})",
            per_iter, matches
        );

        // Strategy 7: Transmute
        #[cfg(target_endian = "little")]
        {
            let start = Instant::now();
            for _ in 0..num_iterations {
                bench.strategy_sparse_u64_transmute(&internal_bv, &mut out);
            }
            let elapsed = start.elapsed();
            let per_iter = elapsed.as_nanos() as f64 / num_iterations as f64 / 1000.0;
            let matches = out == expected_out;
            println!(
                "Strategy 7 (transmute):           {:.2}µs/iter (match: {})",
                per_iter, matches
            );
        }

        // Strategy 8: Collect then fill
        let start = Instant::now();
        for _ in 0..num_iterations {
            bench.strategy_collect_then_fill(&internal_bv, &mut out);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_nanos() as f64 / num_iterations as f64 / 1000.0;
        let matches = out == expected_out;
        println!(
            "Strategy 8 (collect then fill):   {:.2}µs/iter (match: {})",
            per_iter, matches
        );
    }
}
