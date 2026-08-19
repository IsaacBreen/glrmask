use crate::runtime::Constraint;
use crate::runtime::artifact::{
    BackedInternalTokenBufMasks, DenseBufMaskRows, InternalTokenBufMasks,
    PackedInternalTokenBufMask,
};
use crate::automata::regex::Expr;
use crate::ds::weight::Weight;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::io::Read;

const CONSTRAINT_MAGIC: [u8; 8] = *b"GLRCONS\0";
const LEGACY_CONSTRAINT_VERSION: u16 = 7;
const PREVIOUS_COMPRESSED_CONSTRAINT_VERSION: u16 = 9;
const PREVIOUS_EXPRLESS_CONSTRAINT_VERSION: u16 = 10;
const PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION: u16 = 11;
const PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION: u16 = 12;
const PREVIOUS_UNCOMPRESSED_CONSTRAINT_VERSION: u16 = 13;
const PREVIOUS_SECTIONED_CONSTRAINT_VERSION: u16 = 14;
const PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION: u16 = 15;
const PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION: u16 = 16;
const PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION: u16 = 17;
const PREVIOUS_EXTERNAL_RUNTIME_CONSTRAINT_VERSION: u16 = 18;
const CONSTRAINT_VERSION: u16 = 19;
const CONSTRAINT_HEADER_LEN: usize = CONSTRAINT_MAGIC.len() + 2 + 8;
const COMPRESSED_PAYLOAD_HEADER_LEN: usize = 8;
const CONSTRAINT_COMPRESSION_LEVEL: i32 = 1;
const V14_SECTION_MAGIC: [u8; 4] = *b"S14\0";
const V14_SECTION_HEADER_LEN: usize = V14_SECTION_MAGIC.len() + 4 * 8;
const V15_SECTION_MAGIC: [u8; 4] = *b"S15\0";
const V15_SECTION_HEADER_LEN: usize = V15_SECTION_MAGIC.len() + 5 * 8;
const V16_SECTION_MAGIC: [u8; 4] = *b"S16\0";
const V16_SECTION_HEADER_LEN: usize = V16_SECTION_MAGIC.len() + 5 * 8;
const V17_SECTION_MAGIC: [u8; 4] = *b"S17\0";
const V17_SECTION_HEADER_LEN: usize = V17_SECTION_MAGIC.len() + 6 * 8;
const V18_SECTION_MAGIC: [u8; 4] = *b"S18\0";
const V18_SECTION_HEADER_LEN: usize = V18_SECTION_MAGIC.len() + 9 * 8;
const V19_SECTION_MAGIC: [u8; 4] = *b"S19\0";
const V19_SECTION_HEADER_LEN: usize = V19_SECTION_MAGIC.len() + 10 * 8;
const PREVIOUS_CURRENT_CORE_MAGIC: [u8; 4] = *b"C19\0";
const PREVIOUS_CURRENT_CORE_HEADER_LEN: usize = PREVIOUS_CURRENT_CORE_MAGIC.len() + 2 * 8;
const CURRENT_CORE_MAGIC: [u8; 4] = *b"C20\0";
const CURRENT_CORE_HEADER_LEN: usize = CURRENT_CORE_MAGIC.len() + 4 + 2 * 8;
const CURRENT_CORE_FLAG_OMIT_TSID_INVERSE: u32 = 1;

#[inline]
fn uses_external_runtime_sections(version: u16) -> bool {
    version == CONSTRAINT_VERSION || version == PREVIOUS_EXTERNAL_RUNTIME_CONSTRAINT_VERSION
}

#[derive(Serialize)]
struct ConstraintArtifactV10Ref<'a> {
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
}

#[derive(Deserialize)]
struct ConstraintArtifactV10 {
    constraint: Constraint,
    ignore_expr: Option<Expr>,
}

#[derive(Serialize)]
struct ConstraintArtifactV11Ref<'a> {
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
}

#[derive(Deserialize)]
struct ConstraintArtifactV11 {
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
}

#[derive(Serialize)]
struct ConstraintArtifactV12Ref<'a> {
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
    parser_state_domain_labels: &'a [i32],
}

#[derive(Deserialize)]
struct ConstraintArtifactV12 {
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    parser_state_domain_labels: Vec<i32>,
}

#[derive(Serialize)]
struct ConstraintArtifactV13Ref<'a> {
    /// Packed pool for all Weight-bearing Constraint fields outside parser_dwa.
    /// parser_dwa has its own packed pool because its transition topology is
    /// encoded together with the weight references.
    weight_pool: &'a [u8],
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
    parser_state_domain_labels: &'a [i32],
    /// Portable runtime-ready mapping from each internal token to the output
    /// token-mask fragments it represents. Recomputing this from
    /// internal_token_to_tokens is a substantial part of large-constraint load.
    internal_token_buf_masks: &'a [InternalTokenBufMasks],
}

struct PooledWeightDecodeActivator;

impl<'de> Deserialize<'de> for PooledWeightDecodeActivator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        let weights = crate::ds::weight::unpack_pooled_weights(&bytes)
            .map_err(serde::de::Error::custom)?;
        crate::ds::weight::begin_pooled_weight_serde_decode(weights);
        Ok(Self)
    }
}

#[derive(Deserialize)]
struct ConstraintArtifactV13 {
    _weight_pool: PooledWeightDecodeActivator,
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    parser_state_domain_labels: Vec<i32>,
    internal_token_buf_masks: Vec<InternalTokenBufMasks>,
}

#[derive(Serialize)]
struct ConstraintArtifactV14CoreRef<'a> {
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
    parser_state_domain_labels: &'a [i32],
    internal_token_buf_masks: &'a [InternalTokenBufMasks],
}

#[derive(Deserialize)]
struct ConstraintArtifactV14Core {
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    parser_state_domain_labels: Vec<i32>,
    internal_token_buf_masks: Vec<InternalTokenBufMasks>,
}

#[derive(Serialize)]
struct ConstraintArtifactV18CoreRef<'a> {
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
    parser_state_domain_labels: &'a [i32],
}

#[derive(Deserialize)]
struct ConstraintArtifactV18Core {
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    parser_state_domain_labels: Vec<i32>,
}

#[derive(Serialize)]
struct ConstraintArtifactCurrentCoreBaseRef<'a> {
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    parser_state_domain_labels: &'a [i32],
}

#[derive(Deserialize)]
struct ConstraintArtifactCurrentCoreBase {
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    parser_state_domain_labels: Vec<i32>,
}

fn decode_current_core(
    input: &[u8],
) -> Result<(ConstraintArtifactCurrentCoreBase, Option<Vec<u8>>), String> {
    let (header_len, flags, base_len, expr_len) = if input.starts_with(&CURRENT_CORE_MAGIC) {
        if input.len() < CURRENT_CORE_HEADER_LEN {
            return Err("truncated current constraint core header".to_owned());
        }
        let flags = u32::from_le_bytes(
            input[4..8]
                .try_into()
                .expect("current core flags have fixed width"),
        );
        if flags & !CURRENT_CORE_FLAG_OMIT_TSID_INVERSE != 0 {
            return Err("unsupported current constraint core flags".to_owned());
        }
        let base_len = u64::from_le_bytes(
            input[8..16]
                .try_into()
                .expect("current core base length has fixed width"),
        );
        let expr_len = u64::from_le_bytes(
            input[16..24]
                .try_into()
                .expect("current core expression length has fixed width"),
        );
        (CURRENT_CORE_HEADER_LEN, flags, base_len, expr_len)
    } else if input.starts_with(&PREVIOUS_CURRENT_CORE_MAGIC) {
        if input.len() < PREVIOUS_CURRENT_CORE_HEADER_LEN {
            return Err("truncated previous current-core header".to_owned());
        }
        let base_len = u64::from_le_bytes(
            input[4..12]
                .try_into()
                .expect("previous current-core base length has fixed width"),
        );
        let expr_len = u64::from_le_bytes(
            input[12..20]
                .try_into()
                .expect("previous current-core expression length has fixed width"),
        );
        (PREVIOUS_CURRENT_CORE_HEADER_LEN, 0, base_len, expr_len)
    } else {
        return Err("invalid current constraint core header".to_owned());
    };
    let base_len = usize::try_from(base_len)
        .map_err(|_| "current core base length does not fit platform".to_owned())?;
    let expr_len = usize::try_from(expr_len)
        .map_err(|_| "current core expression length does not fit platform".to_owned())?;
    let base_end = header_len
        .checked_add(base_len)
        .ok_or_else(|| "current core base length overflow".to_owned())?;
    let expr_end = base_end
        .checked_add(expr_len)
        .ok_or_else(|| "current core expression length overflow".to_owned())?;
    if expr_end != input.len() {
        return Err("invalid current constraint core section lengths".to_owned());
    }
    let previous_omit_tsid_inverse =
        crate::runtime::artifact::internal_tsid_inverse_artifact_serde::set_omit(
            flags & CURRENT_CORE_FLAG_OMIT_TSID_INVERSE != 0,
        );
    let decoded = bincode::deserialize::<ConstraintArtifactCurrentCoreBase>(
        &input[header_len..base_end],
    );
    crate::runtime::artifact::internal_tsid_inverse_artifact_serde::set_omit(
        previous_omit_tsid_inverse,
    );
    let base = decoded.map_err(|err| err.to_string())?;
    let exprs = (expr_len != 0).then(|| input[base_end..expr_end].to_vec());
    Ok((base, exprs))
}

struct DecodedConstraintCore {
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    terminal_exprs_blob: Option<Vec<u8>>,
    parser_state_domain_labels: Vec<i32>,
    internal_token_buf_masks: Vec<InternalTokenBufMasks>,
}

#[derive(Serialize)]
struct TokenMaskCacheTailRef<'a> {
    guarded_shift_index: &'a [rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >],
    seed_terminal_dense: &'a crate::runtime::artifact::SeedTerminalDenseMasks,
    seed_universe_dense: &'a [u64],
    word_group_sparse_masks: &'a [InternalTokenBufMasks],
    word_group_sparse_prefix_entries: &'a [usize],
    quad_group_sparse_masks: &'a [InternalTokenBufMasks],
    quad_group_dense_masks: &'a [Option<Box<[u32]>>],
    byte_group_sparse_masks: &'a [InternalTokenBufMasks],
    byte_group_dense_masks: &'a [Option<Box<[u32]>>],
    word_group_sparse_total_entries: usize,
    word_group_sparse_max_entries: usize,
    all_tokens_buf_mask: &'a [u32],
    total_internal_buf_cost: usize,
    heavy_token_indices: &'a [usize],
    heavy_total_cost: usize,
    light_avg_cost_x256: usize,
    internal_token_buf_op_costs: &'a [usize],
    word_group_buf_op_costs: &'a [usize],
}

#[derive(Deserialize)]
struct TokenMaskCacheTail {
    guarded_shift_index: Vec<rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >>,
    seed_terminal_dense: crate::runtime::artifact::SeedTerminalDenseMasks,
    seed_universe_dense: Vec<u64>,
    word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    word_group_sparse_prefix_entries: Vec<usize>,
    quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    word_group_sparse_total_entries: usize,
    word_group_sparse_max_entries: usize,
    all_tokens_buf_mask: Box<[u32]>,
    total_internal_buf_cost: usize,
    heavy_token_indices: Vec<usize>,
    heavy_total_cost: usize,
    light_avg_cost_x256: usize,
    internal_token_buf_op_costs: Vec<usize>,
    word_group_buf_op_costs: Vec<usize>,
}

#[derive(Serialize)]
struct TokenMaskCacheIrregularRef<'a> {
    guarded_shift_index: &'a [rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >],
    seed_terminal_dense: &'a crate::runtime::artifact::SeedTerminalDenseMasks,
    seed_universe_dense: &'a [u64],
    quad_group_sparse_masks: &'a [InternalTokenBufMasks],
    quad_group_dense_masks: &'a [Option<Box<[u32]>>],
    byte_group_sparse_masks: &'a [InternalTokenBufMasks],
    byte_group_dense_masks: &'a [Option<Box<[u32]>>],
}

#[derive(Deserialize)]
struct TokenMaskCacheIrregular {
    guarded_shift_index: Vec<rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >>,
    seed_terminal_dense: crate::runtime::artifact::SeedTerminalDenseMasks,
    seed_universe_dense: Vec<u64>,
    quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
}

enum TokenMaskCacheArtifact {
    Full {
        tail: TokenMaskCacheTail,
        word_group_prefix_buf_masks: DenseBufMaskRows,
    },
    Fast {
        irregular: TokenMaskCacheIrregular,
        word_group_sparse_masks: Vec<InternalTokenBufMasks>,
        word_group_prefix_buf_masks: DenseBufMaskRows,
    },
    WordSparse(Vec<InternalTokenBufMasks>),
}

fn encode_word_sparse_token_mask_cache(constraint: &Constraint) -> Vec<u8> {
    const MAGIC: &[u8; 4] = b"TWS1";
    const MAX_BYTES: usize = 512 * 1024;
    let expected_groups = constraint.internal_token_count().div_ceil(64);
    if constraint.word_group_sparse_masks.len() != expected_groups {
        return Vec::new();
    }
    let entry_count = constraint
        .word_group_sparse_masks
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    let encoded_len = 12usize
        .saturating_add((expected_groups + 1).saturating_mul(4))
        .saturating_add(entry_count.saturating_mul(6));
    if encoded_len > MAX_BYTES {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(encoded_len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(expected_groups as u32).to_le_bytes());
    out.extend_from_slice(&(entry_count as u32).to_le_bytes());
    let mut end = 0u32;
    out.extend_from_slice(&end.to_le_bytes());
    for group in &constraint.word_group_sparse_masks {
        end = end.saturating_add(group.len() as u32);
        out.extend_from_slice(&end.to_le_bytes());
    }
    for group in &constraint.word_group_sparse_masks {
        for &(word, bits) in group {
            out.extend_from_slice(&word.to_le_bytes());
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), encoded_len);
    out
}

fn decode_word_sparse_token_mask_cache(input: &[u8]) -> Result<Vec<InternalTokenBufMasks>, String> {
    const HEADER_LEN: usize = 12;
    if input.len() < HEADER_LEN || !input.starts_with(b"TWS1") {
        return Err("invalid sparse word-group cache header".to_owned());
    }
    let group_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let entry_count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
    let offsets_bytes = (group_count + 1)
        .checked_mul(4)
        .ok_or_else(|| "sparse word-group cache offsets overflow".to_owned())?;
    let entries_bytes = entry_count
        .checked_mul(6)
        .ok_or_else(|| "sparse word-group cache entries overflow".to_owned())?;
    let expected = HEADER_LEN
        .checked_add(offsets_bytes)
        .and_then(|n| n.checked_add(entries_bytes))
        .ok_or_else(|| "sparse word-group cache length overflow".to_owned())?;
    if input.len() != expected {
        return Err("invalid sparse word-group cache length".to_owned());
    }
    let offsets_body = &input[HEADER_LEN..HEADER_LEN + offsets_bytes];
    let mut offsets = Vec::<u32>::with_capacity(group_count + 1);
    for bytes in offsets_body.chunks_exact(4) {
        offsets.push(u32::from_le_bytes(bytes.try_into().unwrap()));
    }
    if offsets.first().copied() != Some(0)
        || offsets.last().copied() != Some(entry_count as u32)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err("invalid sparse word-group cache offsets".to_owned());
    }
    let entries = &input[HEADER_LEN + offsets_bytes..];
    let mut groups = Vec::with_capacity(group_count);
    for group in 0..group_count {
        let start = offsets[group] as usize;
        let end = offsets[group + 1] as usize;
        let mut decoded = Vec::with_capacity(end - start);
        for entry in start..end {
            let pos = entry * 6;
            decoded.push((
                u16::from_le_bytes(entries[pos..pos + 2].try_into().unwrap()),
                u32::from_le_bytes(entries[pos + 2..pos + 6].try_into().unwrap()),
            ));
        }
        groups.push(decoded);
    }
    Ok(groups)
}

#[inline]
fn append_cache_u32s(out: &mut Vec<u8>, values: &[u32]) {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
        };
        out.extend_from_slice(bytes);
    } else {
        for &value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn decode_cache_u32_rows(
    input: &[u8],
    pos: &mut usize,
    rows: usize,
    row_len: usize,
) -> Result<DenseBufMaskRows, String> {
    if !DenseBufMaskRows::prefer_flat(rows, row_len) {
        let row_bytes = row_len
            .checked_mul(4)
            .ok_or_else(|| "token-mask prefix row byte length overflow".to_owned())?;
        let mut decoded_rows = Vec::with_capacity(rows);
        for _ in 0..rows {
            let end = pos
                .checked_add(row_bytes)
                .ok_or_else(|| "token-mask prefix row offset overflow".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated token-mask prefix row".to_owned())?;
            let mut row = Vec::<u32>::with_capacity(row_len);
            if cfg!(target_endian = "little") {
                unsafe {
                    row.set_len(row_len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        row.as_mut_ptr().cast::<u8>(),
                        row_bytes,
                    );
                }
            } else {
                row.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())),
                );
            }
            *pos = end;
            decoded_rows.push(row.into_boxed_slice());
        }
        return DenseBufMaskRows::from_rows(decoded_rows);
    }
    let values = rows
        .checked_mul(row_len)
        .ok_or_else(|| "token-mask prefix dimensions overflow".to_owned())?;
    let byte_len = values
        .checked_mul(4)
        .ok_or_else(|| "token-mask prefix byte length overflow".to_owned())?;
    let end = pos
        .checked_add(byte_len)
        .ok_or_else(|| "token-mask prefix offset overflow".to_owned())?;
    let bytes = input
        .get(*pos..end)
        .ok_or_else(|| "truncated token-mask prefix".to_owned())?;
    let mut flat = Vec::<u32>::with_capacity(values);
    if cfg!(target_endian = "little") {
        unsafe {
            flat.set_len(values);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), flat.as_mut_ptr().cast::<u8>(), byte_len);
        }
    } else {
        flat.extend(
            bytes
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())),
        );
    }
    *pos = end;
    DenseBufMaskRows::from_flat(flat.into_boxed_slice(), rows, row_len)
}

fn encode_token_mask_cache(constraint: &Constraint) -> Vec<u8> {
    const MAGIC: &[u8; 4] = b"TMC4";
    const HEADER_LEN: usize = 24;
    const MAX_PREFIX_BYTES: usize = 1024 * 1024;
    const MAX_CACHE_BYTES: usize = 2 * 1024 * 1024;
    if !constraint.token_mask_caches_ready() {
        return Vec::new();
    }
    let mask_words = constraint.mask_len();
    let prefix_rows = constraint.word_group_prefix_buf_masks.len();
    let word_groups = constraint.word_group_sparse_masks.len();
    let word_entries = constraint
        .word_group_sparse_masks
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    let prefix_bytes = prefix_rows
        .saturating_mul(mask_words)
        .saturating_mul(std::mem::size_of::<u32>());
    if prefix_bytes > MAX_PREFIX_BYTES {
        return encode_word_sparse_token_mask_cache(constraint);
    }
    if prefix_rows != word_groups.saturating_add(1)
        || constraint
            .word_group_prefix_buf_masks
            .iter()
            .any(|row| row.len() != mask_words)
    {
        return Vec::new();
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let tail_started = profile.then(std::time::Instant::now);
    let mut tail = Vec::with_capacity(32 * 1024);
    bincode::serialize_into(
        &mut tail,
        &TokenMaskCacheIrregularRef {
            guarded_shift_index: &constraint.table.guarded_shift_index,
            seed_terminal_dense: &constraint.seed_terminal_dense,
            seed_universe_dense: &constraint.seed_universe_dense,
            quad_group_sparse_masks: &constraint.quad_group_sparse_masks,
            quad_group_dense_masks: &constraint.quad_group_dense_masks,
            byte_group_sparse_masks: &constraint.byte_group_sparse_masks,
            byte_group_dense_masks: &constraint.byte_group_dense_masks,
        },
    )
    .expect("token-mask cache serialization should succeed");
    let tail_ms = tail_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let word_offsets_bytes = (word_groups + 1).saturating_mul(4);
    let word_entries_bytes = word_entries.saturating_mul(6);
    let total_len = HEADER_LEN
        .saturating_add(tail.len())
        .saturating_add(word_offsets_bytes)
        .saturating_add(word_entries_bytes)
        .saturating_add(prefix_bytes);
    if total_len > MAX_CACHE_BYTES {
        return encode_word_sparse_token_mask_cache(constraint);
    }
    let prefix_started = profile.then(std::time::Instant::now);
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(MAGIC);
    for value in [tail.len(), mask_words, word_groups, word_entries, prefix_rows] {
        out.extend_from_slice(
            &u32::try_from(value)
                .expect("token-mask cache dimension fits u32")
                .to_le_bytes(),
        );
    }
    out.extend_from_slice(&tail);
    let mut end = 0u32;
    out.extend_from_slice(&end.to_le_bytes());
    for group in &constraint.word_group_sparse_masks {
        end = end.saturating_add(group.len() as u32);
        out.extend_from_slice(&end.to_le_bytes());
    }
    for group in &constraint.word_group_sparse_masks {
        for &(word, bits) in group {
            out.extend_from_slice(&word.to_le_bytes());
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    for row in &constraint.word_group_prefix_buf_masks {
        append_cache_u32s(&mut out, row);
    }
    debug_assert_eq!(out.len(), total_len);
    if let Some(started) = prefix_started {
        eprintln!(
            "[glrmask/profile][token_mask_cache_encode] tail_ms={tail_ms:.3} body_ms={:.3} tail_bytes={} word_sparse_bytes={} prefix_bytes={} total_bytes={}",
            started.elapsed().as_secs_f64() * 1000.0,
            tail.len(),
            word_offsets_bytes + word_entries_bytes,
            prefix_bytes,
            out.len(),
        );
    }
    out
}

fn decode_token_mask_cache(input: &[u8]) -> Result<TokenMaskCacheArtifact, String> {
    const MAGIC: &[u8; 4] = b"TMC3";
    const HEADER_LEN: usize = 16;
    if input.starts_with(b"TWS1") {
        return decode_word_sparse_token_mask_cache(input).map(TokenMaskCacheArtifact::WordSparse);
    }
    if input.starts_with(b"TMC4") {
        const FAST_HEADER_LEN: usize = 24;
        if input.len() < FAST_HEADER_LEN {
            return Err("invalid fast token-mask cache header".to_owned());
        }
        let read = |offset: usize| {
            u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap()) as usize
        };
        let tail_len = read(4);
        let mask_words = read(8);
        let word_groups = read(12);
        let word_entries = read(16);
        let prefix_rows = read(20);
        if prefix_rows != word_groups.saturating_add(1) {
            return Err("fast token-mask prefix row count mismatch".to_owned());
        }
        let tail_end = FAST_HEADER_LEN
            .checked_add(tail_len)
            .ok_or_else(|| "fast token-mask cache tail overflow".to_owned())?;
        let offsets_bytes = (word_groups + 1)
            .checked_mul(4)
            .ok_or_else(|| "fast token-mask sparse offsets overflow".to_owned())?;
        let entries_bytes = word_entries
            .checked_mul(6)
            .ok_or_else(|| "fast token-mask sparse entries overflow".to_owned())?;
        let prefix_bytes = prefix_rows
            .checked_mul(mask_words)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| "fast token-mask prefix bytes overflow".to_owned())?;
        let expected = tail_end
            .checked_add(offsets_bytes)
            .and_then(|n| n.checked_add(entries_bytes))
            .and_then(|n| n.checked_add(prefix_bytes))
            .ok_or_else(|| "fast token-mask cache length overflow".to_owned())?;
        if expected != input.len() {
            return Err("invalid fast token-mask cache length".to_owned());
        }
        let offsets_start = tail_end;
        let entries_start = offsets_start + offsets_bytes;
        let prefix_start = entries_start + entries_bytes;
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let tail_started = profile.then(std::time::Instant::now);
        let irregular = bincode::deserialize::<TokenMaskCacheIrregular>(
            input
                .get(FAST_HEADER_LEN..tail_end)
                .ok_or_else(|| "truncated fast token-mask cache tail".to_owned())?,
        )
        .map_err(|err| err.to_string())?;
        let tail_ms = tail_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let body_started = profile.then(std::time::Instant::now);
        let mut offsets = Vec::<u32>::with_capacity(word_groups + 1);
        for bytes in input[offsets_start..entries_start].chunks_exact(4) {
            offsets.push(u32::from_le_bytes(bytes.try_into().unwrap()));
        }
        if offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(word_entries as u32)
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("invalid fast token-mask sparse offsets".to_owned());
        }
        let entries = &input[entries_start..prefix_start];
        let mut word_group_sparse_masks = Vec::with_capacity(word_groups);
        for group in 0..word_groups {
            let start = offsets[group] as usize;
            let end = offsets[group + 1] as usize;
            let mut decoded = Vec::with_capacity(end - start);
            for entry in start..end {
                let pos = entry * 6;
                let word = u16::from_le_bytes(entries[pos..pos + 2].try_into().unwrap());
                if word as usize >= mask_words {
                    return Err("fast token-mask sparse word out of range".to_owned());
                }
                decoded.push((
                    word,
                    u32::from_le_bytes(entries[pos + 2..pos + 6].try_into().unwrap()),
                ));
            }
            word_group_sparse_masks.push(decoded);
        }
        let mut pos = prefix_start;
        let word_group_prefix_buf_masks =
            decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?;
        debug_assert_eq!(pos, input.len());
        if let Some(started) = body_started {
            eprintln!(
                "[glrmask/profile][token_mask_cache_decode] tail_ms={tail_ms:.3} body_ms={:.3} tail_bytes={} word_sparse_bytes={} prefix_bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                tail_len,
                offsets_bytes + entries_bytes,
                prefix_bytes,
            );
        }
        return Ok(TokenMaskCacheArtifact::Fast {
            irregular,
            word_group_sparse_masks,
            word_group_prefix_buf_masks,
        });
    }
    if input.len() < HEADER_LEN || !input.starts_with(MAGIC) {
        return Err("invalid token-mask cache header".to_owned());
    }
    let read = |offset: usize| {
        u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap()) as usize
    };
    let tail_len = read(4);
    let mask_words = read(8);
    let prefix_rows = read(12);
    let tail_end = HEADER_LEN
        .checked_add(tail_len)
        .ok_or_else(|| "token-mask cache tail overflow".to_owned())?;
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let tail_started = profile.then(std::time::Instant::now);
    let tail = bincode::deserialize::<TokenMaskCacheTail>(
        input
            .get(HEADER_LEN..tail_end)
            .ok_or_else(|| "truncated token-mask cache tail".to_owned())?,
    )
    .map_err(|err| err.to_string())?;
    let tail_ms = tail_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let prefix_started = profile.then(std::time::Instant::now);
    let mut pos = tail_end;
    let word_group_prefix_buf_masks =
        decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?;
    if pos != input.len() {
        return Err("trailing bytes in token-mask cache".to_owned());
    }
    if let Some(started) = prefix_started {
        eprintln!(
            "[glrmask/profile][token_mask_cache_decode] tail_ms={tail_ms:.3} prefix_ms={:.3} tail_bytes={} prefix_bytes={}",
            started.elapsed().as_secs_f64() * 1000.0,
            tail_len,
            input.len().saturating_sub(tail_end),
        );
    }
    Ok(TokenMaskCacheArtifact::Full {
        tail,
        word_group_prefix_buf_masks,
    })
}

fn install_token_mask_cache(
    constraint: &mut Constraint,
    cache: TokenMaskCacheArtifact,
) -> Result<(), String> {
    match cache {
        TokenMaskCacheArtifact::WordSparse(groups) => {
            constraint.word_group_sparse_masks = groups;
            Ok(())
        }
        TokenMaskCacheArtifact::Fast {
            irregular,
            word_group_sparse_masks,
            word_group_prefix_buf_masks,
        } => {
            constraint.table.guarded_shift_index = irregular.guarded_shift_index;
            constraint.seed_terminal_dense = irregular.seed_terminal_dense;
            constraint.seed_universe_dense = irregular.seed_universe_dense.into();
            constraint.word_group_sparse_masks = word_group_sparse_masks;
            constraint.word_group_prefix_buf_masks = word_group_prefix_buf_masks;
            constraint.quad_group_sparse_masks = irregular.quad_group_sparse_masks;
            constraint.quad_group_dense_masks = irregular.quad_group_dense_masks;
            constraint.byte_group_sparse_masks = irregular.byte_group_sparse_masks;
            constraint.byte_group_dense_masks = irregular.byte_group_dense_masks;
            constraint.rebuild_heavy_and_sliding_token_mask_caches();
            constraint.rebuild_token_mask_cache_stats();
            if constraint.token_mask_caches_ready() {
                Ok(())
            } else {
                Err("fast token-mask cache section does not match constraint dimensions".to_owned())
            }
        }
        TokenMaskCacheArtifact::Full {
            tail: cache,
            word_group_prefix_buf_masks,
        } => {
            constraint.table.guarded_shift_index = cache.guarded_shift_index;
            constraint.seed_terminal_dense = cache.seed_terminal_dense;
            constraint.seed_universe_dense = cache.seed_universe_dense.into();
            constraint.word_group_sparse_masks = cache.word_group_sparse_masks;
            constraint.word_group_prefix_buf_masks = word_group_prefix_buf_masks;
            constraint.word_group_sparse_prefix_entries = cache.word_group_sparse_prefix_entries;
            constraint.quad_group_sparse_masks = cache.quad_group_sparse_masks;
            constraint.quad_group_dense_masks = cache.quad_group_dense_masks;
            constraint.byte_group_sparse_masks = cache.byte_group_sparse_masks;
            constraint.byte_group_dense_masks = cache.byte_group_dense_masks;
            constraint.word_group_sparse_total_entries = cache.word_group_sparse_total_entries;
            constraint.word_group_sparse_max_entries = cache.word_group_sparse_max_entries;
            constraint.all_tokens_buf_mask = cache.all_tokens_buf_mask;
            constraint.total_internal_buf_cost = cache.total_internal_buf_cost;
            constraint.heavy_token_indices = cache.heavy_token_indices;
            constraint.heavy_total_cost = cache.heavy_total_cost;
            constraint.light_avg_cost_x256 = cache.light_avg_cost_x256;
            constraint.internal_token_buf_op_costs = cache.internal_token_buf_op_costs;
            constraint.word_group_buf_op_costs = cache.word_group_buf_op_costs;
            constraint.rebuild_heavy_and_sliding_token_mask_caches();
            let rebuilt_heavy_indices = constraint
                .heavy_token_dense_masks
                .iter()
                .enumerate()
                .filter_map(|(index, mask)| mask.is_some().then_some(index))
                .collect::<Vec<_>>();
            if rebuilt_heavy_indices != constraint.heavy_token_indices {
                return Err("token-mask cache heavy-token index mismatch".to_owned());
            }
            if constraint.token_mask_caches_ready() {
                Ok(())
            } else {
                Err("token-mask cache section does not match constraint dimensions".to_owned())
            }
        }
    }
}

fn encode_internal_token_buf_masks(constraint: &Constraint) -> Vec<u8> {
    const MAGIC: &[u8; 4] = b"IBM2";
    const ENTRY_BYTES: usize = std::mem::size_of::<PackedInternalTokenBufMask>();
    let flat_len = constraint.internal_token_buf_flat_len();
    let packed_ready = constraint.internal_token_buf_offsets.len()
        == constraint.internal_token_count().saturating_add(1)
        && constraint
            .internal_token_buf_offsets
            .last()
            .is_some_and(|&end| end as usize == flat_len);
    let group_count = if packed_ready {
        constraint.internal_token_buf_offsets.len().saturating_sub(1)
    } else {
        constraint.internal_token_buf_masks.len()
    };
    let entry_count = if packed_ready {
        flat_len
    } else {
        constraint
            .internal_token_buf_masks
            .iter()
            .map(Vec::len)
            .sum::<usize>()
    };
    let mut out = Vec::with_capacity(
        12usize
            .saturating_add((group_count + 1).saturating_mul(4))
            .saturating_add(entry_count.saturating_mul(ENTRY_BYTES)),
    );
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(group_count as u32).to_le_bytes());
    out.extend_from_slice(&(entry_count as u32).to_le_bytes());
    if packed_ready {
        for &offset in constraint.internal_token_buf_offsets.iter() {
            out.extend_from_slice(&offset.to_le_bytes());
        }
        if let Some(backed) = constraint.backed_internal_token_buf_flat.as_ref() {
            backed.append_wire_bytes(&mut out);
        } else if cfg!(target_endian = "little") {
            let byte_len = entry_count * ENTRY_BYTES;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    constraint.internal_token_buf_flat.as_ptr().cast::<u8>(),
                    byte_len,
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for entry in constraint.internal_token_buf_flat.iter() {
                out.extend_from_slice(&entry.word_idx.to_le_bytes());
                out.extend_from_slice(&0u16.to_le_bytes());
                out.extend_from_slice(&entry.mask.to_le_bytes());
            }
        }
    } else {
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        for mask in &constraint.internal_token_buf_masks {
            end = end.saturating_add(mask.len() as u32);
            out.extend_from_slice(&end.to_le_bytes());
        }
        for mask in &constraint.internal_token_buf_masks {
            for &(word, bits) in mask {
                out.extend_from_slice(&word.to_le_bytes());
                out.extend_from_slice(&0u16.to_le_bytes());
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    out
}

struct DecodedInternalTokenBufMasks {
    flat: Box<[PackedInternalTokenBufMask]>,
    backed: Option<BackedInternalTokenBufMasks>,
    offsets: Box<[u32]>,
}

enum DecodedOriginalTokenMap {
    Materialized(Vec<u32>),
    Packed(std::sync::Arc<
        crate::runtime::artifact::original_token_map_artifact_serde::PackedOriginalTokenMap,
    >),
}

fn decode_internal_token_buf_masks(
    input: &[u8],
    backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
) -> Result<DecodedInternalTokenBufMasks, String> {
    const LEGACY_MAGIC: &[u8; 4] = b"IBM1";
    const FIXED_MAGIC: &[u8; 4] = b"IBM2";
    let fixed = input.starts_with(FIXED_MAGIC);
    if input.len() < 12 || (!fixed && !input.starts_with(LEGACY_MAGIC)) {
        return Err("invalid internal-token buffer-mask section".to_owned());
    }
    let group_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let entry_count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
    let offsets_bytes = (group_count + 1)
        .checked_mul(4)
        .ok_or_else(|| "internal-token buffer-mask offsets overflow".to_owned())?;
    let entry_width = if fixed {
        std::mem::size_of::<PackedInternalTokenBufMask>()
    } else {
        6
    };
    let entries_bytes = entry_count
        .checked_mul(entry_width)
        .ok_or_else(|| "internal-token buffer-mask entries overflow".to_owned())?;
    let expected = 12usize
        .checked_add(offsets_bytes)
        .and_then(|n| n.checked_add(entries_bytes))
        .ok_or_else(|| "internal-token buffer-mask section length overflow".to_owned())?;
    if expected != input.len() {
        return Err("invalid internal-token buffer-mask section length".to_owned());
    }
    let offsets_body = &input[12..12 + offsets_bytes];
    let mut offsets = Vec::<u32>::with_capacity(group_count + 1);
    if cfg!(target_endian = "little") {
        unsafe {
            offsets.set_len(group_count + 1);
            std::ptr::copy_nonoverlapping(
                offsets_body.as_ptr(),
                offsets.as_mut_ptr().cast::<u8>(),
                offsets_bytes,
            );
        }
    } else {
        offsets.extend(
            offsets_body
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())),
        );
    }
    if offsets.first().copied() != Some(0)
        || offsets.last().copied().map(|end| end as usize) != Some(entry_count)
    {
        return Err("invalid internal-token buffer-mask offsets".to_owned());
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("non-monotonic internal-token buffer-mask offsets".to_owned());
    }
    let entries_start = 12 + offsets_bytes;
    let entries = &input[entries_start..];
    let backed = if fixed {
        backing
            .map(|(backing, section_start)| {
                BackedInternalTokenBufMasks::new(
                    backing,
                    section_start + entries_start,
                    entry_count,
                )
            })
            .transpose()?
    } else {
        None
    };
    let mut flat = Vec::<PackedInternalTokenBufMask>::with_capacity(if backed.is_some() {
        0
    } else {
        entry_count
    });
    if backed.is_some() {
        // The retained artifact is the runtime storage; offsets remain owned
        // because they are tiny and hot to index.
    } else if fixed && cfg!(target_endian = "little") {
        unsafe {
            flat.set_len(entry_count);
            std::ptr::copy_nonoverlapping(
                entries.as_ptr(),
                flat.as_mut_ptr().cast::<u8>(),
                entries_bytes,
            );
        }
    } else {
        // The section length was validated above, so every record is present.
        // IBM1 has six-byte records; IBM2 is eight bytes with a two-byte pad.
        unsafe {
            flat.set_len(entry_count);
            let src = entries.as_ptr();
            let dst = flat.as_mut_ptr();
            for entry in 0..entry_count {
                let pos = entry * entry_width;
                let word = u16::from_le(std::ptr::read_unaligned(src.add(pos).cast::<u16>()));
                let bits_offset = if fixed { 4 } else { 2 };
                let bits = u32::from_le(std::ptr::read_unaligned(
                    src.add(pos + bits_offset).cast::<u32>(),
                ));
                std::ptr::write(
                    dst.add(entry),
                    PackedInternalTokenBufMask {
                        word_idx: word,
                        _pad: 0,
                        mask: bits,
                    },
                );
            }
        }
    }
    Ok(DecodedInternalTokenBufMasks {
        flat: flat.into_boxed_slice(),
        backed,
        offsets: offsets.into_boxed_slice(),
    })
}

#[derive(Serialize)]
struct ConstraintArtifactV15RuntimeRef<'a> {
    terminal_live_states: &'a [Vec<u32>],
}

#[derive(Deserialize)]
struct ConstraintArtifactV15Runtime {
    terminal_live_states: Vec<Vec<u32>>,
}

enum DecodedParserDwa {
    Materialized(
        crate::automata::weighted::dwa::DWA,
        Option<crate::automata::weighted::dwa::PackedDwaTokenSetInventory>,
    ),
    Packed(std::sync::Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>),
}

fn v14_sections(payload: &[u8]) -> Result<(&[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V14_SECTION_HEADER_LEN || !payload.starts_with(&V14_SECTION_MAGIC) {
        return Err("invalid v14 constraint section header".to_owned());
    }
    let mut pos = V14_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v14 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value).map_err(|_| "v14 section length does not fit this platform".to_owned())
    };
    let weight_len = take_len()?;
    let dwa_len = take_len()?;
    let table_len = take_len()?;
    let core_len = take_len()?;
    let total = V14_SECTION_HEADER_LEN
        .checked_add(weight_len)
        .and_then(|value| value.checked_add(dwa_len))
        .and_then(|value| value.checked_add(table_len))
        .and_then(|value| value.checked_add(core_len))
        .ok_or_else(|| "overflowing v14 section lengths".to_owned())?;
    if total != payload.len() {
        return Err("invalid v14 constraint section lengths".to_owned());
    }
    let weight_start = V14_SECTION_HEADER_LEN;
    let dwa_start = weight_start + weight_len;
    let table_start = dwa_start + dwa_len;
    let core_start = table_start + table_len;
    Ok((
        &payload[weight_start..dwa_start],
        &payload[dwa_start..table_start],
        &payload[table_start..core_start],
        &payload[core_start..],
    ))
}

fn v15_sections(
    payload: &[u8],
) -> Result<(&[u8], &[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V15_SECTION_HEADER_LEN || !payload.starts_with(&V15_SECTION_MAGIC) {
        return Err("invalid v15 constraint section header".to_owned());
    }
    let mut pos = V15_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v15 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v15 section length does not fit this platform".to_owned())
    };
    let weight_len = take_len()?;
    let dwa_len = take_len()?;
    let table_len = take_len()?;
    let core_len = take_len()?;
    let runtime_len = take_len()?;
    let total = V15_SECTION_HEADER_LEN
        .checked_add(weight_len)
        .and_then(|value| value.checked_add(dwa_len))
        .and_then(|value| value.checked_add(table_len))
        .and_then(|value| value.checked_add(core_len))
        .and_then(|value| value.checked_add(runtime_len))
        .ok_or_else(|| "v15 constraint section lengths overflow".to_owned())?;
    if total != payload.len() {
        return Err("invalid v15 constraint section lengths".to_owned());
    }
    let mut pos = V15_SECTION_HEADER_LEN;
    let weight = &payload[pos..pos + weight_len];
    pos += weight_len;
    let dwa = &payload[pos..pos + dwa_len];
    pos += dwa_len;
    let table = &payload[pos..pos + table_len];
    pos += table_len;
    let core = &payload[pos..pos + core_len];
    pos += core_len;
    let runtime = &payload[pos..pos + runtime_len];
    Ok((weight, dwa, table, core, runtime))
}

fn v16_sections(
    payload: &[u8],
) -> Result<(&[u8], &[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V16_SECTION_HEADER_LEN || !payload.starts_with(&V16_SECTION_MAGIC) {
        return Err("invalid v16 constraint section header".to_owned());
    }
    let mut pos = V16_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v16 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v16 section length does not fit this platform".to_owned())
    };
    let weight_len = take_len()?;
    let dwa_len = take_len()?;
    let table_len = take_len()?;
    let core_len = take_len()?;
    let runtime_len = take_len()?;
    let total = V16_SECTION_HEADER_LEN
        .checked_add(weight_len)
        .and_then(|value| value.checked_add(dwa_len))
        .and_then(|value| value.checked_add(table_len))
        .and_then(|value| value.checked_add(core_len))
        .and_then(|value| value.checked_add(runtime_len))
        .ok_or_else(|| "v16 constraint section lengths overflow".to_owned())?;
    if total != payload.len() {
        return Err("invalid v16 constraint section lengths".to_owned());
    }
    let mut pos = V16_SECTION_HEADER_LEN;
    let weight = &payload[pos..pos + weight_len];
    pos += weight_len;
    let dwa = &payload[pos..pos + dwa_len];
    pos += dwa_len;
    let table = &payload[pos..pos + table_len];
    pos += table_len;
    let core = &payload[pos..pos + core_len];
    pos += core_len;
    let runtime = &payload[pos..pos + runtime_len];
    Ok((weight, dwa, table, core, runtime))
}

fn v17_sections(
    payload: &[u8],
) -> Result<(&[u8], &[u8], &[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V17_SECTION_HEADER_LEN || !payload.starts_with(&V17_SECTION_MAGIC) {
        return Err("invalid v17 constraint section header".to_owned());
    }
    let mut pos = V17_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v17 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v17 section length does not fit this platform".to_owned())
    };
    let weight_len = take_len()?;
    let dwa_len = take_len()?;
    let table_len = take_len()?;
    let core_len = take_len()?;
    let runtime_len = take_len()?;
    let token_bytes_len = take_len()?;
    let total = V17_SECTION_HEADER_LEN
        .checked_add(weight_len)
        .and_then(|value| value.checked_add(dwa_len))
        .and_then(|value| value.checked_add(table_len))
        .and_then(|value| value.checked_add(core_len))
        .and_then(|value| value.checked_add(runtime_len))
        .and_then(|value| value.checked_add(token_bytes_len))
        .ok_or_else(|| "v17 constraint section lengths overflow".to_owned())?;
    if total != payload.len() {
        return Err("invalid v17 constraint section lengths".to_owned());
    }
    let mut pos = V17_SECTION_HEADER_LEN;
    let weight = &payload[pos..pos + weight_len];
    pos += weight_len;
    let dwa = &payload[pos..pos + dwa_len];
    pos += dwa_len;
    let table = &payload[pos..pos + table_len];
    pos += table_len;
    let core = &payload[pos..pos + core_len];
    pos += core_len;
    let runtime = &payload[pos..pos + runtime_len];
    pos += runtime_len;
    let token_bytes = &payload[pos..pos + token_bytes_len];
    Ok((weight, dwa, table, core, runtime, token_bytes))
}

fn v18_sections(
    payload: &[u8],
) -> Result<(&[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V18_SECTION_HEADER_LEN || !payload.starts_with(&V18_SECTION_MAGIC) {
        return Err("invalid v18 constraint section header".to_owned());
    }
    let mut pos = V18_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v18 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v18 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V18_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v18 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v18 constraint section lengths".to_owned());
    }
    let mut pos = V18_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
    ))
}

fn v19_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V19_SECTION_HEADER_LEN || !payload.starts_with(&V19_SECTION_MAGIC) {
        return Err("invalid v19 constraint section header".to_owned());
    }
    let mut pos = V19_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v19 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v19 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V19_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v19 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v19 constraint section lengths".to_owned());
    }
    let mut pos = V19_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
    ))
}

fn constraint_serialized_weight_pool(constraint: &Constraint) -> Vec<Weight> {
    let mut seen = HashSet::new();
    let mut weights = Vec::new();
    let mut push = |weight: &Weight| {
        if seen.insert(weight.ptr_key()) {
            weights.push(weight.clone());
        }
    };

    for weight in constraint.parser_top_accept.values() {
        push(weight);
    }
    for parts in constraint.parser_top_accept_parts.values() {
        for weight in parts {
            push(weight);
        }
    }
    for weight in constraint.direct_regular_l1_complete_by_terminal.values() {
        push(weight);
    }
    for weight in constraint.possible_matches.values() {
        push(weight);
    }
    weights
}

fn attach_packed_non_dwa_weights(
    constraint: &mut Constraint,
    pool: std::sync::Arc<crate::ds::weight::PackedRuntimeWeightPool>,
    ids: Vec<u32>,
) -> Result<(), String> {
    let mut ids = ids.into_iter();
    let mut parser_top_accept = BTreeMap::new();
    for &label in constraint.parser_top_accept.keys() {
        let id = ids
            .next()
            .ok_or_else(|| "missing packed Weight id for parser_top_accept".to_owned())?;
        parser_top_accept.insert(label, id);
    }

    let mut parser_top_accept_parts = BTreeMap::new();
    for (&label, parts) in &constraint.parser_top_accept_parts {
        let mut part_ids = Vec::with_capacity(parts.len());
        for _ in parts {
            part_ids.push(
                ids.next().ok_or_else(|| {
                    "missing packed Weight id for parser_top_accept_parts".to_owned()
                })?,
            );
        }
        parser_top_accept_parts.insert(label, part_ids);
    }

    let mut direct_regular_l1_complete_by_terminal = BTreeMap::new();
    for &terminal in constraint.direct_regular_l1_complete_by_terminal.keys() {
        let id = ids.next().ok_or_else(|| {
            "missing packed Weight id for direct_regular_l1_complete_by_terminal".to_owned()
        })?;
        direct_regular_l1_complete_by_terminal.insert(terminal, id);
    }

    let mut possible_matches = BTreeMap::new();
    for &terminal in constraint.possible_matches.keys() {
        let id = ids
            .next()
            .ok_or_else(|| "missing packed Weight id for possible_matches".to_owned())?;
        possible_matches.insert(terminal, id);
    }
    if ids.next().is_some() {
        return Err("unused packed Weight ids after current constraint core decode".to_owned());
    }

    constraint.packed_non_dwa_weights = Some(std::sync::Arc::new(
        crate::runtime::artifact::PackedNonDwaWeights {
            pool,
            parser_top_accept,
            parser_top_accept_parts,
            direct_regular_l1_complete_by_terminal,
            possible_matches,
        },
    ));
    Ok(())
}

fn invert_original_token_map(
    original_to_internal: &[u32],
    expected_group_count: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let group_count = if expected_group_count != 0 {
        expected_group_count
    } else {
        let Some(max_internal) = original_to_internal
            .iter()
            .copied()
            .filter(|&internal| internal != u32::MAX)
            .max()
        else {
            return Ok(Vec::new());
        };
        usize::try_from(max_internal)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "internal token vocabulary is too large".to_owned())?
    };
    let mut counts = vec![0usize; group_count];
    for &internal in original_to_internal {
        if internal == u32::MAX {
            continue;
        }
        let Some(count) = counts.get_mut(internal as usize) else {
            return Err("original-token map contains an out-of-range internal token".to_owned());
        };
        *count += 1;
    }
    let mut groups = counts
        .into_iter()
        .map(Vec::<u32>::with_capacity)
        .collect::<Vec<_>>();
    for (original, &internal) in original_to_internal.iter().enumerate() {
        if internal == u32::MAX {
            continue;
        }
        let original = u32::try_from(original)
            .map_err(|_| "original token id exceeds u32".to_owned())?;
        groups[internal as usize].push(original);
    }
    Ok(groups)
}

fn envelope(version: u16, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(CONSTRAINT_HEADER_LEN + payload.len());
    bytes.extend_from_slice(&CONSTRAINT_MAGIC);
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

impl Constraint {
    /// Serialize this compiled constraint to a versioned binary artifact.
    ///
    /// v13 intentionally stores the compact wire representation uncompressed.
    /// The wire format itself must be small and fast enough that loading does
    /// not depend on a decompression pass to hide structural redundancy.
    pub fn save(&self) -> Vec<u8> {
        if let Some(bytes) = &self.serialized_artifact_cache {
            return bytes.as_ref().clone();
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        if std::env::var_os("GLRMASK_PROFILE_CACHE_ARTIFACT").is_some() {
            let started = std::time::Instant::now();
            let cache_sizes = [
                ("word_sparse", bincode::serialized_size(&self.word_group_sparse_masks).unwrap_or(0)),
                ("word_prefix", bincode::serialized_size(&self.word_group_prefix_buf_masks).unwrap_or(0)),
                ("word_sparse_prefix", bincode::serialized_size(&self.word_group_sparse_prefix_entries).unwrap_or(0)),
                ("pair", bincode::serialized_size(&self.pair_word_group_buf_masks).unwrap_or(0)),
                ("quad", bincode::serialized_size(&self.quad_word_group_buf_masks).unwrap_or(0)),
                ("super", bincode::serialized_size(&self.super_word_group_buf_masks).unwrap_or(0)),
                ("mega", bincode::serialized_size(&self.mega_word_group_buf_masks).unwrap_or(0)),
                ("giga", bincode::serialized_size(&self.giga_word_group_buf_masks).unwrap_or(0)),
                ("all", bincode::serialized_size(&self.all_tokens_buf_mask).unwrap_or(0)),
                ("heavy", bincode::serialized_size(&self.heavy_token_dense_masks).unwrap_or(0)),
                ("flat", bincode::serialized_size(&self.internal_token_buf_flat).unwrap_or(0)),
                ("offsets", bincode::serialized_size(&self.internal_token_buf_offsets).unwrap_or(0)),
                ("op_costs", bincode::serialized_size(&self.internal_token_buf_op_costs).unwrap_or(0)),
                ("word_costs", bincode::serialized_size(&self.word_group_buf_op_costs).unwrap_or(0)),
            ];
            eprintln!(
                "[glrmask/profile][runtime_mask_cache_sizes] {}",
                cache_sizes
                    .iter()
                    .map(|(name, bytes)| format!("{name}={bytes}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            let cache_a = bincode::serialize(&(
                &self.word_group_sparse_masks,
                &self.word_group_prefix_buf_masks,
                &self.word_group_sparse_prefix_entries,
                &self.pair_word_group_buf_masks,
                &self.quad_word_group_buf_masks,
                &self.super_word_group_buf_masks,
                &self.mega_word_group_buf_masks,
                &self.giga_word_group_buf_masks,
                &self.all_tokens_buf_mask,
            ))
            .expect("runtime mask cache profiling serialization should succeed");
            let cache_b = bincode::serialize(&(
                &self.heavy_token_dense_masks,
                &self.internal_token_buf_flat,
                &self.internal_token_buf_offsets,
                self.total_internal_buf_cost,
                &self.heavy_token_indices,
                self.heavy_total_cost,
                self.light_avg_cost_x256,
                &self.internal_token_buf_op_costs,
                &self.word_group_buf_op_costs,
            ))
            .expect("runtime mask cache profiling serialization should succeed");
            eprintln!(
                "[glrmask/profile][runtime_mask_cache_candidate] ms={:.3} bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                cache_a.len() + cache_b.len(),
            );
        }
        let total_started = profile.then(std::time::Instant::now);
        const PARALLEL_ASSEMBLY_MAX_PACKED_DWA_BYTES: usize = 1024 * 1024;
        let packed_dwa_wire_len = self
            .packed_parser_dwa
            .as_ref()
            .and_then(|packed| packed.fast_wire_len());
        let parallel_assembly_candidate = self.packed_parser_dwa.is_none()
            || packed_dwa_wire_len
                .is_some_and(|len| len <= PARALLEL_ASSEMBLY_MAX_PACKED_DWA_BYTES);
        const DIRECT_TOKENIZER_MIN_BYTES: usize = 400 * 1024;
        let direct_tokenizer_len = parallel_assembly_candidate
            .then(|| crate::automata::lexer::tokenizer::artifact_serde::fast_len(&self.tokenizer))
            .flatten()
            .filter(|&len| len >= DIRECT_TOKENIZER_MIN_BYTES);
        let ((token_bytes, (original_token_map, (tokenizer, internal_token_buf_masks))), ((weight_pool, core), (dwa, table, runtime, token_mask_cache))) = rayon::join(
            || rayon::join(
                || {
                    let started = profile.then(std::time::Instant::now);
                    let bytes = crate::runtime::artifact::shared_packed_token_bytes(&self.token_bytes);
                    if let Some(started) = started {
                        eprintln!(
                            "[glrmask/profile][constraint_save_section] name=token_bytes ms={:.3} bytes={}",
                            started.elapsed().as_secs_f64() * 1000.0,
                            bytes.len(),
                        );
                    }
                    bytes
                },
                || rayon::join(
                    || {
                        let started = profile.then(std::time::Instant::now);
                        let bytes = crate::runtime::artifact::original_token_map_artifact_serde::to_fast_bytes(
                            self.original_token_map(),
                        );
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=original_token_map ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                bytes.len(),
                            );
                        }
                        bytes
                    },
                    || rayon::join(
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = if direct_tokenizer_len.is_some() {
                                Vec::new()
                            } else {
                                crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes(
                                    &self.tokenizer,
                                )
                            };
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=tokenizer ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    direct_tokenizer_len.unwrap_or(bytes.len()),
                                );
                            }
                            bytes
                        },
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = encode_internal_token_buf_masks(self);
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=internal_token_buf_masks ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            bytes
                        },
                    ),
                ),
            ),
            || rayon::join(
            || {
                let branch_started = profile.then(std::time::Instant::now);
                let weights = constraint_serialized_weight_pool(self);
                let (weight_pool, encoded) = rayon::join(
                    || {
                        let started = profile.then(std::time::Instant::now);
                        let weight_pool = crate::ds::weight::pack_pooled_weights(&weights);
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=weights ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                weight_pool.len(),
                            );
                        }
                        weight_pool
                    },
                    || {
                        // The pooled Weight-id map is thread-local, so install
                        // it inside the core branch rather than on the parent
                        // Rayon worker.  Packing WPL3 only reads the same
                        // immutable Weight slice and is independent once ids
                        // have been defined by stable slice order.
                        crate::ds::weight::begin_pooled_weight_serde_encode(&weights);
                        let previous_external =
                            crate::automata::weighted::dwa::set_external_serde(true);
                        let previous_external_table =
                            crate::compiler::glr::table::artifact_serde::set_external_serde(true);
                        let previous_compact_tokenizer =
                            crate::automata::lexer::tokenizer::set_compact_artifact_serde(true);
                        let previous_external_tokenizer =
                            crate::automata::lexer::tokenizer::set_external_artifact_serde(true);
                        let previous_omit_inverse =
                            crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(true);
                        let previous_packed_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_packed(true);
                        let previous_external_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_external(true);
                        let previous_packed_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_packed(true);
                        let previous_external_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_external(true);
                        let started = profile.then(std::time::Instant::now);
                        // `bincode::serialize` first runs `serialized_size` and
                        // then serializes again. Custom compact serializers
                        // (especially the tokenizer) do real packing work in
                        // both passes. Write once into a generously-sized Vec
                        // instead; capacity growth is much cheaper than
                        // rebuilding every packed field twice.
                        let mut encoded = Vec::with_capacity(3 * 1024 * 1024);
                        encoded.extend_from_slice(&CURRENT_CORE_MAGIC);
                        let omit_tsid_inverse = self.can_defer_internal_tsid_inverse();
                        let core_flags = if omit_tsid_inverse {
                            CURRENT_CORE_FLAG_OMIT_TSID_INVERSE
                        } else {
                            0
                        };
                        encoded.extend_from_slice(&core_flags.to_le_bytes());
                        let base_len_offset = encoded.len();
                        encoded.extend_from_slice(&0u64.to_le_bytes());
                        let expr_len_offset = encoded.len();
                        encoded.extend_from_slice(&0u64.to_le_bytes());
                        let base_start = encoded.len();
                        let previous_omit_tsid_inverse =
                            crate::runtime::artifact::internal_tsid_inverse_artifact_serde::set_omit(
                                omit_tsid_inverse,
                            );
                        let encode_result = bincode::serialize_into(
                            &mut encoded,
                            &ConstraintArtifactCurrentCoreBaseRef {
                                constraint: self,
                                ignore_expr: &self.ignore_expr,
                                parser_state_domain_labels: &self.parser_state_domain_labels,
                            },
                        );
                        crate::runtime::artifact::internal_tsid_inverse_artifact_serde::set_omit(
                            previous_omit_tsid_inverse,
                        );
                        let base_len = encoded.len() - base_start;
                        encoded[base_len_offset..base_len_offset + 8]
                            .copy_from_slice(&(base_len as u64).to_le_bytes());
                        let expr_start = encoded.len();
                        if let Some(exprs) = self.tokenizer.terminal_exprs() {
                            bincode::serialize_into(&mut encoded, exprs)
                                .expect("terminal expression serialization should succeed");
                        } else if let Some(blob) = self.deferred_terminal_exprs_blob.as_deref() {
                            encoded.extend_from_slice(blob);
                        }
                        let expr_len = encoded.len() - expr_start;
                        encoded[expr_len_offset..expr_len_offset + 8]
                            .copy_from_slice(&(expr_len as u64).to_le_bytes());
                        crate::automata::lexer::tokenizer::set_compact_artifact_serde(
                            previous_compact_tokenizer,
                        );
                        crate::automata::lexer::tokenizer::set_external_artifact_serde(
                            previous_external_tokenizer,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_packed(
                            previous_packed_token_bytes,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_external(
                            previous_external_token_bytes,
                        );
                        crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(
                            previous_omit_inverse,
                        );
                        crate::runtime::artifact::original_token_map_artifact_serde::set_packed(
                            previous_packed_original_token_map,
                        );
                        crate::runtime::artifact::original_token_map_artifact_serde::set_external(
                            previous_external_original_token_map,
                        );
                        crate::compiler::glr::table::artifact_serde::set_external_serde(
                            previous_external_table,
                        );
                        crate::automata::weighted::dwa::set_external_serde(previous_external);
                        crate::ds::weight::end_pooled_weight_serde_encode();
                        encode_result.expect("Constraint core serialization should succeed");
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=core ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                encoded.len(),
                            );
                        }
                        encoded
                    },
                );
                if let Some(started) = branch_started {
                    eprintln!(
                        "[glrmask/profile][constraint_save_section] name=weights_core_branch ms={:.3}",
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                (
                    weight_pool,
                    encoded,
                )
            },
            || {
                let ((dwa, table), (runtime, token_mask_cache)) = rayon::join(
                    || {
                        rayon::join(
                    || {
                        // A fresh packed DWA is emitted directly into the final
                        // constraint artifact below. Avoid constructing an
                        // 18+ MB temporary section only to copy it once more.
                        let Some(_packed) = self.packed_parser_dwa.as_ref() else {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = self.parser_dwa.artifact_packed_bytes();
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=dwa ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            return bytes;
                        };
                        Vec::new()
                    },
                    || {
                        let started = profile.then(std::time::Instant::now);
                        let bytes = crate::compiler::glr::table::artifact_serde::to_compact_bytes(&self.table);
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=table ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                bytes.len(),
                            );
                        }
                        bytes
                    },
                        )
                    },
                    || rayon::join(
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = bincode::serialize(&ConstraintArtifactV15RuntimeRef {
                                terminal_live_states: &self.terminal_live_states,
                            })
                            .expect("Constraint runtime metadata serialization should succeed");
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=runtime ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            bytes
                        },
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = encode_token_mask_cache(self);
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=token_mask_cache ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            bytes
                        },
                    ),
                );
                (dwa, table, runtime, token_mask_cache)
            },
            ),
        );

        let dwa_wire_len = self
            .packed_parser_dwa
            .as_ref()
            .and_then(|packed| packed.fast_wire_len())
            .unwrap_or(dwa.len());
        // For ordinary constraints, materializing the small packed-DWA section
        // is much cheaper than serially copying the entire multi-megabyte
        // artifact after every independent serializer has finished.  Once all
        // sections are ordinary byte slices, copy them into disjoint final
        // ranges in parallel. Large JS-like DWAs keep direct emission below to
        // avoid creating a second 10+ MiB DWA buffer.
        let packed_dwa_for_parallel = self
            .packed_parser_dwa
            .as_ref()
            .filter(|_| dwa_wire_len <= PARALLEL_ASSEMBLY_MAX_PACKED_DWA_BYTES)
            .map(|packed| packed.fast_wire_bytes());
        let parallel_dwa = packed_dwa_for_parallel
            .as_deref()
            .or_else(|| self.packed_parser_dwa.is_none().then_some(dwa.as_slice()));
        let tokenizer_wire_len = direct_tokenizer_len.unwrap_or(tokenizer.len());
        let assemble_started = profile.then(std::time::Instant::now);
        let payload_len = V19_SECTION_HEADER_LEN
            + weight_pool.len()
            + dwa_wire_len
            + table.len()
            + core.len()
            + runtime.len()
            + token_bytes.len()
            + original_token_map.len()
            + tokenizer_wire_len
            + internal_token_buf_masks.len()
            + token_mask_cache.len();
        if let Some(dwa_bytes) = parallel_dwa {
            debug_assert_eq!(dwa_bytes.len(), dwa_wire_len);
            let total_len = CONSTRAINT_HEADER_LEN + payload_len;
            let mut bytes = Vec::<u8>::with_capacity(total_len);
            // SAFETY: every byte in the allocation is initialized below before
            // the Vec is observed or returned. The section destinations are
            // disjoint slices split from this one allocation and are each
            // written exactly once.
            unsafe {
                bytes.set_len(total_len);
            }
            let header_len = CONSTRAINT_HEADER_LEN + V19_SECTION_HEADER_LEN;
            let (header, mut body) = bytes.split_at_mut(header_len);
            let mut pos = 0usize;
            header[pos..pos + CONSTRAINT_MAGIC.len()].copy_from_slice(&CONSTRAINT_MAGIC);
            pos += CONSTRAINT_MAGIC.len();
            header[pos..pos + 2].copy_from_slice(&CONSTRAINT_VERSION.to_le_bytes());
            pos += 2;
            header[pos..pos + 8].copy_from_slice(&(payload_len as u64).to_le_bytes());
            pos += 8;
            header[pos..pos + V19_SECTION_MAGIC.len()].copy_from_slice(&V19_SECTION_MAGIC);
            pos += V19_SECTION_MAGIC.len();
            for len in [
                weight_pool.len(),
                dwa_bytes.len(),
                table.len(),
                core.len(),
                runtime.len(),
                token_bytes.len(),
                original_token_map.len(),
                tokenizer_wire_len,
                internal_token_buf_masks.len(),
                token_mask_cache.len(),
            ] {
                header[pos..pos + 8].copy_from_slice(&(len as u64).to_le_bytes());
                pos += 8;
            }
            debug_assert_eq!(pos, header.len());

            let mut copy_jobs = Vec::<(&mut [u8], &[u8])>::with_capacity(10);
            let mut direct_tokenizer_destination = None;
            let sources: [&[u8]; 10] = [
                weight_pool.as_slice(),
                dwa_bytes,
                table.as_slice(),
                core.as_slice(),
                runtime.as_slice(),
                token_bytes.as_slice(),
                original_token_map.as_slice(),
                tokenizer.as_slice(),
                internal_token_buf_masks.as_slice(),
                token_mask_cache.as_slice(),
            ];
            for (index, len) in [
                weight_pool.len(),
                dwa_bytes.len(),
                table.len(),
                core.len(),
                runtime.len(),
                token_bytes.len(),
                original_token_map.len(),
                tokenizer_wire_len,
                internal_token_buf_masks.len(),
                token_mask_cache.len(),
            ]
            .into_iter()
            .enumerate()
            {
                let (section, rest) = body.split_at_mut(len);
                if index == 7 && direct_tokenizer_len.is_some() {
                    direct_tokenizer_destination = Some(section);
                } else {
                    copy_jobs.push((section, sources[index]));
                }
                body = rest;
            }
            debug_assert!(body.is_empty());
            if rayon::current_num_threads() > 1 {
                rayon::scope(|scope| {
                    for (destination, source) in copy_jobs {
                        scope.spawn(move |_| destination.copy_from_slice(source));
                    }
                    if let Some(destination) = direct_tokenizer_destination {
                        scope.spawn(|_| {
                            let started = profile.then(std::time::Instant::now);
                            crate::automata::lexer::tokenizer::artifact_serde::write_fast_bytes(
                                &self.tokenizer,
                                destination,
                            )
                            .expect("precomputed fast tokenizer layout should match final section");
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=tokenizer_direct ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    destination.len(),
                                );
                            }
                        });
                    }
                });
            } else {
                for (destination, source) in copy_jobs {
                    destination.copy_from_slice(source);
                }
                if let Some(destination) = direct_tokenizer_destination {
                    crate::automata::lexer::tokenizer::artifact_serde::write_fast_bytes(
                        &self.tokenizer,
                        destination,
                    )
                    .expect("precomputed fast tokenizer layout should match final section");
                }
            }
            if let Some(started) = assemble_started {
                eprintln!(
                    "[glrmask/profile][constraint_save_assemble] ms={:.3} mode=parallel",
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            if let Some(started) = total_started {
                eprintln!(
                    "[glrmask/profile][constraint_save] total_ms={:.3} bytes={}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    bytes.len(),
                );
            }
            return bytes;
        }
        let mut bytes = Vec::with_capacity(CONSTRAINT_HEADER_LEN + payload_len);
        bytes.extend_from_slice(&CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&CONSTRAINT_VERSION.to_le_bytes());
        let payload_len_offset = bytes.len();
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&V19_SECTION_MAGIC);
        bytes.extend_from_slice(&(weight_pool.len() as u64).to_le_bytes());
        let dwa_len_offset = bytes.len();
        bytes.extend_from_slice(&(dwa.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(table.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(core.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(runtime.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(token_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(original_token_map.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(tokenizer_wire_len as u64).to_le_bytes());
        bytes.extend_from_slice(&(internal_token_buf_masks.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(token_mask_cache.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&weight_pool);
        let dwa_start = bytes.len();
        if let Some(packed) = &self.packed_parser_dwa {
            debug_assert!(dwa.is_empty());
            let started = profile.then(std::time::Instant::now);
            packed.append_fast_wire_bytes(&mut bytes);
            if let Some(started) = started {
                eprintln!(
                    "[glrmask/profile][constraint_save_section] name=dwa_direct ms={:.3} bytes={}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    bytes.len() - dwa_start,
                );
            }
        } else {
            bytes.extend_from_slice(&dwa);
        }
        let dwa_len = bytes.len() - dwa_start;
        bytes[dwa_len_offset..dwa_len_offset + 8]
            .copy_from_slice(&(dwa_len as u64).to_le_bytes());
        bytes.extend_from_slice(&table);
        bytes.extend_from_slice(&core);
        bytes.extend_from_slice(&runtime);
        bytes.extend_from_slice(token_bytes.as_slice());
        bytes.extend_from_slice(&original_token_map);
        if direct_tokenizer_len.is_some() {
            unreachable!("direct tokenizer encoding requires the parallel assembly path");
        } else {
            bytes.extend_from_slice(&tokenizer);
        }
        bytes.extend_from_slice(&internal_token_buf_masks);
        bytes.extend_from_slice(&token_mask_cache);
        let payload_len = bytes.len() - CONSTRAINT_HEADER_LEN;
        bytes[payload_len_offset..payload_len_offset + 8]
            .copy_from_slice(&(payload_len as u64).to_le_bytes());
        if let Some(started) = assemble_started {
            eprintln!(
                "[glrmask/profile][constraint_save_assemble] ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][constraint_save] total_ms={:.3} bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                bytes.len(),
            );
        }
        bytes
    }

    /// Load a compiled constraint from an artifact produced by [`Constraint::save`].
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        Self::load_impl(bytes, None)
    }

    /// Load a compiled constraint while taking ownership of the artifact bytes.
    ///
    /// This avoids the whole-artifact copy otherwise required to retain exact
    /// current-format bytes for a later unchanged [`Constraint::save`].
    pub fn load_owned(bytes: Vec<u8>) -> crate::Result<Self> {
        let backing = std::sync::Arc::new(bytes);
        Self::load_impl(backing.as_slice(), Some(std::sync::Arc::clone(&backing)))
    }

    fn load_impl(
        bytes: &[u8],
        owned_artifact: Option<std::sync::Arc<Vec<u8>>>,
    ) -> crate::Result<Self> {
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let mut decompress_ms = 0.0;
        if bytes.len() < CONSTRAINT_HEADER_LEN || !bytes.starts_with(&CONSTRAINT_MAGIC) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid constraint artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if !matches!(
            version,
            LEGACY_CONSTRAINT_VERSION
                | PREVIOUS_COMPRESSED_CONSTRAINT_VERSION
                | PREVIOUS_EXPRLESS_CONSTRAINT_VERSION
                | PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION
                | PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION
                | PREVIOUS_UNCOMPRESSED_CONSTRAINT_VERSION
                | PREVIOUS_SECTIONED_CONSTRAINT_VERSION
                | PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
                | PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                | PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
                | PREVIOUS_EXTERNAL_RUNTIME_CONSTRAINT_VERSION
                | CONSTRAINT_VERSION
        ) {
            let decompress_started = profile.then(std::time::Instant::now);
            return Err(crate::GlrMaskError::Serialization(format!(
                "unsupported constraint artifact version {version}"
            )));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[10..18]
                .try_into()
                .expect("constraint artifact header has fixed width"),
        ))
        .map_err(|_| {
            crate::GlrMaskError::Serialization(
                "constraint artifact payload length does not fit this platform".to_owned(),
            )
        })?;
        if bytes.len() != CONSTRAINT_HEADER_LEN.saturating_add(payload_len) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid constraint artifact payload length".to_owned(),
            ));
        }
        // v17 runtime sections may retain zero-copy views into the artifact.
        // If the caller did not transfer ownership, make the one compatibility
        // copy up front so every retained view and the unchanged-resave cache
        // share the same backing allocation.
        let current_backing = if uses_external_runtime_sections(version)
            || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
        {
            Some(
                owned_artifact
                    .clone()
                    .unwrap_or_else(|| std::sync::Arc::new(bytes.to_vec())),
            )
        } else {
            None
        };
        let section_bytes = current_backing
            .as_ref()
            .map_or(bytes, |backing| backing.as_slice());
        let payload = &section_bytes[CONSTRAINT_HEADER_LEN..];
        let mut raw;
        let serialized = if matches!(
            version,
            PREVIOUS_COMPRESSED_CONSTRAINT_VERSION
                | PREVIOUS_EXPRLESS_CONSTRAINT_VERSION
                | PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION
                | PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION
        ) {
            let decompress_started = profile.then(std::time::Instant::now);
            if payload.len() < COMPRESSED_PAYLOAD_HEADER_LEN {
                return Err(crate::GlrMaskError::Serialization(
                    "invalid compressed constraint artifact payload".to_owned(),
                ));
            }
            let raw_len = usize::try_from(u64::from_le_bytes(
                payload[..COMPRESSED_PAYLOAD_HEADER_LEN]
                    .try_into()
                    .expect("compressed constraint payload header has fixed width"),
            ))
            .map_err(|_| {
                crate::GlrMaskError::Serialization(
                    "uncompressed constraint artifact length does not fit this platform".to_owned(),
                )
            })?;
            let compressed = &payload[COMPRESSED_PAYLOAD_HEADER_LEN..];
            let frame_len = zstd::zstd_safe::get_frame_content_size(compressed)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            if frame_len.is_some_and(|frame_len| frame_len != raw_len as u64) {
                return Err(crate::GlrMaskError::Serialization(
                    "invalid uncompressed constraint artifact length".to_owned(),
                ));
            }

            // Do not reserve the untrusted declared size up front. Stream into
            // a growing buffer and stop after one byte beyond the declared
            // length, so malformed artifacts cannot trigger an immediate huge
            // allocation merely by forging the envelope.
            let output_limit = raw_len.checked_add(1).ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "uncompressed constraint artifact length is too large".to_owned(),
                )
            })?;
            let decoder = zstd::stream::read::Decoder::with_buffer(compressed)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            raw = Vec::new();
            decoder
                .take(output_limit as u64)
                .read_to_end(&mut raw)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            if raw.len() != raw_len {
                return Err(crate::GlrMaskError::Serialization(
                    "invalid uncompressed constraint artifact length".to_owned(),
                ));
            }
            decompress_ms = decompress_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            raw.as_slice()
        } else {
            payload
        };
        let deserialize_started = profile.then(std::time::Instant::now);
        let mut packed_dwa_inventory = None;
        let mut constraint = if uses_external_runtime_sections(version)
            || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
            || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
            || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
            || version == PREVIOUS_SECTIONED_CONSTRAINT_VERSION
        {
            let (
                weight_section,
                dwa_section,
                table_section,
                core_section,
                runtime_section,
                token_bytes_section,
                original_token_map_section,
                tokenizer_section,
                internal_token_buf_masks_section,
                token_mask_cache_section,
            ) =
                if version == CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache) = v19_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                    )
                } else if version == PREVIOUS_EXTERNAL_RUNTIME_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks) = v18_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        None,
                    )
                } else if version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes) = v17_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, Some(runtime), Some(token_bytes), None, None, None, None)
                } else if version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime) = v16_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, Some(runtime), None, None, None, None, None)
                } else if version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime) = v15_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, Some(runtime), None, None, None, None, None)
                } else {
                    let (weight, dwa, table, core) = v14_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, None, None, None, None, None, None)
                };
            let (((dwa_result, (table_result, runtime_result)), ((tokenizer_result, original_token_map_result), (internal_token_buf_masks_result, token_mask_cache_result))), core_result) = rayon::join(
                || rayon::join(
                    || rayon::join(
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let result = if uses_external_runtime_sections(version)
                                || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
                                || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                                || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
                            {
                                let decoded = if dwa_section.starts_with(b"DWF1")
                                    || dwa_section.starts_with(b"DWF2")
                                {
                                    crate::automata::weighted::dwa::PackedRuntimeDwa::from_fast_wire_bytes(
                                        dwa_section,
                                    )
                                } else {
                                    crate::automata::weighted::dwa::PackedRuntimeDwa::from_packed_bytes(
                                        dwa_section,
                                    )
                                };
                                decoded.map(|dwa| {
                                    DecodedParserDwa::Packed(std::sync::Arc::new(dwa))
                                })
                            } else {
                                crate::automata::weighted::dwa::DWA::from_artifact_packed_bytes(
                                    dwa_section,
                                )
                                .map(|(dwa, inventory)| {
                                    DecodedParserDwa::Materialized(dwa, inventory)
                                })
                            };
                            if let Some(started) = started {
                                eprintln!("[glrmask/profile][constraint_section] name=dwa ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                            }
                            result
                        },
                        || {
                            rayon::join(
                                || {
                                    let started = profile.then(std::time::Instant::now);
                                    let result = crate::compiler::glr::table::artifact_serde::from_compact_bytes(table_section);
                                    if let Some(started) = started {
                                        eprintln!("[glrmask/profile][constraint_section] name=table ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                                    }
                                    result
                                },
                                || -> Result<Option<ConstraintArtifactV15Runtime>, String> {
                                    let Some(runtime_section) = runtime_section else {
                                        return Ok(None);
                                    };
                                    let started = profile.then(std::time::Instant::now);
                                    let result = bincode::deserialize::<ConstraintArtifactV15Runtime>(
                                        runtime_section,
                                    )
                                    .map(Some)
                                    .map_err(|err| err.to_string());
                                    if let Some(started) = started {
                                        eprintln!("[glrmask/profile][constraint_section] name=runtime ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                                    }
                                    result
                                },
                            )
                        },
                    ),
                    || rayon::join(
                        || rayon::join(
                            || -> Result<Option<crate::automata::lexer::tokenizer::Tokenizer>, String> {
                            let Some(section) = tokenizer_section else {
                                return Ok(None);
                            };
                            let started = profile.then(std::time::Instant::now);
                            let result = if uses_external_runtime_sections(version) {
                                let backing = current_backing.as_ref().ok_or_else(|| {
                                    "current tokenizer has no artifact backing".to_owned()
                                })?;
                                let base = backing.as_ptr() as usize;
                                let section_start = (section.as_ptr() as usize)
                                    .checked_sub(base)
                                    .ok_or_else(|| {
                                        "tokenizer section does not belong to artifact backing"
                                            .to_owned()
                                    })?;
                                crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes_backed(
                                    section,
                                    std::sync::Arc::clone(backing),
                                    section_start,
                                )
                                .map(Some)
                            } else {
                                crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes(
                                    section,
                                )
                                .map(Some)
                            };
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_section] name=tokenizer ms={:.3}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                            result
                            },
                            || -> Result<Option<DecodedOriginalTokenMap>, String> {
                            let Some(section) = original_token_map_section else {
                                return Ok(None);
                            };
                            let started = profile.then(std::time::Instant::now);
                            let result = if uses_external_runtime_sections(version) {
                                let backing = current_backing.as_ref().ok_or_else(|| {
                                    "current original-token map has no artifact backing".to_owned()
                                })?;
                                let base = backing.as_ptr() as usize;
                                let section_start = (section.as_ptr() as usize)
                                    .checked_sub(base)
                                    .ok_or_else(|| {
                                        "original-token map section does not belong to artifact backing"
                                            .to_owned()
                                    })?;
                                crate::runtime::artifact::original_token_map_artifact_serde::PackedOriginalTokenMap::parse_backed(
                                    std::sync::Arc::clone(backing),
                                    section_start,
                                    section.len(),
                                )
                                .map(|packed| {
                                    Some(DecodedOriginalTokenMap::Packed(std::sync::Arc::new(
                                        packed,
                                    )))
                                })
                            } else {
                                crate::runtime::artifact::original_token_map_artifact_serde::from_fast_bytes(section)
                                    .map(|map| Some(DecodedOriginalTokenMap::Materialized(map)))
                            };
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_section] name=original_token_map ms={:.3}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                            result
                            },
                        ),
                        || rayon::join(
                            || -> Result<Option<DecodedInternalTokenBufMasks>, String> {
                                let Some(section) = internal_token_buf_masks_section else {
                                    return Ok(None);
                                };
                                let started = profile.then(std::time::Instant::now);
                                let backing = current_backing
                                    .as_ref()
                                    .map(|backing| {
                                        let base = backing.as_ptr() as usize;
                                        let section_start = (section.as_ptr() as usize)
                                            .checked_sub(base)
                                            .ok_or_else(|| {
                                                "internal-token buffer-mask section does not belong to artifact backing"
                                                    .to_owned()
                                            })?;
                                        Ok::<_, String>((
                                            std::sync::Arc::clone(backing),
                                            section_start,
                                        ))
                                    })
                                    .transpose()?;
                                let result = decode_internal_token_buf_masks(section, backing).map(Some);
                                if let Some(started) = started {
                                    eprintln!(
                                        "[glrmask/profile][constraint_section] name=internal_token_buf_masks ms={:.3}",
                                        started.elapsed().as_secs_f64() * 1000.0,
                                    );
                                }
                                result
                            },
                            || -> Result<Option<TokenMaskCacheArtifact>, String> {
                                let Some(section) = token_mask_cache_section else {
                                    return Ok(None);
                                };
                                if section.is_empty() {
                                    return Ok(None);
                                }
                                let started = profile.then(std::time::Instant::now);
                                let result = decode_token_mask_cache(section).map(Some);
                                if let Some(started) = started {
                                    eprintln!(
                                        "[glrmask/profile][constraint_section] name=token_mask_cache ms={:.3}",
                                        started.elapsed().as_secs_f64() * 1000.0,
                                    );
                                }
                                result
                            },
                        ),
                    ),
                ),
                || -> Result<
                    (
                        DecodedConstraintCore,
                        Option<std::sync::Arc<crate::runtime::artifact::token_bytes_artifact_serde::PackedTokenBytes>>,
                        Option<(
                            std::sync::Arc<crate::ds::weight::PackedRuntimeWeightPool>,
                            Vec<u32>,
                        )>,
                    ),
                    String,
                > {
                    let section_started = profile.then(std::time::Instant::now);
                    let weight_count = if uses_external_runtime_sections(version)
                        || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
                        || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                        || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
                    {
                        Some(crate::ds::weight::PackedRuntimeWeightPool::peek_weight_count(
                            weight_section,
                        )?)
                    } else {
                        None
                    };

                    let decode_core = || -> Result<_, String> {
                        if let Some(weight_count) = weight_count {
                            crate::ds::weight::begin_pooled_weight_serde_deferred_decode(
                                weight_count,
                            );
                        } else {
                            let weights_started = profile.then(std::time::Instant::now);
                            let weights = crate::ds::weight::unpack_pooled_weights(weight_section)?;
                            if let Some(started) = weights_started {
                                eprintln!("[glrmask/profile][constraint_section] name=weights ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                            }
                            crate::ds::weight::begin_pooled_weight_serde_decode(weights);
                        }
                        let previous_external =
                            crate::automata::weighted::dwa::set_external_serde(true);
                        let previous_external_table =
                            crate::compiler::glr::table::artifact_serde::set_external_serde(true);
                        let previous_compact_tokenizer =
                            crate::automata::lexer::tokenizer::set_compact_artifact_serde(true);
                        let previous_external_tokenizer =
                            crate::automata::lexer::tokenizer::set_external_artifact_serde(
                                uses_external_runtime_sections(version),
                            );
                        let previous_omit_inverse =
                            crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(
                                true,
                            );
                        let previous_packed_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_packed(
                                true,
                            );
                        let previous_external_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_external(
                                uses_external_runtime_sections(version),
                            );
                        let previous_packed_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_packed(true);
                        let previous_external_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_external(
                                uses_external_runtime_sections(version)
                                    || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION,
                            );
                        let previous_defer_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_defer_unpack(
                                version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                                    || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION,
                            );
                        let core_started = profile.then(std::time::Instant::now);
                        let decoded = if uses_external_runtime_sections(version) {
                            if core_section.starts_with(&CURRENT_CORE_MAGIC)
                                || core_section.starts_with(&PREVIOUS_CURRENT_CORE_MAGIC)
                            {
                                decode_current_core(core_section)
                                .map(|(artifact, terminal_exprs_blob)| DecodedConstraintCore {
                                    constraint: artifact.constraint,
                                    ignore_expr: artifact.ignore_expr,
                                    terminal_exprs: None,
                                    terminal_exprs_blob,
                                    parser_state_domain_labels: artifact.parser_state_domain_labels,
                                    internal_token_buf_masks: Vec::new(),
                                })
                            } else {
                                bincode::deserialize::<ConstraintArtifactV18Core>(core_section)
                                    .map(|artifact| DecodedConstraintCore {
                                        constraint: artifact.constraint,
                                        ignore_expr: artifact.ignore_expr,
                                        terminal_exprs: artifact.terminal_exprs,
                                        terminal_exprs_blob: None,
                                        parser_state_domain_labels: artifact.parser_state_domain_labels,
                                        internal_token_buf_masks: Vec::new(),
                                    })
                                    .map_err(|err| err.to_string())
                            }
                        } else {
                            bincode::deserialize::<ConstraintArtifactV14Core>(core_section)
                                .map(|artifact| DecodedConstraintCore {
                                    constraint: artifact.constraint,
                                    ignore_expr: artifact.ignore_expr,
                                    terminal_exprs: artifact.terminal_exprs,
                                    terminal_exprs_blob: None,
                                    parser_state_domain_labels: artifact.parser_state_domain_labels,
                                    internal_token_buf_masks: artifact.internal_token_buf_masks,
                                })
                                .map_err(|err| err.to_string())
                        };
                        let deferred_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::take_deferred();
                        let deferred_weight_ids = if weight_count.is_some() {
                            crate::ds::weight::take_pooled_weight_serde_deferred_ids()
                        } else {
                            Vec::new()
                        };
                        if let Some(started) = core_started {
                            eprintln!("[glrmask/profile][constraint_section] name=core_bincode ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                        }
                        crate::automata::lexer::tokenizer::set_compact_artifact_serde(
                            previous_compact_tokenizer,
                        );
                        crate::automata::lexer::tokenizer::set_external_artifact_serde(
                            previous_external_tokenizer,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_packed(
                            previous_packed_token_bytes,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_external(
                            previous_external_token_bytes,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_defer_unpack(
                            previous_defer_token_bytes,
                        );
                        crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(
                            previous_omit_inverse,
                        );
                        crate::runtime::artifact::original_token_map_artifact_serde::set_packed(
                            previous_packed_original_token_map,
                        );
                        crate::runtime::artifact::original_token_map_artifact_serde::set_external(
                            previous_external_original_token_map,
                        );
                        crate::compiler::glr::table::artifact_serde::set_external_serde(
                            previous_external_table,
                        );
                        crate::automata::weighted::dwa::set_external_serde(previous_external);
                        crate::ds::weight::end_pooled_weight_serde_decode();
                        decoded.map(|artifact| {
                            (artifact, deferred_token_bytes, deferred_weight_ids)
                        })
                    };

                    let (artifact, deferred_token_bytes, packed_weights) =
                        if uses_external_runtime_sections(version)
                            || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
                            || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                            || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
                        {
                            let weights_started = profile.then(std::time::Instant::now);
                            // WPL3 current-format runtime indexing is now only
                            // a small linear framing scan + one section copy.
                            // Running that tiny job as another nested Rayon
                            // branch competes with the much heavier core/DWA
                            // decoders and increases wall time on Windows.
                            let packed_weights =
                                crate::ds::weight::PackedRuntimeWeightPool::from_packed_bytes(
                                    weight_section,
                                )?;
                            if let Some(started) = weights_started {
                                eprintln!("[glrmask/profile][constraint_section] name=weights_packed ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                            }
                            let (artifact, deferred_token_bytes, deferred_weight_ids) = decode_core()?;
                            (
                                artifact,
                                deferred_token_bytes,
                                Some((std::sync::Arc::new(packed_weights), deferred_weight_ids)),
                            )
                        } else {
                            let (artifact, deferred_token_bytes, _) = decode_core()?;
                            (artifact, deferred_token_bytes, None)
                        };
                    if let Some(started) = section_started {
                        eprintln!("[glrmask/profile][constraint_section] name=core_total ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                    }
                    Ok((artifact, deferred_token_bytes, packed_weights))
                },
            );
            let parser_dwa = dwa_result.map_err(crate::GlrMaskError::Serialization)?;
            let table = table_result.map_err(crate::GlrMaskError::Serialization)?;
            let runtime = runtime_result.map_err(crate::GlrMaskError::Serialization)?;
            let tokenizer = tokenizer_result.map_err(crate::GlrMaskError::Serialization)?;
            let original_token_map =
                original_token_map_result.map_err(crate::GlrMaskError::Serialization)?;
            let external_internal_token_buf_masks =
                internal_token_buf_masks_result.map_err(crate::GlrMaskError::Serialization)?;
            let token_mask_cache =
                token_mask_cache_result.map_err(crate::GlrMaskError::Serialization)?;
            let (artifact, deferred_token_bytes, packed_weights) =
                core_result.map_err(crate::GlrMaskError::Serialization)?;
            let token_bytes_started = profile.then(std::time::Instant::now);
            let external_token_bytes = if let Some(token_bytes_section) = token_bytes_section {
                let backing = current_backing
                    .as_ref()
                    .expect("current-format token section has artifact backing");
                let start = (token_bytes_section.as_ptr() as usize)
                    .checked_sub(section_bytes.as_ptr() as usize)
                    .ok_or_else(|| {
                        crate::GlrMaskError::Serialization(
                            "token-byte section does not belong to artifact backing".to_owned(),
                        )
                    })?;
                Some(std::sync::Arc::new(
                    crate::runtime::artifact::token_bytes_artifact_serde::PackedTokenBytes::parse_backed(
                        std::sync::Arc::clone(backing),
                        start,
                        token_bytes_section.len(),
                    )
                    .map_err(crate::GlrMaskError::Serialization)?,
                ))
            } else {
                None
            };
            let token_bytes_ms = token_bytes_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let mut constraint = artifact.constraint;
            if let Some(tokenizer) = tokenizer {
                constraint.tokenizer = tokenizer;
            }
            if let Some(original_token_map) = original_token_map {
                match original_token_map {
                    DecodedOriginalTokenMap::Materialized(map) => {
                        constraint.original_token_to_internal = map;
                        constraint.packed_original_token_to_internal = None;
                    }
                    DecodedOriginalTokenMap::Packed(map) => {
                        constraint.original_token_to_internal = Vec::new();
                        constraint.packed_original_token_to_internal = Some(map);
                    }
                }
            }
            let attach_weights_started = profile.then(std::time::Instant::now);
            if let Some((pool, ids)) = packed_weights {
                attach_packed_non_dwa_weights(&mut constraint, pool, ids)
                    .map_err(crate::GlrMaskError::Serialization)?;
            }
            let attach_weights_ms = attach_weights_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let attach_dwa_started = profile.then(std::time::Instant::now);
            match parser_dwa {
                DecodedParserDwa::Materialized(parser_dwa, inventory) => {
                    constraint.parser_dwa = parser_dwa;
                    constraint.packed_parser_dwa = None;
                    packed_dwa_inventory = inventory;
                }
                DecodedParserDwa::Packed(parser_dwa) => {
                    constraint.parser_dwa = crate::automata::weighted::dwa::DWA::new(0, 0);
                    constraint.packed_parser_dwa = Some(parser_dwa);
                }
            }
            let attach_dwa_ms = attach_dwa_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            constraint.table = table;
            let invert_started = profile.then(std::time::Instant::now);
            if !uses_external_runtime_sections(version)
                && version != PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
            {
                constraint.internal_token_to_tokens = invert_original_token_map(
                    &constraint.original_token_to_internal,
                    artifact.internal_token_buf_masks.len(),
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            }
            let invert_ms = invert_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            constraint.ignore_expr = artifact.ignore_expr;
            constraint.parser_state_domain_labels = artifact.parser_state_domain_labels;
            constraint.deferred_terminal_exprs_blob = artifact
                .terminal_exprs_blob
                .map(|blob| std::sync::Arc::<[u8]>::from(blob.into_boxed_slice()));
            constraint.deferred_terminal_exprs = Default::default();
            if let Some(decoded) = external_internal_token_buf_masks {
                constraint.internal_token_buf_masks = Vec::new();
                constraint.internal_token_buf_flat = decoded.flat;
                constraint.backed_internal_token_buf_flat = decoded.backed;
                constraint.internal_token_buf_offsets = decoded.offsets;
            } else {
                constraint.internal_token_buf_masks = artifact.internal_token_buf_masks;
                constraint.backed_internal_token_buf_flat = None;
            }
            constraint.packed_token_bytes = external_token_bytes.or(deferred_token_bytes);
            if let Some(cache) = token_mask_cache {
                install_token_mask_cache(&mut constraint, cache)
                    .map_err(crate::GlrMaskError::Serialization)?;
            }
            if let Some(runtime) = runtime {
                constraint.terminal_live_states = runtime.terminal_live_states;
            }
            let restore_exprs_started = profile.then(std::time::Instant::now);
            constraint
                .tokenizer
                .restore_terminal_exprs(artifact.terminal_exprs)
                .map_err(crate::GlrMaskError::Serialization)?;
            let restore_exprs_ms = restore_exprs_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            if profile {
                eprintln!(
                    "[glrmask/profile][constraint_post_decode] token_bytes_ms={token_bytes_ms:.3} attach_weights_ms={attach_weights_ms:.3} attach_dwa_ms={attach_dwa_ms:.3} invert_original_map_ms={invert_ms:.3} restore_exprs_ms={restore_exprs_ms:.3}"
                );
            }
            constraint
        } else if version == PREVIOUS_UNCOMPRESSED_CONSTRAINT_VERSION {
            let previous_dwa_mode = crate::automata::weighted::dwa::set_packed_serde(true);
            let decoded = bincode::deserialize::<ConstraintArtifactV13>(serialized);
            crate::automata::weighted::dwa::set_packed_serde(previous_dwa_mode);
            // The pool activator is the first v13 field and therefore installs
            // the Weight reference table before Constraint is deserialized.
            // Always clear it, including after a malformed later field.
            crate::ds::weight::end_pooled_weight_serde_decode();
            let artifact = decoded
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            let mut constraint = artifact.constraint;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint.parser_state_domain_labels = artifact.parser_state_domain_labels;
            constraint.internal_token_buf_masks = artifact.internal_token_buf_masks;
            constraint
                .tokenizer
                .restore_terminal_exprs(artifact.terminal_exprs)
                .map_err(crate::GlrMaskError::Serialization)?;
            constraint
        } else if version == PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION {
            let artifact: ConstraintArtifactV12 = bincode::deserialize(serialized)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            let mut constraint = artifact.constraint;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint.parser_state_domain_labels = artifact.parser_state_domain_labels;
            constraint
                .tokenizer
                .restore_terminal_exprs(artifact.terminal_exprs)
                .map_err(crate::GlrMaskError::Serialization)?;
            constraint
        } else if version == PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION {
            let artifact: ConstraintArtifactV11 = bincode::deserialize(serialized)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            let mut constraint = artifact.constraint;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint
                .tokenizer
                .restore_terminal_exprs(artifact.terminal_exprs)
                .map_err(crate::GlrMaskError::Serialization)?;
            constraint
        } else if version == PREVIOUS_EXPRLESS_CONSTRAINT_VERSION {
            let artifact: ConstraintArtifactV10 = bincode::deserialize(serialized)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            let mut constraint = artifact.constraint;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint
        } else {
            bincode::deserialize(serialized)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
        };
        let deserialize_ms = deserialize_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if !constraint.parser_state_domain_labels.is_empty() {
            if constraint.parser_state_domain_labels.len() != constraint.table.num_states as usize {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "parser-state domain map has {} entries for {} parser states",
                    constraint.parser_state_domain_labels.len(),
                    constraint.table.num_states,
                )));
            }
            let first_synthetic = constraint.table.num_states as i64;
            let default_label = crate::compiler::glr::labels::DEFAULT_LABEL as i64;
            for &label in &constraint.parser_state_domain_labels {
                if label == i32::MAX {
                    continue;
                }
                let label64 = label as i64;
                if label64 < first_synthetic || label64 >= default_label {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "invalid parser-state domain label {label} for {} parser states",
                        constraint.table.num_states,
                    )));
                }
            }
        }
        let rebuild_started = profile.then(std::time::Instant::now);
        let skip_runtime_rebuild_for_profile =
            std::env::var_os("GLRMASK_SKIP_RUNTIME_REBUILD_FOR_PROFILE").is_some();
        if !skip_runtime_rebuild_for_profile {
            if constraint.uses_dynamic_runtime() {
                constraint.rebuild_dynamic_runtime_caches();
            } else {
                if let Some(inventory) = packed_dwa_inventory.take() {
                    crate::automata::weighted::dwa::install_packed_decode_token_set_inventory(
                        inventory,
                    );
                }
                constraint.rebuild_runtime_caches();
            }
        }
        if let Some(total_started) = total_started {
            let rebuild_ms = rebuild_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            eprintln!(
                "[glrmask/profile][constraint_load] bytes={} decompress_ms={decompress_ms:.3} deserialize_ms={deserialize_ms:.3} rebuild_ms={rebuild_ms:.3} total_ms={:.3}",
                bytes.len(),
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if uses_external_runtime_sections(version)
            || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
            || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
            || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
        {
            constraint.serialized_artifact_cache = current_backing.or_else(|| {
                Some(owned_artifact.unwrap_or_else(|| std::sync::Arc::new(bytes.to_vec())))
            });
        }
        Ok(constraint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vocab;
    use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
    use crate::runtime::CommitTemplateDfas;
    use std::sync::Arc;

    fn tiny_constraint() -> Constraint {
        Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
            &Vocab::new(vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
            ]),
        )
        .unwrap()
    }

    fn ignored_constraint() -> Constraint {
        Constraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a";
            "#,
            &Vocab::new(vec![(0, b"a".to_vec()), (1, b" ".to_vec())]),
        )
        .unwrap()
    }

    #[test]
    fn constraint_envelope_roundtrips_and_rejects_previous_formats() {
        let constraint = tiny_constraint();
        let saved = constraint.save();
        assert!(saved.starts_with(&CONSTRAINT_MAGIC));
        assert!(bincode::deserialize::<Constraint>(&saved).is_err());
        let loaded = Constraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let raw = bincode::serialize(&constraint).unwrap();
        assert!(Constraint::load(&raw)
            .unwrap_err()
            .to_string()
            .contains("header"));

        let mut previous_version = saved;
        previous_version[8..10].copy_from_slice(&2u16.to_le_bytes());
        assert!(Constraint::load(&previous_version)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));

        let mut previous_schema = constraint.save();
        previous_schema[8..10].copy_from_slice(&8u16.to_le_bytes());
        assert!(Constraint::load(&previous_schema)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
    }

    #[test]
    fn constraint_envelope_loads_legacy_payloads() {
        let constraint = tiny_constraint();
        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            CONSTRAINT_VERSION
        );
        let raw = bincode::serialize(&constraint).unwrap();

        let loaded = Constraint::load(&envelope(LEGACY_CONSTRAINT_VERSION, &raw))
            .expect("legacy artifact should remain loadable");

        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn constraint_envelope_loads_previous_compressed_payload_without_ignore_descriptor() {
        let constraint = ignored_constraint();
        assert!(constraint.ignore_expr.is_some());
        let raw = bincode::serialize(&constraint).unwrap();
        let compressed = zstd::bulk::compress(&raw, CONSTRAINT_COMPRESSION_LEVEL).unwrap();
        let mut payload = Vec::with_capacity(COMPRESSED_PAYLOAD_HEADER_LEN + compressed.len());
        payload.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        payload.extend_from_slice(&compressed);

        let loaded = Constraint::load(&envelope(
            PREVIOUS_COMPRESSED_CONSTRAINT_VERSION,
            &payload,
        ))
        .expect("the previous compressed wire layout should remain loadable");

        assert!(loaded.ignore_expr.is_none());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn constraint_envelope_loads_previous_exprless_v10_payload() {
        let constraint = ignored_constraint();
        let raw = bincode::serialize(&ConstraintArtifactV10Ref {
            constraint: &constraint,
            ignore_expr: &constraint.ignore_expr,
        })
        .unwrap();
        let compressed = zstd::bulk::compress(&raw, CONSTRAINT_COMPRESSION_LEVEL).unwrap();
        let mut payload = Vec::with_capacity(COMPRESSED_PAYLOAD_HEADER_LEN + compressed.len());
        payload.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        payload.extend_from_slice(&compressed);

        let loaded = Constraint::load(&envelope(PREVIOUS_EXPRLESS_CONSTRAINT_VERSION, &payload))
            .expect("v10 exprless artifact should remain loadable");

        assert_eq!(loaded.ignore_expr, constraint.ignore_expr);
        assert!(loaded.tokenizer.terminal_exprs().is_none());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn constraint_envelope_loads_previous_v11_terminal_expr_payload() {
        let constraint = ignored_constraint();
        let raw = bincode::serialize(&ConstraintArtifactV11Ref {
            constraint: &constraint,
            ignore_expr: &constraint.ignore_expr,
            terminal_exprs: constraint.tokenizer.terminal_exprs(),
        })
        .unwrap();
        let compressed = zstd::bulk::compress(&raw, CONSTRAINT_COMPRESSION_LEVEL).unwrap();
        let mut payload = Vec::with_capacity(COMPRESSED_PAYLOAD_HEADER_LEN + compressed.len());
        payload.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        payload.extend_from_slice(&compressed);

        let loaded = Constraint::load(&envelope(
            PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION,
            &payload,
        ))
        .expect("v11 terminal-expression artifact should remain loadable");

        assert_eq!(loaded.ignore_expr, constraint.ignore_expr);
        assert_eq!(loaded.tokenizer.terminal_exprs(), constraint.tokenizer.terminal_exprs());
        assert!(loaded.parser_state_domain_labels.is_empty());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_constraint_artifact_rejects_invalid_parser_state_domain_map() {
        let mut constraint = ignored_constraint();
        constraint.parser_state_domain_labels = vec![0];
        let error = Constraint::load(&constraint.save()).unwrap_err().to_string();
        assert!(error.contains("parser-state domain map") || error.contains("domain label"));
    }

    #[test]
    fn current_constraint_artifact_preserves_parser_state_domain_labels() {
        let mut constraint = ignored_constraint();
        constraint.parser_state_domain_labels =
            vec![i32::MAX; constraint.table.num_states as usize];
        if let Some(first) = constraint.parser_state_domain_labels.first_mut() {
            *first = constraint.table.num_states as i32;
        }
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert_eq!(
            loaded.parser_state_domain_labels,
            constraint.parser_state_domain_labels,
        );
    }

    #[test]
    fn current_constraint_artifact_preserves_global_ignore_descriptor() {
        let constraint = ignored_constraint();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(constraint.ignore_expr.is_some());
        assert_eq!(loaded.ignore_expr, constraint.ignore_expr);
        assert!(
            loaded.tokenizer.terminal_exprs().is_none(),
            "current load should defer terminal expression reconstruction",
        );
        assert_eq!(
            loaded.retained_terminal_exprs(),
            constraint.tokenizer.terminal_exprs(),
            "current artifacts should lazily retain terminal proof expressions",
        );
        if constraint.can_defer_internal_tsid_inverse() {
            assert!(
                loaded.internal_tsid_to_states.is_empty(),
                "current scalar-state artifacts should defer the redundant TSID inverse",
            );
            assert_eq!(
                loaded.internal_tsid_groups(),
                constraint.internal_tsid_to_states.as_slice(),
                "deferred TSID inverse must reconstruct exactly",
            );
        }
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn constraint_envelope_rejects_invalid_compressed_payloads() {
        let constraint = tiny_constraint();
        let raw = bincode::serialize(&constraint).unwrap();
        let compressed = zstd::bulk::compress(&raw, CONSTRAINT_COMPRESSION_LEVEL).unwrap();

        let mut wrong_raw_len = Vec::with_capacity(8 + compressed.len());
        wrong_raw_len.extend_from_slice(&((raw.len() + 1) as u64).to_le_bytes());
        wrong_raw_len.extend_from_slice(&compressed);
        assert!(Constraint::load(&envelope(
            PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION,
            &wrong_raw_len,
        ))
            .unwrap_err()
            .to_string()
            .contains("uncompressed"));

        assert!(Constraint::load(&envelope(
            PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION,
            &[0; 8],
        ))
        .is_err());
    }

    #[test]
    fn constraint_envelope_rejects_version_and_length_mismatches() {
        let constraint = tiny_constraint();
        let mut wrong_version = constraint.save();
        wrong_version[8..10].copy_from_slice(&(CONSTRAINT_VERSION + 1).to_le_bytes());
        assert!(Constraint::load(&wrong_version)
            .unwrap_err()
            .to_string()
            .contains("version"));

        let mut wrong_length = constraint.save();
        wrong_length[10..18].copy_from_slice(&0u64.to_le_bytes());
        assert!(Constraint::load(&wrong_length)
            .unwrap_err()
            .to_string()
            .contains("payload length"));
    }

    #[test]
    fn constraint_roundtrip_preserves_commit_template_dfas() {
        let mut constraint = tiny_constraint();
        let mut pop = UnweightedDfa::new();
        let accepted = pop.add_state();
        pop.add_transition(pop.start_state, 7, accepted);
        pop.set_accepting(accepted, true);
        let template = CommitTemplateDfas {
            pop,
            read: UnweightedDfa::default(),
            push: UnweightedDfa::default(),
            pop_to_read: vec![None; 2],
            pop_to_push: vec![None; 2],
            read_to_push: Vec::new(),
        };
        constraint.template_dfas_by_terminal = vec![None, Some(Arc::new(template.clone()))];

        let loaded = Constraint::load(&constraint.save()).expect("template artifact should load");
        let loaded_template = loaded.template_dfas_by_terminal[1]
            .as_deref()
            .expect("serialized template should survive load");
        let loaded_fast_template = loaded.fast_template_dfas_by_terminal[1]
            .as_deref()
            .expect("runtime template transition cache should be rebuilt after load");
        assert_eq!(loaded_template.pop, template.pop);
        assert_eq!(loaded_template.read, template.read);
        assert_eq!(loaded_template.push, template.push);
        assert_eq!(loaded_template.pop_to_read, template.pop_to_read);
        assert_eq!(loaded_template.pop_to_push, template.pop_to_push);
        assert_eq!(loaded_template.read_to_push, template.read_to_push);
        assert_eq!(loaded_fast_template.pop.start_state, template.pop.start_state);
        assert_eq!(
            loaded_fast_template.pop.states[accepted as usize].is_accepting,
            template.pop.states[accepted as usize].is_accepting
        );
        assert_eq!(
            loaded_fast_template.pop.states[template.pop.start_state as usize]
                .transitions
                .get(7),
            Some(accepted)
        );
    }

}
