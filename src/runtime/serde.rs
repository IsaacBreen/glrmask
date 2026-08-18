use crate::runtime::Constraint;
use crate::runtime::artifact::InternalTokenBufMasks;
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
const CONSTRAINT_VERSION: u16 = 16;
const CONSTRAINT_HEADER_LEN: usize = CONSTRAINT_MAGIC.len() + 2 + 8;
const COMPRESSED_PAYLOAD_HEADER_LEN: usize = 8;
const CONSTRAINT_COMPRESSION_LEVEL: i32 = 1;
const V14_SECTION_MAGIC: [u8; 4] = *b"S14\0";
const V14_SECTION_HEADER_LEN: usize = V14_SECTION_MAGIC.len() + 4 * 8;
const V15_SECTION_MAGIC: [u8; 4] = *b"S15\0";
const V15_SECTION_HEADER_LEN: usize = V15_SECTION_MAGIC.len() + 5 * 8;
const V16_SECTION_MAGIC: [u8; 4] = *b"S16\0";
const V16_SECTION_HEADER_LEN: usize = V16_SECTION_MAGIC.len() + 5 * 8;

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

fn invert_original_token_map(original_to_internal: &[u32]) -> Result<Vec<Vec<u32>>, String> {
    let Some(max_internal) = original_to_internal
        .iter()
        .copied()
        .filter(|&internal| internal != u32::MAX)
        .max()
    else {
        return Ok(Vec::new());
    };
    let group_count = usize::try_from(max_internal)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "internal token vocabulary is too large".to_owned())?;
    let mut groups = vec![Vec::<u32>::new(); group_count];
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
            return bytes.to_vec();
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let ((weight_pool, core), (dwa, table, runtime)) = rayon::join(
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
                        let previous_omit_inverse =
                            crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(true);
                        let previous_packed_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_packed(true);
                        let previous_packed_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_packed(true);
                        let started = profile.then(std::time::Instant::now);
                        // `bincode::serialize` first runs `serialized_size` and
                        // then serializes again. Custom compact serializers
                        // (especially the tokenizer) do real packing work in
                        // both passes. Write once into a generously-sized Vec
                        // instead; capacity growth is much cheaper than
                        // rebuilding every packed field twice.
                        let mut encoded = Vec::with_capacity(3 * 1024 * 1024);
                        let encode_result = bincode::serialize_into(
                            &mut encoded,
                            &ConstraintArtifactV14CoreRef {
                                constraint: self,
                                ignore_expr: &self.ignore_expr,
                                terminal_exprs: self.tokenizer.terminal_exprs(),
                                parser_state_domain_labels: &self.parser_state_domain_labels,
                                internal_token_buf_masks: &self.internal_token_buf_masks,
                            },
                        );
                        crate::automata::lexer::tokenizer::set_compact_artifact_serde(
                            previous_compact_tokenizer,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_packed(
                            previous_packed_token_bytes,
                        );
                        crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(
                            previous_omit_inverse,
                        );
                        crate::runtime::artifact::original_token_map_artifact_serde::set_packed(
                            previous_packed_original_token_map,
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
                let ((dwa, table), runtime) = rayon::join(
                    || {
                        rayon::join(
                    || {
                        let started = profile.then(std::time::Instant::now);
                        let bytes = self.packed_parser_dwa.as_ref().map_or_else(
                            || self.parser_dwa.artifact_packed_bytes(),
                            |packed| packed.fast_wire_bytes(),
                        );
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=dwa ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                bytes.len(),
                            );
                        }
                        bytes
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
                );
                (dwa, table, runtime)
            },
        );

        let payload_len =
            V16_SECTION_HEADER_LEN
                + weight_pool.len()
                + dwa.len()
                + table.len()
                + core.len()
                + runtime.len();
        let mut bytes = Vec::with_capacity(CONSTRAINT_HEADER_LEN + payload_len);
        bytes.extend_from_slice(&CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&CONSTRAINT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload_len as u64).to_le_bytes());
        bytes.extend_from_slice(&V16_SECTION_MAGIC);
        bytes.extend_from_slice(&(weight_pool.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(dwa.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(table.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(core.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(runtime.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&weight_pool);
        bytes.extend_from_slice(&dwa);
        bytes.extend_from_slice(&table);
        bytes.extend_from_slice(&core);
        bytes.extend_from_slice(&runtime);
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
        let payload = &bytes[CONSTRAINT_HEADER_LEN..];
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
        let mut constraint = if version == CONSTRAINT_VERSION
            || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
            || version == PREVIOUS_SECTIONED_CONSTRAINT_VERSION
        {
            let (weight_section, dwa_section, table_section, core_section, runtime_section) =
                if version == CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime) = v16_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, Some(runtime))
                } else if version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime) = v15_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, Some(runtime))
                } else {
                    let (weight, dwa, table, core) = v14_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, None)
                };
            let ((dwa_result, (table_result, runtime_result)), core_result) = rayon::join(
                || {
                    rayon::join(
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let result = if version == CONSTRAINT_VERSION
                                || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
                            {
                                let decoded = if dwa_section.starts_with(b"DWF1") {
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
                                    let result = bincode::deserialize::<ConstraintArtifactV15Runtime>(runtime_section)
                                        .map(Some)
                                        .map_err(|err| err.to_string());
                                    if let Some(started) = started {
                                        eprintln!("[glrmask/profile][constraint_section] name=runtime ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                                    }
                                    result
                                },
                            )
                        },
                    )
                },
                || -> Result<
                    (
                        ConstraintArtifactV14Core,
                        Option<std::sync::Arc<crate::runtime::artifact::token_bytes_artifact_serde::PackedTokenBytes>>,
                        Option<(
                            std::sync::Arc<crate::ds::weight::PackedRuntimeWeightPool>,
                            Vec<u32>,
                        )>,
                    ),
                    String,
                > {
                    let section_started = profile.then(std::time::Instant::now);
                    let weight_count = if version == CONSTRAINT_VERSION
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
                        let previous_omit_inverse =
                            crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(
                                true,
                            );
                        let previous_packed_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_packed(
                                true,
                            );
                        let previous_packed_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_packed(true);
                        let previous_defer_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_defer_unpack(
                                version == CONSTRAINT_VERSION
                                    || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION,
                            );
                        let core_started = profile.then(std::time::Instant::now);
                        let decoded = bincode::deserialize::<ConstraintArtifactV14Core>(core_section)
                            .map_err(|err| err.to_string());
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
                        crate::runtime::artifact::token_bytes_artifact_serde::set_packed(
                            previous_packed_token_bytes,
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
                        if version == CONSTRAINT_VERSION
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
            let (artifact, deferred_token_bytes, packed_weights) =
                core_result.map_err(crate::GlrMaskError::Serialization)?;
            let mut constraint = artifact.constraint;
            if let Some((pool, ids)) = packed_weights {
                attach_packed_non_dwa_weights(&mut constraint, pool, ids)
                    .map_err(crate::GlrMaskError::Serialization)?;
            }
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
            constraint.table = table;
            constraint.internal_token_to_tokens =
                invert_original_token_map(&constraint.original_token_to_internal)
                    .map_err(crate::GlrMaskError::Serialization)?;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint.parser_state_domain_labels = artifact.parser_state_domain_labels;
            constraint.internal_token_buf_masks = artifact.internal_token_buf_masks;
            constraint.packed_token_bytes = deferred_token_bytes;
            if let Some(runtime) = runtime {
                constraint.terminal_live_states = runtime.terminal_live_states;
            }
            constraint
                .tokenizer
                .restore_terminal_exprs(artifact.terminal_exprs)
                .map_err(crate::GlrMaskError::Serialization)?;
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
        if let Some(total_started) = total_started {
            let rebuild_ms = rebuild_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            eprintln!(
                "[glrmask/profile][constraint_load] bytes={} decompress_ms={decompress_ms:.3} deserialize_ms={deserialize_ms:.3} rebuild_ms={rebuild_ms:.3} total_ms={:.3}",
                bytes.len(),
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if version == CONSTRAINT_VERSION
            || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
        {
            constraint.serialized_artifact_cache = Some(std::sync::Arc::from(bytes));
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
        assert_eq!(
            loaded.tokenizer.terminal_exprs(),
            constraint.tokenizer.terminal_exprs(),
            "current artifacts should retain terminal proof expressions",
        );
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
