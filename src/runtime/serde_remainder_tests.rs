use glrmask_vocab::__private::VocabExt as _;
use super::*;

fn old_runtime_wire(
    metadata: &ConstraintArtifactCurrentRuntimeRef<'_>,
    residual: &[u8],
) -> Vec<u8> {
    let mut meta = Vec::with_capacity(256 * 1024);
    bincode::serialize_into(&mut meta, metadata).unwrap();
    let mut out = Vec::with_capacity(CURRENT_RUNTIME_HEADER_LEN + meta.len() + residual.len());
    out.extend_from_slice(&CURRENT_RUNTIME_MAGIC);
    out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    out.extend_from_slice(&(residual.len() as u64).to_le_bytes());
    out.extend_from_slice(&meta);
    out.extend_from_slice(residual);
    out
}

#[test]
fn direct_runtime_metadata_writer_preserves_every_byte() {
    let terminal_live_states = vec![vec![0, 3, 1000], vec![], vec![7]];
    let ids = vec![1u32, 4, 7];
    let rows = vec![0u64, u64::MAX, 0x1234_5678_9abc_def0];

    let metadata = ConstraintArtifactCurrentRuntimeRef {
        template_dynamic_proofs: None,
        terminal_live_states: &terminal_live_states,
        segmented_runtime: None,
        dynamic_mask_vocab: None,
        virtual_runtimes: Vec::new(),
        static_virtual_residual_mask: None,
        packed_dwa_dense_mask_ids: &ids,
        packed_dwa_dense_mask_rows: &rows,
    };

    for size in [0usize, 1, 31, 4096, 300_000] {
        let residual = (0..size).map(|index| index as u8).collect::<Vec<_>>();
        assert_eq!(
            encode_current_runtime_wire(&metadata, &residual),
            old_runtime_wire(&metadata, &residual),
        );
    }
}

#[test]
fn runtime_writer_still_decodes_with_the_existing_reader() {
    let metadata = ConstraintArtifactCurrentRuntimeRef {
        template_dynamic_proofs: None,
        terminal_live_states: &[],
        segmented_runtime: None,
        dynamic_mask_vocab: None,
        virtual_runtimes: Vec::new(),
        static_virtual_residual_mask: None,
        packed_dwa_dense_mask_ids: &[],
        packed_dwa_dense_mask_rows: &[],
    };

    let bytes = encode_current_runtime_wire(&metadata, &[]);
    let backing = Arc::new(bytes);
    let (decoded, residual) =
        decode_current_runtime_wire(backing.as_slice(), Arc::clone(&backing)).unwrap();
    assert!(decoded.template_dynamic_proofs.is_none());
    assert!(decoded.terminal_live_states.is_empty());
    assert!(decoded.virtual_runtimes.is_empty());
    assert!(residual.is_none());
}

#[test]
fn runtime_section_length_corruption_remains_an_error() {
    let metadata = ConstraintArtifactCurrentRuntimeRef {
        template_dynamic_proofs: None,
        terminal_live_states: &[],
        segmented_runtime: None,
        dynamic_mask_vocab: None,
        virtual_runtimes: Vec::new(),
        static_virtual_residual_mask: None,
        packed_dwa_dense_mask_ids: &[],
        packed_dwa_dense_mask_rows: &[],
    };

    for range in [4..12, 12..20] {
        let mut bytes = encode_current_runtime_wire(&metadata, &[]);
        bytes[range].copy_from_slice(&u64::MAX.to_le_bytes());
        let backing = Arc::new(bytes);
        assert!(
            decode_current_runtime_wire(backing.as_slice(), Arc::clone(&backing))
                .is_err()
        );
    }
}

#[test]
fn ordinary_composition_metadata_unwrap_has_identical_bytes() {
    let vocab = crate::Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let constraint = crate::Grammar::glrm("start root; nt root = 'a' 'b'?;")
        .compile_with_build_options(
            &vocab,
            MaskMode::C.build_options(),
        )
        .unwrap();

    let base = encode_composition_metadata_base_for_save(&constraint);
    let expected = if let Some(wire) =
        crate::compiler::boundary_precomputed_completion::saved_wire(&constraint)
    {
        crate::compiler::boundary_precomputed_completion::wrap_envelope(base, wire)
            .unwrap()
    } else {
        crate::compiler::boundary_precomputed_completion::split_envelope(&base)
            .unwrap()
            .0
            .to_vec()
    };
    assert_eq!(encode_composition_metadata_for_save(&constraint), expected);
}

fn sparse_wire(map_len: u64, live: u64, entries: &[(u32, u32)]) -> Vec<u8> {
    let mut out = b"SRM3".to_vec();
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&12u64.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // terminal
    out.extend_from_slice(&0u32.to_le_bytes()); // offset
    for n in [map_len, live, 0, 0, 1, 0] { out.extend_from_slice(&n.to_le_bytes()); }
    out.extend_from_slice(b"TKF3");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    for &(index, state) in entries {
        out.extend_from_slice(&index.to_le_bytes());
        out.extend_from_slice(&state.to_le_bytes());
    }
    out
}

#[test]
fn malformed_sparse_projection_counts_and_entries_fail_before_expansion() {
    for bytes in [
        sparse_wire(u64::MAX, 0, &[]),
        sparse_wire(1, u64::MAX, &[]),
        sparse_wire(1, 1, &[]),
        sparse_wire(1, 1, &[(1, 0)]),
        sparse_wire(1, 1, &[(0, 1)]),
        sparse_wire(3, 2, &[(1, 0), (1, 0)]),
    ] {
        let backing = Arc::new(bytes);
        assert!(decode_static_virtual_residual_mask_wire(&backing, Arc::clone(&backing)).is_err());
    }
    // Even a plausible dense dimension cannot cause repeated large allocations
    // in sibling or nested payloads, as admission uses the shared budget.
    let bytes = Arc::new(sparse_wire(8 * 1024 * 1024, 0, &[]));
    let budget = LoadBudget::new(0);
    budget.charge(48 << 20).unwrap();
    assert!(decode_static_virtual_residual_mask_wire_budgeted(&bytes, Arc::clone(&bytes), &budget).is_err());
}

#[test]
fn malformed_compressed_composition_lengths_are_rejected() {
    let mut bytes = COMPOSITION_METADATA_ZSTD_MAGIC.to_vec();
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    bytes.push(0);
    assert!(validate_composition_metadata_wire(&bytes).is_err());
    assert!(decode_composition_metadata(&bytes).is_err());
}
