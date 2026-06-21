use glrmask_runtime::{RuntimeArtifact, ARTIFACT_MAGIC, ARTIFACT_VERSION};

fn envelope(version: u16, payload_len: u64, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&ARTIFACT_MAGIC);
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn rejects_obsolete_envelope_version() {
    let error = RuntimeArtifact::from_bytes(envelope(1, 0, &[])).unwrap_err();
    assert!(error.to_string().contains("version 1"));
}

#[test]
fn rejects_mismatched_payload_length() {
    let error = RuntimeArtifact::from_bytes(envelope(ARTIFACT_VERSION, 3, &[0, 1])).unwrap_err();
    assert!(error.to_string().contains("payload length"));
}
