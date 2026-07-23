use crate::runtime::Constraint;

const CONSTRAINT_MAGIC: [u8; 8] = *b"GLRCONS\0";
const CONSTRAINT_VERSION: u16 = 4;
const CONSTRAINT_HEADER_LEN: usize = CONSTRAINT_MAGIC.len() + 2 + 8;

impl Constraint {
    /// Serialize this compiled constraint to a versioned binary artifact.
    pub fn save(&self) -> Vec<u8> {
        let payload = bincode::serialize(self).expect("Constraint serialization should succeed");
        let mut bytes = Vec::with_capacity(CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&CONSTRAINT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// Load a compiled constraint from an artifact produced by [`Constraint::save`].
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        if bytes.len() < CONSTRAINT_HEADER_LEN || !bytes.starts_with(&CONSTRAINT_MAGIC) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid constraint artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if version != CONSTRAINT_VERSION {
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
        let mut constraint: Constraint =
            bincode::deserialize(&bytes[CONSTRAINT_HEADER_LEN..])
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        constraint.rebuild_runtime_caches();
        Ok(constraint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vocab;

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
}
