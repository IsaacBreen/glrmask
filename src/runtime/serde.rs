use crate::runtime::Constraint;

impl Constraint {
    pub fn save(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Constraint serialization should succeed")
    }

    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        let mut constraint: Self = bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        constraint.rebuild_runtime_caches();
        Ok(constraint)
    }
}
