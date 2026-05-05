use crate::runtime::Constraint;

impl Constraint {
    pub fn save(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Constraint serialization should succeed")
    }

    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        let profile = std::env::var_os("GLRMASK_PROFILE_LOAD").is_some();
        let total_started = std::time::Instant::now();
        let deserialize_started = std::time::Instant::now();
        let mut constraint: Self = bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        if profile {
            eprintln!(
                "[glrmask/profile][load] phase=deserialize ms={:.3} bytes={}",
                deserialize_started.elapsed().as_secs_f64() * 1000.0,
                bytes.len(),
            );
        }
        let rebuild_started = std::time::Instant::now();
        constraint.rebuild_runtime_caches();
        if profile {
            eprintln!(
                "[glrmask/profile][load] phase=rebuild_runtime_caches ms={:.3} total_ms={:.3}",
                rebuild_started.elapsed().as_secs_f64() * 1000.0,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(constraint)
    }
}
