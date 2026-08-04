use glrmask_artifact::__private::equiv_types::InternalIdMap;
use glrmask_weighted_automata::weighted_u32::dwa::DWA;

#[derive(Debug, Clone, Copy, Default)]
pub struct TerminalDwaPhaseProfile {
    pub id_map_ms: f64,
    pub terminal_dwa_ms: f64,
    pub compact_ms: f64,
    pub split_terminal_dwa_total_ms: f64,
    pub global_merge_ms: f64,
}

#[derive(Debug, Clone)]
pub struct LocalIdMapTerminalDwa {
    pub id_map: InternalIdMap,
    pub dwa: DWA,
    pub profile: TerminalDwaPhaseProfile,
}

impl TerminalDwaPhaseProfile {
    pub fn total_ms(self) -> f64 {
        self.id_map_ms + self.terminal_dwa_ms + self.compact_ms
    }

    pub fn add_assign(&mut self, other: Self) {
        self.id_map_ms += other.id_map_ms;
        self.terminal_dwa_ms += other.terminal_dwa_ms;
        self.compact_ms += other.compact_ms;
        self.split_terminal_dwa_total_ms += other.split_terminal_dwa_total_ms;
        self.global_merge_ms += other.global_merge_ms;
    }
}

pub fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}
