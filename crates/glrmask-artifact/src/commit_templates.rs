use glrmask_finite_automata::automata::unweighted_u32::dfa::DFA as UnweightedDfa;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CommitTemplateDfas {
    pub pop: UnweightedDfa,
    pub read: UnweightedDfa,
    pub push: UnweightedDfa,
    pub pop_to_read: Vec<Option<u32>>,
    pub pop_to_push: Vec<Option<u32>>,
    pub read_to_push: Vec<Option<u32>>,
}
