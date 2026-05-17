use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateEquivalencePassKind {
    MaxLength,
}

impl StateEquivalencePassKind {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "max_length" => Ok(Self::MaxLength),
            other => Err(format!(
                "unknown state-equivalence pass `{other}`; expected one of: max_length"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateEquivalenceScope {
    Global,
    L2p,
}

pub(crate) trait StateEquivalencePass {
    type Statistic;

    fn name(&self) -> &'static str;

    fn compute_statistic(&self, vocab: &Vocab) -> Self::Statistic;

    fn compute_state_map(
        &self,
        tokenizer: &Tokenizer,
        statistic: &Self::Statistic,
        initial_state_map: Option<&ManyToOneIdMap>,
        active_groups: Option<&[bool]>,
    ) -> ManyToOneIdMap;
}