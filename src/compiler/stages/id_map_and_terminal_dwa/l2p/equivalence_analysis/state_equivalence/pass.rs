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
