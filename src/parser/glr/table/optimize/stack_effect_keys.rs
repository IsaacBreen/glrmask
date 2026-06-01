enum StackEffectActionKey {
    Shift(u32, bool),
    StackShifts(Vec<StackShift>),
    GuardedStackShifts(Vec<GuardedStackShift>),
    Reduce(NonterminalID, u32),
    Split,
    Accept,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackEffectKey {
    origin_state: u32,
    state: u32,
    tid: TerminalID,
    action: StackEffectActionKey,
    frame: StackEffectFrame,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackEffectVisitKey {
    state: u32,
    tid: TerminalID,
    action_tag: u8,
    frame: StackEffectFrame,
}

#[derive(Clone)]
struct StackEffectResult {
    effects: Vec<GuardedStackShift>,
    origin_dependent: bool,
}
