use crate::glr::table::StateID as ParserStateID;
use crate::precompute4::full_dwa::FullDWABuildError;
use crate::precompute4::weighted_automata::common::Label;

pub const DEFAULT_TRANSITION_SYMBOL: Label = Label::MAX;

pub fn encode_symbol_i16(id: ParserStateID) -> Result<Label, FullDWABuildError> {
    if id.0 > Label::MAX as usize {
        panic!("Transition symbol out of range: {:?}", id);
        Err(FullDWABuildError::ParserStateIdOutOfRange { state_id: id })
    } else {
        Ok(id.0 as Label)
    }
}

// Negative codes for stack-hitching: Label::MIN + id. Requires parser state IDs to fit in Label.
pub fn encode_negative_i16(id: ParserStateID) -> Result<Label, FullDWABuildError> {
    if id.0 > Label::MAX as usize {
        panic!("Negative transition symbol out of range: {:?}", id);
        Err(FullDWABuildError::ParserStateIdOutOfRange { state_id: id })
    } else {
        Ok(Label::MIN + id.0 as Label)
    }
}

// Returns (is_positive_label, parser_state_id).
pub fn decode_symbol_i16(code: Label) -> Result<(bool, ParserStateID), ()> {
    if code >= 0 {
        Ok((true, ParserStateID(code as usize)))
    } else {
        Ok((false, ParserStateID(code.wrapping_sub(Label::MIN) as usize)))
    }
}

pub fn is_default_transition(code: Label) -> bool { code == DEFAULT_TRANSITION_SYMBOL }
