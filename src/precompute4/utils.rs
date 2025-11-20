use crate::glr::table::StateID as ParserStateID;
use crate::precompute4::full_dwa::FullDWABuildError;

pub const DEFAULT_TRANSITION_SYMBOL: i16 = i16::MAX;

pub fn encode_symbol_i16(id: ParserStateID) -> Result<i16, FullDWABuildError> {
    if id.0 > i16::MAX as usize {
        Err(FullDWABuildError::ParserStateIdOutOfRange { state_id: id })
    } else {
        Ok(id.0 as i16)
    }
}

// Negative codes for stack-hitching: i16::MIN + id. Requires parser state IDs to fit in i16.
pub fn encode_negative_i16(id: ParserStateID) -> Result<i16, FullDWABuildError> {
    if id.0 > i16::MAX as usize {
        Err(FullDWABuildError::ParserStateIdOutOfRange { state_id: id })
    } else {
        Ok(i16::MIN + id.0 as i16)
    }
}

// Returns (is_positive_label, parser_state_id).
pub fn decode_symbol_i16(code: i16) -> Result<(bool, ParserStateID), ()> {
    if code >= 0 {
        Ok((true, ParserStateID(code as usize)))
    } else {
        Ok((false, ParserStateID(code.wrapping_sub(i16::MIN) as usize)))
    }
}

pub fn is_default_transition(code: i16) -> bool { code == DEFAULT_TRANSITION_SYMBOL }
