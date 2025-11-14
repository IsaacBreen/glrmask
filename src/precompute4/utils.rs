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

pub fn encode_negative_i16(id: ParserStateID) -> Result<i16, FullDWABuildError> {
    // Negative codes for stack-hitching. We store as i16::MIN + id.
    // This requires that parser state IDs are not too large.
    if id.0 > (i16::MAX as usize) {
        Err(FullDWABuildError::ParserStateIdOutOfRange { state_id: id })
    } else {
        Ok(i16::MIN + (id.0 as i16))
    }
}

pub fn decode_symbol_i16(code: i16) -> Result<(bool, ParserStateID), ()> {
    if code == DEFAULT_TRANSITION_SYMBOL {
        Err(())
    } else if code >= 0 {
        Ok((false, ParserStateID(code as usize)))
    } else {
        Ok((true, ParserStateID((code.wrapping_sub(i16::MIN)) as usize)))
    }
}

pub fn is_default_transition(code: i16) -> bool {
    code == DEFAULT_TRANSITION_SYMBOL
}