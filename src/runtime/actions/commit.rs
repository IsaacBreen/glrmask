



#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::state::ConstraintState;

impl<'a> ConstraintState<'a> {
    
    
    
    
    
    
    pub fn commit_token(
        &mut self,
        token_id: u32,
    ) {
        unimplemented!()
    }

    
    
    
    
    pub fn commit_bytes(&mut self, bytes: &[u8]) {
        unimplemented!()
    }

    
    
    
    pub fn commit_tokens(&mut self, tokens: &[u32]) {
        unimplemented!()
    }

    
    pub(crate) fn process_bytes_raw(&mut self, bytes: &[u8]) {
        unimplemented!()
    }
}
