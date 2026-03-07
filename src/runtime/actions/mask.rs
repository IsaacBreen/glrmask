


#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::state::ConstraintState;

pub struct MaskView<'state, 'constraint> {
    state: &'state ConstraintState<'constraint>,
}

impl<'a> ConstraintState<'a> {
    
    pub fn mask_view(&self) -> MaskView<'_, 'a> {
        MaskView { state: self }
    }
}

impl MaskView<'_, '_> {
    
    
    
    
    pub fn mask(&self) -> Vec<u32> {
        let _ = self.state;
        unimplemented!()
    }

    
    
    
    
    pub fn fill_mask(&self, buf: &mut [u32]) {
        let _ = self.state;
        let _ = buf;
        unimplemented!()
    }
}
