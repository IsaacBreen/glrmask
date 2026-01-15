
use sep1::dwa_i32::rangeset::RangeSet;

fn main() {
    println!("Testing RangeSet behavior");

    let w1 = RangeSet::from_item(1);
    println!("w1 (from_item 1): {:?}", w1);

    let max_val = 5usize;
    let w_all = RangeSet::ones(max_val + 1);
    println!("w_all: {:?}", w_all);

    let w_int = &w_all & &w1;
    println!("w_int (all & w1): {:?}", w_int);

    let w0 = RangeSet::from_item(0);
    println!("w0 (from_item 0): {:?}", w0);

    // Simulate nwa_special_map logic
    let current_tokens = RangeSet::ones(max_val + 1);
    let edge_weight = RangeSet::from_item(1);
    
    let intersection = &current_tokens & &edge_weight;
    println!("intersection: {:?}", intersection);

    // Test exact scenario from visual debug
    // 1..=MAX
    let mut w_wtf = RangeSet::ones(max_val + 1);
    w_wtf.remove(0);
    println!("w_remove_0: {:?}", w_wtf);

    let w_inv0 = &RangeSet::ones(max_val + 1) - &RangeSet::from_item(0);
    println!("!w0: {:?}", w_inv0);
    
    let w_inv1 = &RangeSet::ones(max_val + 1) - &RangeSet::from_item(1);
    println!("!w1: {:?}", w_inv1);
}
