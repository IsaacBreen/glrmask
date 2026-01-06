
use sep1::precompute4::weighted_automata::rangeset::RangeSet;
use range_set_blaze::RangeSetBlaze;

fn main() {
    println!("Testing RangeSet behavior");

    let w1 = RangeSet::from_item(1);
    println!("w1 (from_item 1): {:?}", w1);

    let w_all = RangeSet::all();
    println!("w_all: {:?}", w_all);

    let w_int = &w_all & &w1;
    println!("w_int (all & w1): {:?}", w_int);

    let w0 = RangeSet::from_item(0);
    println!("w0 (from_item 0): {:?}", w0);

    // Simulate nwa_special_map logic
    let mut current_tokens = RangeSet::all();
    let edge_weight = RangeSet::from_item(1);
    
    let intersection = &current_tokens & &edge_weight;
    println!("intersection: {:?}", intersection);

    // Test exact scenario from visual debug
    // 1..=MAX
    let mut w_wtf = RangeSet::all();
    w_wtf.remove(0);
    println!("w_remove_0: {:?}", w_wtf);

    let w_inv0 = !RangeSet::from_item(0);
    println!("!w0: {:?}", w_inv0);
    
    let w_inv1 = !RangeSet::from_item(1);
    println!("!w1: {:?}", w_inv1);
}
