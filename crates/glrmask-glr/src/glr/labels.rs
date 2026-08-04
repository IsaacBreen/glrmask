pub const DEFAULT_LABEL: i32 = i32::MAX - 1;

pub fn encode_positive_label(state: u32) -> i32 {
    state as i32
}

pub fn encode_negative_label(state: u32) -> i32 {
    i32::MIN.wrapping_add(state as i32)
}

pub fn is_negative_label(label: i32) -> bool {
    label < 0 && label != DEFAULT_LABEL
}

pub fn negative_to_positive_label(label: i32) -> i32 {
    label.wrapping_sub(i32::MIN)
}
