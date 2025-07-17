use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};
use crate::json_serialization::{JSONConvertible, JSONNode};

#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct U8Set {
    pub(crate) x: u128,
    pub(crate) y: u128,
}

// Assuming u8 implements JSONConvertible like this (or similar):
// (You might need to add this if it's not already part of your json_serialization module)
/*
impl JSONConvertible for u8 {
    fn to_json(&self) -> JSONNode {
        // JSON numbers are often f64. u8 fits perfectly.
        JSONNode::Number(*self as f64)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Number(n) => {
                if n.fract() == 0.0 && n >= 0.0 && n <= 255.0 {
                    Ok(n as u8)
                } else {
                    Err(format!("Number {} is not a valid u8", n))
                }
            }
            // Optionally, support strings if your u8s might be serialized that way
            // JSONNode::String(s) => s.parse::<u8>().map_err(|e| format!("Invalid u8 string: {}", e)),
            _ => Err("Expected JSONNode::Number for u8".to_string()),
        }
    }
}
*/

// Space-efficient JSON conversion for U8Set using ranges.
impl JSONConvertible for U8Set {
    fn to_json(&self) -> JSONNode {
        if self.is_empty() {
            return JSONNode::Array(Vec::new());
        }

        let mut members_json = Vec::new();
        let mut iter = self.iter(); // self.iter() is guaranteed to be sorted

        // Get the first item to initialize current_start and current_prev
        // This is safe due to the is_empty() check above.
        let first_val = iter.next().unwrap();
        let mut current_start = first_val;
        let mut current_prev = first_val;

        for val in iter {
            if val == current_prev + 1 {
                // Continues the range
                current_prev = val;
            } else {
                // End of current range/item, push it
                if current_start == current_prev {
                    // Single item
                    members_json.push(JSONNode::Int(current_start as i128));
                } else {
                    // Range
                    members_json.push(JSONNode::Array(vec![
                        JSONNode::Int(current_start as i128),
                        JSONNode::Int(current_prev as i128),
                    ]));
                }
                // Start a new range/item
                current_start = val;
                current_prev = val;
            }
        }

        // Push the last accumulated range/item
        if current_start == current_prev {
            members_json.push(JSONNode::Int(current_start as i128));
        } else {
            members_json.push(JSONNode::Array(vec![
                JSONNode::Int(current_start as i128),
                JSONNode::Int(current_prev as i128),
            ]));
        }

        JSONNode::Array(members_json)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut set = U8Set::none();
                for item_node in arr {
                    match item_node {
                        JSONNode::Int(n) => {
                            if n <= u8::MAX as i128 {
                                set.insert(n as u8);
                            } else {
                                return Err(format!("Number {} is too large for u8", n));
                            }
                        }
                        JSONNode::Array(pair_arr) => {
                            if pair_arr.len() == 2 {
                                let start_val = match &pair_arr[0] {
                                    JSONNode::Int(n) => {
                                        if *n <= u8::MAX as i128 { *n as u8 }
                                        else { return Err(format!("Start of range {} is too large for u8", n)); }
                                    }
                                    _ => return Err(format!("Expected JSONNode::Int for start of range value, got {:?}", pair_arr[0])),
                                };
                                let end_val = match &pair_arr[1] {
                                    JSONNode::Int(n) => {
                                        if *n <= u8::MAX as i128 { *n as u8 }
                                        else { return Err(format!("End of range {} is too large for u8", n)); }
                                    }
                                    _ => return Err(format!("Expected JSONNode::Int for end of range value, got {:?}", pair_arr[1])),
                                };

                                if start_val > end_val {
                                    return Err(format!("Range start {} > end {} is invalid", start_val, end_val));
                                }
                                for val_in_range in start_val..=end_val {
                                    set.insert(val_in_range);
                                }
                            } else {
                                return Err("Range array in U8Set JSON must have 2 elements".to_string());
                            }
                        }
                        _ => return Err("U8Set JSON array elements must be Int (single value) or 2-element Array of Ints (range)".to_string()),
                    }
                }
                Ok(set)
            }
            _ => Err("Expected JSONNode::Array for U8Set".to_string()),
        }
    }
}


impl Default for U8Set {
    fn default() -> Self {
        Self::none()
    }
}

impl U8Set {
    #[inline]
    fn is_set(&self, index: u8) -> bool {
        if index < 128 {
            self.x & (1 << index) != 0
        } else {
            self.y & (1 << (index - 128)) != 0
        }
    }

    #[inline]
    fn set_bit(&mut self, index: u8) {
        if index < 128 {
            self.x |= 1 << index;
        } else {
            self.y |= 1 << (index - 128);
        }
    }

    #[inline]
    fn clear_bit(&mut self, index: u8) {
        if index < 128 {
            self.x &= !(1 << index);
        } else {
            self.y &= !(1 << (index - 128));
        }
    }

    #[inline]
    fn update(&mut self, other: &Self) {
        self.x |= other.x;
        self.y |= other.y;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.x.count_ones() as usize + self.y.count_ones() as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x == 0 && self.y == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.x = 0;
        self.y = 0;
    }

    #[inline]
    pub fn all() -> Self {
        U8Set { x: u128::MAX, y: u128::MAX }
    }

    #[inline]
    pub fn none() -> Self {
        U8Set { x: 0, y: 0 }
    }

    #[inline]
    pub fn new() -> Self {
        Self::none()
    }


    pub fn from_u8(p0: u8) -> U8Set {
        let mut result = U8Set::none();
        result.insert(p0);
        result
    }


    pub fn from_u8_range(start: u8, end: u8) -> U8Set {
        Self::from_match_fn(move |i| start <= i && i <= end)
    }

    pub fn from_char(p0: char) -> U8Set {
        Self::from_chars(&p0.to_string())
    }

    pub fn from_char_negation(p0: char) -> U8Set {
        let mut result = U8Set::none();
        result.insert(p0 as u8);
        result.complement()
    }

    pub fn from_byte_range(range: impl IntoIterator<Item = u8>) -> U8Set {
        let mut result = U8Set::none();
        for c in range {
            result.insert(c);
        }
        result
    }

    pub fn from_char_negation_range(range: impl IntoIterator<Item = u8>) -> U8Set {
        Self::from_byte_range(range).complement()
    }

    pub fn from_slice(slice: &[u8]) -> Self {
        let mut result = Self::none();
        for byte in slice {
            result.insert(*byte);
        }
        result
    }

    #[inline]
    pub fn insert(&mut self, value: u8) -> bool {
        if self.contains(value) {
            false
        } else {
            self.set_bit(value);
            true
        }
    }


    #[inline]
    pub fn remove(&mut self, value: u8) -> bool {
        if !self.contains(value) {
            false
        } else {
            self.clear_bit(value);
            true
        }
    }

    pub fn without(&self, value: impl Into<u8>) -> Self {
        let mut result = *self;
        result.remove(value.into());
        result
    }

    pub fn difference(&self, other: &Self) -> Self {
        let mut result = *self;
        result.x &= !other.x;
        result.y &= !other.y;
        result
    }

    #[inline]
    pub fn contains(&self, value: impl Into<u8>) -> bool {
        self.is_set(value.into())
    }


    pub fn from_byte(byte: u8) -> Self {
        let mut result = Self::none();
        result.insert(byte);
        result
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut result = Self::none();
        for byte in bytes {
            result.insert(*byte);
        }
        result
    }

    pub fn from_chars(chars: &str) -> Self {
        let mut result = Self::none();
        for c in chars.chars() {
            assert!(c.is_ascii(), "Character {} is not a valid ASCII u8 value", c);
            result.insert(c as u8);
        }
        result
    }


    pub fn from_chars_negation(chars: &str) -> Self {
        Self::from_chars(chars).complement()
    }


    pub fn from_str(s: &str) -> Self {
        Self::from_chars(s)
    }

    pub fn from_match_fn<F>(f: F) -> Self
    where
        F: Fn(u8) -> bool,
    {
        let mut result = Self::none();
        for i in 0..=255 {
            if f(i) {
                result.insert(i);
            }
        }
        result
    }

    pub fn from_range(start: u8, end: u8) -> Self {
        Self::from_match_fn(move |i| start <= i && i <= end)
    }

    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=255).filter(move |&i| self.contains(i))
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        U8Set {
            x: self.x | other.x,
            y: self.y | other.y,
        }
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        U8Set {
            x: self.x & other.x,
            y: self.y & other.y,
        }
    }


    #[inline]
    pub fn complement(&self) -> Self {
        U8Set {
            x: !self.x,
            y: !self.y,
        }
    }
}


impl BitOr for &U8Set {
    type Output = U8Set;
    fn bitor(self, other: &U8Set) -> U8Set { self.union(other) }
}
impl BitAnd for &U8Set {
    type Output = U8Set;
    fn bitand(self, other: &U8Set) -> U8Set { self.intersection(other) }
}
impl BitOr for U8Set {
    type Output = U8Set;
    fn bitor(self, other: U8Set) -> U8Set { &self | &other }
}
impl BitAnd for U8Set {
    type Output = U8Set;
    fn bitand(self, other: U8Set) -> U8Set { &self & &other }
}
impl BitOrAssign for U8Set {
    fn bitor_assign(&mut self, other: U8Set) { self.update(&other); }
}
impl BitAndAssign for U8Set {
    fn bitand_assign(&mut self, other: U8Set) {
        self.x &= other.x;
        self.y &= other.y;
    }
}

impl std::fmt::Debug for U8Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ranges = Vec::new();
        if !self.is_empty() {
            let mut iter = self.iter();
            // Safe to unwrap due to is_empty check
            let first = iter.next().unwrap();
            let mut current_start = first;
            let mut current_prev = first;

            for i in iter {
                if i == current_prev + 1 {
                    current_prev = i;
                } else {
                    ranges.push((current_start, current_prev));
                    current_start = i;
                    current_prev = i;
                }
            }
            ranges.push((current_start, current_prev));
        }

        let format_byte = |b: u8| -> String {
            if b.is_ascii_graphic() || b == b' ' {
                format!("'{}'", b as char)
            } else {
                format!("0x{:02x}", b)
            }
        };

        let mut parts = Vec::new();
        for &(start_val, end_val) in &ranges {
            // Using saturating_sub to handle u8 boundaries gracefully.
            let len = end_val.saturating_sub(start_val) as usize + 1;
            if len >= 3 {
                parts.push(format!("{} - {}", format_byte(start_val), format_byte(end_val)));
            } else {
                parts.push(format_byte(start_val));
                if len == 2 {
                    parts.push(format_byte(end_val));
                }
            }
        }

        const MAX_DEBUG_PARTS: usize = 16;
        let parts_str = if parts.len() > MAX_DEBUG_PARTS {
            let mut s = parts[..MAX_DEBUG_PARTS].join(", ");
            s.push_str(", ...");
            s
        } else {
            parts.join(", ")
        };

        write!(f, "U8Set([{}], len: {})", parts_str, self.len())
    }
}

impl std::fmt::Display for U8Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ranges = Vec::new();
        let mut current_start = None;
        let mut current_prev = None;

        for i in self.iter() {
            match (current_start, current_prev) {
                (None, None) => { // First item in a potential new range
                    current_start = Some(i);
                }
                (Some(_start_val), Some(p_val)) if i == p_val + 1 => { // Continues a range
                    // Just update prev, start remains the same
                }
                (Some(s_val), Some(p_val)) => { // End of a range, start of a new one
                    ranges.push((s_val, p_val));
                    current_start = Some(i);
                }
                _ => unreachable!("Invalid state in U8Set Display fmt"),
            }
            current_prev = Some(i);
        }

        if let Some(s_val) = current_start {
            if let Some(p_val) = current_prev {
                ranges.push((s_val, p_val));
            }
        }

        write!(f, "[")?;
        for (i, (start_val, end_val)) in ranges.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            // Helper to format char or byte hex
            let format_byte = |b: u8, ff: &mut std::fmt::Formatter<'_>| {
                if b.is_ascii_graphic() || b == b' ' { // Check for printable ASCII or space
                    write!(ff, "'{}'", b as char)
                } else {
                    write!(ff, "0x{:02x}", b) // Non-printable as hex
                }
            };

            if start_val == end_val {
                format_byte(*start_val, f)?;
            } else if end_val - start_val == 1 { // Two consecutive items, print both
                format_byte(*start_val, f)?;
                write!(f, ", ")?;
                format_byte(*end_val, f)?;
            } else { // A range
                format_byte(*start_val, f)?;
                write!(f, "..")?;
                format_byte(*end_val, f)?;
            }
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use crate::json_serialization::JSONConvertible;
    use super::*;

    #[test]
    fn test_u8set_basic_ops() {
        let mut set = U8Set::none();
        assert!(set.insert(b'a'));
        assert!(set.insert(b'b'));
        assert!(!set.insert(b'a'));
        assert!(set.contains(b'a'));
        assert!(set.contains(b'b'));
        assert!(!set.contains(b'c'));
        assert_eq!(set.len(), 2);
        assert!(set.remove(b'a'));
        assert!(!set.remove(b'c'));
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
        set.clear();
        assert!(set.is_empty());

        let set1 = U8Set::from_chars("abc");
        let set2 = U8Set::from_chars("bcd");
        let union = &set1 | &set2;
        let intersection = &set1 & &set2;
        assert_eq!(union.len(), 4); // abcd
        assert_eq!(intersection.len(), 2); // bc

        let even_set = U8Set::from_match_fn(|x| x % 2 == 0);
        assert!(even_set.contains(0));
        assert!(even_set.contains(2));
        assert!(!even_set.contains(1));
        assert_eq!(even_set.len(), 128);
    }

    #[test]
    fn test_u8set_json_serialization() {
        let mut set = U8Set::none();
        set.insert(10);
        set.insert(20);
        set.insert(30);

        let json_node = set.to_json();
        match json_node {
            JSONNode::Array(ref arr) => {
                assert_eq!(arr.len(), 3, "JSON array should have 3 individual numbers");
                assert_eq!(arr[0], JSONNode::Int(10), "First element should be 10");
                assert_eq!(arr[1], JSONNode::Int(20), "Second element should be 20");
                assert_eq!(arr[2], JSONNode::Int(30), "Third element should be 30");
            }
            _ => panic!("Expected JSONNode::Array"),
        }

        let deserialized_set = U8Set::from_json(json_node).unwrap();
        assert_eq!(deserialized_set.len(), 3);
        assert!(deserialized_set.contains(10));
        assert!(deserialized_set.contains(20));
        assert!(deserialized_set.contains(30));
        assert!(!deserialized_set.contains(40));
        assert_eq!(set, deserialized_set);

        // Test empty set
        let empty_set = U8Set::none();
        let empty_json = empty_set.to_json();
        match empty_json {
            JSONNode::Array(ref arr) => assert!(arr.is_empty()),
            _ => panic!("Expected JSONNode::Array for empty set"),
        }
        let deserialized_empty = U8Set::from_json(empty_json).unwrap();
        assert!(deserialized_empty.is_empty());
        assert_eq!(empty_set, deserialized_empty);

        // Test full set (might be slow to construct JSON, but check logic)
        let full_set = U8Set::all();
        let full_json = full_set.to_json();
        match full_json {
            JSONNode::Array(ref arr) => {
                assert_eq!(arr.len(), 1, "Full set should serialize to a single range array");
                match &arr[0] {
                    JSONNode::Array(range_arr) => {
                        assert_eq!(range_arr.len(), 2, "Range array should have two elements");
                        assert_eq!(range_arr[0], JSONNode::Int(0), "Range start should be 0");
                        assert_eq!(range_arr[1], JSONNode::Int(255), "Range end should be 255");
                    }
                    _ => panic!("Expected inner JSONNode::Array for the range"),
                }
            }
            _ => panic!("Expected JSONNode::Array for full set"),
        }
        let deserialized_full = U8Set::from_json(full_json).unwrap();
        assert_eq!(deserialized_full.len(), 256);
        assert_eq!(full_set, deserialized_full);
    }

    #[test]
    fn test_u8set_json_serialization_ranges() {
        let mut set = U8Set::none();
        // {0, 1, 2, 5, 10, 11, 12, 14}
        set.insert(0); set.insert(1); set.insert(2); // Range 0-2
        set.insert(5); // Single
        set.insert(10); set.insert(11); set.insert(12); // Range 10-12
        set.insert(14); // Single

        let json_node = set.to_json();
        match json_node {
            JSONNode::Array(ref arr) => {
                assert_eq!(arr.len(), 4, "Expected 4 items: [0..2], 5, [10..12], 14");
                assert_eq!(arr[0], JSONNode::Array(vec![JSONNode::Int(0), JSONNode::Int(2)]));
                assert_eq!(arr[1], JSONNode::Int(5));
                assert_eq!(arr[2], JSONNode::Array(vec![JSONNode::Int(10), JSONNode::Int(12)]));
                assert_eq!(arr[3], JSONNode::Int(14));
            }
            _ => panic!("Expected JSONNode::Array"),
        }

        let deserialized_set = U8Set::from_json(json_node).unwrap();
        assert_eq!(deserialized_set.len(), 8, "Deserialized set should have 8 members");
        assert_eq!(set, deserialized_set, "Deserialized set should match original");
    }

    #[test]
    fn test_u8set_display_and_debug() {
        let set1 = U8Set::from_bytes(&[b'a', b'b', b'c', b'z', 0, 1, 2, 15]);
        // Display: ['a'..'c', 'z', 0x00..0x02, 0x0f] (order might vary based on iter)
        // Iteration is 0..255, so it will be sorted.
        assert_eq!(format!("{}", set1), "[0x00..0x02, 0x0f, 'a'..'c', 'z']");

        let set2 = U8Set::from_bytes(&[b'a', b'c', b'e']);
        assert_eq!(format!("{}", set2), "['a', 'c', 'e']");

        let set3 = U8Set::from_bytes(&[10, 12, 11]); // Order of insertion doesn't matter
        assert_eq!(format!("{}", set3), "[0x0a..0x0c]"); // Display will sort

        let set4 = U8Set::from_bytes(&[b'h', b'e', b'l', b'l', b'o']);
        assert_eq!(format!("{}", set4), "['e', 'h', 'l', 'o']"); // 'l' is unique

        // Debug format
        assert_eq!(format!("{:?}", set1), "U8Set([0x00 - 0x02, 0x0f, 'a' - 'c', 'z'], len: 8)");
        assert_eq!(format!("{:?}", set2), "U8Set(['a', 'c', 'e'], len: 3)");
        assert_eq!(format!("{:?}", set3), "U8Set([0x0a - 0x0c], len: 3)");
        assert_eq!(format!("{:?}", set4), "U8Set(['e', 'h', 'l', 'o'], len: 4)");

        // Test with many non-consecutive items to check truncation
        let mut many_items_set = U8Set::new();
        for i in (0..50).step_by(2) {
            many_items_set.insert(i); // 0, 2, 4, ... 48. 25 items.
        }
        assert_eq!(
            format!("{:?}", many_items_set),
            "U8Set([0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, ...], len: 25)"
        );
    }
}
