use std::collections::{BTreeMap, HashMap, BTreeSet, HashSet};
use std::hash::Hash;
use std::marker::Sized;
use bimap::BiBTreeMap;

// Import the derive macro
use json_convertible_derive::JSONConvertible;

// --- JSONNode Enum ---
#[derive(Debug, Clone, PartialEq)]
pub enum JSONNode {
    Null,
    Bool(bool),
    Int(i128),
    UInt(u128),
    Float(f64),
    String(String),
    Array(Vec<JSONNode>),
    Object(BTreeMap<String, JSONNode>),
}

// --- JSONConvertible Trait ---
pub trait JSONConvertible: Sized {
    fn to_json(&self) -> JSONNode;
    fn from_json(node: JSONNode) -> Result<Self, String>;
}

// --- Implementations for Primitives ---

impl JSONConvertible for () {
    fn to_json(&self) -> JSONNode { JSONNode::Null }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Null => Ok(()),
            _ => Err("Expected JSONNode::Null for unit type".to_string()),
        }
    }
}

impl JSONConvertible for bool {
    fn to_json(&self) -> JSONNode { JSONNode::Bool(*self) }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Bool(b) => Ok(b),
            _ => Err("Expected JSONNode::Bool for bool".to_string()),
        }
    }
}

// --- Implementations for numeric primitives without precision loss ---
macro_rules! impl_json_for_signed {
    ($($t:ty),*) => {
        $(
            impl JSONConvertible for $t {
                fn to_json(&self) -> JSONNode { JSONNode::Int(*self as i128) }
                fn from_json(node: JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::Int(n)   => {
                            if n >= (<$t>::MIN as i128) && n <= (<$t>::MAX as i128) {
                                Ok(n as $t)
                            } else {
                                Err(format!("Integer {} out of range for {}", n, stringify!($t)))
                            }
                        },
                        JSONNode::UInt(u)  => {
                            // Check if u fits into the positive range of $t
                            if u <= (<$t>::MAX as u128) {
                                Ok(u as $t)
                            } else {
                                Err(format!("Unsigned integer {} too large for {}", u, stringify!($t)))
                            }
                        },
                        other => Err(format!("Expected JSONNode::Int or JSONNode::UInt for {}, got {:?}", stringify!($t), other)),
                    }
                }
            }
        )*
    };
}

macro_rules! impl_json_for_unsigned {
    ($($t:ty),*) => {
        $(
            impl JSONConvertible for $t {
                fn to_json(&self) -> JSONNode { JSONNode::UInt(*self as u128) }
                fn from_json(node: JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::UInt(u) => {
                            if u <= (<$t>::MAX as u128) {
                                Ok(u as $t)
                            } else {
                                Err(format!("Unsigned integer {} out of range for {}", u, stringify!($t)))
                            }
                        },
                        JSONNode::Int(n)  => {
                            if n >= 0 {
                                let u_n = n as u128;
                                if u_n <= (<$t>::MAX as u128) {
                                    Ok(u_n as $t)
                                } else {
                                    Err(format!("Integer {} (as u128) out of range for {}", n, stringify!($t)))
                                }
                            } else {
                                Err(format!("Negative integer {} cannot be converted to {}", n, stringify!($t)))
                            }
                        },
                        other => Err(format!("Expected JSONNode::UInt or non-negative JSONNode::Int for {}, got {:?}", stringify!($t), other)),
                    }
                }
            }
        )*
    };
}

macro_rules! impl_json_for_float {
    ($($t:ty),*) => {
        $(
            impl JSONConvertible for $t {
                fn to_json(&self) -> JSONNode { JSONNode::Float(*self as f64) }
                fn from_json(node: JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::Float(f) => Ok(f as $t),
                        JSONNode::Int(n)   => Ok(n as $t), // Standard Rust cast behavior, potential precision loss for large i128
                        JSONNode::UInt(u)  => Ok(u as $t), // Standard Rust cast behavior, potential precision loss for large u128
                        other => Err(format!("Expected JSONNode::Float, ::Int, or ::UInt for {}, got {:?}", stringify!($t), other)),
                    }
                }
            }
        )*
    };
}

// Signed integers
impl_json_for_signed!(i8, i16, i32, i64, isize, i128);
// Unsigned integers
impl_json_for_unsigned!(u8, u16, u32, u64, usize, u128);
// Floating-point types
impl_json_for_float!(f32, f64);


impl JSONConvertible for String {
    fn to_json(&self) -> JSONNode { JSONNode::String(self.clone()) }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::String(s) => Ok(s),
            _ => Err("Expected JSONNode::String for String".to_string()),
        }
    }
}

impl<'a> JSONConvertible for &'a str {
    fn to_json(&self) -> JSONNode { JSONNode::String(self.to_string()) }
    fn from_json(_node: JSONNode) -> Result<Self, String> {
        Err("Cannot deserialize into &str, deserialize into String instead".to_string())
    }
}

// --- Implementations for Option ---
impl<T: JSONConvertible> JSONConvertible for Option<T> {
    fn to_json(&self) -> JSONNode {
        match self {
            Some(val) => val.to_json(),
            None => JSONNode::Null,
        }
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Null => Ok(None),
            _ => T::from_json(node).map(Some),
        }
    }
}

// --- Implementations for Collections ---

// Generic Vec<T>
impl<T: JSONConvertible> JSONConvertible for Vec<T> {
    fn to_json(&self) -> JSONNode {
        JSONNode::Array(self.iter().map(|item| item.to_json()).collect())
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => arr.into_iter().map(T::from_json).collect(),
            _ => Err("Expected JSONNode::Array for Vec<T>".to_string()),
        }
    }
}


impl<T: JSONConvertible + Ord> JSONConvertible for BTreeSet<T> {
    fn to_json(&self) -> JSONNode {
        JSONNode::Array(self.iter().map(|item| item.to_json()).collect())
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => arr.into_iter().map(T::from_json).collect(),
            _ => Err("Expected JSONNode::Array for BTreeSet<T>".to_string()),
        }
    }
}

impl<T: JSONConvertible + Eq + Hash + Ord> JSONConvertible for HashSet<T> {
    fn to_json(&self) -> JSONNode {
        let mut vec: Vec<_> = self.iter().collect();
        vec.sort(); // Sort for deterministic output
        JSONNode::Array(vec.iter().map(|item| item.to_json()).collect())
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => arr.into_iter().map(T::from_json).collect(),
            _ => Err("Expected JSONNode::Array for HashSet<T>".to_string()),
        }
    }
}

impl<K, V> JSONConvertible for BTreeMap<K, V>
where
    K: JSONConvertible + Ord, // Keys in BTreeMap must be Ord
    V: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        // BTreeMap is already sorted by key, so iter() gives deterministic order
        let pairs = self.iter().map(|(k, v)| {
            JSONNode::Array(vec![k.to_json(), v.to_json()])
        }).collect();
        JSONNode::Array(pairs)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut map = BTreeMap::new();
                for pair_node in arr {
                    match pair_node {
                        JSONNode::Array(mut pair_vec) if pair_vec.len() == 2 => {
                            let v_node = pair_vec.pop().unwrap();
                            let k_node = pair_vec.pop().unwrap();
                            map.insert(K::from_json(k_node)?, V::from_json(v_node)?);
                        }
                        _ => return Err("Expected 2-element array for BTreeMap entry".to_string()),
                    }
                }
                Ok(map)
            }
            _ => Err("Expected JSONNode::Array for BTreeMap<K, V>".to_string()),
        }
    }
}

impl<K, V> JSONConvertible for HashMap<K, V>
where
    K: JSONConvertible + Eq + Hash + Ord, // Ord needed for sorting for deterministic JSON
    V: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let mut sorted_pairs: Vec<_> = self.iter().collect();
        // Sort by key for deterministic JSON output
        sorted_pairs.sort_by_key(|(k, _)| *k); // Requires K: Ord

        let pairs = sorted_pairs.into_iter().map(|(k, v)| {
            JSONNode::Array(vec![k.to_json(), v.to_json()])
        }).collect();
        JSONNode::Array(pairs)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut map = HashMap::new();
                for pair_node in arr {
                    match pair_node {
                        JSONNode::Array(mut pair_vec) if pair_vec.len() == 2 => {
                            let v_node = pair_vec.pop().unwrap();
                            let k_node = pair_vec.pop().unwrap();
                            map.insert(K::from_json(k_node)?, V::from_json(v_node)?);
                        }
                        _ => return Err("Expected 2-element array for HashMap entry".to_string()),
                    }
                }
                Ok(map)
            }
            _ => Err("Expected JSONNode::Array for HashMap<K, V>".to_string()),
        }
    }
}

impl<L, R> JSONConvertible for BiBTreeMap<L, R>
where
    L: JSONConvertible + Ord + Eq, // BiBTreeMap requires Ord + Eq for both left and right
    R: JSONConvertible + Ord + Eq,
{
    fn to_json(&self) -> JSONNode {
        // BiBTreeMap iter() is sorted by the left value (L)
        let pairs = self.iter().map(|(l, r)| {
            JSONNode::Array(vec![l.to_json(), r.to_json()])
        }).collect();
        JSONNode::Array(pairs)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut map = BiBTreeMap::new();
                for pair_node in arr {
                    match pair_node {
                        JSONNode::Array(mut pair_vec) if pair_vec.len() == 2 => {
                            let r_node = pair_vec.pop().unwrap();
                            let l_node = pair_vec.pop().unwrap();
                            map.insert(L::from_json(l_node)?, R::from_json(r_node)?);
                        }
                        _ => return Err("Expected 2-element array for BiBTreeMap entry".to_string()),
                    }
                }
                Ok(map)
            }
            _ => Err("Expected JSONNode::Array for BiBTreeMap<L, R>".to_string()),
        }
    }
}

// --- Tests (optional, but good for verifying) ---
#[cfg(test)]
mod tests {
    use super::*; // Imports JSONNode, JSONConvertible, MyStruct, etc.

    // Example struct using the derive
    #[derive(Debug, Clone, PartialEq, JSONConvertible)]
    struct MyStruct {
        field1: i32,
        field2: String,
        optional_field: Option::<bool>,
        list_of_numbers: Vec::<u32>,
        byte_buffer: Vec::<u8>,
    }

    // Example generic struct using the derive
    #[derive(Debug, Clone, PartialEq, JSONConvertible)]
    struct GenericStruct<T: JSONConvertible, U: JSONConvertible> {
        item_t: T,
        item_u: U,
        description: String,
    }

    // Example unit struct using the derive
    #[derive(Debug, Clone, PartialEq, JSONConvertible)]
    struct MyUnitStruct;

    #[test]
    fn test_my_struct_serialization_deserialization() {
        let original = MyStruct {
            field1: 42,
            field2: "hello".to_string(),
            optional_field: Some(true),
            list_of_numbers: vec![1, 2, 3],
            byte_buffer: vec![10, 20, 30],
        };

        let json_node = original.to_json();

        // Expected JSON structure (BTreeMap for object means fields are sorted by key)
        // {
        //   "byte_buffer": [UInt(10), UInt(20), UInt(30)],
        //   "field1": Int(42),
        //   "field2": "hello",
        //   "list_of_numbers": [UInt(1), UInt(2), UInt(3)],
        //   "optional_field": true
        // }

        if let JSONNode::Object(obj) = &json_node {
            assert_eq!(obj.get("field1"), Some(&JSONNode::Int(42)));
            assert_eq!(obj.get("field2"), Some(&JSONNode::String("hello".to_string())));
            assert_eq!(obj.get("optional_field"), Some(&JSONNode::Bool(true)));
            if let Some(JSONNode::Array(arr)) = obj.get("list_of_numbers") {
                 assert_eq!(arr, &vec![JSONNode::UInt(1), JSONNode::UInt(2), JSONNode::UInt(3)]);
            } else {
                panic!("list_of_numbers not found or not an array");
            }
            if let Some(JSONNode::Array(arr)) = obj.get("byte_buffer") {
                 assert_eq!(arr, &vec![JSONNode::UInt(10), JSONNode::UInt(20), JSONNode::UInt(30)]);
            } else {
                panic!("byte_buffer not found or not an array");
            }
        } else {
            panic!("Expected JSONNode::Object, got {:?}", json_node);
        }

        let deserialized = MyStruct::from_json(json_node).expect("Deserialization failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_my_struct_optional_none() {
        let original = MyStruct {
            field1: 1,
            field2: "world".to_string(),
            optional_field: None,
            list_of_numbers: vec![],
            byte_buffer: vec![],
        };
        let json_node = original.to_json();
        // { "byte_buffer": [], "field1": Int(1), "field2": "world", "list_of_numbers": [], "optional_field": null }
        if let JSONNode::Object(obj) = &json_node {
             assert_eq!(obj.get("optional_field"), Some(&JSONNode::Null));
             assert_eq!(obj.get("field1"), Some(&JSONNode::Int(1)));
        } else {
            panic!("Expected JSONNode::Object");
        }
        let deserialized = MyStruct::from_json(json_node).expect("Deserialization failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_generic_struct_serialization() {
        let original = GenericStruct {
            item_t: 123i32,
            item_u: "test_string".to_string(),
            description: "A generic item".to_string(),
        };
        let json_node = original.to_json();
        // {
        //   "description": "A generic item",
        //   "item_t": Int(123),
        //   "item_u": "test_string"
        // }
        if let JSONNode::Object(obj) = &json_node {
            assert_eq!(obj.get("item_t"), Some(&JSONNode::Int(123)));
        } else {
            panic!("Expected JSONNode::Object");
        }
        let deserialized = GenericStruct::<i32, String>::from_json(json_node).expect("Deserialization failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_generic_struct_with_custom_type() {
        let my_s = MyStruct {
            field1: 1, field2: "inner".to_string(), optional_field: None, list_of_numbers: vec![], byte_buffer: vec![]
        };
        let original = GenericStruct {
            item_t: my_s.clone(),
            item_u: MyUnitStruct,
            description: "Struct with custom types".to_string(),
        };
        let json_node = original.to_json();
        let deserialized = GenericStruct::<MyStruct, MyUnitStruct>::from_json(json_node).expect("Deserialization failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_unit_struct_serialization() {
        let original = MyUnitStruct;
        let json_node = original.to_json();
        assert_eq!(json_node, JSONNode::Object(BTreeMap::new())); // Derive macro makes unit structs empty objects
        let deserialized = MyUnitStruct::from_json(json_node).expect("Deserialization failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_btreemap_serialization() {
        let mut map = BTreeMap::new();
        map.insert("b_key".to_string(), 20i32);
        map.insert("a_key".to_string(), 10i32);

        let json_node = map.to_json();
        // Expected: [["a_key", Int(10)], ["b_key", Int(20)]] (sorted by key)
        match json_node {
            JSONNode::Array(pairs) => {
                assert_eq!(pairs.len(), 2);
                // Check first pair
                match &pairs[0] {
                    JSONNode::Array(pair1) => {
                        assert_eq!(pair1[0], JSONNode::String("a_key".to_string()));
                        assert_eq!(pair1[1], JSONNode::Int(10));
                    }
                    _ => panic!("Expected array for pair"),
                }
                // Check second pair
                match &pairs[1] {
                    JSONNode::Array(pair2) => {
                        assert_eq!(pair2[0], JSONNode::String("b_key".to_string()));
                        assert_eq!(pair2[1], JSONNode::Int(20));
                    }
                    _ => panic!("Expected array for pair"),
                }
            }
            _ => panic!("Expected JSONNode::Array for BTreeMap"),
        }

        let deserialized: BTreeMap<String, i32> = BTreeMap::from_json(map.to_json()).unwrap();
        assert_eq!(map, deserialized);
    }

    #[test]
    fn test_vec_u8_specialization() {
        let data: Vec<u8> = vec![1, 2, 255];
        let json_node = data.to_json();
        // Expected: [UInt(1), UInt(2), UInt(255)]
        match json_node {
            JSONNode::Array(ref nums) => {
                assert_eq!(nums.len(), 3);
                assert_eq!(nums[0], JSONNode::UInt(1));
                assert_eq!(nums[1], JSONNode::UInt(2));
                assert_eq!(nums[2], JSONNode::UInt(255));
            }
            _ => panic!("Expected JSONNode::Array for Vec<u8>"),
        }
        let deserialized: Vec<u8> = Vec::from_json(json_node).unwrap();
        assert_eq!(data, deserialized);

        // Test invalid u8 from various JSONNode number types
        let invalid_json_uint_too_large = JSONNode::Array(vec![JSONNode::UInt(256)]);
        assert!(Vec::<u8>::from_json(invalid_json_uint_too_large).is_err());

        let invalid_json_int_too_large = JSONNode::Array(vec![JSONNode::Int(256)]);
        assert!(Vec::<u8>::from_json(invalid_json_int_too_large).is_err());

        let invalid_json_int_neg = JSONNode::Array(vec![JSONNode::Int(-1)]);
        assert!(Vec::<u8>::from_json(invalid_json_int_neg).is_err());

        let invalid_json_float = JSONNode::Array(vec![JSONNode::Float(10.5)]);
        assert!(Vec::<u8>::from_json(invalid_json_float).is_err()); // Floats not convertible to u8 by default

        // Test valid u8 from Int
        let valid_json_int = JSONNode::Array(vec![JSONNode::Int(128)]);
        let deserialized_from_int: Vec<u8> = Vec::from_json(valid_json_int).unwrap();
        assert_eq!(deserialized_from_int, vec![128u8]);
    }

    #[test]
    fn test_large_numbers() {
        let large_u128 = u128::MAX;
        let json_u128 = large_u128.to_json();
        assert_eq!(json_u128, JSONNode::UInt(u128::MAX));
        let deserialized_u128 = u128::from_json(json_u128).unwrap();
        assert_eq!(large_u128, deserialized_u128);

        let large_i128 = i128::MIN;
        let json_i128 = large_i128.to_json();
        assert_eq!(json_i128, JSONNode::Int(i128::MIN));
        let deserialized_i128 = i128::from_json(json_i128).unwrap();
        assert_eq!(large_i128, deserialized_i128);

        let float_val = 123.456f64;
        let json_float = float_val.to_json();
        assert_eq!(json_float, JSONNode::Float(123.456f64));
        let deserialized_float = f64::from_json(json_float).unwrap();
        assert_eq!(float_val, deserialized_float);

        // Test conversion from Int to Float
        let int_node = JSONNode::Int(123);
        let float_from_int = f64::from_json(int_node).unwrap();
        assert_eq!(float_from_int, 123.0f64);

        // Test conversion from UInt to Float
        let uint_node = JSONNode::UInt(456);
        let float_from_uint = f64::from_json(uint_node).unwrap();
        assert_eq!(float_from_uint, 456.0f64);

        // Test that integer types do not deserialize from Float
        let float_node_exact_int = JSONNode::Float(123.0);
        assert!(i32::from_json(float_node_exact_int.clone()).is_err());
        assert!(u32::from_json(float_node_exact_int.clone()).is_err());

        let float_node_inexact_int = JSONNode::Float(123.7);
        assert!(i32::from_json(float_node_inexact_int.clone()).is_err());
        assert!(u32::from_json(float_node_inexact_int.clone()).is_err());

        // Test range errors for integer types
        assert!(i8::from_json(JSONNode::Int(128)).is_err()); // 128 is out of range for i8
        assert!(i8::from_json(JSONNode::UInt(128)).is_err());// 128 is out of range for i8
        assert!(u8::from_json(JSONNode::Int(256)).is_err()); // 256 is out of range for u8
        assert!(u8::from_json(JSONNode::UInt(256)).is_err());// 256 is out of range for u8
        assert!(u8::from_json(JSONNode::Int(-1)).is_err());  // -1 is out of range for u8
    }
}