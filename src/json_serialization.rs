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
    Number(String), // Store numbers as strings to preserve precision
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

macro_rules! impl_json_for_number {
    ($($t:ty),*) => {
        $(
            impl JSONConvertible for $t {
                fn to_json(&self) -> JSONNode { JSONNode::Number(self.to_string()) }
                fn from_json(node: JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::Number(s) => {
                            s.parse::<$t>().map_err(|e| format!("Failed to parse number string '{}' as {}: {}", s, stringify!($t), e))
                        }
                        _ => Err(format!("Expected JSONNode::Number(String) for {}", stringify!($t))),
                    }
                }
            }
        )*
    };
}

impl_json_for_number!(usize, isize, u16, u32, u64, i8, i16, i32, i64, f32, f64);

// Specific implementation for u8 to handle range/integer validation
impl JSONConvertible for u8 {
    fn to_json(&self) -> JSONNode { JSONNode::Number(self.to_string()) }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Number(s) => {
                s.parse::<u8>().map_err(|e| format!("Failed to parse number string '{}' as u8: {}", s, e))
            }
            _ => Err("Expected JSONNode::Number(String) for u8".to_string()),
        }
    }
}

impl JSONConvertible for u128 {
    fn to_json(&self) -> JSONNode { JSONNode::Number(self.to_string()) }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Number(s) => {
                s.parse::<u128>().map_err(|e| format!("Failed to parse number string '{}' as u128: {}", s, e))
            }
            _ => Err("Expected JSONNode::Number(String) for u128".to_string()),
        }
    }
}
impl JSONConvertible for i128 {
    fn to_json(&self) -> JSONNode { JSONNode::Number(self.to_string()) }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Number(s) => {
                s.parse::<i128>().map_err(|e| format!("Failed to parse number string '{}' as i128: {}", s, e))
            }
            _ => Err("Expected JSONNode::Number(String) for i128".to_string()),
        }
    }
}

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
        vec.sort();
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
    K: JSONConvertible + Ord,
    V: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
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
    K: JSONConvertible + Eq + Hash + Ord,
    V: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let mut sorted_pairs: Vec<_> = self.iter().collect();
        sorted_pairs.sort_by_key(|(k, _)| *k);

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
    L: JSONConvertible + Ord + Eq,
    R: JSONConvertible + Ord + Eq,
{
    fn to_json(&self) -> JSONNode {
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
        list_of_numbers: Vec::<u32>, // Uses generic Vec<T>
        byte_buffer: Vec::<u8>,      // Uses specialized Vec<u8> (handled by generic Vec + specific u8)
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

        // Expected JSON structure representation within JSONNode (BTreeMap for object means fields are sorted by key)
        // {
        //   "byte_buffer": ["10", "20", "30"],
        //   "field1": "42",
        //   "field2": "hello",
        //   "list_of_numbers": ["1", "2", "3"],
        //   "optional_field": true
        // }

        if let JSONNode::Object(obj) = &json_node {
            assert_eq!(obj.get("field1"), Some(&JSONNode::Number("42".to_string())));
            assert_eq!(obj.get("field2"), Some(&JSONNode::String("hello".to_string())));
            assert_eq!(obj.get("optional_field"), Some(&JSONNode::Bool(true)));
            if let Some(JSONNode::Array(arr)) = obj.get("list_of_numbers") {
                 assert_eq!(arr, &vec![JSONNode::Number("1".to_string()), JSONNode::Number("2".to_string()), JSONNode::Number("3".to_string())]);
            } else {
                panic!("list_of_numbers not found or not an array");
            }
            if let Some(JSONNode::Array(arr)) = obj.get("byte_buffer") {
                 assert_eq!(arr, &vec![JSONNode::Number("10".to_string()), JSONNode::Number("20".to_string()), JSONNode::Number("30".to_string())]);
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
        // Expected JSON structure representation within JSONNode: { "byte_buffer": [], "field1": "1", "field2": "world", "list_of_numbers": [], "optional_field": null }
        if let JSONNode::Object(obj) = &json_node {
             assert_eq!(obj.get("optional_field"), Some(&JSONNode::Null));
             assert_eq!(obj.get("field1"), Some(&JSONNode::Number("1".to_string())));
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
        // Expected JSON structure representation within JSONNode:
        // {
        //   "description": "A generic item",
        //   "item_t": "123",
        //   "item_u": "test_string"
        // }
        if let JSONNode::Object(obj) = &json_node {
             assert_eq!(obj.get("item_t"), Some(&JSONNode::Number("123".to_string())));
             assert_eq!(obj.get("item_u"), Some(&JSONNode::String("test_string".to_string())));
             assert_eq!(obj.get("description"), Some(&JSONNode::String("A generic item".to_string())));
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
        assert_eq!(json_node, JSONNode::Object(BTreeMap::new()));
        let deserialized = MyUnitStruct::from_json(json_node).expect("Deserialization failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_btreemap_serialization() {
        let mut map = BTreeMap::new();
        map.insert("b_key".to_string(), 20);
        map.insert("a_key".to_string(), 10);

        let json_node = map.to_json();
        // Expected JSON structure representation within JSONNode: [["a_key", "10"], ["b_key", "20"]] (sorted by key)
        match json_node {
            JSONNode::Array(pairs) => {
                assert_eq!(pairs.len(), 2);
                // Check first pair
                match &pairs[0] {
                    JSONNode::Array(pair1) => {
                        assert_eq!(pair1[0], JSONNode::String("a_key".to_string()));
                        assert_eq!(pair1[1], JSONNode::Number("10".to_string()));
                    }
                    _ => panic!("Expected array for pair"),
                }
                // Check second pair
                match &pairs[1] {
                    JSONNode::Array(pair2) => {
                        assert_eq!(pair2[0], JSONNode::String("b_key".to_string()));
                        assert_eq!(pair2[1], JSONNode::Number("20".to_string()));
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
        // Expected JSON structure representation within JSONNode: ["1", "2", "255"]
        match json_node {
            JSONNode::Array(ref nums) => {
                assert_eq!(nums.len(), 3);
                assert_eq!(nums[0], JSONNode::Number("1".to_string()));
                assert_eq!(nums[1], JSONNode::Number("2".to_string()));
                assert_eq!(nums[2], JSONNode::Number("255".to_string()));
            }
            _ => panic!("Expected JSONNode::Array for Vec<u8>"),
        }
        let deserialized: Vec<u8> = Vec::from_json(json_node).unwrap();
        assert_eq!(data, deserialized);

        // Test invalid u8 from string
        let invalid_json = JSONNode::Array(vec![JSONNode::Number("256".to_string())]);
        assert!(Vec::<u8>::from_json(invalid_json).is_err());
        let invalid_json_fract = JSONNode::Array(vec![JSONNode::Number("10.5".to_string())]);
        assert!(Vec::<u8>::from_json(invalid_json_fract).is_err());
        let invalid_json_text = JSONNode::Array(vec![JSONNode::Number("abc".to_string())]);
        assert!(Vec::<u8>::from_json(invalid_json_text).is_err());
    }

     #[test]
     fn test_f64_precision() {
        let val: f64 = 0.1 + 0.2; // This often results in a value slightly different from 0.3 in f64
        let val_str = format!("{:?}", val); // Use debug formatting to see the exact f64 value
        let json_node = val.to_json();

        match json_node {
            JSONNode::Number(s) => {
                // The string representation in the JSONNode should match the exact f64 value
                // Note: The parse back might still have f64 precision issues if you were to
                // check equality of f64 values, but the string representation in the JSONNode
                // is preserved.
                 assert_eq!(s, val_str);
                 let deserialized: f64 = f64::from_json(JSONNode::Number(s)).unwrap();
                 // Direct f64 equality check might fail for floating point inaccuracies
                 // assert_eq!(val, deserialized); // This could fail

                 // A better check is to compare their string representations after deserialization
                 assert_eq!(format!("{:?}", val), format!("{:?}", deserialized));

            }
            _ => panic!("Expected JSONNode::Number(String)"),
        }

         let large_int_f64 = 9007199254740992.0_f64; // An integer within f64 range but beyond i64::MAX
         let large_int_f64_str = large_int_f64.to_string();
         let json_node_large = large_int_f64.to_json();
         match json_node_large {
             JSONNode::Number(s) => {
                 assert_eq!(s, large_int_f64_str);
                 let deserialized: f64 = f64::from_json(JSONNode::Number(s)).unwrap();
                 assert_eq!(format!("{:?}", large_int_f64), format!("{:?}", deserialized));
             }
             _ => panic!("Expected JSONNode::Number(String)"),
         }

         // Test a number that requires more precision than f64 offers
         let high_precision_str = "0.1234567890123456789"; // More decimal places than f64 can precisely store
         let json_node_hp = JSONNode::Number(high_precision_str.to_string());

         // Deserializing this back into f64 WILL lose precision
         let deserialized_hp: f64 = f64::from_json(json_node_hp.clone()).unwrap();
         // The deserialized f64 will not be exactly equal to the original high_precision_str when formatted back
         // assert_ne!(format!("{:?}", deserialized_hp), high_precision_str); // This is expected

         // The point is that the *JSONNode* itself holds the precise string:
         match json_node_hp {
             JSONNode::Number(s) => {
                 assert_eq!(s, high_precision_str);
             }
             _ => panic!("Expected JSONNode::Number(String)"),
         }
     }

    #[test]
    fn test_i128_u128_serialization_deserialization() {
        let large_i128: i128 = 12345678901234567890123456789012345_i128;
        let large_u128: u128 = 98765432109876543210987654321098765_u128;

        let json_i128 = large_i128.to_json();
        let json_u128 = large_u128.to_json();

        match json_i128 {
            JSONNode::Number(s) => assert_eq!(s, large_i128.to_string()),
            _ => panic!("Expected JSONNode::Number(String) for i128"),
        }

        match json_u128 {
            JSONNode::Number(s) => assert_eq!(s, large_u128.to_string()),
            _ => panic!("Expected JSONNode::Number(String) for u128"),
        }

        let deserialized_i128 = i128::from_json(json_i128).unwrap();
        let deserialized_u128 = u128::from_json(json_u128).unwrap();

        assert_eq!(large_i128, deserialized_i128);
        assert_eq!(large_u128, deserialized_u128);

        // Test deserializing an out-of-range string for i128/u128
        let too_large_i128_str = "1701411834604692317316873037158841057280"; // i128::MAX + 1
        let invalid_i128_json = JSONNode::Number(too_large_i128_str.to_string());
        assert!(i128::from_json(invalid_i128_json).is_err());

         let too_large_u128_str = "3402823669209384634633746074317682114560"; // u128::MAX + 1
        let invalid_u128_json = JSONNode::Number(too_large_u128_str.to_string());
        assert!(u128::from_json(invalid_u128_json).is_err());

        let invalid_number_format_str = "not_a_number";
        let invalid_format_json = JSONNode::Number(invalid_number_format_str.to_string());
        assert!(i128::from_json(invalid_format_json.clone()).is_err());
        assert!(u128::from_json(invalid_format_json.clone()).is_err());
    }
}
