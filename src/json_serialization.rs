use std::collections::{BTreeMap, HashMap, BTreeSet, HashSet};
use std::hash::Hash;
use std::marker::Sized;
use bimap::BiBTreeMap;

// Import the derive macro
use json_convertible_derive::JSONConvertible;

// Added for arbitrary precision numbers
use bigdecimal::BigDecimal;
use num_traits::{ToPrimitive, FromPrimitive};
use std::str::FromStr;
use std::convert::TryFrom; // Needed for TryFrom<i128>

// --- JSONNode Enum ---
#[derive(Debug, Clone, PartialEq)]
pub enum JSONNode {
    Null,
    Bool(bool),
    Number(BigDecimal), // Store numbers as BigDecimal for precision
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

// Macro for integer types using BigDecimal
macro_rules! impl_json_for_integer {
    ($($t:ty),*) => {
        $(
            impl JSONConvertible for $t {
                fn to_json(&self) -> JSONNode {
                    JSONNode::Number(BigDecimal::from(*self))
                }
                fn from_json(node: JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::Number(n) => n
                            .to_i128() // Use i128 as intermediate for range checking
                            .and_then(|v| <$t>::try_from(v).ok()) // Try converting i128 to target type
                            .ok_or_else(|| format!("Cannot convert '{}' to {}", n, stringify!($t))),
                        _ => Err(format!("Expected JSONNode::Number(BigDecimal) for {}", stringify!($t))),
                    }
                }
            }
        )*
    };
}

impl_json_for_integer!(
    usize, isize, u8, u16, u32, u64,
    i8,  i16,  i32, i64,
    i128, u128
);

// Specific implementations for floating point types
impl JSONConvertible for f64 {
    fn to_json(&self) -> JSONNode {
        // Round-trip through a string representation to capture the exact bit pattern
        let s = format!("{:?}", self);
        let dec = BigDecimal::from_str(&s).expect("valid f64 debug string");
        JSONNode::Number(dec)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Number(n) =>
                n.to_f64().ok_or_else(|| format!("Cannot convert '{}' to f64", n)),
            _ => Err("Expected JSONNode::Number(BigDecimal) for f64".into()),
        }
    }
}

impl JSONConvertible for f32 {
    fn to_json(&self) -> JSONNode {
        // Round-trip through a string representation to capture the exact bit pattern
        let s = format!("{:?}", self);
        let dec = BigDecimal::from_str(&s).expect("valid f32 debug string");
        JSONNode::Number(dec)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Number(n) =>
                n.to_f32().ok_or_else(|| format!("Cannot convert '{}' to f32", n)),
            _ => Err("Expected JSONNode::Number(BigDecimal) for f32".into()),
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
    use bigdecimal::BigDecimal;
    use std::str::FromStr;

    // Example struct using the derive
    #[derive(Debug, Clone, PartialEq, JSONConvertible)]
    struct MyStruct {
        field1: i32,
        field2: String,
        optional_field: Option::<bool>,
        list_of_numbers: Vec::<u32>, // Uses generic Vec<T>
        byte_buffer: Vec::<u8>,      // Uses generic Vec + specific u8
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
        //   "byte_buffer": [10, 20, 30], // Now stores BigDecimal
        //   "field1": 42, // Now stores BigDecimal
        //   "field2": "hello",
        //   "list_of_numbers": [1, 2, 3], // Now stores BigDecimal
        //   "optional_field": true
        // }

        if let JSONNode::Object(obj) = &json_node {
            assert_eq!(obj.get("field1"), Some(&JSONNode::Number(BigDecimal::from(42))));
            assert_eq!(obj.get("field2"), Some(&JSONNode::String("hello".to_string())));
            assert_eq!(obj.get("optional_field"), Some(&JSONNode::Bool(true)));
            if let Some(JSONNode::Array(arr)) = obj.get("list_of_numbers") {
                 assert_eq!(arr, &vec![
                     JSONNode::Number(BigDecimal::from(1)),
                     JSONNode::Number(BigDecimal::from(2)),
                     JSONNode::Number(BigDecimal::from(3))
                 ]);
            } else {
                panic!("list_of_numbers not found or not an array");
            }
            if let Some(JSONNode::Array(arr)) = obj.get("byte_buffer") {
                 assert_eq!(arr, &vec![
                     JSONNode::Number(BigDecimal::from(10)),
                     JSONNode::Number(BigDecimal::from(20)),
                     JSONNode::Number(BigDecimal::from(30))
                 ]);
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
        // Expected JSON structure representation within JSONNode: { "byte_buffer": [], "field1": 1, "field2": "world", "list_of_numbers": [], "optional_field": null }
        if let JSONNode::Object(obj) = &json_node {
             assert_eq!(obj.get("optional_field"), Some(&JSONNode::Null));
             assert_eq!(obj.get("field1"), Some(&JSONNode::Number(BigDecimal::from(1))));
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
        //   "item_t": 123, // Now stores BigDecimal
        //   "item_u": "test_string"
        // }
        if let JSONNode::Object(obj) = &json_node {
             assert_eq!(obj.get("item_t"), Some(&JSONNode::Number(BigDecimal::from(123))));
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
        // Expected JSON structure representation within JSONNode: [["a_key", 10], ["b_key", 20]] (sorted by key, numbers are BigDecimal)
        match json_node {
            JSONNode::Array(pairs) => {
                assert_eq!(pairs.len(), 2);
                // Check first pair
                match &pairs[0] {
                    JSONNode::Array(pair1) => {
                        assert_eq!(pair1[0], JSONNode::String("a_key".to_string()));
                        assert_eq!(pair1[1], JSONNode::Number(BigDecimal::from(10)));
                    }
                    _ => panic!("Expected array for pair"),
                }
                // Check second pair
                match &pairs[1] {
                    JSONNode::Array(pair2) => {
                        assert_eq!(pair2[0], JSONNode::String("b_key".to_string()));
                        assert_eq!(pair2[1], JSONNode::Number(BigDecimal::from(20)));
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
    fn test_vec_u8_serialization_deserialization() {
        let data: Vec<u8> = vec![1, 2, 255];
        let json_node = data.to_json();
        // Expected JSON structure representation within JSONNode: [1, 2, 255] (BigDecimal)
        match json_node {
            JSONNode::Array(ref nums) => {
                assert_eq!(nums.len(), 3);
                assert_eq!(nums[0], JSONNode::Number(BigDecimal::from(1)));
                assert_eq!(nums[1], JSONNode::Number(BigDecimal::from(2)));
                assert_eq!(nums[2], JSONNode::Number(BigDecimal::from(255)));
            }
            _ => panic!("Expected JSONNode::Array for Vec<u8>"),
        }
        let deserialized: Vec<u8> = Vec::from_json(json_node).unwrap();
        assert_eq!(data, deserialized);

        // Test invalid u8 from BigDecimal
        let invalid_json_too_large = JSONNode::Array(vec![JSONNode::Number(BigDecimal::from(256))]);
        assert!(Vec::<u8>::from_json(invalid_json_too_large).is_err());
        let invalid_json_negative = JSONNode::Array(vec![JSONNode::Number(BigDecimal::from(-1))]);
        assert!(Vec::<u8>::from_json(invalid_json_negative).is_err());
         let invalid_json_fractional = JSONNode::Array(vec![JSONNode::Number(BigDecimal::from_str("10.5").unwrap())]);
        assert!(Vec::<u8>::from_json(invalid_json_fractional).is_err());

        // Test deserializing a non-Number node
        let invalid_json_text = JSONNode::Array(vec![JSONNode::String("abc".to_string())]);
        assert!(Vec::<u8>::from_json(invalid_json_text).is_err());
         let invalid_json_bool = JSONNode::Array(vec![JSONNode::Bool(true)]);
        assert!(Vec::<u8>::from_json(invalid_json_bool).is_err());
    }

     #[test]
     fn test_f64_precision() {
        let val: f64 = 0.1 + 0.2; // This often results in a value slightly different from 0.3 in f64
        let val_str = format!("{:?}", val); // Use debug formatting to see the exact f64 value
        let json_node = val.to_json(); // This will create BigDecimal from val_str

        match json_node {
            JSONNode::Number(n) => {
                // The BigDecimal representation in the JSONNode should match the exact f64 value's string representation
                 assert_eq!(n, BigDecimal::from_str(&val_str).unwrap());

                 let deserialized: f64 = f64::from_json(JSONNode::Number(n.clone())).unwrap();
                 // Direct f64 equality check might fail for floating point inaccuracies
                 // assert_eq!(val, deserialized); // This could fail

                 // A better check is to compare their string representations after deserialization
                 assert_eq!(format!("{:?}", val), format!("{:?}", deserialized));

            }
            _ => panic!("Expected JSONNode::Number(BigDecimal)"),
        }

         let large_int_f64 = 9007199254740992.0_f64; // An integer within f64 range but beyond i64::MAX
         let large_int_f64_str = format!("{:?}", large_int_f64); // Debug format might show .0
         let json_node_large = large_int_f64.to_json(); // Creates BigDecimal from this string

         match json_node_large {
             JSONNode::Number(n) => {
                 assert_eq!(n, BigDecimal::from_str(&large_int_f64_str).unwrap());
                 let deserialized: f64 = f64::from_json(JSONNode::Number(n)).unwrap();
                 assert_eq!(format!("{:?}", large_int_f64), format!("{:?}", deserialized));
             }
             _ => panic!("Expected JSONNode::Number(BigDecimal)"),
         }

         // Test a number that requires more precision than f64 offers
         let high_precision_str = "0.1234567890123456789"; // More decimal places than f64 can precisely store
         let high_precision_decimal = BigDecimal::from_str(high_precision_str).unwrap();
         let json_node_hp = JSONNode::Number(high_precision_decimal.clone());

         // The JSONNode holds the precise BigDecimal:
         match json_node_hp {
             JSONNode::Number(ref n) => {
                 assert_eq!(n, &high_precision_decimal);
             }
             _ => panic!("Expected JSONNode::Number(BigDecimal)"),
         }

         // Deserializing this back into f64 WILL lose precision
         let deserialized_hp: f64 = f64::from_json(json_node_hp.clone()).unwrap();
         // The deserialized f64 will not be exactly equal to the original high_precision_str when formatted back
         assert_ne!(format!("{:?}", deserialized_hp), high_precision_str); // This is expected

         // Test deserializing non-numeric JSONNode into f64
         assert!(f64::from_json(JSONNode::String("1.0".to_string())).is_err());
         assert!(f64::from_json(JSONNode::Null).is_err());
     }

    #[test]
    fn test_i128_u128_serialization_deserialization() {
        let large_i128: i128 = 12345678901234567890123456789012345_i128;
        let large_u128: u128 = 98765432109876543210987654321098765_u128;

        let json_i128 = large_i128.to_json();
        let json_u128 = large_u128.to_json();

        match json_i128 {
            JSONNode::Number(n) => assert_eq!(n, BigDecimal::from(large_i128)),
            _ => panic!("Expected JSONNode::Number(BigDecimal) for i128"),
        }

        match json_u128 {
            JSONNode::Number(n) => assert_eq!(n, BigDecimal::from(large_u128)),
            _ => panic!("Expected JSONNode::Number(BigDecimal) for u128"),
        }

        let deserialized_i128 = i128::from_json(json_i128).unwrap();
        let deserialized_u128 = u128::from_json(json_u128).unwrap();

        assert_eq!(large_i128, deserialized_i128);
        assert_eq!(large_u128, deserialized_u128);

        // Test deserializing an out-of-range BigDecimal for i128/u128
        let too_large_i128_bd = BigDecimal::from_str("1701411834604692317316873037158841057280").unwrap(); // i128::MAX + 1
        let invalid_i128_json = JSONNode::Number(too_large_i128_bd);
        assert!(i128::from_json(invalid_i128_json).is_err());

         let too_large_u128_bd = BigDecimal::from_str("3402823669209384634633746074317682114560").unwrap(); // u128::MAX + 1
        let invalid_u128_json = JSONNode::Number(too_large_u128_bd);
        assert!(u128::from_json(invalid_u128_json).is_err());

         let invalid_negative_u128_bd = BigDecimal::from(-1);
        let invalid_negative_u128_json = JSONNode::Number(invalid_negative_u128_bd);
        assert!(u128::from_json(invalid_negative_u128_json).is_err());


        // Test deserializing non-Number node
        let invalid_format_json_string = JSONNode::String("123".to_string());
        assert!(i128::from_json(invalid_format_json_string.clone()).is_err());
        assert!(u128::from_json(invalid_format_json_string).is_err());

         let invalid_format_json_bool = JSONNode::Bool(false);
         assert!(i128::from_json(invalid_format_json_bool.clone()).is_err());
        assert!(u128::from_json(invalid_format_json_bool).is_err());

         let invalid_format_json_null = JSONNode::Null;
         assert!(i128::from_json(invalid_format_json_null.clone()).is_err());
        assert!(u128::from_json(invalid_format_json_null).is_err());
    }
}
