use std::collections::{BTreeMap, HashMap, BTreeSet, HashSet};
use std::hash::Hash; // Required for HashMap/HashSet keys if we were to use them directly

// --- JSON Node Definition ---

#[derive(Debug, Clone, PartialEq)]
pub enum JSONNode {
    Null,
    Bool(bool),
    Number(f64),    // For f32, f64, and smaller integers
    BigInt(String), // For u64, i64, u128, i128 (represented as strings)
    String(String),
    Array(Vec<JSONNode>),
    // For structs, this will be an array of (key, value) pairs.
    // For BTreeMap/HashMap, this will be an array of [key, value] arrays.
    Object(Vec<(String, JSONNode)>),
}

// --- JSONConvertible Trait ---

pub trait JSONConvertible: Sized {
    fn to_json(&self) -> JSONNode;
    fn from_json(node: &JSONNode) -> Result<Self, String>;
}

// --- Implementations for Primitives ---

impl JSONConvertible for () {
    fn to_json(&self) -> JSONNode {
        JSONNode::Null
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Null => Ok(()),
            _ => Err("Expected JSONNode::Null for ()".to_string()),
        }
    }
}

impl JSONConvertible for bool {
    fn to_json(&self) -> JSONNode {
        JSONNode::Bool(*self)
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Bool(b) => Ok(*b),
            _ => Err("Expected JSONNode::Bool for bool".to_string()),
        }
    }
}

macro_rules! impl_json_for_int_to_f64 {
    ($($t:ty),*) => {
        $(
            impl JSONConvertible for $t {
                fn to_json(&self) -> JSONNode {
                    JSONNode::Number(*self as f64)
                }
                fn from_json(node: &JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::Number(n) => {
                            if *n < <$t>::MIN as f64 || *n > <$t>::MAX as f64 {
                                Err(format!("Number {} out of range for {}", n, stringify!($t)))
                            } else {
                                Ok(*n as $t)
                            }
                        }
                        _ => Err(format!("Expected JSONNode::Number for {}", stringify!($t))),
                    }
                }
            }
        )*
    };
}

impl_json_for_int_to_f64!(u8, i8, u16, i16, u32, i32, usize, isize);

macro_rules! impl_json_for_big_int_to_string {
    ($($t:ty),*) => {
        $(
            impl JSONConvertible for $t {
                fn to_json(&self) -> JSONNode {
                    JSONNode::BigInt(self.to_string())
                }
                fn from_json(node: &JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::BigInt(s) => s.parse::<$t>().map_err(|e| format!("Failed to parse BigInt string '{}' as {}: {}", s, stringify!($t), e)),
                        JSONNode::String(s) => s.parse::<$t>().map_err(|e| format!("Failed to parse String '{}' as {}: {}", s, stringify!($t), e)), // Allow parsing from string too
                        _ => Err(format!("Expected JSONNode::BigInt or JSONNode::String for {}", stringify!($t))),
                    }
                }
            }
        )*
    };
}

impl_json_for_big_int_to_string!(u64, i64, u128, i128);


impl JSONConvertible for f32 {
    fn to_json(&self) -> JSONNode {
        JSONNode::Number(*self as f64)
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Number(n) => Ok(*n as f32),
            _ => Err("Expected JSONNode::Number for f32".to_string()),
        }
    }
}

impl JSONConvertible for f64 {
    fn to_json(&self) -> JSONNode {
        JSONNode::Number(*self)
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Number(n) => Ok(*n),
            _ => Err("Expected JSONNode::Number for f64".to_string()),
        }
    }
}

impl JSONConvertible for String {
    fn to_json(&self) -> JSONNode {
        JSONNode::String(self.clone())
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::String(s) => Ok(s.clone()),
            _ => Err("Expected JSONNode::String for String".to_string()),
        }
    }
}

// --- Implementations for Basic Collections ---

impl<T: JSONConvertible> JSONConvertible for Vec<T> {
    fn to_json(&self) -> JSONNode {
        JSONNode::Array(self.iter().map(|item| item.to_json()).collect())
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => arr.iter().map(T::from_json).collect(),
            _ => Err("Expected JSONNode::Array for Vec<T>".to_string()),
        }
    }
}

impl<T: JSONConvertible> JSONConvertible for Option<T> {
    fn to_json(&self) -> JSONNode {
        match self {
            Some(value) => value.to_json(),
            None => JSONNode::Null,
        }
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Null => Ok(None),
            _ => T::from_json(node).map(Some),
        }
    }
}

impl<T: JSONConvertible> JSONConvertible for Box<T> {
    fn to_json(&self) -> JSONNode {
        (**self).to_json()
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        T::from_json(node).map(Box::new)
    }
}

// BTreeMap<K, V> -> Array of [K_json, V_json] pairs
impl<K, V> JSONConvertible for BTreeMap<K, V>
where
    K: JSONConvertible + Ord, // Ord required by BTreeMap
    V: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let pairs: Vec<JSONNode> = self
            .iter()
            .map(|(k, v)| JSONNode::Array(vec![k.to_json(), v.to_json()]))
            .collect();
        JSONNode::Array(pairs)
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut map = BTreeMap::new();
                for pair_node in arr {
                    match pair_node {
                        JSONNode::Array(pair_vec) if pair_vec.len() == 2 => {
                            let key = K::from_json(&pair_vec[0])?;
                            let value = V::from_json(&pair_vec[1])?;
                            map.insert(key, value);
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

// HashMap<K, V> -> Array of [K_json, V_json] pairs
impl<K, V> JSONConvertible for HashMap<K, V>
where
    K: JSONConvertible + Eq + Hash, // Eq + Hash required by HashMap
    V: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let pairs: Vec<JSONNode> = self
            .iter()
            .map(|(k, v)| JSONNode::Array(vec![k.to_json(), v.to_json()]))
            .collect();
        // Note: Order is not guaranteed for HashMap serialization
        JSONNode::Array(pairs)
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut map = HashMap::new();
                for pair_node in arr {
                    match pair_node {
                        JSONNode::Array(pair_vec) if pair_vec.len() == 2 => {
                            let key = K::from_json(&pair_vec[0])?;
                            let value = V::from_json(&pair_vec[1])?;
                            map.insert(key, value);
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

// BTreeSet<T> -> Array of T_json
impl<T> JSONConvertible for BTreeSet<T>
where
    T: JSONConvertible + Ord, // Ord required by BTreeSet
{
    fn to_json(&self) -> JSONNode {
        JSONNode::Array(self.iter().map(|item| item.to_json()).collect())
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => arr.iter().map(T::from_json).collect(),
            _ => Err("Expected JSONNode::Array for BTreeSet<T>".to_string()),
        }
    }
}

// HashSet<T> -> Array of T_json
impl<T> JSONConvertible for HashSet<T>
where
    T: JSONConvertible + Eq + Hash, // Eq + Hash required by HashSet
{
    fn to_json(&self) -> JSONNode {
        // Note: Order is not guaranteed for HashSet serialization
        JSONNode::Array(self.iter().map(|item| item.to_json()).collect())
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => arr.iter().map(T::from_json).collect(),
            _ => Err("Expected JSONNode::Array for HashSet<T>".to_string()),
        }
    }
}

// --- Helper for serializing structs to JSONNode::Object ---
// Expects a slice of (field_name, field_json_node)
pub fn struct_to_json_object(fields: Vec<(&str, JSONNode)>) -> JSONNode {
    JSONNode::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

// --- Helper for deserializing structs from JSONNode::Object ---
// Provides a BTreeMap for easy lookup of field values by name.
pub fn json_object_to_btreemap(node: &JSONNode) -> Result<BTreeMap<String, JSONNode>, String> {
    match node {
        JSONNode::Object(obj_fields) => {
            let mut map = BTreeMap::new();
            for (name, value_node) in obj_fields {
                map.insert(name.clone(), value_node.clone());
            }
            Ok(map)
        }
        _ => Err("Expected JSONNode::Object to deserialize struct".to_string()),
    }
}

// --- Implementation for Vec<u8> ---
impl JSONConvertible for Vec<u8> {
    fn to_json(&self) -> JSONNode {
        // Represent Vec<u8> as a base64 string for compactness and to handle non-UTF8 bytes
        JSONNode::String(base64_sim::encode_bytes(self))
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::String(s) => base64_sim::decode_to_bytes(s)
                .map_err(|e| format!("Base64 decoding failed for Vec<u8>: {}", e)),
            _ => Err("Expected JSONNode::String (base64) for Vec<u8>".to_string()),
        }
    }
}

// --- Simple Base64-like encoding/decoding for Vec<u8> ---
// This is a very basic base64-like utility. For robust base64, a proper library would be better.
// This avoids adding an external dependency as per the request.
mod base64_sim {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode_bytes(bytes: &[u8]) -> String {
        let mut result = String::new();
        let mut iter = bytes.chunks_exact(3);

        for chunk in &mut iter {
            let b1 = chunk[0];
            let b2 = chunk[1];
            let b3 = chunk[2];

            result.push(CHARS[(b1 >> 2) as usize] as char);
            result.push(CHARS[(((b1 & 0x03) << 4) | (b2 >> 4)) as usize] as char);
            result.push(CHARS[(((b2 & 0x0F) << 2) | (b3 >> 6)) as usize] as char);
            result.push(CHARS[(b3 & 0x3F) as usize] as char);
        }

        let remainder = iter.remainder();
        if !remainder.is_empty() {
            let b1 = remainder[0];
            result.push(CHARS[(b1 >> 2) as usize] as char);
            if remainder.len() == 1 {
                result.push(CHARS[((b1 & 0x03) << 4) as usize] as char);
                result.push('=');
                result.push('=');
            } else { // remainder.len() == 2
                let b2 = remainder[1];
                result.push(CHARS[(((b1 & 0x03) << 4) | (b2 >> 4)) as usize] as char);
                result.push(CHARS[((b2 & 0x0F) << 2) as usize] as char);
                result.push('=');
            }
        }
        result
    }

    fn val(c: char) -> Result<u8, String> {
        match c {
            'A'..='Z' => Ok(c as u8 - b'A'),
            'a'..='z' => Ok(c as u8 - b'a' + 26),
            '0'..='9' => Ok(c as u8 - b'0' + 52),
            '+' => Ok(62),
            '/' => Ok(63),
            _ => Err(format!("Invalid base64 character: {}", c)),
        }
    }

    pub fn decode_to_bytes(s: &str) -> Result<Vec<u8>, String> {
        let mut result = Vec::new();
        let mut buf = [0u8; 4];
        let mut buf_idx = 0;
        let mut padding = 0;

        for char_c in s.chars() {
            if char_c == '=' {
                padding += 1;
                buf[buf_idx] = 0; // Placeholder, actual value doesn't matter for padding
            } else if padding > 0 {
                return Err("Invalid base64: data after padding".to_string());
            } else {
                buf[buf_idx] = val(char_c)?;
            }
            buf_idx += 1;

            if buf_idx == 4 {
                result.push((buf[0] << 2) | (buf[1] >> 4));
                if padding < 2 {
                    result.push(((buf[1] & 0x0F) << 4) | (buf[2] >> 2));
                }
                if padding < 1 {
                    result.push(((buf[2] & 0x03) << 6) | buf[3]);
                }
                buf_idx = 0;
            }
        }
        if buf_idx != 0 {
             return Err("Invalid base64: trailing characters".to_string());
        }
        Ok(result)
    }
}

// --- Implementation for BiBTreeMap<L, R> ---
// Serializes as an array of [L_json, R_json] pairs, like BTreeMap.
// For deserialization, it collects pairs and inserts them.
// Note: BiBTreeMap requires L and R to be Ord.
use bimap::BiBTreeMap;
impl<L, R> JSONConvertible for BiBTreeMap<L, R>
where
    L: JSONConvertible + Ord + Clone, // Clone needed for insertion during deserialization
    R: JSONConvertible + Ord + Clone, // Clone needed for insertion during deserialization
{
    fn to_json(&self) -> JSONNode {
        let pairs: Vec<JSONNode> = self
            .iter()
            .map(|(l, r)| JSONNode::Array(vec![l.to_json(), r.to_json()]))
            .collect();
        JSONNode::Array(pairs)
    }

    fn from_json(node: &JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut map = BiBTreeMap::new();
                for pair_node in arr {
                    match pair_node {
                        JSONNode::Array(pair_vec) if pair_vec.len() == 2 => {
                            let left = L::from_json(&pair_vec[0])?;
                            let right = R::from_json(&pair_vec[1])?;
                            // BiBTreeMap insert can fail if uniqueness is violated.
                            // For simplicity, we assume valid BiBTreeMap structure in JSON.
                            // A more robust impl might handle insert errors.
                            map.insert(left, right);
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

// --- Implementation for Arc<Mutex<T>> ---
use std::sync::{Arc, Mutex};
impl<T: JSONConvertible> JSONConvertible for Arc<Mutex<T>> {
    fn to_json(&self) -> JSONNode {
        let guard = self.lock().expect("Mutex poisoned during to_json");
        guard.to_json()
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        T::from_json(node).map(|val| Arc::new(Mutex::new(val)))
    }
}
