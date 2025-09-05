use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::marker::Sized;
use bimap::BiBTreeMap;

// Import the derive macro
use json_convertible_derive::JSONConvertible;

// Add these lines for serde_json
use serde_json::Value as SerdeValue;
use serde_json::Map as SerdeMap; // BTreeMap<String, SerdeValue> is SerdeMap
use std::convert::TryInto;
use std::io::{Read, Write};
use kdam::BarExt;
// Added for streaming

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
    Object(BTreeMap<String, JSONNode>), // BTreeMap for sorted keys
}

impl JSONNode {
    // Kind name of this node (concise)
    pub fn kind(&self) -> &'static str {
        match self {
            JSONNode::Null => "Null",
            JSONNode::Bool(_) => "Bool",
            JSONNode::Int(_) => "Int",
            JSONNode::UInt(_) => "UInt",
            JSONNode::Float(_) => "Float",
            JSONNode::String(_) => "String",
            JSONNode::Array(_) => "Array",
            JSONNode::Object(_) => "Object",
        }
    }

    // Short preview with a default max length budget for strings/keys
    pub fn short_preview(&self) -> String {
        self.short_preview_limit(40)
    }

    // Short, human-focused preview that avoids dumping large payloads
    pub fn short_preview_limit(&self, max_len: usize) -> String {
        match self {
            JSONNode::Null => "Null".to_string(),
            JSONNode::Bool(b) => format!("Bool({})", b),
            JSONNode::Int(i) => format!("Int({})", i),
            JSONNode::UInt(u) => format!("UInt({})", u),
            JSONNode::Float(f) => {
                if f.is_nan() {
                    "Float(NaN)".to_string()
                } else if f.is_infinite() {
                    if f.is_sign_positive() { "Float(+inf)".to_string() } else { "Float(-inf)".to_string() }
                } else {
                    format!("Float({})", f)
                }
            }
            JSONNode::String(s) => {
                let len = s.len();
                let mut preview: String = s.chars().take(max_len).collect();
                if s.chars().count() > max_len {
                    preview.push('…');
                }
                format!("String(len={}, \"{}\")", len, preview)
            }
            JSONNode::Array(arr) => format!("Array(len={})", arr.len()),
            JSONNode::Object(obj) => {
                let len = obj.len();
                let mut keys = Vec::new();
                for k in obj.keys().take(3) {
                    let mut p: String = k.chars().take(max_len).collect();
                    if k.chars().count() > max_len { p.push('…'); }
                    keys.push(format!("\"{}\"", p));
                }
                if keys.is_empty() {
                    format!("Object(len={})", len)
                } else {
                    let extra = if len > 3 { ", …" } else { "" };
                    format!("Object(len={}, keys=[{}{}])", len, keys.join(", "), extra)
                }
            }
        }
    }

    // New method to convert JSONNode to serde_json::Value
    pub fn to_serde_value(&self) -> SerdeValue {
        match self {
            JSONNode::Null => SerdeValue::Null,
            JSONNode::Bool(b) => SerdeValue::Bool(*b),
            JSONNode::Int(i) => SerdeValue::Number(serde_json::Number::from_i128(*i).expect(format!("Int {} out of range for serde_json::Value", i).as_str())),
            JSONNode::UInt(u) => SerdeValue::Number(serde_json::Number::from_u128(*u).expect(format!("UInt {} out of range for serde_json::Value", u).as_str())),
            JSONNode::Float(f) => {
                // serde_json::Number::from_f64 returns None for NaN/Infinity
                // We'll convert such cases to SerdeValue::Null, a common practice.
                serde_json::Number::from_f64(*f)
                    .map(SerdeValue::Number)
                    .unwrap_or(SerdeValue::Null)
            }
            JSONNode::String(s) => SerdeValue::String(s.clone()),
            JSONNode::Array(arr) => {
                SerdeValue::Array(arr.iter().map(|node| node.to_serde_value()).collect())
            }
            JSONNode::Object(obj) => {
                let mut map = SerdeMap::new();
                for (k, v) in obj {
                    map.insert(k.clone(), v.to_serde_value());
                }
                SerdeValue::Object(map)
            }
        }
    }

    // New method to convert serde_json::Value to JSONNode
    pub fn from_serde_value(s_val: SerdeValue) -> Result<Self, String> {
        match s_val {
            SerdeValue::Null => Ok(JSONNode::Null),
            SerdeValue::Bool(b) => Ok(JSONNode::Bool(b)),
            SerdeValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(JSONNode::Int(i as i128))
                } else if let Some(u) = n.as_u64() {
                    Ok(JSONNode::UInt(u as u128))
                } else if let Some(f) = n.as_f64() {
                    Ok(JSONNode::Float(f))
                } else {
                    Err(format!("Unsupported number type in serde_json::Value: {}", n))
                }
            }
            SerdeValue::String(s) => Ok(JSONNode::String(s)),
            SerdeValue::Array(arr) => {
                let mut nodes = Vec::with_capacity(arr.len());
                for val in arr {
                    nodes.push(Self::from_serde_value(val)?);
                }
                Ok(JSONNode::Array(nodes))
            }
            SerdeValue::Object(obj_map) => {
                let mut btree_map = BTreeMap::new();
                for (k, v) in obj_map {
                    btree_map.insert(k, Self::from_serde_value(v)?);
                }
                Ok(JSONNode::Object(btree_map))
            }
        }
    }

    // Update to_json_string to use serde_json
    pub fn to_json_string(&self) -> String {
        let serde_value = self.to_serde_value();
        // serde_json::to_string can fail, though less likely for SerdeValue -> String
        // For simplicity here, unwrap, but consider error handling for production
        serde_json::to_string(&serde_value).unwrap_or_else(|e| {
            // Fallback or panic for critical error
            eprintln!("Critical error: Failed to serialize SerdeValue to string: {}", e);
            // A minimal JSON representation of an error or "null"
            "{\"error\":\"serialization_failed\"}".to_string()
        })
    }

    // Update from_json_string to use serde_json
    pub fn from_json_string(s: &str) -> Result<JSONNode, String> {
        let serde_value: SerdeValue = serde_json::from_str(s)
            .map_err(|e| format!("Failed to parse JSON string with serde_json: {}", e))?;
        Self::from_serde_value(serde_value)
    }

    // New method to write JSONNode directly to a writer
    pub fn to_writer<W: Write>(&self, writer: W) -> Result<(), String> {
        let serde_value = self.to_serde_value();
        serde_json::to_writer(writer, &serde_value)
            .map_err(|e| format!("Failed to write JSONNode to writer: {}", e))
    }

    // New method to read JSONNode directly from a reader
    pub fn from_json_reader<R: Read>(reader: R) -> Result<JSONNode, String> {
        let serde_value: SerdeValue = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to read JSONNode from reader: {}", e))?;
        Self::from_serde_value(serde_value)
    }

    pub fn into_object(self) -> Result<BTreeMap<String, JSONNode>, String> {
        match self {
            JSONNode::Object(obj) => Ok(obj),
            other => Err(format!(
                "Expected JSONNode::Object, got {}",
                other.short_preview()
            )),
        }
    }
}

// --- JSONConvertible Trait ---
pub trait JSONConvertible: Sized {
    fn to_json(&self) -> JSONNode;
    fn from_json(node: JSONNode) -> Result<Self, String>;

    // Default implementation for streaming serialization
    fn to_writer<W: Write>(&self, writer: W) -> Result<(), String> {
        self.to_json().to_writer(writer)
    }

    // Default implementation for streaming deserialization
    fn from_json_reader<R: Read>(reader: R) -> Result<Self, String> {
        JSONNode::from_json_reader(reader).and_then(Self::from_json)
    }
}

// --- Implementations for Primitives ---

impl JSONConvertible for () {
    fn to_json(&self) -> JSONNode { JSONNode::Null }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Null => Ok(()),
            other => Err(format!(
                "Expected JSONNode::Null for unit type (), got {}",
                other.short_preview()
            )),
        }
    }
}

macro_rules! impl_json_for_tuple {
    ( $($T:ident : $idx:tt),+ ) => {
        impl<$($T: JSONConvertible),+> JSONConvertible for ($($T,)+) {
            fn to_json(&self) -> JSONNode {
                JSONNode::Array(vec![
                    $(self.$idx.to_json()),+
                ])
            }

            fn from_json(node: JSONNode) -> Result<Self, String> {
                match node {
                    JSONNode::Array(arr) => {
                        const N: usize = [$(stringify!($T)),+].len();
                        if arr.len() != N {
                            return Err(format!(
                                "Expected JSONNode::Array with {} elements for tuple, got {}",
                                N,
                                arr.len()
                            ));
                        }
                        Ok(($(
                            $T::from_json(arr[$idx].clone())
                                .map_err(|e| {
                                    format!("At $[{}]: {}", $idx, e)
                                })?
                        ,)+))
                    }
                    other => Err(format!("Expected JSON array for tuple, got {}", other.short_preview())),
                }
            }

            // Implement to_writer for tuples
            fn to_writer<W: Write>(&self, mut writer: W) -> Result<(), String> {
                write!(writer, "[").map_err(|e| e.to_string())?;
                let mut first = true;
                $(
                    if !first {
                        write!(writer, ",").map_err(|e| e.to_string())?;
                    }
                    self.$idx.to_writer(&mut writer)?;
                    first = false;
                )+
                write!(writer, "]").map_err(|e| e.to_string())?;
                Ok(())
            }

            // Implement from_json_reader for tuples
            fn from_json_reader<R: Read>(reader: R) -> Result<Self, String> {
                // This is more complex for tuples directly from reader without an intermediate JSONNode.
                // For now, use the default that goes via JSONNode.
                JSONNode::from_json_reader(reader).and_then(Self::from_json)
            }
        }
    };
}

impl_json_for_tuple!(T0:0);
impl_json_for_tuple!(T0:0, T1:1);
impl_json_for_tuple!(T0:0, T1:1, T2:2);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3, T4:4);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7, T8:8);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7, T8:8, T9:9);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7, T8:8, T9:9, T10:10);
impl_json_for_tuple!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7, T8:8, T9:9, T10:10, T11:11);

impl JSONConvertible for bool {
    fn to_json(&self) -> JSONNode { JSONNode::Bool(*self) }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Bool(b) => Ok(b),
            other => Err(format!(
                "Expected JSON boolean for bool, got {}",
                other.short_preview()
            )),
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
                        other => Err(format!(
                            "Expected JSON integer (Int/UInt) for {}, got {}",
                            stringify!($t), other.short_preview()
                        )),
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
                        other => Err(format!(
                            "Expected JSON unsigned integer (UInt) or non-negative Int for {}, got {}",
                            stringify!($t), other.short_preview()
                        )),
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
                        other => Err(format!(
                            "Expected JSON number (Float/Int/UInt) for {}, got {}",
                            stringify!($t), other.short_preview()
                        )),
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
            other => Err(format!(
                "Expected JSON string for String, got {}",
                other.short_preview()
            )),
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
            JSONNode::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for (i, item) in arr.into_iter().enumerate() {
                    match T::from_json(item) {
                        Ok(v) => out.push(v),
                        Err(e) => return Err(format!("While deserializing Vec<T> at $[{}]: {}", i, e)),
                    }
                }
                Ok(out)
            }
            other => Err(format!(
                "Expected JSON array for Vec<T>, got {}",
                other.short_preview()
            )),
        }
    }
    fn to_writer<W: Write>(&self, mut writer: W) -> Result<(), String> {
        write!(writer, "[").map_err(|e| e.to_string())?;
        let mut first = true;
        for item in self {
            if !first {
                write!(writer, ",").map_err(|e| e.to_string())?;
            }
            item.to_writer(&mut writer)?;
            first = false;
        }
        write!(writer, "]").map_err(|e| e.to_string())?;
        Ok(())
    }

    fn from_json_reader<R: Read>(reader: R) -> Result<Self, String> {
        JSONNode::from_json_reader(reader).and_then(Self::from_json)
    }
}

// Generic array
impl<T: JSONConvertible, const N: usize> JSONConvertible for [T; N] {
    fn to_json(&self) -> JSONNode {
        // Convert array elements to JSONNode and wrap in a JSON array
        JSONNode::Array(self.iter().map(|item| item.to_json()).collect())
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                if arr.len() != N {
                    return Err(format!(
                        "Expected array of length {} for [T; N], got length {}",
                        N, arr.len()
                    ));
                }
                // Convert with path-aware errors
                let mut vec: Vec<T> = Vec::with_capacity(N);
                for (i, v) in arr.into_iter().enumerate() {
                    match T::from_json(v) {
                        Ok(val) => vec.push(val),
                        Err(e) => return Err(format!("While deserializing [T; {}] at $[{}]: {}", N, i, e)),
                    }
                }
                // Then try to convert Vec<T> into [T; N]; error if length mismatch
                vec.try_into()
                    .map_err(|v: Vec<T>| {
                        format!(
                            "Expected array of length {} for [T; N], got length {}",
                            N,
                            v.len()
                        )
                    })
            }
            other => Err(format!(
                "Expected JSON array for [T; {}], got {}",
                N, other.short_preview()
            )),
        }
    }
}


impl<T: JSONConvertible + Ord> JSONConvertible for BTreeSet<T> {
    fn to_json(&self) -> JSONNode {
        JSONNode::Array(self.iter().map(|item| item.to_json()).collect())
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut set = BTreeSet::new();
                for (i, item) in arr.into_iter().enumerate() {
                    match T::from_json(item) {
                        Ok(v) => { set.insert(v); }
                        Err(e) => return Err(format!("While deserializing BTreeSet<T> at $[{}]: {}", i, e)),
                    }
                }
                Ok(set)
            }
            other => Err(format!(
                "Expected JSON array for BTreeSet<T>, got {}",
                other.short_preview()
            )),
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
            JSONNode::Array(arr) => {
                let mut set = HashSet::new();
                for (i, item) in arr.into_iter().enumerate() {
                    match T::from_json(item) {
                        Ok(v) => { set.insert(v); }
                        Err(e) => return Err(format!("While deserializing HashSet<T> at $[{}]: {}", i, e)),
                    }
                }
                Ok(set)
            }
            other => Err(format!(
                "Expected JSON array for HashSet<T>, got {}",
                other.short_preview()
            )),
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
                for (i, pair_node) in arr.into_iter().enumerate() {
                    match pair_node {
                        JSONNode::Array(mut pair_vec) if pair_vec.len() == 2 => {
                            let v_node = pair_vec.pop().unwrap();
                            let k_node = pair_vec.pop().unwrap();
                            let key = K::from_json(k_node)
                                .map_err(|e| format!("While deserializing BTreeMap<K, V> at $[{}][0] (key): {}", i, e))?;
                            let val = V::from_json(v_node)
                                .map_err(|e| format!("While deserializing BTreeMap<K, V> at $[{}][1] (value): {}", i, e))?;
                            map.insert(key, val);
                        }
                        other => return Err(format!(
                            "Expected 2-element array for BTreeMap entry at $[{}], got {}",
                            i,
                            other.short_preview()
                        )),
                    }
                }
                Ok(map)
            }
            other => Err(format!(
                "Expected JSON array of [key, value] pairs for BTreeMap<K, V>, got {}",
                other.short_preview()
            )),
        }
    }

    fn to_writer<W: Write>(&self, mut writer: W) -> Result<(), String> {
        write!(writer, "[").map_err(|e| e.to_string())?;
        let mut first = true;
        for (k, v) in self {
            if !first {
                write!(writer, ",").map_err(|e| e.to_string())?;
            }
            write!(writer, "[").map_err(|e| e.to_string())?;
            k.to_writer(&mut writer)?;
            write!(writer, ",").map_err(|e| e.to_string())?;
            v.to_writer(&mut writer)?;
            write!(writer, "]").map_err(|e| e.to_string())?;
            first = false;
        }
        write!(writer, "]").map_err(|e| e.to_string())?;
        Ok(())
    }

    fn from_json_reader<R: Read>(mut reader: R) -> Result<Self, String> {
        // This is a simplified streaming reader for BTreeMap assuming it's an array of [key, value] pairs.
        // It's not a full JSON parser, but sufficient for the expected format.
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        let json_node = JSONNode::from_json_string(&String::from_utf8(buf).map_err(|e| e.to_string())?)?;
        Self::from_json(json_node)
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
                for (i, pair_node) in arr.into_iter().enumerate() {
                    match pair_node {
                        JSONNode::Array(mut pair_vec) if pair_vec.len() == 2 => {
                            let v_node = pair_vec.pop().unwrap();
                            let k_node = pair_vec.pop().unwrap();
                            let key = K::from_json(k_node)
                                .map_err(|e| format!("While deserializing HashMap<K, V> at $[{}][0] (key): {}", i, e))?;
                            let val = V::from_json(v_node)
                                .map_err(|e| format!("While deserializing HashMap<K, V> at $[{}][1] (value): {}", i, e))?;
                            map.insert(key, val);
                        }
                        other => return Err(format!(
                            "Expected 2-element array for HashMap entry at $[{}], got {}",
                            i, other.short_preview()
                        )),
                    }
                }
                Ok(map)
            }
            other => Err(format!(
                "Expected JSON array of [key, value] pairs for HashMap<K, V>, got {}",
                other.short_preview()
            )),
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
                for (i, pair_node) in arr.into_iter().enumerate() {
                    match pair_node {
                        JSONNode::Array(mut pair_vec) if pair_vec.len() == 2 => {
                            let r_node = pair_vec.pop().unwrap();
                            let l_node = pair_vec.pop().unwrap();
                            let l = L::from_json(l_node)
                                .map_err(|e| format!("While deserializing BiBTreeMap<L, R> at $[{}][0] (left/key): {}", i, e))?;
                            let r = R::from_json(r_node)
                                .map_err(|e| format!("While deserializing BiBTreeMap<L, R> at $[{}][1] (right/value): {}", i, e))?;
                            map.insert(l, r);
                        }
                        other => return Err(format!(
                            "Expected 2-element array for BiBTreeMap entry at $[{}], got {}",
                            i, other.short_preview()
                        )),
                    }
                }
                Ok(map)
            }
            other => Err(format!(
                "Expected JSON array of [left, right] pairs for BiBTreeMap<L, R>, got {}",
                other.short_preview()
            )),
        }
    }
}

// --- Tests (optional, but good for verifying) ---
#[cfg(test)]
mod tests {
    use super::*; // Imports JSONNode, JSONConvertible, MyStruct, etc.
    use std::io::Cursor;
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

    #[test]
    fn test_json_string_conversion() {
        let original = MyStruct {
            field1: 42,
            field2: "hello \"world\" \\ / \\b \\f \n \r \t".to_string(),
            optional_field: Some(true),
            list_of_numbers: vec![1, 2, 3],
            byte_buffer: vec![10, 20, 30],
        };

        let json_node_via_trait = original.to_json();
        let json_string = json_node_via_trait.to_json_string();

        // Expected string from serde_json (keys will be sorted due to BTreeMap in JSONNode::Object
        // and SerdeMap in serde_json::Value::Object if built from a sorted iterator,
        // or if serde_json::to_string sorts them by default for `Map`).
        // serde_json sorts object keys by default when serializing.
        let expected_json_string = r#"{"byte_buffer":[10,20,30],"field1":42,"field2":"hello \"world\" \\ / \b \f \n \r \t","list_of_numbers":[1,2,3],"optional_field":true}"#;

        // We can parse the expected string and our generated string to SerdeValue and compare them
        // to avoid issues with exact string formatting (e.g. spacing if pretty print was used).
        let serde_val_generated: SerdeValue = serde_json::from_str(&json_string).unwrap();
        let serde_val_expected: SerdeValue = serde_json::from_str(expected_json_string).unwrap();

        assert_eq!(serde_val_generated, serde_val_expected);

        // Test deserialization from string
        let parsed_node = JSONNode::from_json_string(&json_string).expect("Failed to parse JSON string back to JSONNode");
        assert_eq!(json_node_via_trait, parsed_node);

        let deserialized_struct = MyStruct::from_json(parsed_node).expect("Deserialization from parsed_node failed");
        assert_eq!(original, deserialized_struct);
    }

    #[test]
    fn test_non_finite_float_serialization() {
        let node_nan = JSONNode::Float(f64::NAN);
        assert_eq!(node_nan.to_json_string(), "null");

        let node_inf = JSONNode::Float(f64::INFINITY);
        assert_eq!(node_inf.to_json_string(), "null");

        let node_neg_inf = JSONNode::Float(f64::NEG_INFINITY);
        assert_eq!(node_neg_inf.to_json_string(), "null");

        // Test deserialization of null back to float (will become 0.0 or error based on from_serde_value)
        // Current from_serde_value for SerdeValue::Null will become JSONNode::Null.
        // If you want null in JSON to become a specific float (e.g. 0.0 or NaN),
        // you'd adjust JSONConvertible for Option<f64> or f64 itself.
        // Here, we are testing JSONNode::from_json_string directly.
        let parsed_nan_node = JSONNode::from_json_string("null").unwrap();
        assert_eq!(parsed_nan_node, JSONNode::Null); // serde_json parses "null" to SerdeValue::Null
                                                     // then from_serde_value converts SerdeValue::Null to JSONNode::Null
    }

    #[test]
    fn test_tuple_serialization() {
        let original: (i32, String, bool) = (123, "hello".to_string(), true);
        let json_node = original.to_json();
        assert_eq!(
            json_node,
            JSONNode::Array(vec![
                JSONNode::Int(123),
                JSONNode::String("hello".to_string()),
                JSONNode::Bool(true)
            ])
        );

        let deserialized = <(i32, String, bool)>::from_json(json_node).unwrap();
        assert_eq!(original, deserialized);

        // Test single-element tuple
        let single_tuple: (i32,) = (42,);
        let json_node_single = single_tuple.to_json();
        assert_eq!(json_node_single, JSONNode::Array(vec![JSONNode::Int(42)]));
        let deserialized_single = <(i32,)>::from_json(json_node_single).unwrap();
        assert_eq!(single_tuple, deserialized_single);

        // Test wrong number of elements for deserialization
        let wrong_node = JSONNode::Array(vec![JSONNode::Int(1), JSONNode::Int(2)]);
        assert!(<(i32, String, bool)>::from_json(wrong_node).is_err());
    }

    #[test]
    fn test_streaming_serialization_deserialization() {
        let original = MyStruct {
            field1: 42,
            field2: "streaming test".to_string(),
            optional_field: Some(false),
            list_of_numbers: vec![10, 20],
            byte_buffer: vec![1, 2, 3],
        };

        let mut buffer = Vec::new();
        original.to_writer(&mut buffer).unwrap();

        let deserialized = MyStruct::from_json_reader(Cursor::new(buffer)).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_streaming_btreemap() {
        let mut original_map = BTreeMap::new();
        original_map.insert("key1".to_string(), 100);
        original_map.insert("key2".to_string(), 200);

        let mut buffer = Vec::new();
        original_map.to_writer(&mut buffer).unwrap();

        let deserialized_map: BTreeMap<String, i32> = BTreeMap::from_json_reader(Cursor::new(buffer)).unwrap();
        assert_eq!(original_map, deserialized_map);
    }

    #[test]
    fn test_streaming_vec() {
        let original_vec = vec![1, 2, 3];
        let mut buffer = Vec::new();
        original_vec.to_writer(&mut buffer).unwrap();

        let deserialized_vec: Vec<i32> = Vec::from_json_reader(Cursor::new(buffer)).unwrap();
        assert_eq!(original_vec, deserialized_vec);
    }
}

