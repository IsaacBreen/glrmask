use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
use serde_json::Map as SerdeMap;
use serde_json::Value as SerdeValue;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::convert::TryInto;
use std::hash::Hash;
use std::io::{Read, Write};

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

impl JSONNode {
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

    pub fn short_preview(&self) -> String {
        self.short_preview_limit(40)
    }

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
                    if f.is_sign_positive() {
                        "Float(+inf)".to_string()
                    } else {
                        "Float(-inf)".to_string()
                    }
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
                    if k.chars().count() > max_len {
                        p.push('…');
                    }
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

    pub fn to_serde_value(&self) -> SerdeValue {
        match self {
            JSONNode::Null => SerdeValue::Null,
            JSONNode::Bool(b) => SerdeValue::Bool(*b),
            JSONNode::Int(i) => serde_json::Number::from_i128(*i)
                .map(SerdeValue::Number)
                .unwrap_or_else(|| {
                    panic!("Int {} out of range for serde_json::Value::Number", i)
                }),
            JSONNode::UInt(u) => serde_json::Number::from_u128(*u)
                .map(SerdeValue::Number)
                .unwrap_or_else(|| {
                    panic!("UInt {} out of range for serde_json::Value::Number", u)
                }),
            JSONNode::Float(f) => serde_json::Number::from_f64(*f)
                .map(SerdeValue::Number)
                .unwrap_or(SerdeValue::Null),
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

    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            eprintln!("Critical error: failed to serialize JSONNode: {e}");
            "{\"error\":\"serialization_failed\"}".to_string()
        })
    }

    pub fn from_json_string(s: &str) -> Result<JSONNode, String> {
        serde_json::from_str(s).map_err(|e| format!("Failed to parse JSON string: {e}"))
    }

    pub fn to_writer<W: Write>(&self, writer: W) -> Result<(), String> {
        serde_json::to_writer(writer, self)
            .map_err(|e| format!("Failed to write JSONNode to writer: {e}"))
    }

    pub fn from_reader<R: Read>(reader: R) -> Result<JSONNode, String> {
        serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to read JSONNode from reader: {e}"))
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

    fn to_writer<W: Write>(&self, writer: W) -> Result<(), String> {
        self.to_json().to_writer(writer)
    }

    fn from_reader<R: Read>(reader: R) -> Result<Self, String> {
        JSONNode::from_reader(reader).and_then(Self::from_json)
    }
}

// --- Implementations for Primitives ---

impl JSONConvertible for () {
    fn to_json(&self) -> JSONNode {
        JSONNode::Null
    }
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
                JSONNode::Array(vec![$(self.$idx.to_json()),+])
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
                                .map_err(|e| format!("At $[{}]: {}", $idx, e))?
                        ,)+))
                    }
                    other => Err(format!("Expected JSON array for tuple, got {}", other.short_preview())),
                }
            }

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

            fn from_reader<R: Read>(reader: R) -> Result<Self, String> {
                JSONNode::from_reader(reader).and_then(Self::from_json)
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
    fn to_json(&self) -> JSONNode {
        JSONNode::Bool(*self)
    }
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

macro_rules! impl_json_for_signed {
    ($($t:ty),*) => {
        $(
            impl JSONConvertible for $t {
                fn to_json(&self) -> JSONNode {
                    JSONNode::Int(*self as i128)
                }
                fn from_json(node: JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::Int(n) => {
                            if n >= (<$t>::MIN as i128) && n <= (<$t>::MAX as i128) {
                                Ok(n as $t)
                            } else {
                                Err(format!("Integer {} out of range for {}", n, stringify!($t)))
                            }
                        }
                        JSONNode::UInt(u) => {
                            if u <= (<$t>::MAX as u128) {
                                Ok(u as $t)
                            } else {
                                Err(format!("Unsigned integer {} too large for {}", u, stringify!($t)))
                            }
                        }
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
                fn to_json(&self) -> JSONNode {
                    JSONNode::UInt(*self as u128)
                }
                fn from_json(node: JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::UInt(u) => {
                            if u <= (<$t>::MAX as u128) {
                                Ok(u as $t)
                            } else {
                                Err(format!("Unsigned integer {} out of range for {}", u, stringify!($t)))
                            }
                        }
                        JSONNode::Int(n) => {
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
                        }
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
                fn to_json(&self) -> JSONNode {
                    JSONNode::Float(*self as f64)
                }
                fn from_json(node: JSONNode) -> Result<Self, String> {
                    match node {
                        JSONNode::Float(f) => Ok(f as $t),
                        JSONNode::Int(n)   => Ok(n as $t),
                        JSONNode::UInt(u)  => Ok(u as $t),
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

impl_json_for_signed!(i8, i16, i32, i64, isize, i128);
impl_json_for_unsigned!(u8, u16, u32, u64, usize, u128);
impl_json_for_float!(f32, f64);

impl JSONConvertible for String {
    fn to_json(&self) -> JSONNode {
        JSONNode::String(self.clone())
    }
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
    fn to_json(&self) -> JSONNode {
        JSONNode::String(self.to_string())
    }
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
                        Err(e) => {
                            return Err(format!(
                                "While deserializing Vec<T> at $[{}]: {}",
                                i, e
                            ))
                        }
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

    fn from_reader<R: Read>(reader: R) -> Result<Self, String> {
        JSONNode::from_reader(reader).and_then(Self::from_json)
    }
}

impl<T: JSONConvertible, const N: usize> JSONConvertible for [T; N] {
    fn to_json(&self) -> JSONNode {
        JSONNode::Array(self.iter().map(|item| item.to_json()).collect())
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                if arr.len() != N {
                    return Err(format!(
                        "Expected array of length {} for [T; N], got length {}",
                        N,
                        arr.len()
                    ));
                }
                let mut vec: Vec<T> = Vec::with_capacity(N);
                for (i, v) in arr.into_iter().enumerate() {
                    match T::from_json(v) {
                        Ok(val) => vec.push(val),
                        Err(e) => {
                            return Err(format!(
                                "While deserializing [T; {}] at $[{}]: {}",
                                N, i, e
                            ))
                        }
                    }
                }
                vec.try_into().map_err(|v: Vec<T>| {
                    format!(
                        "Expected array of length {} for [T; N], got length {}",
                        N,
                        v.len()
                    )
                })
            }
            other => Err(format!(
                "Expected JSON array for [T; {}], got {}",
                N,
                other.short_preview()
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
                        Ok(v) => {
                            set.insert(v);
                        }
                        Err(e) => {
                            return Err(format!(
                                "While deserializing BTreeSet<T> at $[{}]: {}",
                                i, e
                            ))
                        }
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
        vec.sort();
        JSONNode::Array(vec.iter().map(|item| item.to_json()).collect())
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut set = HashSet::new();
                for (i, item) in arr.into_iter().enumerate() {
                    match T::from_json(item) {
                        Ok(v) => {
                            set.insert(v);
                        }
                        Err(e) => {
                            return Err(format!(
                                "While deserializing HashSet<T> at $[{}]: {}",
                                i, e
                            ))
                        }
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
    K: JSONConvertible + Ord,
    V: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let pairs = self
            .iter()
            .map(|(k, v)| JSONNode::Array(vec![k.to_json(), v.to_json()]))
            .collect();
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
                            let key = K::from_json(k_node).map_err(|e| {
                                format!(
                                    "While deserializing BTreeMap<K, V> at $[{}][0] (key): {}",
                                    i, e
                                )
                            })?;
                            let val = V::from_json(v_node).map_err(|e| {
                                format!(
                                    "While deserializing BTreeMap<K, V> at $[{}][1] (value): {}",
                                    i, e
                                )
                            })?;
                            map.insert(key, val);
                        }
                        other => {
                            return Err(format!(
                                "Expected 2-element array for BTreeMap entry at $[{}], got {}",
                                i,
                                other.short_preview()
                            ))
                        }
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

    fn from_reader<R: Read>(reader: R) -> Result<Self, String> {
        JSONNode::from_reader(reader).and_then(Self::from_json)
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
        let pairs = sorted_pairs
            .into_iter()
            .map(|(k, v)| JSONNode::Array(vec![k.to_json(), v.to_json()]))
            .collect();
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
                            let key = K::from_json(k_node).map_err(|e| {
                                format!(
                                    "While deserializing HashMap<K, V> at $[{}][0] (key): {}",
                                    i, e
                                )
                            })?;
                            let val = V::from_json(v_node).map_err(|e| {
                                format!(
                                    "While deserializing HashMap<K, V> at $[{}][1] (value): {}",
                                    i, e
                                )
                            })?;
                            map.insert(key, val);
                        }
                        other => {
                            return Err(format!(
                                "Expected 2-element array for HashMap entry at $[{}], got {}",
                                i,
                                other.short_preview()
                            ))
                        }
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
    L: JSONConvertible + Ord + Eq,
    R: JSONConvertible + Ord + Eq,
{
    fn to_json(&self) -> JSONNode {
        let pairs = self
            .iter()
            .map(|(l, r)| JSONNode::Array(vec![l.to_json(), r.to_json()]))
            .collect();
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
                            let l = L::from_json(l_node).map_err(|e| {
                                format!(
                                    "While deserializing BiBTreeMap<L, R> at $[{}][0] (left/key): {}",
                                    i, e
                                )
                            })?;
                            let r = R::from_json(r_node).map_err(|e| {
                                format!(
                                    "While deserializing BiBTreeMap<L, R> at $[{}][1] (right/value): {}",
                                    i, e
                                )
                            })?;
                            map.insert(l, r);
                        }
                        other => {
                            return Err(format!(
                                "Expected 2-element array for BiBTreeMap entry at $[{}], got {}",
                                i,
                                other.short_preview()
                            ))
                        }
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

// --- Helper Functions for Byte Serialization ---

/// Converts a byte slice to JSON, preferring String encoding when valid UTF-8.
/// Falls back to Array of integers for non-UTF-8 data.
pub fn bytes_to_json(bytes: &[u8]) -> JSONNode {
    match String::from_utf8(bytes.to_vec()) {
        Ok(s) => JSONNode::String(s),
        Err(_) => JSONNode::Array(bytes.iter().map(|&b| JSONNode::UInt(b as u128)).collect()),
    }
}

/// Reconstructs a byte vector from JSON (either String or Array format).
pub fn json_to_bytes(node: JSONNode) -> Result<Vec<u8>, String> {
    match node {
        JSONNode::String(s) => Ok(s.into_bytes()),
        JSONNode::Array(arr) => {
            let mut bytes = Vec::with_capacity(arr.len());
            for item in arr {
                let val = u8::from_json(item)?;
                bytes.push(val);
            }
            Ok(bytes)
        }
        _ => Err("Expected String or Array for bytes".to_string()),
    }
}

impl serde::Serialize for JSONNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.to_serde_value().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for JSONNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serde_value = SerdeValue::deserialize(deserializer)?;
        JSONNode::from_serde_value(serde_value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[derive(Debug, Clone, PartialEq, JSONConvertible)]
    struct MyStruct {
        field1: i32,
        field2: String,
        optional_field: Option<bool>,
        list_of_numbers: Vec<u32>,
        byte_buffer: Vec<u8>,
    }

    #[derive(Debug, Clone, PartialEq, JSONConvertible)]
    struct GenericStruct<T: JSONConvertible, U: JSONConvertible> {
        item_t: T,
        item_u: U,
        description: String,
    }

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

        if let JSONNode::Object(obj) = &json_node {
            assert_eq!(obj.get("field1"), Some(&JSONNode::Int(42)));
            assert_eq!(
                obj.get("field2"),
                Some(&JSONNode::String("hello".to_string()))
            );
            assert_eq!(obj.get("optional_field"), Some(&JSONNode::Bool(true)));
            if let Some(JSONNode::Array(arr)) = obj.get("list_of_numbers") {
                assert_eq!(
                    arr,
                    &vec![
                        JSONNode::UInt(1),
                        JSONNode::UInt(2),
                        JSONNode::UInt(3)
                    ]
                );
            } else {
                panic!("list_of_numbers not found or not an array");
            }
            if let Some(JSONNode::Array(arr)) = obj.get("byte_buffer") {
                assert_eq!(
                    arr,
                    &vec![
                        JSONNode::UInt(10),
                        JSONNode::UInt(20),
                        JSONNode::UInt(30)
                    ]
                );
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
        if let JSONNode::Object(obj) = &json_node {
            assert_eq!(obj.get("item_t"), Some(&JSONNode::Int(123)));
        } else {
            panic!("Expected JSONNode::Object");
        }
        let deserialized =
            GenericStruct::<i32, String>::from_json(json_node).expect("Deserialization failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_generic_struct_with_custom_type() {
        let my_s = MyStruct {
            field1: 1,
            field2: "inner".to_string(),
            optional_field: None,
            list_of_numbers: vec![],
            byte_buffer: vec![],
        };
        let original = GenericStruct {
            item_t: my_s.clone(),
            item_u: MyUnitStruct,
            description: "Struct with custom types".to_string(),
        };
        let json_node = original.to_json();
        let deserialized =
            GenericStruct::<MyStruct, MyUnitStruct>::from_json(json_node).expect(
                "Deserialization failed",
            );
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
        map.insert("b_key".to_string(), 20i32);
        map.insert("a_key".to_string(), 10i32);

        let json_node = map.to_json();
        match json_node {
            JSONNode::Array(pairs) => {
                assert_eq!(pairs.len(), 2);
                match &pairs[0] {
                    JSONNode::Array(pair1) => {
                        assert_eq!(pair1[0], JSONNode::String("a_key".to_string()));
                        assert_eq!(pair1[1], JSONNode::Int(10));
                    }
                    _ => panic!("Expected array for pair"),
                }
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

        let invalid_json_uint_too_large = JSONNode::Array(vec![JSONNode::UInt(256)]);
        assert!(Vec::<u8>::from_json(invalid_json_uint_too_large).is_err());

        let invalid_json_int_too_large = JSONNode::Array(vec![JSONNode::Int(256)]);
        assert!(Vec::<u8>::from_json(invalid_json_int_too_large).is_err());

        let invalid_json_int_neg = JSONNode::Array(vec![JSONNode::Int(-1)]);
        assert!(Vec::<u8>::from_json(invalid_json_int_neg).is_err());

        let invalid_json_float = JSONNode::Array(vec![JSONNode::Float(10.5)]);
        assert!(Vec::<u8>::from_json(invalid_json_float).is_err());

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

        let int_node = JSONNode::Int(123);
        let float_from_int = f64::from_json(int_node).unwrap();
        assert_eq!(float_from_int, 123.0f64);

        let uint_node = JSONNode::UInt(456);
        let float_from_uint = f64::from_json(uint_node).unwrap();
        assert_eq!(float_from_uint, 456.0f64);

        let float_node_exact_int = JSONNode::Float(123.0);
        assert!(i32::from_json(float_node_exact_int.clone()).is_err());
        assert!(u32::from_json(float_node_exact_int.clone()).is_err());

        let float_node_inexact_int = JSONNode::Float(123.7);
        assert!(i32::from_json(float_node_inexact_int.clone()).is_err());
        assert!(u32::from_json(float_node_inexact_int.clone()).is_err());

        assert!(i8::from_json(JSONNode::Int(128)).is_err());
        assert!(i8::from_json(JSONNode::UInt(128)).is_err());
        assert!(u8::from_json(JSONNode::Int(256)).is_err());
        assert!(u8::from_json(JSONNode::UInt(256)).is_err());
        assert!(u8::from_json(JSONNode::Int(-1)).is_err());
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

        let expected_json_string = r#"{"byte_buffer":[10,20,30],"field1":42,"field2":"hello \"world\" \\ / \\b \\f \n \r \t","list_of_numbers":[1,2,3],"optional_field":true}"#;

        let serde_val_generated: SerdeValue = serde_json::from_str(&json_string).unwrap();
        let serde_val_expected: SerdeValue =
            serde_json::from_str(expected_json_string).unwrap();

        assert_eq!(serde_val_generated, serde_val_expected);

        let parsed_node =
            JSONNode::from_json_string(&json_string).expect("Failed to parse JSON string");
        let deserialized_struct =
            MyStruct::from_json(parsed_node).expect("Deserialization from parsed_node failed");
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

        let parsed_nan_node = JSONNode::from_json_string("null").unwrap();
        assert_eq!(parsed_nan_node, JSONNode::Null);
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

        let single_tuple: (i32,) = (42,);
        let json_node_single = single_tuple.to_json();
        assert_eq!(json_node_single, JSONNode::Array(vec![JSONNode::Int(42)]));
        let deserialized_single = <(i32,)>::from_json(json_node_single).unwrap();
        assert_eq!(single_tuple, deserialized_single);

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

        let deserialized = MyStruct::from_reader(Cursor::new(buffer)).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_streaming_btreemap() {
        let mut original_map = BTreeMap::new();
        original_map.insert("key1".to_string(), 100);
        original_map.insert("key2".to_string(), 200);

        let mut buffer = Vec::new();
        original_map.to_writer(&mut buffer).unwrap();

        let deserialized_map: BTreeMap<String, i32> =
            BTreeMap::from_reader(Cursor::new(buffer)).unwrap();
        assert_eq!(original_map, deserialized_map);
    }

    #[test]
    fn test_streaming_vec() {
        let original_vec = vec![1, 2, 3];
        let mut buffer = Vec::new();
        original_vec.to_writer(&mut buffer).unwrap();

        let deserialized_vec: Vec<i32> = Vec::from_reader(Cursor::new(buffer)).unwrap();
        assert_eq!(original_vec, deserialized_vec);
    }
}
