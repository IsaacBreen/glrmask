//! Complete, borrowed schema observations. Equality and cache hashing use
//! one typed node representation, with a location policy for lowering.
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use serde_json::Value;

use super::ast::{AdditionalProperties, NumberSchema, ObjectSchema, Schema, SchemaKind, SchemaType, StringSchema};

#[derive(Clone, Copy)]
pub(super) struct SchemaShape<'a> {
    root: &'a Schema,
    locations: LocationPolicy<'a>,
}

#[derive(Clone, Copy)]
enum LocationPolicy<'a> {
    Ignore,
    Relative(&'a str),
    Absolute,
}

#[derive(PartialEq, Hash)]
enum Location<'a> {
    Ignored,
    Root,
    Relative(&'a str),
    Absolute(&'a str),
}

#[derive(PartialEq, Hash)]
struct Node<'a> {
    location: Location<'a>,
    additional_site: bool,
    kind: Kind<'a>,
}

#[derive(PartialEq, Hash)]
enum Kind<'a> {
    Any,
    Never,
    Ref(&'a str),
    Assertions(Assertions<'a>),
}

#[derive(PartialEq, Hash)]
struct Assertions<'a> {
    types: Option<&'a [SchemaType]>,
    constant: Option<LiteralShape<'a>>,
    enumeration: Option<LiteralList<'a>>,
    object: Option<Object<'a>>,
    array: Option<(usize, Option<usize>, usize)>,
    string: Option<&'a StringSchema>,
    number: Option<Number<'a>>,
    // Child counts delimit each ordered child list in the preorder stream.
    any_of: usize,
    one_of: usize,
    all_of: usize,
    not: bool,
}

/// A literal grammar observes spelling and ordered object entries, not
/// merely serde_json's map/value equality. The same observations feed hash
/// and equality, including the distinction between -0.0 and 0.0 spellings.
#[derive(Clone, Copy)]
struct LiteralShape<'a>(&'a Value);

#[derive(Clone, Copy)]
struct LiteralList<'a>(&'a [Value]);

impl PartialEq for LiteralList<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().map(LiteralShape).eq(other.0.iter().map(LiteralShape))
    }
}

impl Hash for LiteralList<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for value in self.0 { LiteralShape(value).hash(state); }
    }
}

impl PartialEq for LiteralShape<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self.0, other.0) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Number(a), Value::Number(b)) => a.to_string() == b.to_string(),
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => LiteralList(a) == LiteralList(b),
            (Value::Object(a), Value::Object(b)) => a.iter().map(|(k, v)| (k, LiteralShape(v)))
                .eq(b.iter().map(|(k, v)| (k, LiteralShape(v)))),
            _ => false,
        }
    }
}

impl Hash for LiteralShape<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self.0).hash(state);
        match self.0 {
            Value::Null => {}
            Value::Bool(value) => value.hash(state),
            Value::Number(value) => value.to_string().hash(state),
            Value::String(value) => value.hash(state),
            Value::Array(values) => LiteralList(values).hash(state),
            Value::Object(map) => {
                map.len().hash(state);
                for (key, value) in map {
                    key.hash(state);
                    LiteralShape(value).hash(state);
                }
            }
        }
    }
}

#[derive(PartialEq, Hash)]
struct Object<'a> {
    property_keys: Vec<&'a str>,
    pattern_keys: Vec<&'a str>,
    required: &'a BTreeSet<String>,
    required_order: &'a [String],
    dependencies: &'a BTreeMap<String, BTreeSet<String>>,
    min: usize,
    max: Option<usize>,
    property_names: bool,
    additional: u8,
}

impl<'a> From<&'a ObjectSchema> for Object<'a> {
    fn from(object: &'a ObjectSchema) -> Self {
        Self {
            property_keys: object.properties.iter().map(|p| p.name.as_str()).collect(),
            pattern_keys: object.pattern_properties.iter().map(|p| p.pattern.as_str()).collect(),
            required: &object.required,
            required_order: &object.required_order,
            dependencies: &object.property_dependencies,
            min: object.min_properties,
            max: object.max_properties,
            property_names: object.property_names.is_some(),
            additional: match &object.additional_properties {
                AdditionalProperties::AllowAny => 0,
                AdditionalProperties::Deny => 1,
                AdditionalProperties::Schema(_) => 2,
            },
        }
    }
}

#[derive(PartialEq, Hash)]
struct Number<'a> {
    integer: bool,
    minimum: Option<u64>,
    maximum: Option<u64>,
    exclusive_minimum: bool,
    exclusive_maximum: bool,
    multiple_of: Option<u64>,
    format: Option<&'a str>,
}

impl<'a> From<&'a NumberSchema> for Number<'a> {
    fn from(number: &'a NumberSchema) -> Self {
        let bits = |value: f64| if value == 0.0 { 0 } else { value.to_bits() };
        Self {
            integer: number.integer,
            minimum: number.minimum.map(bits),
            maximum: number.maximum.map(bits),
            exclusive_minimum: number.exclusive_minimum,
            exclusive_maximum: number.exclusive_maximum,
            multiple_of: number.multiple_of.map(bits),
            format: number.format.as_deref(),
        }
    }
}

impl<'a> SchemaShape<'a> {
    pub(super) fn semantic(root: &'a Schema) -> Self {
        Self { root, locations: LocationPolicy::Ignore }
    }

    pub(super) fn lowering(root: &'a Schema) -> Self {
        // Object-choice recursion checks can compare a branch's original
        // location with a ref target. Relative locations are insufficient
        // when a subtree contains refs. Reference-free subtrees can still
        // share lowering results across structurally identical sites.
        let locations = if root.descendants().any(|node| matches!(node.kind, SchemaKind::Ref(_))) {
            LocationPolicy::Absolute
        } else {
            LocationPolicy::Relative(&root.location)
        };
        Self { root, locations }
    }

    fn observe(&self, schema: &'a Schema) -> Node<'a> {
        let location = match self.locations {
            LocationPolicy::Ignore => Location::Ignored,
            LocationPolicy::Absolute => Location::Absolute(&schema.location),
            LocationPolicy::Relative(root) if schema.location == root => Location::Root,
            LocationPolicy::Relative(root) => schema.location.strip_prefix(root)
                .filter(|suffix| suffix.starts_with('/'))
                .map_or(Location::Absolute(&schema.location), Location::Relative),
        };
        let kind = match &schema.kind {
            SchemaKind::Any => Kind::Any,
            SchemaKind::Never => Kind::Never,
            SchemaKind::Ref(pointer) => Kind::Ref(pointer),
            SchemaKind::Assertions(a) => Kind::Assertions(Assertions {
                types: a.types.as_deref(),
                constant: a.const_value.as_ref().map(LiteralShape),
                enumeration: a.enum_values.as_deref().map(LiteralList),
                object: a.object.as_ref().map(Object::from),
                array: a.array.as_ref().map(|a| (a.min_items, a.max_items, a.prefix_items.len())),
                string: a.string.as_ref(),
                number: a.number.as_ref().map(Number::from),
                any_of: a.any_of.len(),
                one_of: a.one_of.len(),
                all_of: a.all_of.len(),
                not: a.not.is_some(),
            }),
        };
        Node {
            location,
            additional_site: !matches!(self.locations, LocationPolicy::Ignore)
                && schema.location.ends_with("/additionalProperties"),
            kind,
        }
    }

    pub(super) fn equals(self, other: SchemaShape<'_>) -> bool {
        self.root.descendants().map(|node| self.observe(node))
            .eq(other.root.descendants().map(|node| other.observe(node)))
    }

    pub(super) fn fingerprint(self) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        for node in self.root.descendants() {
            self.observe(node).hash(&mut hasher);
        }
        hasher.finish()
    }
}

pub(super) fn schemas_shape_equivalent(left: &Schema, right: &Schema) -> bool {
    SchemaShape::semantic(left).equals(SchemaShape::semantic(right))
}
