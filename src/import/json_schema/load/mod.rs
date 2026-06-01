//! JSON Schema loader.
//!
//! The loader is the only importer phase that may inspect arbitrary
//! `serde_json::Value` keyword maps.  Its output is the typed `schema::*` model.
//! It deliberately does not construct grammar expressions.

mod collect;
mod keywords;
mod pointers;
mod shape;
mod typed;

pub(crate) use self::typed::load_document;
