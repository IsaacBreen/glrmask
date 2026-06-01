use super::Schema;

/// Array-specific assertions after loading.
#[derive(Debug, Clone)]
pub(crate) struct ArraySchema {
    pub(crate) items: Box<Schema>,
    pub(crate) prefix_items: Vec<Schema>,
    pub(crate) min_items: usize,
    pub(crate) max_items: Option<usize>,
}

impl Default for ArraySchema {
    fn default() -> Self {
        Self {
            items: Box::new(Schema::any("<implicit-array-items>")),
            prefix_items: Vec::new(),
            min_items: 0,
            max_items: None,
        }
    }
}
