use crate::import::ast::GrammarExpr;

use super::ast::ArraySchema;
use super::error::{ImportResult, SchemaImportError};
use super::lower::{lit, never, seq, Lowerer, ITEM_SEPARATOR};

impl<'a> Lowerer<'a> {
    pub(crate) fn lower_array(&mut self, schema: &ArraySchema) -> ImportResult<GrammarExpr> {
        if !schema.prefix_items.is_empty() {
            return Err(SchemaImportError::new(
                "prefixItems is parsed but not lowered by the simple importer",
            ));
        }
        if schema.max_items.is_some_and(|max| max < schema.min_items) {
            return Ok(never());
        }

        let item = self.lower_schema(&schema.items)?;
        let body = self.array_body(item, schema.min_items, schema.max_items);
        Ok(seq(vec![lit("["), body, lit("]")]))
    }

    fn array_body(&self, item: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {
        match max {
            Some(0) => GrammarExpr::Epsilon,
            Some(max) => GrammarExpr::SeparatedSequence {
                items: vec![(
                    GrammarExpr::RepeatRange { expr: Box::new(item), min, max },
                    min > 0,
                )],
                separator: Box::new(lit(ITEM_SEPARATOR)),
                allow_empty: min == 0,
            },
            None if min == 0 => GrammarExpr::SeparatedSequence {
                items: vec![(GrammarExpr::RepeatOne(Box::new(item)), false)],
                separator: Box::new(lit(ITEM_SEPARATOR)),
                allow_empty: true,
            },
            None => {
                let mut items = (0..min)
                    .map(|_| (item.clone(), true))
                    .collect::<Vec<_>>();
                items.push((GrammarExpr::RepeatOne(Box::new(item)), false));
                GrammarExpr::SeparatedSequence {
                    items,
                    separator: Box::new(lit(ITEM_SEPARATOR)),
                    allow_empty: false,
                }
            }
        }
    }
}
