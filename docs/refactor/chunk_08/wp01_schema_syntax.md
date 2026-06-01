# Schema syntax package

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Keep schema structs value-semantic rather than grammar-semantic.
2. Rename vague AST concepts to SchemaDocument, SchemaDefinition, SchemaAssertions, ObjectSchema, ArraySchema, StringSchema, NumberSchema.
3. Ensure every schema node has a location string suitable for diagnostics.
4. Do not add GrammarExpr, token, automata, parser, or runtime imports to schema/.
5. Future: exact numeric representation should live in schema/number.rs, not lower/number.rs.

## Definition of done

- The source tree has an obvious home for: keep schema structs value-semantic rather than grammar-semantic.
- The source tree has an obvious home for: rename vague ast concepts to schemadocument, schemadefinition, schemaassertions, objectschema, arrayschema, stringschema, numberschema.
- The source tree has an obvious home for: ensure every schema node has a location string suitable for diagnostics.
- The source tree has an obvious home for: do not add grammarexpr, token, automata, parser, or runtime imports to schema/.
- The source tree has an obvious home for: future: exact numeric representation should live in schema/number.rs, not lower/number.rs.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
