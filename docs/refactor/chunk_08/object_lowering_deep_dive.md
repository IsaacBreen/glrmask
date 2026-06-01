# Object lowering deep dive
Object lowering is the hardest part of the importer because JSON object schemas
combine set-like maps with sequence-like JSON text.  A JSON object value has no
semantic key order, but a JSON text does.  Therefore the lowerer must either
accept all legal key permutations or choose and document a canonicalization.

The current design accepts permutations using grammar structure, while adding
performance strategies for large optional property families.  The object lowerer
should ultimately be split into:

1. fixed-key closed objects;
2. open objects with additionalProperties;
3. pattern-property maps;
4. anyOf/object-variant factoring;
5. residual branch/shadow-owner logic;
6. large literal-key trie factoring.

Every split must preserve the invariant that a key covered by multiple schemas
must satisfy all applicable schemas.
