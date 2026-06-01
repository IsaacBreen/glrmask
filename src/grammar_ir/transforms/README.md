# Grammar IR transforms

Transforms preserve the named grammar denotation while changing its shape before
flat lowering. They should not allocate `GrammarDef` ids and should not depend on
GLR table construction.
