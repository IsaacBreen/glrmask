# Object key-order semantics

JSON object values are maps.  JSON object texts are sequences of key-value pairs.
JSON Schema validates the map, not the source order.  Therefore a grammar for an
object schema must accept every key order that decodes to a valid map unless the
crate deliberately documents a canonicalization strategy.

The current lowerer uses sequence grammars, NFA bodies, and factoring to accept
many key orders while controlling grammar size.  Any object cleanup must keep
these invariants explicit:

1. required keys appear at least once;
2. duplicate-key policy is documented;
3. optional keys may appear or not according to property bounds;
4. additional keys obey exclusion and additionalProperties policy;
5. pattern-covered keys satisfy all matching schemas.
