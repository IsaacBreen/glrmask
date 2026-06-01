# Serialization compatibility contract

Serialized constraints are versioned envelopes. Derived runtime caches are not serialized and must be rebuilt deterministically after load. Legacy direct-bincode loading remains as a compatibility fallback until a major release removes it.
