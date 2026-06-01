# Regex lowering semantics for strings

JSON Schema regexes are over decoded string characters.  The grammar regexes in
this crate are over encoded JSON string body bytes.  Lowering a regex therefore
requires a semantic translation:

```text
decoded regex language R ⊆ UnicodeScalar*
JSON-body encoding relation EncBody
emitted body regex B such that EncBody(R) ⊆/=/ B
```

Exact lowering should be preferred.  If a regex feature cannot be translated,
choose one of:

- reject the schema;
- broaden to all JSON strings with an explicit diagnostic/comment;
- support only an anchored/simple subset with a documented contract.
