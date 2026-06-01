# `cache_types.rs` deep dive

## Summary

The file names the cache fields as cache fields.  This matters because several of these fields look semantically important when read in isolation, but they are only accelerators for materializing masks or walking automata.

## Local invariants

- The file should have one mathematical subject.
- The file should not import parser/compiler modules unless its subject requires them.
- The file should not add public API accidentally.
- If a method is only a cache helper, keep it inside the artifact cache namespace.

## Review questions

1. Can the file be explained without mentioning unrelated runtime algorithms?
2. Does every type name expose its coordinate system or cache role?
3. Would a paper reader know where this file fits in the mathematical pipeline?
4. Is the file a good candidate for future compile-repair after this structural pass?

## Follow-up candidates

- Add rustdoc examples after the API stabilizes.
- Add compile-time assertions only after the no-compile cleanup phase ends.
- Use this file as the unit of review in the next mechanical repair pass.
