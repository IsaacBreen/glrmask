# Suggested next chunks

After this chunk, the most valuable next work is one of:

1. split `fast_path.rs` into a directory of individual fast-path proofs;
2. split `profiled.rs` and remove duplicated control flow through an observer abstraction;
3. clean `runtime/mask/mod.rs` and `runtime/mask_mapping.rs` using the same denotation-first approach;
4. replace broad `use super::*` imports in Commit with explicit imports;
5. begin the first compile-repair pass for chunks 00--12.

If continuing without compile, the next purely architectural chunk should be runtime Mask/mask-mapping cleanup. If switching to compile repair, start with module path and visibility errors in Commit because this chunk intentionally widened local helper visibility.
