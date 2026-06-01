# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## GLR table split

The table subsystem now has a clearer ownership line:

- `action.rs`: action data structures.
- `row.rs`: row maps and row-level access.
- `build.rs`: LR/GLR item-set construction and table assembly.
- `options.rs`: table construction/optimization policy.
- `optimize.rs`: facade over table optimization fragments.
- `optimize/`: individual optimization families.

## Mathematical objects in the table

A `GLRTable` is not just an LR parse table.  It is an optimized execution relation over parser stack states.  Its rows can contain compact stack effects, not just classic shift/reduce entries.

## Two-row semantics

`action` rows encode execution.  `advance` rows encode admission.  A state/terminal pair may have an optimized action whose guard has to be checked against lower stack states; admission rows are still used as a fast prefilter.  Table optimization passes must preserve the invariant that admission rows do not claim impossible top-state support after state remapping or synthetic state insertion.

## Why `options.rs` matters

Previously, environment variables were read from several algorithmic locations.  That makes the mathematical table transformation harder to inspect because policy is mixed into transformation code.  `GLRTableOptions` is a local typed policy object for this subsystem.
