# Number lowering deep dive
Number lowering should eventually be exact over JSON numeric lexical space.  The
current code uses `f64` for schema storage and regex generation helpers for
ranges.  This is acceptable as inherited behavior during structural cleanup, but
not ideal for publication.  The target is a decimal-rational schema layer and a
number-lowerer that can state exactness for each combination of range and
multipleOf constraints.
