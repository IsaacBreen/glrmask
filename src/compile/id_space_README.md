# ID-space contracts

A `ManyToOneIdMap` is a quotient map from original ids to internal ids. A `MappedArtifact<T>` is a dependent pair `(artifact, id_map)`: every numeric id inside the artifact is meaningful only in the coordinates described by the map. Compaction is therefore a change of coordinates, not just a data-size optimization.
