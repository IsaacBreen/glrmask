# Chunk 03 follow-up backlog

The phase graph is a foundation.  This backlog lists the next refinements that should happen without undoing the graph.


## P1: Recover safe phase parallelism

Make every phase return `PhaseOutput<T>` with a local profile delta.  Then `rayon::join` can combine independent phases without mutably borrowing the same profile.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P1: Replace environment-only compile decisions

Thread a resolved internal compile options struct through the phase graph.  Public `CompileOptions` can lower into that internal type.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P1: Move terminal-DWA partition env flags

Pair-partition mode, partition count, objective, global max-length, and related flags should live in typed options, not terminal-DWA internals.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P2: Create structured CompileReport

Keep flat profile fields for benchmarks but add phase records, artifact size records, option records, and warnings.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P2: Rename compatibility shims out of compiler namespace

Once downstream code imports new paths, remove or shrink `compiler::compile` and orphan `compiler/pipeline.rs`.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P1: Separate parser stack-effect facts from GLR implementation

Create a `compile/parser_facts` or `compile/stack_effects` module that exposes parser-side contracts without paper overemphasis on GLR/LR.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P2: Move final Constraint field initialization behind a builder

`finalize.rs` should eventually use a `RuntimeArtifactBuilder` to reduce giant struct literal fragility.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P2: Audit profile side effects outside pipeline

Terminal-DWA and scan-relation internals still print directly in places.  Centralize through sinks/reporting.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P3: Add architecture tests

Add text-level or unit-level tests that assert no `std::env`/`eprintln!` in pipeline modules and that phase modules remain under size thresholds.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.

## P1: Document exact equivalence boundaries

Write one doc for Terminal-DWA equivalence vs CanMatch equivalence vs final internal coordinate equivalence.

Definition of done: implementation is moved without weakening the phase graph, docs are updated, and compatibility shims are either preserved intentionally or removed intentionally.
