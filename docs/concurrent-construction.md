# Concurrent grammar construction

The public construction operations are synchronous. `prepare`, `compile`,
`link`, and `save` return their finished result; they do not return background
jobs or continue building after returning.

## Independent assembly handles

`Grammar` and `PreparedGrammar` have mutable, single-item `bind(name, value)`
methods. An independent request starts by cloning a reusable handle:

```python
request = template.clone()
request.bind("MARK", marker)
constraint = request.link()
```

Clones share immutable definitions and binding snapshots. A successful binding
changes only its receiver. Attaching a child captures that child's current
configuration; subsequent edits to the original child do not affect the parent.
An error leaves the receiver's binding configuration unchanged. Already returned
constraints and already captured compile/link snapshots remain stable.

Python assignment (`request = template`) is an alias, not a clone. Concurrent
specialization should use separate handles. Concurrent conflicting access to the
same mutable Python handle can report a borrow/busy error rather than silently
lose an update. Never use repeated updates to one shared handle as a throughput
benchmark for independent requests.

## Caller-owned Rayon pools

Rust callers can choose an executor and CPU budget for independent construction:

```rust
use glrmask::{BuildOptions, Constraint, Optimization, PreparedGrammar};
use rayon::prelude::*;

fn instantiate_many(
    template: &PreparedGrammar,
    markers: &[u32],
    workers: usize,
) -> Result<Vec<Constraint>, Box<dyn std::error::Error>> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;
    Ok(pool.install(|| {
        markers.par_iter().map(|&id| {
            let mut request = template.clone();
            request.bind("MARK", glrmask::TokenSet::from(id))?;
            request.link_with(BuildOptions::default().optimization(Optimization::FastBuild))
        }).collect::<glrmask::Result<Vec<_>>>()
    })?)
}
```

In a service, retain the caller's pool rather than creating it for every batch.
Construction already running on a Rayon worker stays in that pool; it is not
redirected to GLRMask's capped default pool or its single-thread tiny-build
helper. The composition finalizer likewise respects caller-owned execution.
Internal parallel work uses the existing pool's budget rather than creating a
pool for every request. This does not promise that one job uses all available
workers or that every algorithm benefits from internal parallelism.

Outside an existing Rayon context, GLRMask retains its default single-build
policy: a capped compilation pool (six threads on Windows, ten elsewhere), with
a serial helper for suitable tiny dynamic builds. An explicit parallel
`GLRMASK_COMPILE_THREADS` or `RAYON_NUM_THREADS` budget also applies to those
tiny builds, rather than funnelling them through the serial helper. Configure
environment variables before first compilation; owned pools initialize lazily.

Setting a library pool to one worker is **not** a request for many independent
OS threads to run on their own callers. For native caller-controlled execution,
use the caller's Rayon context, or configure a shared library budget appropriate
to the whole workload. Keep outer request concurrency and inner compilation
parallelism within one deliberate CPU budget.

## Python's GIL boundary

Substantial native source/prepared binding, requirement analysis, preparation,
compilation, linking and serialization run outside the GIL. Snapshot capture is
cheap; Python argument and result conversion still require Python access.

The language boundary is not free. A workload consisting of many extremely small
Python calls can be limited by Python dispatch and object construction even when
the Rust calculation is parallel. Measure complete request preparation/linking
as well as isolated binding calls, and avoid attributing GIL overhead to native
mask computation.

## What scaling measurements establish

Measure independent jobs with persistent workers and equal work. Report total
throughput and per-job latency separately. Test shared warm templates, unrelated
templates, cold and warm vocabulary artifacts, and both single- and multi-socket
placement where available. Physical cores and hardware threads are different
resources.

Useful native throughput efficiency is `X(p) / (p * X(1))`, where `X(p)` is jobs
per second at `p` workers under the same per-job workload. Memory bandwidth,
allocation, cache-line traffic and NUMA placement can limit scaling before every
core is busy. A test on a smaller machine is not evidence of 128-core scaling.

Construction metadata and validation must not become runtime mask lookups.
Normal packed `fill_mask()` keeps its separate implementation and should be
qualified both alone and after construction/forcing activity. Shared hardware
contention is reported separately from additional work in the mask code path.

## Measured qualification

See the [15 October 2026 core-batch report](validation/core-batch-2026-10-15.md)
for matched measurements, exact correctness gates and remaining trade-offs.
