# Documentation synchronization

`GUIDE.md` is the canonical source for repository usage documentation.
`scripts/sync-guide.py` adds the README title and License section. The Guide
sync GitHub workflow checks that the generated README is current.

Benchmark blocks and light/dark SVGs are generated from a reviewed public
aggregate snapshot by `scripts/sync-benchmarks.py`. To update measurements,
add a dated snapshot only after checking its source revision, measurement date,
hardware, APIs, policies, cohort, outcomes and raw-file digests against retained
CFA results. Keep other datasets and experimental results separate. Update the
generator's snapshot/asset paths and companion benchmark report together. Its
`PROVENANCE_SHA256` pins the reviewed source, date, hardware, cohorts, APIs,
outcomes, policies and evidence digests: changing them fails explicitly until
the prose and report have been reviewed and that digest updated. Then run:

```bash
python scripts/sync-benchmarks.py
python scripts/sync-guide.py
python scripts/sync-benchmarks.py --check
python scripts/sync-guide.py --check
```

CI checks the generated prose, tables, SVGs and README against that snapshot.
It does not fetch a changing live website or qualify new benchmark results.

## Website relationship

In `IsaacBreen/personal-website-2026`, `scripts/sync-glrmask-guide.mjs` imports
`GUIDE.md` from an explicit source revision (or `GLRMASK_GUIDE_SOURCE` locally).
It generates `src/content/glrmask-prototypes/handbook/guide.md`. Website prebuild
runs that sync; predev uses the local source.

The website intentionally transforms installation into tabs, resolves relative
documentation links, reformats examples for its layout, and omits the Guide's
introduction, abbreviated implementation and Performance section. Its
`PrototypeArticle.astro` combines `handbook/nutshell.md`,
`CanonicalBenchmarks.astro`, and the generated Getting started article on the
Overview page. `CanonicalBenchmarks.astro` reads separate reviewed JSON Schema
and JavaScript data files. Overview introduction and interactive figures have
their own website presentation; the full Overview is not a verbatim README.

Website edits do not write back into this repository. A website change to shared
API facts belongs in `GUIDE.md` too; a newly qualified benchmark belongs in both
aggregate sources. To detect benchmark drift against a website checkout:

```bash
python scripts/sync-benchmarks.py --check --website-root /path/to/website
```

This compares exact revisions, schema/position counts, membership hashes,
outcomes, statistics and tail data. It reads aggregate JSON only. Coordinate
website updates separately and verify its pinned Guide revision and transforms
against current source before deployment; changing this repository cannot
update an already deployed website.
