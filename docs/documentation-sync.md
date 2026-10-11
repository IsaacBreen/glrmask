# Documentation synchronization

The website Overview is the baseline for the public introduction, section order
and Getting started prose. `docs/content/` now holds that shared content.
`scripts/sync-guide.py` generates:

- `README.md`: GitHub Overview, separate static JSON/JavaScript benchmarks,
  Getting started, a short How it works link, and License.
- `README.rust.md`: the same introduction and benchmarks, Rust installation and
  quickstart, How it works, and License. Cargo selects this file.
- `python/README.md` and `python/PYTHON.md`: the Python variant. `PYTHON.md` is the
  package metadata input; the other copy is the GitHub directory landing page.
- `GUIDE.md`: shared introduction and quickstarts plus the full technical
  reference retained in `docs/content/reference.md`.
- `docs/website-content.json`: a versioned shared-content payload for the website
  owner. It includes the approved introduction, links, dynamic attribution,
  installation commands, usage, quickstarts, cleaned usage details, chart spec
  and dataset paths. Its canonical content digest detects stale imports.

Edit the shared sources, not generated files. The package variants contain
only their own language's installation and quickstart. Their images and
benchmark-detail links are absolute; the GitHub README uses relative theme-aware
SVG references. Each context omits its own top-level Website/GitHub link.

## Benchmarks and plots

Reviewed public snapshots live in `docs/benchmarks/`. `scripts/sync-benchmarks.py`
generates the benchmark Markdown, light/dark SVGs and registry PNGs. The shared
chart spec `docs/content/tail-chart.json` preserves the website's inverse
log-exceedance axis, log latency axis, series palette, clean axes and lack of a
grid. The static figures show Static versus llguidance. JSON samples are joined
by straight segments; JavaScript uses the website’s adaptive monotone cubic
display policy through selected measured anchors. Additional anchors constrain
latency error to 2% at every supplied quantile. Neither figure extrapolates;
measurements and statistics remain unchanged. JSON and
JavaScript remain separate, including their source identities and timing limits.
Dynamic is introduced at the end of the JSON section with its own TBM plot:
GLRMask is faded, GLRMask dynamic and llguidance remain prominent. A TTFM table
follows; no second TBM table is added. The README deep link requires the website to initialize
the JSON dataset and Dynamic mode from `?dataset=json&mode=dynamic#benchmarks`.
The documentation link and website URL-state implementation must be published
together; a fragment-only benchmark link does not select Dynamic.

Use `rsvg-convert` from librsvg to regenerate opaque, 2× registry PNGs. Each PNG
records its source SVG digest; checks validate that digest and all PNG chunk
checksums without needing a renderer in CI.

JavaScript image URLs carry a generated-content digest so cached figures follow
the current rendering; JSON image URLs and assets remain unchanged.

```bash
python scripts/sync-benchmarks.py --render-png
python scripts/sync-guide.py
python scripts/sync-benchmarks.py --check
python scripts/sync-guide.py --check
```

Measurement updates require independent retained-CFA verification first. The
JSON and JavaScript provenance pins cover dates, source revisions, cohort,
APIs/timing policies, outcomes and evidence identities. Review the corresponding
public report and caption before changing a pin. Do not silently combine other
sources, experimental results, or package-version measurements. CI checks
snapshot-to-prose/table/asset and shared-source-to-document consistency.

## Website integration contract

In `IsaacBreen/personal-website-2026`, import `docs/website-content.json` from an
explicit published GLRMask source revision (or a provided local source path).
Reject unknown `schema` values and verify `content_sha256` over the payload
without that key, using sorted-key compact JSON, UTF-8, and SHA-256. Import the
same revision's benchmark snapshots and `tail-chart.json`. Avoid fetching a
moving branch independently for each file.

Render Overview in `overview_order`: introduction, context-specific links,
benchmarks, Getting started. Website presentation can retain interactive
JSON/JavaScript and Static/Dynamic controls. Use `introduction`, `dynamic`,
`usage`, `quickstarts`, and `usage_details` directly; remove old regex prose
corrections from `scripts/sync-glrmask-guide.mjs`. Keep installation tabs and
syntax highlighting as presentation. Map the shared log domains, grid/palette
and observed-support floor into `CanonicalTailPlot.astro`; website interaction
may expose Dynamic while the static README figures show only Static/llguidance.

The website owner must perform this integration in their checkout. This
repository generates the contract but does not change an independently
deployed website. Its build should fail if generated content or the recorded
source revision/digest is stale. To compare the current website aggregates:

```bash
python scripts/sync-benchmarks.py --check --website-root /path/to/website
```

## Package archive checks

Python's distinct `PYTHON.md` filename prevents a root GitHub README collision
when maturin relocates `pyproject.toml` into the source archive. The root path
dependency must also include its selected `README.rust.md`.

```bash
cargo package --no-verify --allow-dirty --locked --offline -p glrmask
maturin sdist --manifest-path python/Cargo.toml --out /tmp/glrmask-docs-dist
python scripts/check-readme-archives.py --crate /path/to/glrmask.crate --sdist /path/to/glrmask.tar.gz
# For any future, explicitly authorized package build:
python scripts/check-readme-archives.py --wheel /path/to/glrmask.whl
```

Those source-archive checks do not compile or publish a package. Existing
registry versions retain their uploaded descriptions. Before a future package
release, publish the referenced documentation assets, then verify descriptions
in the newly built artifacts and their actual registry rendering. This docs
workflow does not authorize a new version or replace an existing release.
