# Core-batch qualification drivers

These scripts run existing baseline/candidate executables or installed Python packages; they never change a production library or compile an alternate masking backend.

* `qualify_native_pair.py`: alternating fresh-process normal-mask/commit/lifecycle comparison, with separate full-mask captures and artifact hashes.
* `qualify_scaling_pair.py`: identical native setup driver in persistent caller-owned pools; reports complete workload/worker rows. Supply real hardware topology when interpreting results.
* `verification_force_pair.py` plus `review_force_bench.py`: matched strict-token-forcing comparison, including empty/singleton/multiple-ID cases and before-mask versus existing-mask queries. Python packages need not expose the same historical forcing name; each uses strict ID semantics, not canonical tokenization.
* `review_api_conformance_v2.py`: installed-public-API differential and clone/binding/concurrency checks.
* Rust oracle programs: compile externally against the chosen built `glrmask` library with `internal-api`, or adapt them into a test fixture. They cover arbitrary byte frontiers, passive markers, scoped lexers and 128-worker event-cache liveness. The reference methods are intentionally not public library API.

The native runner sources are `examples/core_batch_bench.rs` and `examples/core_batch_scaling.rs`. Build both revisions with the same production release flags/features and separate target directories; preserve the built baseline before candidate edits. `--features internal-api` is needed only for qualification helpers. Run timed controllers without competing builds or timed jobs. A lack of repeatability flags is not a universal proof of no latency regression. The ordinary masks must match exactly.

Example commands (replace artifact paths; output directories must not exist):

```sh
python scripts/qualification/core_batch/qualify_native_pair.py --baseline BASELINE_BENCH --candidate CANDIDATE_BENCH --baseline-ref BASE_SHA --candidate-ref CANDIDATE_SHA --output normal-mask --vocab 8192 --repetitions 5000 --pairs 4
python scripts/qualification/core_batch/qualify_scaling_pair.py --baseline BASELINE_SCALING --candidate CANDIDATE_SCALING --output setup-scaling --workers 16 --jobs-per-worker 32 --pairs 3
python scripts/qualification/core_batch/verification_force_pair.py --baseline-site BASELINE_SITE --candidate-site CANDIDATE_SITE --output strict-forcing --repetitions 2000 --pairs 3
python scripts/qualification/core_batch/review_api_conformance_v2.py --site CANDIDATE_SITE --output public-conformance.json
```

No controller installs packages or downloads a model/vocabulary. The compact benchmark vocabulary is synthetic, so results must not be described as a full JSON-Schema corpus or serving-model sweep. Conventional CPython dispatch/object conversion is part of the Python forcing timings.

`public_consumer_smoke.rs` is a compile-only public API fixture. Use it as
`src/main.rs` in a separate Cargo package with a path dependency on this checkout
and `default-features = false`; do not enable `internal-api`. Its Rust
`forced_tokens()` calls propagate `Result` with `?`.
