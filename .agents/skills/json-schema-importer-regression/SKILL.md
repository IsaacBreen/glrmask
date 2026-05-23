---
name: json-schema-importer-regression
description: Temporary workflow for fixing glrmask JSON-schema importer TBM or build-time regressions after the modular rewrite, using the old monolithic importer only as diagnostic evidence.
---

# JSON Schema Importer Regression

Use this temporary skill when optimizing glrmask JSON-schema importer regressions after the modular importer rewrite, especially when old `src/import/json_schema.rs` had better TBM or build timings than current modular code.

## Targets

- TBM must be below `12us`; `12us` is the hard ceiling.
- Commit must be below `10us`; `10us` is the hard ceiling.
- Build time should be below `10s` for all cases except one or two schemas that were already genuinely slow before the rewrite.
- Treat old pre-overhaul timings as diagnostic baselines, not as a reason to restore the old importer.

## Known Baseline Boundary

- Last monolithic importer baseline: `8512868acef0e01b241247fe0d3ffcfb02b991a6` (`Move JSON schema importer tests to separate file`).
- Modular rewrite commit: `08d8d1764ded847e0e1c05dd5734d38b072b3194` (`Replace JSON schema importer with modular rewrite`).
- For importer-regression work, compare current emitted grammar and stabilized timings against `8512868ace` before assuming a parser/GSS fix is required.
- Cross-product evidence on `/tmp/o9961_ctx_tbm_candidate2.json` showed old-emitted GLRM running on the current runtime recovered most of the advantage (`8.083us` vs current-emitted/current-runtime `13.125us`, old full wheel `9.125us`). Treat this as proof that the primary lever is importer-emitted grammar shape.

## Principles

- Do not rip out the modular importer or paste back old `json_schema.rs`. The old importer was messy and had bad edge behavior; it is evidence, not architecture.
- Do not use `glrmask_native`-only CFA full reports, chunked reports, or broad
  sweep artifacts for importer-regression decisions. Full/report artifacts must
  include the normal comparison framework set so plots, discrepancies, and
  relative regressions remain available. Never set
  `FORCE_DISABLE_LLGUIDANCE_NATIVE` unless the human explicitly asks for a
  narrow glrmask-only diagnostic in that turn.
- Use old grammar/timing/profile artifacts to identify what shape got worse: object body factoring, `anyOf`/`oneOf` branching, pattern/additional-property mixing, bounded string chunks, enum grouping, or terminal sharing.
- Stay on the importer unless fresh cross-product evidence disproves it. Do not drift into parser/GSS/runtime redesign while old-emitted grammar on current runtime is faster.
- Prefer simple current-architecture fixes in `src/import/json_schema/*` that produce cleaner grammars and fewer parser paths.
- When the user asks to optimize JSON-schema build time, keep the lever in JSON-schema importing/lowering first. Optimize downstream compiler cost by changing the generated grammar shape, terminal/key factoring, object-body NFA structure, and sharing decisions. Do not drift into generic compiler or runtime optimization until importer-shape evidence shows there is no importer-side mitigation left.
- For importer-driven build regressions, measure the downstream phase that grew, but interpret it as feedback on import shape: e.g. too many byte-split terminals, too many ExprNFA states/symbols, duplicated value subexpressions, or excessive parser/terminal-DWA work induced by lowering. The preferred fix is a cleaner emitted schema grammar that preserves the TBM win.
- Make good use of `ExprNFA` for object lowering, but inspect the resulting grammar. The NFA structure and emitted grammar structure both affect runtime.
- Simpler grammars usually win: fewer wrapper choices, fewer duplicated object bodies, fewer overlapping key/value alternatives, fewer live parser paths at close tokens.
- Do not keep a patch just because it looks like the old grammar. Keep it only if stabilized timings improve and targets are met or the patch is a measured prerequisite with no regression.
- Exception: when the human explicitly directs a grammar/NFA structure change,
  keep that structural direction even if the first measurement regresses, then
  treat the slowdown as the next regression to mitigate. Record the tradeoff
  clearly instead of reverting by default.

## Workflow

1. Start from stabilized evidence, not single noisy profiles.
   - Run or reuse focused `example-specific`/slow-step reports.
   - Compare against old importer baselines only for the same problem/example/step when possible.

2. Profile the exact current slow step.
   - Record token, prefix, schema location, mask/commit/TBM, parser path counts, tokenizer states, and whether the cost is mask, commit, build, or mixed.
   - If profile buckets do not explain the cost, add narrow instrumentation before editing.

3. Compare current grammar with old grammar.
   - Use old commit `8512868acef0e01b241247fe0d3ffcfb02b991a6` or the relevant pre-overhaul commit.
   - Use `make show-grammar-glrmask` in `constraint-framework-analysis` for corpus examples when possible; it is the fastest way to inspect the emitted GLRM grammar without detouring into runtime internals.
   - Look for structural differences, not code to copy wholesale.
   - Useful old-source command:
     ```bash
     git show 8512868acef0e01b241247fe0d3ffcfb02b991a6:src/import/json_schema.rs
     ```
   - For minimized/custom schemas, dump GLRM with `_glrmask.dump_json_schema_grammar_glrm(schema_json)` from isolated old/current wheels and compare the dumps directly.
   - When switching old/current frequently, prefer isolated wheel or environment paths over repeated global rebuilds. Keep both old and current dump/run artifacts in `/tmp` with explicit names, and record which binary/wheel produced each artifact.

4. Name the exact modular fix before editing.
   - Examples of acceptable targets: a narrower `ExprNFA` object body, required-property `anyOf` factoring, open-object variant factoring, pattern/additional-property key sharing, compact bounded-string shape, or enum terminal grouping.
   - For build-time regressions caused by a runtime-oriented importer shape, name the importer representation to change and the downstream cost it should reduce, such as replacing per-byte literal-key paths with grouped prefix terminals, sharing repeated value expressions behind nonterminals, or reducing duplicated ExprNFA suffix states while keeping the same parser frontier at the hot token.
   - Avoid schema-specific hacks and broad heuristics that only hide one benchmark.

5. Validate with the same focused artifact and then a nearby broader check.
   - First gate importer-shape patches on the warmed minimized reproducer `/tmp/o9961_ctx_tbm_candidate2.json` when the change targets the `o9961` anyOf/object-family regression.
   - For that reproducer, ignore cold round 1; use warmed round 2 or later. Round 1 includes one-time initialization spikes and is not comparable to CFA stabilized rows.
   - For TBM work, run focused timings with enough repetitions and report before/after rows.
   - For build-time work, report before/after build wall time, dominant downstream profile buckets, and the importer-shape stats that explain them (grammar rules, terminal count, ExprNFA states/symbols, parser/DWA state counts when available).
   - If the patch worsens a neighboring known hotspot, revert or narrow it before reporting success.


## Pre-Commit Evidence Checklist

Before committing importer-performance work, verify the commit message and body name the exact problem/schema ID(s) and examples improved, the metric improved (`TBM`, `commit`, `mask`, or build time), the threshold used for accept/reject, and the before/after measurements for those named cases when known. Include measurement artifact or log paths when available. If validation is partial, the commit body must explicitly state what was run, what was not run, and why the remaining measurement was skipped. Also record any paired tradeoff, especially when an importer shape improves runtime but changes build time or vice versa.

Do not accept vague bodies such as "improves TBM" or "reduces build time"; each claimed win must be tied to named cases and numbers.

## Red Flags

- Restoring old monolithic importer logic wholesale.
- Keeping a patch that changes grammar shape but misses `TBM < 12us`.
- Treating `25us` as acceptable TBM.
- Optimizing one case by disabling an existing runtime/table/compression optimization.
- Trusting raw `profile_step` totals over stabilized sweep rows.
- Ignoring build-time regressions while improving TBM.
- Letting overlapping `anyOf` object alternatives survive as top-level branch choices when a simpler `ExprNFA` can represent the same recognizer language.

## Notes Discipline

Keep the dated notes in `/Users/isaacbreen/Projects2/gcg-paper/notes/` current while working. Record:

- old baseline commit and timings,
- current slow-step command/log paths,
- profile and grammar dump paths,
- patch hypothesis,
- before/after stabilized timings,
- keep/revert decision and reason.

Do not report success, pause the investigation, or commit importer-performance work until the dated note has been updated for the completed chunk. The note must include artifact paths and the keep/revert decision, not just a final summary reconstructed later.
