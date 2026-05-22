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

## Principles

- Do not rip out the modular importer or paste back old `json_schema.rs`. The old importer was messy and had bad edge behavior; it is evidence, not architecture.
- Use old grammar/timing/profile artifacts to identify what shape got worse: object body factoring, `anyOf`/`oneOf` branching, pattern/additional-property mixing, bounded string chunks, enum grouping, or terminal sharing.
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
   - Look for structural differences, not code to copy wholesale.
   - Useful old-source command:
     ```bash
     git show 8512868acef0e01b241247fe0d3ffcfb02b991a6:src/import/json_schema.rs
     ```

4. Name the exact modular fix before editing.
   - Examples of acceptable targets: a narrower `ExprNFA` object body, required-property `anyOf` factoring, open-object variant factoring, pattern/additional-property key sharing, compact bounded-string shape, or enum terminal grouping.
   - For build-time regressions caused by a runtime-oriented importer shape, name the importer representation to change and the downstream cost it should reduce, such as replacing per-byte literal-key paths with grouped prefix terminals, sharing repeated value expressions behind nonterminals, or reducing duplicated ExprNFA suffix states while keeping the same parser frontier at the hot token.
   - Avoid schema-specific hacks and broad heuristics that only hide one benchmark.

5. Validate with the same focused artifact and then a nearby broader check.
   - For TBM work, run focused timings with enough repetitions and report before/after rows.
   - For build-time work, report before/after build wall time, dominant downstream profile buckets, and the importer-shape stats that explain them (grammar rules, terminal count, ExprNFA states/symbols, parser/DWA state counts when available).
   - If the patch worsens a neighboring known hotspot, revert or narrow it before reporting success.

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
