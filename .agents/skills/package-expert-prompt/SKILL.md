---
name: package-expert-prompt
description: Build a self-contained expert prompt zip for offline or external review. Use when a user asks to make an expert zip, package source for an expert, prepare an expert prompt bundle, or send a prompt plus code snapshot for specialized analysis.
---

# Package Expert Prompt

## Goal

Create a zip that lets an expert return a complete technical solution without
access to the live workspace. Include a clear prompt, the smallest useful source
snapshot, and explicit instructions that the expert should reason
mathematically.

## Output Location

Put generated expert prompt zip files in `/tmp` unless the user specifies
otherwise:

```text
/tmp/
```

Name bundles after the question or area, for example:

```text
/tmp/shared-state-equivalence-expert.zip
```

Do not place generated expert zips under `.agents/` unless the user explicitly
asks for that location.

## Bundle Contents

At minimum include:

- `TASK.md`: the exact question for the expert, with relevant background,
  constraints, what to inspect, and the expected output.
- `references/`: a subfolder containing all source files, tests, logs,
  fixtures, schemas, notes, or other material needed to reason about the task.

The zip should contain only these two top-level entries:

```text
TASK.md
references/
```

Do not put `README.md`, source folders, patches, logs, or any other files at the
archive root. Put every supporting file under `references/`.

Tell the expert that their returned artifact should itself be a zip containing:

- A Markdown document explaining the solution in full, including the reasoning,
  invariants, correctness argument, complexity or performance implications, and
  validation plan.
- For implementation tasks, a complete working implementation as the primary
  deliverable: one or more patch files or full modified files that can be
  applied to the included codebase. Do not ask merely for a plan, blueprint, or
  pseudocode when the user wants code. Pseudocode is acceptable only as
  supplementary explanation or when the user explicitly asks for design only.
- Tests and validation commands needed to prove the implementation.

The expert's returned zip should not include unrelated or unmodified files. If
there are multiple viable solutions, ask the expert to choose and implement the
recommended one as a concrete patch, then describe other options as tradeoffs.
Do not accept "enough code detail to implement it" as the target for coding
tasks; require actual patch files or full modified source files.

For Rust codebase questions, prefer including:

- `src/`
- `tests/` when tests are relevant to the question
- `Cargo.toml`
- `Cargo.lock`

Add extra files only when they materially help the expert: schemas, fixtures,
benchmarks, logs, or focused notes. Avoid large generated outputs, build
artifacts, `.git/`, `target/`, virtualenvs, and unrelated datasets.

## Prompt Requirements

Write `TASK.md` yourself. Do not leave the key question for a worker to infer.
Include:

- The concrete problem or design question.
- The files or modules likely to matter.
- Current assumptions and known constraints.
- What the expert should deliver. For coding tasks, this must say "return a
  complete implementation patch" and list expected tests. Ask for critique,
  algorithm proposals, bug diagnosis, or review checklists only when the user
  asked for analysis rather than implementation.
- Any non-goals or approaches already rejected.

Also instruct the expert to think mathematically: define the objects involved,
state invariants, reason about equivalence or correctness precisely, and explain
why the proposed algorithm preserves the required semantics.

If the expert is expected to review a partially implemented change, include the
current intended semantics and any open risks.

Do not mention `.agents/` paths, local storage conventions, or where the zip was
created in `TASK.md`; those details are for the local packaging workflow only,
not for the expert prompt.

## Build Workflow

Create a temporary staging directory under `/tmp`, copy the selected files into
it, then zip from inside the staging directory so paths inside the archive are
clean.

Example:

```bash
repo=/Users/isaacbreen/Projects2/glrmask2
name=shared-state-equivalence-expert
stage=/tmp/${name}-stage
out="/tmp/${name}.zip"

rm -rf "$stage"
mkdir -p "$stage"
cp "$repo/Cargo.toml" "$repo/Cargo.lock" "$stage/"
mkdir -p "$stage/references"
mv "$stage/Cargo.toml" "$stage/Cargo.lock" "$stage/references/"
cp -R "$repo/src" "$stage/references/src"
cp -R "$repo/tests" "$stage/references/tests"
$EDITOR "$stage/TASK.md"

mkdir -p "$(dirname "$out")"
(cd "$stage" && zip -r "$out" . -x 'target/*' '.git/*')
unzip -l "$out" | sed -n '1,120p'
```

Prefer deterministic shell commands over manual Finder or GUI zipping. Do not
overwrite an existing bundle without checking whether it is still needed.

## Delivery

After creating the zip:

1. Inspect the archive listing to confirm only `TASK.md` and `references/` are
   at the archive root, not nested under a temporary parent directory.
2. Report the path and a short inventory to the user.
3. If unified file sending is available, send the zip to `HUMAN`.
4. Run `git status --short` and confirm the zip is untracked unless the user
   asked for it to be committed.
