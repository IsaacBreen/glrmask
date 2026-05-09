---
name: package-expert-prompt
description: Build a self-contained expert prompt zip for offline or external review. Use when a user asks to make an expert zip, package source for an expert, prepare an expert prompt bundle, or send a prompt plus code snapshot for specialized analysis.
---

# Package Expert Prompt

## Goal

Create a zip that gives an expert enough context to answer a focused technical
question without access to the live workspace. Include a clear prompt and the
smallest useful source snapshot.

## Output Location

Use the repository-local output directory unless the user specifies otherwise:

```text
.agents/expert-prompts/
```

Name bundles after the question or area, for example:

```text
.agents/expert-prompts/shared-state-equivalence-expert.zip
```

Keep generated expert zips untracked unless the user explicitly asks to commit
or publish them.

## Bundle Contents

At minimum include:

- `PROMPT.md`: the exact question for the expert, with relevant background,
  constraints, what to inspect, and the expected output.
- `README.md`: brief inventory of the bundle and how it was created.
- Source files needed to reason about the question.

For Rust codebase questions, prefer including:

- `src/`
- `tests/` when tests are relevant to the question
- `Cargo.toml`
- `Cargo.lock`

Add extra files only when they materially help the expert: schemas, fixtures,
benchmarks, logs, or focused notes. Avoid large generated outputs, build
artifacts, `.git/`, `target/`, virtualenvs, and unrelated datasets.

## Prompt Requirements

Write `PROMPT.md` yourself. Do not leave the key question for a worker to infer.
Include:

- The concrete problem or design question.
- The files or modules likely to matter.
- Current assumptions and known constraints.
- What the expert should deliver, such as critique, algorithm proposal, bug
  diagnosis, or review checklist.
- Any non-goals or approaches already rejected.

If the expert is expected to review a partially implemented change, include the
current intended semantics and any open risks.

## Build Workflow

Create a temporary staging directory under `/tmp`, copy the selected files into
it, then zip from inside the staging directory so paths inside the archive are
clean.

Example:

```bash
repo=/Users/isaacbreen/Projects2/glrmask2
name=shared-state-equivalence-expert
stage=/tmp/${name}
out="$repo/.agents/expert-prompts/${name}.zip"

rm -rf "$stage"
mkdir -p "$stage"
cp "$repo/Cargo.toml" "$repo/Cargo.lock" "$stage/"
cp -R "$repo/src" "$stage/src"
cp -R "$repo/tests" "$stage/tests"
$EDITOR "$stage/PROMPT.md"
cat > "$stage/README.md" <<'EOF'
# Expert Bundle

This archive contains a focused prompt plus source snapshot for expert review.
Start with PROMPT.md.
EOF

mkdir -p "$(dirname "$out")"
(cd "$stage" && zip -r "$out" . -x 'target/*' '.git/*')
unzip -l "$out" | sed -n '1,120p'
```

Prefer deterministic shell commands over manual Finder or GUI zipping. Do not
overwrite an existing bundle without checking whether it is still needed.

## Delivery

After creating the zip:

1. Inspect the archive listing to confirm `PROMPT.md` and source files are at
   the archive root, not nested under a temporary parent directory.
2. Report the path and a short inventory to the user.
3. If unified file sending is available, send the zip to `HUMAN`.
4. Run `git status --short` and confirm the zip is untracked unless the user
   asked for it to be committed.

