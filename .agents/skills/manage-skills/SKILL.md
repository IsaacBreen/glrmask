---
name: manage-skills
description: Create, update, split, move, and git-commit Codex skills. Use when adding a new skill, editing an existing skill, converting notes or workflows into skills, improving skills after use, splitting one skill into multiple skills, moving skills between global and repo-local locations, or ensuring skill changes are committed for self-improvement over time.
---

# Manage Skills

## Goal

Maintain skills as durable agent knowledge. Keep them concise, triggerable, and committed so future agents inherit improvements.

Use the built-in `$skill-creator` guidance when available, especially for new skills, interface metadata, and progressive disclosure structure.

## Location

Put skills where the user asks.

When the user asks for repo-local skills, use:

```text
$REPO_ROOT/.agents/skills/<skill-name>/
```

When the user asks for globally discoverable skills, use:

```text
${CODEX_HOME:-$HOME/.codex}/skills/<skill-name>/
```

If the location is ambiguous, prefer repo-local `.agents/skills` when the skill encodes project or team workflow. Prefer global skills only for broadly reusable personal workflows.

## Naming

Use lowercase hyphen-case with letters, digits, and hyphens only. Prefer short verb-led names such as `manage-skills`, `debug-parser`, or `publish-release`.

Keep folder name, frontmatter `name`, and invocation examples aligned.

## Creating Skills

Initialize new skills with the skill-creator initializer when available:

```bash
python3 /path/to/skill-creator/scripts/init_skill.py <skill-name> --path <skills-dir> --interface display_name='...' --interface short_description='...' --interface default_prompt='Use $<skill-name> to ...'
```

Then replace the template `SKILL.md` with concise, imperative instructions. Include only the information a future agent needs to do the work.

Use `references/` for detailed or long supporting material that should be loaded only when needed. Use `scripts/` for deterministic repeated operations. Use `assets/` only for files that are copied or used in outputs.

Do not add auxiliary documentation such as `README.md`, `INSTALLATION_GUIDE.md`, `QUICK_REFERENCE.md`, or changelogs unless the user explicitly asks for them.

## Updating Skills

Inspect the existing skill before editing. Preserve useful trigger language in frontmatter, but make it specific enough that the skill triggers only in appropriate contexts.

When splitting a skill, keep each skill's frontmatter scoped to its own task and add a short pointer to the sibling skill where handoff is useful.

When moving a skill, remove the old copy after confirming the new copy is in the requested location and has the expected files.

Keep `SKILL.md` lean. Move long examples, runbooks, schemas, policies, or copied notes into `references/` and mention when to read them.

## Git Discipline

Skill changes should be git committed.

Before committing, inspect `git status --short` and stage only skill-related files unless the user explicitly asks to include other work. Do not stage unrelated code, generated scratch files, or dirty files you did not intentionally change.

Use a focused commit message, for example:

```bash
git add .agents/skills/<skill-name>
git commit -m "Add <skill-name> skill"
```

When updating multiple related skills together, commit them together only if the change is conceptually one update. Otherwise make separate commits.

After committing, report the commit hash and any remaining unrelated dirty files.
