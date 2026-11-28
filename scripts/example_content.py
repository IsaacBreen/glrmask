"""Render tested Markdown snippets without their hidden setup."""
def visible_example(markdown):
    lines = markdown.splitlines()
    language = lines[0][3:]
    assert lines[0].startswith("```") and lines[-1] == "```"
    visible = [lines[0]]
    hidden = False
    for line in lines[1:-1]:
        if line in ("# [hidden]", "// [hidden]"):
            assert not hidden
            hidden = True
        elif line in ("# [/hidden]", "// [/hidden]"):
            assert hidden
            hidden = False
        elif not hidden and not (language == "rust" and line.startswith("# ")):
            visible.append(line)
    assert not hidden
    return "\n".join([*visible, lines[-1]])

def full_example(markdown):
    """Return the exact runnable example, removing only authoring markers."""
    lines = markdown.splitlines()
    rust = lines[0] == "```rust"
    return "\n".join(line[2:] if rust and line.startswith("# ") else line
                     for line in lines if line not in ("# [hidden]", "# [/hidden]", "// [hidden]", "// [/hidden]"))


def expandable_example(markdown):
    short, full = visible_example(markdown), full_example(markdown)
    if short == full:
        return short
    return short + "\n\n<details>\n<summary>Full example</summary>\n\n" + full + "\n\n</details>"
