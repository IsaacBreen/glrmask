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
