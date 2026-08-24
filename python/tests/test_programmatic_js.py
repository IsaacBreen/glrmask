import glrmask
import glrmask._glrmask as extension


def test_programmatic_js_compiler_is_not_public() -> None:
    assert not hasattr(glrmask, "ProgrammaticJsCompiler")
    assert not hasattr(extension, "ProgrammaticJsCompiler")
    assert "ProgrammaticJsCompiler" not in extension.__all__
