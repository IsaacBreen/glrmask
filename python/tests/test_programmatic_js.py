import glrmask


def _vocab() -> glrmask.Vocab:
    pieces = [
        b"tools",
        b".lookup(",
        b")",
        b"{",
        b"}",
        b"status",
        b'"status"',
        b": ",
        b"customer",
        b".status",
        b'"open"',
        b'"closed"',
        b'"bogus"',
        b"x",
        b" ? ",
        b" : ",
        b";",
    ]
    return glrmask.Vocab.from_id_to_bytes(dict(enumerate(pieces)))


def _accepts(constraint: glrmask.Constraint, source: str) -> bool:
    state = constraint.start()
    try:
        state.commit_bytes(source.encode())
    except ValueError:
        return False
    return state.is_finished()


def test_programmatic_js_compiler_shared_parts_and_schema_semantics() -> None:
    vocab = _vocab()

    parent = glrmask.ProgrammaticJsCompiler.compile_parent(vocab)
    dynamic_value = glrmask.ProgrammaticJsCompiler.compile_dynamic_value(vocab)
    condition = glrmask.ProgrammaticJsCompiler.compile_condition(vocab)
    compiler = glrmask.ProgrammaticJsCompiler.from_components(
        parent,
        dynamic_value,
        condition,
    )

    schema = r'''{
      "type":"object",
      "properties":{"status":{"enum":["open","closed"]}},
      "required":["status"],
      "additionalProperties":false
    }'''
    compiled_schema = compiler.compile_schema(schema, vocab)
    dispatcher = compiler.compile_dispatcher({"lookup": compiled_schema}, vocab)
    constraint = compiler.compose_dispatcher(dispatcher, vocab)

    assert _accepts(constraint, 'tools.lookup({status: "open"});')
    assert _accepts(constraint, "tools.lookup({status: customer.status});")
    assert _accepts(constraint, 'tools.lookup({status: x ? "open" : "closed"});')
    assert not _accepts(constraint, 'tools.lookup({status: "bogus"});')
    assert not _accepts(constraint, 'tools.lookup({status: "open" + x});')
    assert not _accepts(constraint, 'tools.lookup({status: x ? "open" : "bogus"});')
    assert not _accepts(constraint, "tools.lookup(customer);")
    assert not _accepts(constraint, 'tools.unknown({status: "open"});')
