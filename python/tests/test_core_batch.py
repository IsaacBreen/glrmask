"""The public mutable/snapshot and canonical-forcing contracts (no fixture adapter)."""

import copy
from concurrent.futures import ThreadPoolExecutor

import glrmask
import numpy as np
import pytest


MODES = [glrmask.Optimization.AUTO, glrmask.Optimization.FAST_BUILD,
         glrmask.Optimization.BALANCED, glrmask.Optimization.FAST_RUNTIME]


def vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"ab", 17: b"", 29: b""})


def test_bind_is_single_item_mutation_and_each_failure_is_atomic():
    v = vocab()
    grammar = glrmask.Grammar.from_glrm('extern token M start r nt r=M')
    alias = grammar
    untouched = grammar.clone()
    assert grammar.bind("M", glrmask.Token(17)) is None
    assert alias.requirements == []
    assert [r.name for r in untouched.requirements] == ["M"]
    for name, value in [("M", glrmask.Token(29)), ("absent", glrmask.Token(17))]:
        with pytest.raises(ValueError):
            grammar.bind(name, value)
        assert grammar.requirements == []
    for method in [lambda x: x.clone(), copy.copy, copy.deepcopy]:
        fork = method(untouched)
        assert fork.bind("M", glrmask.Token(29)) is None
        assert untouched.requirements[0].name == "M"
        assert np.flatnonzero(fork.compile(v).start().mask()).tolist() == [29]


@pytest.mark.parametrize("prepared", [False, True])
def test_child_and_sibling_snapshots_survive_later_edits(prepared):
    v = vocab()
    child = glrmask.Grammar.from_glrm('extern token M start r nt r=M')
    parent = glrmask.Grammar.from_glrm('extern grammar L extern grammar R start r nt r=L R')
    if prepared:
        child, parent = child.prepare(v), parent.prepare(v)
    assert parent.bind("L", child) is None
    assert parent.bind("R", child) is None
    frozen = parent.clone()
    child.bind("M", glrmask.Token(29))
    assert [r.name for r in parent.requirements] == ["L::M", "R::M"]
    assert parent.bind("L::M", glrmask.Token(17)) is None
    assert [r.name for r in parent.requirements] == ["R::M"]
    assert [r.name for r in frozen.requirements] == ["L::M", "R::M"]
    if prepared:
        parent = glrmask.PreparedGrammar.load(parent.save(), vocab=v)
    parent.bind("R::M", glrmask.Token(29))
    constraint = parent.link() if prepared else parent.compile(v)
    state = constraint.start()
    assert state.forced_tokens() == [17, 29]
    for token in state.forced_tokens():
        assert state.mask()[token]
        state.commit_token(token)
    assert state.is_accepting()
    assert not hasattr(state, "forced")
    assert not hasattr(state, "forced_token_ids")
    assert not hasattr(state, "forced_bytes")
    assert not hasattr(state, "mask_if_at_most"), "bounded query must stay private"


@pytest.mark.parametrize("mode", MODES)
def test_exact_values_widen_to_grammar_without_becoming_printed_bytes(mode):
    v = vocab()
    for value in [glrmask.Token(17), v.tokens([17, 29])]:
        source = glrmask.Grammar.from_glrm('extern grammar X start r nt r=X')
        prepared = source.prepare(v)
        source.bind("X", value)
        prepared.bind("X", value)
        a = source.compile(v, optimization=mode)
        b = prepared.link(optimization=mode)
        assert np.array_equal(a.start().mask(), b.start().mask())
        for constraint in [a, b, glrmask.Constraint.load(b.save())]:
            assert constraint.start().mask()[17]
            state = constraint.start()
            with pytest.raises(ValueError):
                state.commit_bytes(b"a")
            state = constraint.start()
            state.commit_token(17)
            assert state.is_accepting()


@pytest.mark.parametrize("mode", MODES)
def test_canonical_forcing_is_read_only_and_stops_at_acceptance(mode):
    v = vocab()
    state = glrmask.Grammar.from_ebnf('r ::= "ab"').compile(v, optimization=mode).start()
    assert state.forced_tokens() == [2]  # longest spelling of the forced bytes
    state.commit_token(0)
    assert state.forced_tokens() == [1]
    before = state.mask().copy()
    assert state.forced_tokens() == [1]
    assert np.array_equal(before, state.mask())
    state.commit_token(1)
    assert state.is_accepting() and state.forced_tokens() == []


def test_independent_parallel_mutation_keeps_shared_template_open():
    v = vocab()
    template = glrmask.Grammar.from_glrm('extern token M start r nt r=M').prepare(v)

    def specialize(index):
        request = template.clone()
        token = 17 if index % 2 == 0 else 29
        request.bind("M", glrmask.Token(token))
        compiled = request.link(optimization=glrmask.Optimization.FAST_BUILD)
        return compiled.start().forced_tokens(), token

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(specialize, range(32)))
    assert all(ids == [token] for ids, token in results)
    assert [r.name for r in template.requirements] == ["M"]


def test_incremental_many_bindings_and_requirements_preserve_all_snapshots():
    size = 512
    text = "\n".join(f"extern token E{i}" for i in range(size)) + "\nstart r nt r=E0"
    g = glrmask.Grammar.from_glrm(text)
    original = g.clone()
    assert len(g.requirements) == size
    for i in range(size):
        assert g.bind(f"E{i}", glrmask.Token(17)) is None
    assert g.requirements == []
    assert len(original.requirements) == size
    assert [r.name for r in original.requirements] == sorted(f"E{i}" for i in range(size))


@pytest.mark.parametrize("mode", MODES)
def test_shortest_braces_are_compositional_not_parser_priority(mode):
    v = glrmask.Vocab.from_id_to_bytes({0: b"ac", 1: b"abc", 2: b"a", 3: b"b", 4: b"c"})
    local = glrmask.Grammar.from_glrm('t A=/a|ab/ t T=shortest { A } "c" start r nt r=T')
    whole = glrmask.Grammar.from_glrm('t A=/a|ab/ t T=shortest { A "c" } start r nt r=T')
    assert local.compile(v, optimization=mode).start().mask()[0]
    assert not local.compile(v, optimization=mode).start().mask()[1]
    mask = whole.compile(v, optimization=mode).start().mask()
    assert mask[0] and mask[1]
