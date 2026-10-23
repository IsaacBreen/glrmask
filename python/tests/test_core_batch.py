"""The public mutable/snapshot and canonical-forcing contracts (no fixture adapter)."""

import copy
from concurrent.futures import ThreadPoolExecutor

import glrmask
import numpy as np
import pytest


MODES = [glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_BUILD,
         glrmask.MaskStrategy.TOKEN_CLASS_TRIE, glrmask.MaskStrategy.FAST_RUNTIME]


def vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"ab", 17: b"", 29: b""})


def test_bind_is_single_item_mutation_and_each_failure_is_atomic():
    v = vocab()
    grammar = glrmask.Module(glrmask.Grammar.from_glrm('extern token M start r nt r=M'))
    alias = grammar
    untouched = grammar.copy()
    assert grammar.bind("M", glrmask.TokenSet([17])) is None
    assert list(alias.externs.unbound) == []
    assert list(untouched.externs.unbound) == ["M"]
    for name, value in [("M", glrmask.TokenSet([29])), ("absent", glrmask.TokenSet([17]))]:
        with pytest.raises(ValueError):
            grammar.bind(name, value)
        assert list(grammar.externs.unbound) == []
    for method in [lambda x: x.copy(), copy.copy, copy.deepcopy]:
        fork = method(untouched)
        assert fork.bind("M", glrmask.TokenSet([29])) is None
        assert list(untouched.externs.unbound) == ["M"]
        assert np.flatnonzero(fork.compile(v).start().mask()).tolist() == [29]


def test_child_and_sibling_snapshots_survive_later_edits():
    v = vocab()
    child_17 = glrmask.Module(glrmask.Grammar.from_glrm('extern token M start r nt r=M'))
    child_17.bind("M", glrmask.TokenSet([17]))
    child_29 = glrmask.Module(glrmask.Grammar.from_glrm('extern token M start r nt r=M'))
    child_29.bind("M", glrmask.TokenSet([29]))

    parent = glrmask.Module(glrmask.Grammar.from_glrm('extern grammar L extern grammar R start r nt r=L R'))
    parent.precompile(v)
    assert parent.bind("L", child_17) is None
    assert parent.bind("R", child_29) is None

    constraint = parent.compile(v)
    state = constraint.start()
    assert state.forced_tokens() == [17, 29]
    for token in state.forced_tokens():
        assert state.mask()[token]
        assert state.commit_token(token)
    assert state.is_accepting()
    assert not hasattr(state, "forced")
    assert not hasattr(state, "forced_token_ids")
    assert not hasattr(state, "forced_bytes")
    assert not hasattr(state, "mask_if_at_most"), "bounded query must stay private"


@pytest.mark.parametrize("mode", MODES)
def test_exact_values_widen_to_grammar_without_becoming_printed_bytes(mode):
    v = vocab()
    for value in [glrmask.TokenSet([17]), glrmask.TokenSet([17, 29])]:
        m1 = glrmask.Module(glrmask.Grammar.from_glrm('extern grammar X start r nt r=X'))
        m2 = glrmask.Module(glrmask.Grammar.from_glrm('extern grammar X start r nt r=X'))
        m2.precompile(v, mask_strategy=mode)
        m1.bind("X", value)
        m2.bind("X", value)
        a = m1.compile(v, mask_strategy=mode)
        b = m2.compile(v, mask_strategy=mode)
        assert np.array_equal(a.start().mask(), b.start().mask())
        for constraint in [a, b, glrmask.Constraint.load(b.save(), v)]:
            assert constraint.start().mask()[17]
            state = constraint.start()
            assert not state.commit_bytes(b"a")
            state = constraint.start()
            assert state.commit_token(17)
            assert state.is_accepting()


@pytest.mark.parametrize("mode", MODES)
def test_canonical_forcing_is_read_only_and_stops_at_acceptance(mode):
    v = vocab()
    state = glrmask.Grammar.from_ebnf('r ::= "ab"').compile(v, mask_strategy=mode).start()
    assert state.forced_tokens() == [2]  # longest spelling of the forced bytes
    assert state.commit_token(0)
    assert state.forced_tokens() == [1]
    before = state.mask().copy()
    assert state.forced_tokens() == [1]
    assert np.array_equal(before, state.mask())
    assert state.commit_token(1)
    assert state.is_accepting() and state.forced_tokens() == []


def test_independent_parallel_mutation_keeps_shared_template_open():
    v = vocab()
    template = glrmask.Module(glrmask.Grammar.from_glrm('extern token M start r nt r=M'))
    template.precompile(v)

    def specialize(index):
        request = template.copy()
        token = 17 if index % 2 == 0 else 29
        request.bind("M", glrmask.TokenSet([token]))
        compiled = request.compile(v)
        return compiled.start().forced_tokens(), token

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(specialize, range(32)))
    assert all(ids == [token] for ids, token in results)
    assert list(template.externs.unbound) == ["M"]


def test_incremental_many_bindings_and_requirements_preserve_all_snapshots():
    size = 512
    text = "\n".join(f"extern token E{i}" for i in range(size)) + "\nstart r nt r=E0"
    g = glrmask.Module(glrmask.Grammar.from_glrm(text))
    original = g.copy()
    assert len(g.externs.unbound) == size
    for i in range(size):
        assert g.bind(f"E{i}", glrmask.TokenSet([17])) is None
    assert len(g.externs.unbound) == 0
    assert len(original.externs.unbound) == size
