import numpy as np
import pytest
import glrmask

@pytest.mark.parametrize("n", [0, 33, 63, 64])
def test_mask_domain(n):
    v = glrmask.Vocab({n-1: b"a"} if n else {})
    c = glrmask.Grammar.from_lark('start: "a"').compile(v)
    s = c.start()
    assert c.mask_bit_len == s.mask_bit_len == n
    w = (n+31)//32
    p = s.mask()
    assert p.dtype == np.uint32 and p.shape == (w,)
    assert p.flags.writeable and p.flags.c_contiguous
    assert np.array_equal(s.mask(), p)
    dense = np.unpackbits(p.astype('<u4').view(np.uint8), bitorder='little')[:n].astype(bool)
    assert dense.dtype == np.bool_ and dense.shape == (n,)
    assert np.flatnonzero(dense).tolist() == ([n-1] if n else [])
    for i in range(w*32):
        assert bool(int(p[i//32]) & (1 << (i%32))) == (dense[i] if i < n else False)
    for dtype in [np.uint32, bool, np.int32, np.float32, np.bool_, int, None, "uint32", np.dtype(np.uint32)]:
        with pytest.raises(TypeError):
            s.mask(dtype=dtype)
    with pytest.raises(TypeError):
        s.mask(size=n)
    for length in [w,w+2]:
        buf = np.full(length, 0xffffffff, dtype=np.uint32)
        s.fill_mask(buf)
        assert np.array_equal(buf[:w], p) and not np.any(buf[w:])
    with pytest.raises(TypeError):
        s.fill_mask(np.zeros(w + 1, dtype=np.int32))
    with pytest.raises(TypeError):
        s.fill_mask(dense)
    if w:
        buf = np.full(w-1, 0xffffffff, dtype=np.uint32)
        with pytest.raises(glrmask.GLRMaskError) as exc:
            s.fill_mask(buf)
        assert exc.value.kind == "buffer_too_small"
        assert np.all(buf == 0xffffffff)
        assert s.commit_token(n-1)
        s.fill_mask(p)
        assert np.array_equal(p, s.mask())
        assert s.mask_bit_len == n


def test_threaded_fill_mask_matches_sequential():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    v = glrmask.Vocab({i: bytes([97 + i % 3]) for i in range(4097)})
    c = glrmask.Grammar.from_lark('start: "a" "b" "c"').compile(v)
    states = [c.start() for _ in range(8)]
    for i, s in enumerate(states):
        for token in range(i % 3):
            assert s.commit_token(token)
    expected = [s.mask().copy() for s in states]
    # Use fresh states so the first parallel fill need not hit a cached mask.
    states = [s.copy() for s in states]
    barrier = Barrier(len(states))
    def fill(i):
        # Odd threads fill an int32 buffer through a zero-copy uint32 view.
        out = np.zeros((c.mask_bit_len+31)//32+2, dtype=np.int32 if i % 2 else np.uint32)
        barrier.wait(timeout=30)
        for _ in range(5):
            s = states[i]
            s.fill_mask(out.view(np.uint32))
            np.testing.assert_array_equal(out[:-2].view(np.uint32), expected[i])
            assert not np.any(out[-2:])
        return out
    with ThreadPoolExecutor(max_workers=len(states)) as pool:
        results = list(pool.map(fill, range(len(states))))
    assert len(results) == len(states)

def test_mask_bit_len_is_read_only():
    c = glrmask.Grammar.from_lark('start: "a"').compile(glrmask.Vocab({0: b"a"}))
    for owner in [c, c.start()]:
        assert owner.mask_bit_len == 1
        with pytest.raises(AttributeError):
            owner.mask_bit_len = 3

@pytest.mark.parametrize("n", [0, 33, 63, 64])
def test_cached_mask_bit_len_survives_load_and_state_copy(n):
    v = glrmask.Vocab({n-1: b"a"} if n else {})
    original = glrmask.Grammar.from_lark('start: "a"').compile(v)
    restored = glrmask.Constraint.load(original.save(), v)
    assert original.mask_bit_len == restored.mask_bit_len == n
    state = restored.start()
    assert state.copy().mask_bit_len == n
    if n:
        assert state.commit_token(n-1)
    assert state.mask_bit_len == state.copy().mask_bit_len == n

def test_cached_mask_bit_len_includes_exact_token():
    v = glrmask.Vocab({0: b"a", 63: b""})
    module = glrmask.Module(glrmask.Grammar.from_glrm("extern token T start r nt r=T"))
    module.bind("T", glrmask.TokenSet([63]))
    c = module.compile(v)
    for constraint in [c, glrmask.Constraint.load(c.save(), v)]:
        state = constraint.start()
        assert constraint.mask_bit_len == state.mask_bit_len == 64
        assert state.commit_token(63)
        assert state.mask_bit_len == 64


def test_fill_mask_rejects_unaligned_and_swapped_endian_but_accepts_row():
    state = glrmask.Grammar.from_lark('start: "a"').compile(glrmask.Vocab({0: b"a"})).start()
    unaligned = np.ndarray((1,), dtype=np.uint32, buffer=bytearray(5), offset=1)
    nonnative = np.zeros(1, dtype=np.dtype(np.uint32).newbyteorder("S"))
    for invalid in (unaligned, nonnative):
        before = invalid.tobytes()
        with pytest.raises(TypeError):
            state.fill_mask(invalid)
        assert invalid.tobytes() == before
    matrix = np.full((2, 2), 0xffffffff, dtype=np.uint32)
    state.fill_mask(matrix[1])
    assert matrix.tolist() == [[0xffffffff, 0xffffffff], [1, 0]]
