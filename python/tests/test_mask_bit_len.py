import numpy as np
import pytest
import glrmask

@pytest.mark.parametrize("n", [0, 33, 63, 64])
def test_mask_domain(n):
    v = glrmask.Vocab({n-1: b"a"} if n else {})
    c = glrmask.Grammar.from_lark('start: "a"').compile(v)
    s = c.start()
    assert c.mask_bit_len() == s.mask_bit_len() == n
    w = (n+31)//32
    p = s.mask()
    assert p.dtype == np.uint32 and p.shape == (w,)
    assert p.flags.writeable and p.flags.c_contiguous
    assert np.array_equal(s.mask(dtype=np.uint32), p)
    dense = s.mask(dtype=bool)
    assert dense.dtype == np.bool_ and dense.shape == (n,)
    assert np.flatnonzero(dense).tolist() == ([n-1] if n else [])
    for i in range(w*32):
        assert bool(int(p[i//32]) & (1 << (i%32))) == (dense[i] if i < n else False)
    for dtype in [np.int32, np.float32, np.bool_, int, None, "uint32", np.dtype(np.uint32)]:
        with pytest.raises(TypeError):
            s.mask(dtype=dtype)
    with pytest.raises(TypeError):
        s.mask(size=n)
    for dtype in [np.uint32, np.int32]:
        for length in [w,w+2]:
            buf = np.full(length, -1 if dtype is np.int32 else 0xffffffff, dtype=dtype)
            s.fill_mask(buf)
            assert np.array_equal(buf[:w].view(np.uint32), p) and not np.any(buf[w:])
    with pytest.raises(TypeError):
        s.fill_mask(dense)
    if w:
        buf = np.full(w-1, 0xffffffff, dtype=np.uint32)
        with pytest.raises(ValueError):
            s.fill_mask(buf)
        assert np.all(buf == 0xffffffff)
        assert s.commit_token(n-1)
        s.fill_mask(p)
        assert np.array_equal(p, s.mask())
        assert s.mask_bit_len() == n
