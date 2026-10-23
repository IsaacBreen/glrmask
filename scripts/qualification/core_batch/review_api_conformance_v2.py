"""Independent installed-wheel checks for the GLRMask core batch.

No private Rust/Python helpers and no production edits. Canonical forcing is
checked for cold/warm determinism, admission and read-only behavior. The optional
historical strict contract additionally checks singleton-ID inevitability.
The independent canonical-language oracle lives in tests/forcing_semantics_oracle.rs.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--site", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--forcing-contract", choices=("canonical", "strict"), default="canonical")
args = parser.parse_args()
if args.output.exists():
    raise SystemExit(f"Refusing to overwrite receipt: {args.output}")
sys.path.insert(0, str(args.site))
import glrmask as g
import numpy as np
assert str(Path(g.__file__).resolve()).casefold().startswith(str(args.site.resolve()).casefold()), g.__file__
modes = [g.Optimization.AUTO,g.Optimization.FAST_BUILD,g.Optimization.BALANCED,g.Optimization.FAST_RUNTIME]
started=time.perf_counter()
counters={"forcing_contract":args.forcing_contract,"constraints":0,"states":0,"forced_ids_replayed":0,"bindings":0}
digest=hashlib.sha256()

def strict_reference(state):
    cursor=state.clone()
    out=[]
    for _ in range(10_000):
        if cursor.is_accepting():break
        ids=np.flatnonzero(cursor.mask())
        if len(ids)!=1:break
        id=int(ids[0]);out.append(id);cursor.commit_token(id)
    return out

def verify_state(state):
    # Clone has no current-generation mask cache. Check this path before
    # intentionally computing the source mask for the warm-cache comparison.
    marker_before=state.marker_events()
    cold=state.clone()
    forced_cold=cold.forced_tokens()
    expected=strict_reference(cold) if args.forcing_contract=="strict" else forced_cold
    assert forced_cold==expected,(forced_cold,expected)
    before=state.mask().copy()
    acceptance=state.is_accepting()
    forced=state.forced_tokens()
    assert forced==expected,(forced,expected,np.flatnonzero(before).tolist())
    assert np.array_equal(state.mask(),before)
    assert state.is_accepting()==acceptance
    assert state.marker_events()==marker_before, "forcing or masking altered passive marker events"
    assert before.dtype==np.bool_ and before.itemsize==1
    packed=np.full((len(before)+31)//32+2,-1,dtype=np.int32)
    state.fill_mask(packed)
    unpacked=np.array([(int(packed[i//32])>>(i%32))&1 for i in range(len(before))],dtype=np.bool_)
    assert np.array_equal(unpacked,before)
    assert packed[-2:].tolist()==[0,0]
    cursor=state.clone()
    for id in forced:
        assert not cursor.is_accepting()
        allowed=np.flatnonzero(cursor.mask()).tolist()
        assert id in allowed,(id,allowed)
        if args.forcing_contract=="strict":assert allowed==[id]
        cursor.commit_token(id)
        counters["forced_ids_replayed"]+=1
    counters["states"]+=1
    digest.update(before.tobytes())
    digest.update(bytes([acceptance]))
    digest.update(json.dumps(forced,separators=(",",":")).encode())

def walk(c):
    counters["constraints"]+=1
    assert not hasattr(c.start(),"forced")
    assert not hasattr(c.start(),"forced_token_ids")
    assert not hasattr(c.start(),"forced_bytes")
    assert not hasattr(c.start(),"mask_if_at_most")
    todo=[(c.start(),0)]
    while todo:
        state,depth=todo.pop()
        verify_state(state)
        if depth==2:continue
        for id in np.flatnonzero(state.mask()).tolist()[:8]:
            child=state.clone();child.commit_token(int(id));todo.append((child,depth+1))

v=g.Vocab.from_id_to_bytes({0:b"a",1:b"b",2:b"ab",3:b"abc",4:b"a",5:b" ",6:b"c",7:b"",8:b"x",32:b"\xc3",33:b"\xa9",34:chr(233).encode(),67:b"a",99:b"",130:b""})
for source in [
    'start root nt root="ab" "c"?',
    'start root t X=/a|ab/ nt root=X "c"?',
    'start root ignore SPACE t SPACE=/ +/ nt root="ab"',
    'start root t X=(/a|ab/ - (/a|ab/ b/./s+)) nt root=X "b"?',
    'start root nt root="\u00e9" "a"?',
    'start root nt root=eps',
    'start root nt root=@mark(1) "a" @mark(2) "b" @mark(3)',
    'start root t X=(/a|ab/ - (/a|ab/ b/./s+)) nt root=@mark(11) X @mark(12) "c"?',
]:
    for mode in modes:
        for ends in [[],[130],[4]]:
            c=g.Grammar.from_glrm(source).compile(v,optimization=mode,end_tokens=ends)
            walk(c)
            restored=g.Constraint.load(c.save())
            verify_state(restored.start())

for source in ['extern token M start root nt root=M "a"?',
               'extern token M start root nt root="ab" | M "c"',
               'extern grammar M start root nt root=M "b"',
               'extern token M start root nt root=@mark(7) M @mark(8) "b" @mark(9)']:
    for values in [[],[0],[4],[7],[99],[0,4,7,99]]:
        for mode in modes:
            desc=g.Grammar.from_glrm(source)
            assert desc.bind("M",g.TokenSet(values)) is None
            walk(desc.compile(v,optimization=mode,end_tokens=[130]))

# Snapshot identities, aliases, save/load, strong per-bind failure semantics.
for prepared in [False,True]:
    child=g.Grammar.from_glrm('extern token M start root nt root="a" M')
    parent=g.Grammar.from_glrm('extern grammar L extern grammar R start root nt root=L R')
    if prepared:child=child.prepare(v);parent=parent.prepare(v)
    alias=parent
    assert parent.bind("L",child) is None
    assert parent.bind("R",child) is None
    assert alias is parent
    old=parent.clone()
    copies=[copy.copy(parent),copy.deepcopy(parent)]
    one,two=copy.deepcopy([parent,parent])
    assert one is two and one is not parent
    copies.append(one)
    assert child.bind("M",g.Token(99)) is None
    assert [r.name for r in parent.requirements]==["L::M","R::M"]
    for branch in [parent,*copies]:
        assert branch.bind("L::M",g.Token(7)) is None
        assert branch.bind("R::M",g.Token(99)) is None
        before=[(r.name,r.kind) for r in branch.requirements]
        try:branch.bind("L::M",g.Token(99))
        except ValueError:pass
        else:raise AssertionError("duplicate bind accepted")
        assert [(r.name,r.kind) for r in branch.requirements]==before
        c=branch.link() if prepared else branch.compile(v)
        walk(c)
    assert len(old.requirements)==2
    if prepared:
        load=g.PreparedGrammar.load(old.save(),v)
        assert len(load.requirements)==2

# Independent mutable handles share definitions, not updates.
source="\n".join(f"extern token E{i}" for i in range(128))+"\nstart root nt root=E0"
for prepared in [False,True]:
    template=g.Grammar.from_glrm(source)
    if prepared:template=template.prepare(v)
    def job(seed):
        branch=template.clone()
        names=list(range(128));random.Random(seed).shuffle(names)
        for name in names:assert branch.bind(f"E{name}",g.Token(99)) is None
        assert branch.requirements==[]
        assert len(template.requirements)==128
        return 128
    with ThreadPoolExecutor(max_workers=8) as pool:
        counters["bindings"]+=sum(pool.map(job,range(24)))

args.output.parent.mkdir(parents=True,exist_ok=True)
receipt={"status":"passed","package":g.__file__,"elapsed_seconds":time.perf_counter()-started,"checks":counters,"trace_sha256":digest.hexdigest(),"note":"Independent public installed-wheel oracle, all four optimization modes. Not a performance benchmark."}
args.output.write_text(json.dumps(receipt,indent=2),encoding="utf8")
print(json.dumps(receipt,indent=2))
