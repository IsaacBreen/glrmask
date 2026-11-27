"""Small canonical assembly example; run against the installed glrmask wheel."""
from glrmask import Expr, Grammar, Module, MaskStrategy, Vocab

vocab = Vocab({0: b'xay', 1: b'xby'})
child = Module(Grammar.from_lark('start: "a"'))
child.precompile(vocab, mask_strategy=MaskStrategy.FAST_MASKS)
parent = Module(Grammar.from_glrm(
    'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
))
parent.bind('child', child)
restored = Module.load(parent.save())
assert restored.compile(vocab).start().mask().tolist() == [1]

choice = Expr.choice(child, Grammar.from_lark('start: "b"'))
parent2 = Module(Grammar.from_glrm(
    'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
))
parent2.bind('child', choice)
assert parent2.compile(vocab).start().mask().tolist() == [3]
