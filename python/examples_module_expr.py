"""Small canonical assembly example; run against the installed glrmask wheel."""
from glrmask import Expr, Grammar, MaskStrategy, Module, Vocab

vocab = Vocab.from_id_to_bytes({0: b'xay', 1: b'xby'})
child = Module(Grammar.from_ebnf('start ::= "a"'),
               mask_strategy=MaskStrategy.TOKEN_CLASS_TRIE)
child.precompile(vocab)
parent = Module(Grammar.from_glrm(
    'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
))
parent.bind(child=child)
restored = Module.load(parent.save(), vocab)
assert restored.compile(vocab).start().mask().tolist() == [True, False]

choice = Expr.choice(child, Grammar.from_ebnf('start ::= "b"'))
parent2 = Module(Grammar.from_glrm(
    'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
))
parent2.bind(child=choice)
assert parent2.compile(vocab).start().mask().tolist() == [True, True]
