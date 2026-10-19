"""Small canonical assembly example; run against the installed glrmask wheel."""
from glrmask import Expr, Grammar, Module, ModuleOptions, Optimization, Vocab

vocab = Vocab.from_id_to_bytes({0: b'xay', 1: b'xby'})
child = Module(Grammar.from_ebnf('start ::= "a"'),
               options=ModuleOptions(optimization=Optimization.BALANCED))
child.precompile(vocab)
parent = Grammar.from_glrm(
    'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
).module()
parent.bind('child', child)
restored = Module.load(parent.save(), vocab)
assert restored.compile(vocab).start().mask().tolist() == [True, False]

choice = Expr.choice([child, Grammar.from_ebnf('start ::= "b"')])
assembled = Module(Expr.sequence([
    Grammar.from_ebnf('start ::= "x"'), choice, Grammar.from_ebnf('start ::= "y"')
]))
assert assembled.compile(vocab).start().mask().tolist() == [True, True]
