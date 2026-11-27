//! Expression operators have no source, lexical scope or path segment of their own.
use super::*;
use super::module::sealed::Binding;

/// A grammar expression.
#[derive(Debug, Clone)]
pub struct Expr(pub(crate) Arc<ExprNode>);

#[derive(Debug, Clone)]
pub(crate) enum ExprLeaf {
    Grammar(Grammar),
    Module(Module),
    TokenSet(TokenSet),
}

#[derive(Debug, Clone)]
pub(crate) enum ExprNode {
    Leaf(ExprLeaf),
    Literal(Vec<u8>),
    Choice(Vec<Expr>),
    Sequence(Vec<Expr>),
}

impl From<&Expr> for Expr {
    fn from(e: &Expr) -> Self { e.clone() }
}
impl From<Module> for Expr {
    fn from(m: Module) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::Module(m))))
    }
}
impl From<&Module> for Expr {
    fn from(m: &Module) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::Module(m.clone()))))
    }
}
impl From<Grammar> for Expr {
    fn from(g: Grammar) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::Grammar(g))))
    }
}
impl From<&Grammar> for Expr {
    fn from(g: &Grammar) -> Self {
        Expr::from(g.clone())
    }
}
impl From<TokenSet> for Expr {
    fn from(t: TokenSet) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::TokenSet(t))))
    }
}
impl From<&TokenSet> for Expr {
    fn from(t: &TokenSet) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::TokenSet(t.clone()))))
    }
}

impl Expr {
    /// Matches exactly `bytes`.
    pub fn literal(bytes: impl AsRef<[u8]>) -> Self {
        Self(Arc::new(ExprNode::Literal(bytes.as_ref().to_vec())))
    }

    /// Matches `parts` in order.
    pub fn sequence(parts: impl IntoIterator<Item = Expr>) -> Self {
        Self(Arc::new(ExprNode::Sequence(parts.into_iter().collect())))
    }

    /// Matches any one of `parts`.
    pub fn choice(parts: impl IntoIterator<Item = Expr>) -> Self {
        let list: Vec<Expr> = parts.into_iter().collect();
        if list.len() == 1 {
            list.into_iter().next().unwrap()
        } else {
            Self(Arc::new(ExprNode::Choice(list)))
        }
    }

    pub(crate) fn skeleton(&self, allow_modules: bool) -> Result<(Grammar, Vec<(String, Binding)>)> {
        fn walk(
            expr: &Expr,
            leaves: &mut Vec<(String, Binding)>,
            externs: &mut Vec<(String, ExternKind)>,
            depth: usize,
            count: &mut usize,
            allow_modules: bool,
        ) -> Result<crate::grammar::ast::GrammarExpr> {
            *count += 1;
            if depth > graph::MAX_DEPTH || *count > graph::MAX_NODES {
                return Err(Error::ResourceLimit("the expression is too large or too deeply nested".into()));
            }
            Ok(match expr.0.as_ref() {
                ExprNode::Leaf(leaf) => {
                    let name = format!("leaf{}", leaves.len());
                    let (kind, binding) = match leaf {
                        ExprLeaf::Grammar(g) => (ExternKind::Grammar, Binding::Grammar(g.clone())),
                        ExprLeaf::Module(m) => {
                            if !allow_modules {
                                return Err(Error::InvalidInput("a Module can't be part of a Grammar; bind it to a Module instead".into()));
                            }
                            (ExternKind::Grammar, Binding::Module(m.clone()))
                        },
                        ExprLeaf::TokenSet(t) => {
                            if !allow_modules {
                                return Err(Error::InvalidInput("token sets can only be bound to a Module".into()));
                            }
                            (ExternKind::Token, Binding::TokenSet(t.clone()))
                        },
                    };
                    externs.push((name.clone(), kind));
                    leaves.push((name.clone(), binding));
                    crate::grammar::ast::GrammarExpr::Ref(name)
                }
                ExprNode::Literal(bytes) => {
                    crate::grammar::ast::GrammarExpr::Literal(bytes.clone())
                }
                ExprNode::Choice(parts) => {
                    if parts.is_empty() {
                        crate::grammar::ast::GrammarExpr::Choice(Vec::new())
                    } else if parts.len() == 1 {
                        walk(&parts[0], leaves, externs, depth + 1, count, allow_modules)?
                    } else {
                        let walked = parts.iter().map(|p| walk(p, leaves, externs, depth + 1, count, allow_modules)).collect::<Result<Vec<_>>>()?;
                        crate::grammar::ast::GrammarExpr::Choice(walked)
                    }
                }
                ExprNode::Sequence(parts) => {
                    if parts.is_empty() {
                        crate::grammar::ast::GrammarExpr::Epsilon
                    } else if parts.len() == 1 {
                        walk(&parts[0], leaves, externs, depth + 1, count, allow_modules)?
                    } else {
                        let walked = parts.iter().map(|p| walk(p, leaves, externs, depth + 1, count, allow_modules)).collect::<Result<Vec<_>>>()?;
                        crate::grammar::ast::GrammarExpr::Sequence(walked)
                    }
                }
            })
        }
        let mut leaves = Vec::new();
        let mut externs = Vec::new();
        let body = walk(self, &mut leaves, &mut externs, 0, &mut 0, allow_modules)?;
        let named = crate::grammar::ast::NamedGrammar {
            start: "expr_root".into(),
            rules: vec![crate::grammar::ast::NamedRule {
                name: "expr_root".into(),
                expr: body,
                is_terminal: false,
                is_internal: false,
            }],
            ignore: None,
            lexer_partitions: BTreeMap::new(),
            lexer_literal_partitions: BTreeMap::new(),
            default_lexer_partition: None,
        };
        let grammar = Grammar::new(GrammarSource::Constructed);
        let _ = grammar.parsed_component.set(ParsedComponent::from_named(named, &externs).map_err(crate::error::from_grammar));
        grammar.component()?;
        Ok((grammar, leaves))
    }
}

pub(crate) fn check_expr_unbound(expr: &Expr) -> Vec<String> {
    let mut unbound = Vec::new();
    fn walk(e: &Expr, out: &mut Vec<String>) {
        match e.0.as_ref() {
            ExprNode::Leaf(ExprLeaf::Grammar(g)) => {
                for u in g.externs().unbound().iter() {
                    out.push(u.name().to_string());
                }
            }
            ExprNode::Leaf(ExprLeaf::Module(m)) => {
                for u in m.externs().unbound().iter() {
                    out.push(u.name().to_string());
                }
            }
            ExprNode::Choice(parts) | ExprNode::Sequence(parts) => {
                for p in parts {
                    walk(p, out);
                }
            }
            _ => {}
        }
    }
    walk(expr, &mut unbound);
    unbound
}
