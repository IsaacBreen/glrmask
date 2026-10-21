//! Expression operators have no source, lexical scope or path segment of their own.
use super::*;

/// Immutable unordered union or concatenation of independently scoped grammars.
#[derive(Debug, Clone)]
pub struct Expr(pub(crate) Arc<ExprNode>);

#[derive(Debug, Clone)]
pub(crate) enum ExprLeaf {
    Module(Module),
    Token(Token),
    ExactTokens(ExactTokens),
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
impl From<Grammar<'_>> for Expr {
    fn from(g: Grammar<'_>) -> Self {
        Expr::from(Module::new(&g))
    }
}
impl From<&Grammar<'_>> for Expr {
    fn from(g: &Grammar<'_>) -> Self {
        Expr::from(Module::new(g))
    }
}
impl From<Token> for Expr {
    fn from(t: Token) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::Token(t))))
    }
}
impl From<&Token> for Expr {
    fn from(t: &Token) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::Token(*t))))
    }
}
impl From<ExactTokens> for Expr {
    fn from(t: ExactTokens) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::ExactTokens(t))))
    }
}
impl From<&ExactTokens> for Expr {
    fn from(t: &ExactTokens) -> Self {
        Expr(Arc::new(ExprNode::Leaf(ExprLeaf::ExactTokens(t.clone()))))
    }
}

impl Expr {
    /// Union without ordered-choice priority. Empty choice matches nothing. One-part choice is that part.
    pub fn choice(items: impl IntoIterator<Item = impl Into<Expr>>) -> Self {
        let list: Vec<Expr> = items.into_iter().map(Into::into).collect();
        if list.len() == 1 {
            list.into_iter().next().unwrap()
        } else {
            Self(Arc::new(ExprNode::Choice(list)))
        }
    }

    /// Concatenation; a model token may cross any number of operand boundaries. Empty sequence is epsilon.
    pub fn sequence(items: impl IntoIterator<Item = impl Into<Expr>>) -> Self {
        Self(Arc::new(ExprNode::Sequence(items.into_iter().map(Into::into).collect())))
    }

    /// Match exact bytes (no whitespace or ignore inserted).
    pub fn literal(bytes: impl AsRef<[u8]>) -> Self {
        Self(Arc::new(ExprNode::Literal(bytes.as_ref().to_vec())))
    }

    pub(crate) fn skeleton(&self) -> Result<(Grammar<'static>, Vec<(String, Binding)>)> {
        fn walk(
            expr: &Expr,
            leaves: &mut Vec<(String, Binding)>,
            externs: &mut Vec<(String, ExternalKind)>,
            depth: usize,
            count: &mut usize,
        ) -> Result<crate::grammar::ast::GrammarExpr> {
            *count += 1;
            if depth > graph::MAX_DEPTH || *count > graph::MAX_NODES {
                return Err(Error::Compilation("expression graph exceeds depth/node limits".into()));
            }
            Ok(match expr.0.as_ref() {
                ExprNode::Leaf(leaf) => {
                    let name = format!("leaf{}", leaves.len());
                    let (kind, binding) = match leaf {
                        ExprLeaf::Module(m) => (ExternalKind::Grammar, Binding::from(m.clone())),
                        ExprLeaf::Token(t) => (ExternalKind::Token, Binding::from(*t)),
                        ExprLeaf::ExactTokens(t) => (ExternalKind::Token, Binding::from(t.clone())),
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
                        walk(&parts[0], leaves, externs, depth + 1, count)?
                    } else {
                        let walked = parts.iter().map(|p| walk(p, leaves, externs, depth + 1, count)).collect::<Result<Vec<_>>>()?;
                        crate::grammar::ast::GrammarExpr::Choice(walked)
                    }
                }
                ExprNode::Sequence(parts) => {
                    if parts.is_empty() {
                        crate::grammar::ast::GrammarExpr::Epsilon
                    } else if parts.len() == 1 {
                        walk(&parts[0], leaves, externs, depth + 1, count)?
                    } else {
                        let walked = parts.iter().map(|p| walk(p, leaves, externs, depth + 1, count)).collect::<Result<Vec<_>>>()?;
                        crate::grammar::ast::GrammarExpr::Sequence(walked)
                    }
                }
            })
        }
        let mut leaves = Vec::new();
        let mut externs = Vec::new();
        let body = walk(self, &mut leaves, &mut externs, 0, &mut 0)?;
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
        let _ = grammar.parsed_component.set(ParsedComponent::from_named(named, &externs).map_err(Into::into));
        grammar.component()?;
        Ok((grammar, leaves))
    }
}

pub(crate) fn check_expr_unbound(expr: &Expr) -> Vec<String> {
    let mut unbound = Vec::new();
    fn walk(e: &Expr, out: &mut Vec<String>) {
        match e.0.as_ref() {
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
