//! Expression operators have no source, lexical scope or path segment of their own.
use super::*;

/// Immutable unordered union or concatenation of independently scoped grammars.
#[derive(Debug, Clone)]
pub struct Expr(Arc<ExprNode>);
#[derive(Debug)]
enum ExprNode { Leaf(Module), Choice(Vec<Expr>), Sequence(Vec<Expr>) }
/// Convert a whole grammar or configured module into one expression leaf.
pub trait IntoExpr { fn into_expr(self)->Expr; }
impl IntoExpr for Expr { fn into_expr(self)->Expr {self} }
impl IntoExpr for &Expr { fn into_expr(self)->Expr {self.clone()} }
impl IntoExpr for Module { fn into_expr(self)->Expr {Expr(Arc::new(ExprNode::Leaf(self)))} }
impl IntoExpr for &Module { fn into_expr(self)->Expr {self.clone().into_expr()} }
impl IntoExpr for Grammar<'_> {fn into_expr(self)->Expr {self.module().into_expr()} }
impl IntoExpr for &Grammar<'_> {fn into_expr(self)->Expr {self.module().into_expr()} }

impl Expr {
    /// Union without ordered-choice priority. Empty choice is rejected on conversion to source/module.
    pub fn choice<T:IntoExpr>(items:impl IntoIterator<Item=T>)->Self {
        Self(Arc::new(ExprNode::Choice(items.into_iter().map(IntoExpr::into_expr).collect())))
    }
    /// Concatenation; a model token may cross any number of operand boundaries. Empty sequence is epsilon.
    pub fn sequence<T:IntoExpr>(items:impl IntoIterator<Item=T>)->Self {
        Self(Arc::new(ExprNode::Sequence(items.into_iter().map(IntoExpr::into_expr).collect())))
    }
    fn skeleton(&self)->Result<(Grammar<'static>,Vec<Module>)> {
        fn walk(expr:&Expr,leaves:&mut Vec<Module>,depth:usize,count:&mut usize)->Result<crate::grammar::ast::GrammarExpr> {
            *count+=1;
            if depth>graph::MAX_DEPTH || *count>graph::MAX_NODES {
                return Err(Error::Compilation("expression graph exceeds depth/node limits".into()));
            }
            Ok(match expr.0.as_ref() {
                ExprNode::Leaf(module)=>{let name=format!("leaf{}",leaves.len());leaves.push(module.clone());crate::grammar::ast::GrammarExpr::Ref(name)},
                ExprNode::Choice(parts)=>{
                    if parts.is_empty(){return Err(Error::Compilation("empty Expr::choice is unsupported; use a nonempty choice".into()));}
                    let parts=parts.iter().map(|p|walk(p,leaves,depth+1,count)).collect::<Result<Vec<_>>>()?;
                    crate::grammar::ast::GrammarExpr::Choice(parts)
                },
                ExprNode::Sequence(parts)=>{
                    let parts=parts.iter().map(|p|walk(p,leaves,depth+1,count)).collect::<Result<Vec<_>>>()?;
                    crate::grammar::ast::GrammarExpr::Sequence(parts)
                },
            })
        }
        let mut leaves=Vec::new();let body=walk(self,&mut leaves,0,&mut 0)?;
        let externs=(0..leaves.len()).map(|i|(format!("leaf{i}"),ExternalKind::Grammar)).collect::<Vec<_>>();
        let named=crate::grammar::ast::NamedGrammar {
            start:"expr_root".into(),
            rules:vec![crate::grammar::ast::NamedRule {name:"expr_root".into(),expr:body,is_terminal:false,is_internal:false}],
            ignore:None,lexer_partitions:BTreeMap::new(),lexer_literal_partitions:BTreeMap::new(),default_lexer_partition:None,
        };
        let grammar=Grammar::new(GrammarSource::Constructed);
        let _=grammar.parsed_component.set(ParsedComponent::from_named(named,&externs).map_err(Into::into));
        grammar.component()?;Ok((grammar,leaves))
    }
}
impl Module {
    /// Preserve configured and prepared leaves. Positional paths are `leaf0`, `leaf1`, ...;
    /// grouping operators add no path components.
    pub fn from_expr(expr:&Expr)->Result<Self> {
        let (grammar,leaves)=expr.skeleton()?;
        let mut module=Module::new(&grammar);
        for (i,leaf) in leaves.into_iter().enumerate(){module.bind(format!("leaf{i}"),leaf)?;}
        Ok(module)
    }
}
impl Grammar<'static> {
    /// Only source operands are accepted. Configured or prepared modules require [`Module::from_expr`].
    pub fn from_expr(expr:&Expr)->Result<Self> {
        let (mut grammar,leaves)=expr.skeleton()?;
        for (i,leaf) in leaves.into_iter().enumerate(){grammar.bind(format!("leaf{i}"),leaf.to_source()?)?;}
        Ok(grammar)
    }
}
