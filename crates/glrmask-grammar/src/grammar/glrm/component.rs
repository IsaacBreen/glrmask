//! Parsed, isolated component descriptions shared by source frontends.
//! This is a compiler boundary, not a GLRM text-translation interface.
use super::*;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ParsedComponent {
    scope: Arc<ParsedGlrmScope>,
    declarations: Arc<GlrmExternalDeclarations>,
}

impl ParsedComponent {
    pub fn parse_glrm(source: &str) -> Result<Self, GlrMaskError> {
        let mut parser=GlrmParser::from_source(source)?;
        if parser.version!=GlrmVersion::V1 {
            return Err(parser.locate(err("legacy GLRM syntax is not supported by Grammar; use '=' declarations and explicit typed externs"),0));
        }
        parser.strict_native=true;
        Self::from_scope(parser.parse_root_scope()?)
    }

    /// Unstable repository tooling only. Public Grammar constructors select the
    /// strict native surface above; old fixtures must opt into this separate path.
    #[doc(hidden)]
    pub fn parse_glrm_for_tools(source:&str)->Result<Self,GlrMaskError> {
        let mut parser=GlrmParser::from_source(source)?;
        Self::from_scope(parser.parse_root_scope()?)
    }

    fn from_scope(scope:ParsedGlrmScope) -> Result<Self,GlrMaskError> {
        validate_scope_for_v1_format(&scope,"grammar")?;
        let kinds=collect_external_kinds(&scope);
        let declarations=GlrmExternalDeclarations {names:kinds.keys().cloned().collect(),kinds,spans:collect_external_spans(&scope)};
        Ok(Self {scope:Arc::new(scope),declarations:Arc::new(declarations)})
    }

    /// Construct a component from another frontend's AST. Its lexical identity,
    /// start and ignore remain scoped; no conversion through GLRM source occurs.
    pub fn from_named(grammar: NamedGrammar, externs: &[(String,ExternalKind)]) -> Result<Self,GlrMaskError> {
        Self::from_named_with_ignores(grammar,externs,&[])
    }

    pub fn from_named_with_ignores(grammar:NamedGrammar,externs:&[(String,ExternalKind)],additional_ignores:&[String])->Result<Self,GlrMaskError> {
        let scope=ParsedGlrmScope {
            extra_ignores:additional_ignores.to_vec(),
            rules:grammar.rules,start:grammar.start,ignore:grammar.ignore,
            lexer_partitions:grammar.lexer_partitions,
            lexer_literal_partitions:grammar.lexer_literal_partitions,
            default_lexer_partition:grammar.default_lexer_partition,
            all_literals_partition:None,external_terminals:Vec::new(),
            subgrammars:externs.iter().map(|(name,kind)|ParsedSubgrammar {
                name:name.clone(),body:ParsedSubgrammarBody::External(*kind),source_span:None,
            }).collect(),
        };
        Self::from_scope(scope)
    }

    /// Attach original frontend declaration coordinates without changing any
    /// recognition node or lexical identity. Constructed nodes may have no span.
    pub fn with_source_spans(&self,spans:&BTreeMap<String,SourceSpan>)->Result<Self,GlrMaskError> {
        if spans.keys().any(|key|!self.declarations.kinds.contains_key(key)) {
            return Err(err("source span refers to an unknown component requirement"));
        }
        fn walk(scope:&mut ParsedGlrmScope,prefix:&str,spans:&BTreeMap<String,SourceSpan>) {
            let qualify=|name:&str|if prefix.is_empty(){name.to_owned()}else{format!("{prefix}::{name}")};
            for terminal in &mut scope.external_terminals {
                if let Some(span)=spans.get(&qualify(&terminal.name)) {terminal.source_span=Some(*span);}
            }
            for child in &mut scope.subgrammars {
                let path=qualify(&child.name);
                if let Some(span)=spans.get(&path){child.source_span=Some(*span);}
                if let ParsedSubgrammarBody::Inline(scope)=&mut child.body {walk(scope,&path,spans);}
            }
        }
        let mut scope=self.scope.as_ref().clone();walk(&mut scope,"",spans);Self::from_scope(scope)
    }

    pub fn declarations(&self) -> &GlrmExternalDeclarations { &self.declarations }

    /// Bind exact events structurally. Other dependencies stay open.
    pub fn with_tokens(&self, bindings:&[(&str,&[(u32,u32)])]) -> Result<Self,GlrMaskError> {
        let mut scope=self.scope.as_ref().clone();
        bind_external_terminals(&mut scope,bindings)?;
        Self::from_scope(scope)
    }

    /// Resolve all grammar-valued children, retaining each child's own lexical
    /// scope. Inputs may already contain nested inline components.
    pub fn with_children(&self, children:&[(&str,Self)]) -> Result<Self,GlrMaskError> {
        let mut names=BTreeMap::new();
        for (name,child) in children {
            if names.insert(*name,child).is_some() {return Err(err("duplicate component binding"));}
        }
        let resolve=|path:&str|names.get(path).map(|c|c.scope.as_ref().clone())
            .ok_or_else(||err(&format!("external grammar {path:?} has no binding")));
        let mut scope=self.scope.as_ref().clone();
        let mut consumed=BTreeSet::new();
        bind_inline_subgrammars(&mut scope,"",&resolve,&mut consumed,0,64)?;
        if let Some(name)=names.keys().find(|n|!consumed.contains(**n)) {
            return Err(err(&format!("unknown component binding {name:?}")));
        }
        Self::from_scope(scope)
    }

    pub fn lower_closed(&self) -> Result<NamedGrammar,GlrMaskError> {
        if contains_external_subgrammar(&self.scope) {
            return Err(err("component has unresolved externals"));
        }
        lower_parsed_grammar(self.scope.as_ref().clone(),&BTreeMap::new()).map(|p|p.grammar)
    }

    /// Lower a local body with real late-bound fragment slots. No vocabulary
    /// spelling can substitute for an exact-event binding at this boundary.
    pub fn lower_shell(&self, first_id:u32, reserved:impl IntoIterator<Item=u32>, bindings:&[(&str,&[(u32,u32)])])
        -> Result<GlrmWithExternalSubgrammars,GlrMaskError>
    {
        let mut scope=self.scope.as_ref().clone();
        bind_external_terminals(&mut scope,bindings)?;
        let mut used=reserved.into_iter().collect::<RangeSetBlaze<_>>();
        collect_scope_special_token_ids(&scope,&mut used);
        let names=collect_external_subgrammar_names(&scope);
        let mut candidate=first_id;
        let mut slots=BTreeMap::new();
        for (index,name) in names.iter().enumerate() {
            let id=next_free_token_id(&used,candidate)?;
            used.insert(id);slots.insert(name.clone(),id);
            if index+1<names.len() {candidate=id.checked_add(1).ok_or_else(||err("no token ID available for component placeholder"))?;}
        }
        lower_parsed_grammar(scope,&slots)
    }
}
