//! A parsed Lark document retains vocabulary atoms and schema source spans.
//! The host compiler resolves them through the same component/event IR as GLRM.
use super::*;
use std::collections::BTreeSet;
use std::ops::Range;

#[derive(Debug,Clone,Copy,PartialEq,Eq,Default)]
pub enum LarkProfile {
    #[default]
    Recognition,
    Llm,
}
#[derive(Debug,Clone,PartialEq,Eq)]
pub enum LarkTokenSet {
    Ranges {ranges:Vec<(u32,u32)>,negate:bool},
    Any,
    Named(String),
}
#[derive(Debug,Clone)]
pub struct LarkDocument {
    pub grammar:NamedGrammar,
    pub additional_ignores:Vec<String>,
    pub imports:BTreeSet<String>,
    pub events:Vec<(String,LarkTokenSet)>,
    pub schemas:Vec<(String,Range<usize>)>,
    /// Original atom/declaration spans; repeated @imports retain their first use.
    pub source_spans:BTreeMap<String,SourceSpan>,
}

pub fn parse_lark_document(input:&str,profile:LarkProfile)->Result<LarkDocument,GlrMaskError> {
    let mut lexer=Lexer::with_profile(input,profile);
    let tokens=lexer.tokenize().map_err(|e|e.at(SourceSpan::at(input,lexer.current_start,lexer.pos)))?;
    let mut spans=SourceSpan::tokens(input,&lexer.offsets);
    spans.push(SourceSpan::at(input,input.len(),input.len()));
    let mut parser=Parser::new(tokens);parser.profile=profile;parser.spans=spans;
    let grammar=parser.parse_grammar().map_err(|e|parser.locate(e,parser.pos))?;
    let externs=parser.events.iter().map(|(name,_)|name.clone())
        .chain(parser.schemas.iter().map(|(name,_)|name.clone()))
        .chain(parser.imports.iter().map(|name|format!("@{name}"))).collect::<HashSet<_>>();
    let additional_ignores=parser.ignored.iter().skip(1).cloned().collect::<Vec<_>>();
    let grammar=normalize_lark_named_with_externals(grammar,&externs,&additional_ignores)?;
    Ok(LarkDocument{grammar,additional_ignores,imports:parser.imports,events:parser.events,schemas:parser.schemas,source_spans:parser.source_spans})
}

impl LarkDocument {
    /// Legacy lower-level import supports explicit positive IDs without needing
    /// a vocabulary. Domain-dependent atoms and child composition need the host.
    pub fn into_plain_grammar(self)->Result<NamedGrammar,GlrMaskError> {
        if !self.imports.is_empty() || !self.schemas.is_empty(){
            return Err(GlrMaskError::GrammarParse("this Lark document requires the component-aware Grammar frontend".into()));
        }
        let mut grammar=self.grammar;
        let mut substitutions=HashMap::new();
        for (name,set) in self.events {
            let LarkTokenSet::Ranges{ranges,negate:false}=set else {
                return Err(GlrMaskError::GrammarParse("named/complement/wildcard tokens require a vocabulary-aware Grammar frontend".into()));
            };
            substitutions.insert(name,event_expression(&ranges));
        }
        for rule in &mut grammar.rules {rewrite(&mut rule.expr,&mut |name|substitutions.get(name).cloned());}
        if !self.additional_ignores.is_empty(){
            return crate::grammar::glrm::ParsedComponent::from_named_with_ignores(grammar,&[],&self.additional_ignores)?.lower_closed();
        }
        Ok(grammar)
    }

    /// Return an AST whose ordinary rule names cannot collide with external
    /// @names. The '$' prefix cannot be authored as a source identifier.
    pub fn resolved_additional_ignores(&self)->Vec<String> {
        self.additional_ignores.iter().map(|name|format!("$lark_rule_{name}")).collect()
    }

    pub fn resolve_events(&self,values:&HashMap<String,GrammarExpr>)->NamedGrammar {
        let mut grammar=self.grammar.clone();
        let local_names=grammar.rules.iter().map(|r|r.name.clone()).collect::<HashSet<_>>();
        for rule in &mut grammar.rules {
            rewrite(&mut rule.expr,&mut |name| {
                if let Some(value)=values.get(name){return Some(value.clone());}
                if let Some(external)=name.strip_prefix('@'){return Some(GrammarExpr::Ref(external.to_owned()));}
                local_names.contains(name).then(||GrammarExpr::Ref(format!("$lark_rule_{name}")))
            });
            rule.name=format!("$lark_rule_{}",rule.name);
        }
        grammar.start=format!("$lark_rule_{}",grammar.start);
        grammar.ignore=grammar.ignore.map(|name|format!("$lark_rule_{name}"));
        grammar.lexer_partitions=grammar.lexer_partitions.into_iter().map(|(name,part)|(format!("$lark_rule_{name}"),part)).collect();
        // A component-aware frontend must retain the same safe local regular
        // compression as the plain Lark importer. Calls and multiple independent
        // ignore identities are deliberately not flattened by this pass.
        if self.imports.is_empty() && self.schemas.is_empty() && self.additional_ignores.is_empty() {
            super::optimize_lark_grammar(&mut grammar);
        }
        grammar
    }
}

pub fn event_expression(ranges:&[(u32,u32)])->GrammarExpr {
    let mut atoms=ranges.iter().map(|&(a,b)|if a==b{GrammarExpr::SpecialToken(a)}else{GrammarExpr::SpecialTokenRange(a,b)}).collect::<Vec<_>>();
    if atoms.len()==1{atoms.pop().unwrap()}else{GrammarExpr::Choice(atoms)}
}

fn rewrite(expr:&mut GrammarExpr,reference:&mut impl FnMut(&str)->Option<GrammarExpr>) {
    match expr {
        GrammarExpr::Ref(name)=>{if let Some(value)=reference(name){*expr=value;}},
        GrammarExpr::Grouped(inner)|GrammarExpr::Shortest(inner)|GrammarExpr::Quantified(inner,_)=>rewrite(inner,reference),
        GrammarExpr::Sequence(parts)|GrammarExpr::Choice(parts)=>{for part in parts{rewrite(part,reference);}},
        GrammarExpr::Exclude{expr,exclude}|GrammarExpr::Intersect{expr,intersect:exclude}=>{rewrite(expr,reference);rewrite(exclude,reference);},
        GrammarExpr::SeparatedSequence{items,separator,..}=>{for (item,_) in items{rewrite(item,reference);}rewrite(separator,reference);},
        GrammarExpr::ExprNFA(nfa)=>{for symbol in &mut nfa.symbols{rewrite(symbol,reference);}},
        _=>{},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profiles_and_opaque_atoms_are_explicit() {
        let source=r#"start: <[1,3-8]> <[^2]> <[*]> <|finish|> @payload %json {"type":"string","description":"}"}"#;
        assert!(parse_lark_document(source,LarkProfile::Recognition).is_err());
        let doc=parse_lark_document(source,LarkProfile::Llm).unwrap();
        assert_eq!(doc.events.len(),4);assert_eq!(doc.schemas.len(),1);
        assert!(doc.imports.contains("payload"));
        assert_eq!(&source[doc.schemas[0].1.clone()],r#"{"type":"string","description":"}"}"#);
    }
    #[test]
    fn invalid_event_atoms_and_contexts_fail_closed() {
        for s in ["start: <[8-3]>","start: <[1,]>","start: <[^*]>","start: <[4294967296]>",
                  "start: A\nA: <[1]>","start: A\nA: @child","start: A\nA: %json {}",
                  "start: item\nitem[lazy]: @child","start: <[1]>\n%ignore <[1]>"] {
            assert!(parse_lark_document(s,LarkProfile::Llm).is_err(),"{s}");
        }
    }
    #[test]
    fn same_rule_and_external_name_have_separate_coordinates() {
        let doc=parse_lark_document("start: payload @payload\npayload: \"x\"",LarkProfile::Llm).unwrap();
        let grammar=doc.resolve_events(&HashMap::new());
        assert_eq!(grammar.rules[0].expr,GrammarExpr::Sequence(vec![GrammarExpr::Ref("$lark_rule_payload".into()),GrammarExpr::Ref("payload".into())]));
    }
}
