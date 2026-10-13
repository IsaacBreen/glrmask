use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use super::{choice_or_single, sequence_or_single};
mod document;
pub use document::{LarkDocument,LarkProfile,LarkTokenSet,parse_lark_document};
use crate::{GlrMaskError,SourceSpan};
use crate::automata::lexer::regex::checked_pattern;
use crate::automata::lexer::{
    DFA as LexerDfa,
    compile::compile_expression_labeled_nfa,
    regex::decode_unicode_escape,
};
use crate::grammar::ast::resolve_terminal_subexpressions;
use crate::grammar::flat::GrammarDef;
use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule, Quantifier, lower};
use crate::grammar::factoring::factor_named_grammar;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Terminal(String),
    Literal(String),
    SpecialToken(u32),
    EventSet(LarkTokenSet),
    External(String),
    JsonSchema(std::ops::Range<usize>),
    Regex(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Pipe,
    Star,
    Plus,
    Question,
    Colon,
    Newline,
    Dot,
    Tilde,
    Number(usize),
    Comma,
    Arrow,
    Bang,
    PercentIgnore,
    PercentImport,
    Minus,
}

struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
    offsets:Vec<(usize,usize)>,
    current_start:usize,
    profile: LarkProfile,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer {
            input: input.as_bytes(),
            pos: 0, offsets:Vec::new(), current_start:0,
            profile: LarkProfile::Llm,
        }
    }

    fn with_profile(input:&'a str,profile:LarkProfile)->Self {
        Self {input:input.as_bytes(),pos:0,profile,offsets:Vec::new(),current_start:0}
    }

    fn require_llm(&self, feature:&str)->Result<(),GlrMaskError> {
        if self.profile!=LarkProfile::Llm {
            return Err(GlrMaskError::GrammarParse(format!("{feature} requires the explicit Lark LLM compatibility profile")));
        }
        Ok(())
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.input.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_whitespace_inline(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        while let Some(b) = self.peek() {
            if b == b'\n' {
                break;
            }
            self.pos += 1;
        }
    }

    fn push_utf8_char_from(&mut self, start: usize, output: &mut String) {
        let remaining = std::str::from_utf8(&self.input[start..])
            .expect("Lark source originated from valid UTF-8");
        let character = remaining
            .chars()
            .next()
            .expect("non-ASCII byte must begin a UTF-8 character");
        self.pos = start + character.len_utf8();
        output.push(character);
    }

    fn lex_string(&mut self, quote: u8) -> Result<String, GlrMaskError> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some(b) if b == quote => return Ok(s),
                Some(b'\\') => {
                    let escape_start = self.pos - 1;
                    if let Some((character, next)) = decode_unicode_escape(self.input, escape_start) {
                        self.pos = next;
                        s.push(character);
                        continue;
                    }
                    match self.advance() {
                        Some(b'n') => s.push('\n'),
                        Some(b't') => s.push('\t'),
                        Some(b'r') => s.push('\r'),
                        Some(b'\\') => s.push('\\'),
                        Some(c) if c == quote => s.push(c as char),
                        Some(b'"') => s.push('"'),
                        Some(b'\'') => s.push('\''),
                        Some(b'x') => {
                            let h1 = self.advance().ok_or_else(|| {
                                GlrMaskError::GrammarParse("unterminated \\x escape".into())
                            })?;
                            let h2 = self.advance().ok_or_else(|| {
                                GlrMaskError::GrammarParse("unterminated \\x escape".into())
                            })?;
                            let hex_str = format!("{}{}", h1 as char, h2 as char);
                            let byte = u8::from_str_radix(&hex_str, 16).map_err(|_| {
                                GlrMaskError::GrammarParse(format!(
                                    "invalid \\x escape: \\x{hex_str}"
                                ))
                            })?;
                            s.push(byte as char);
                        }
                        Some(c) => {
                            s.push('\\');
                            if c.is_ascii() {
                                s.push(c as char);
                            } else {
                                self.push_utf8_char_from(self.pos - 1, &mut s);
                            }
                        }
                        None => {
                            return Err(GlrMaskError::GrammarParse("unterminated escape".into()));
                        }
                    }
                }
                Some(b) if b.is_ascii() => s.push(b as char),
                Some(_) => self.push_utf8_char_from(self.pos - 1, &mut s),
                None => return Err(GlrMaskError::GrammarParse("unterminated string".into())),
            }
        }
    }

    fn lex_regex(&mut self) -> Result<String, GlrMaskError> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some(b'/') => return Ok(s),
                Some(b'\\') => {
                    s.push('\\');
                    if let Some(b) = self.advance() {
                        if b.is_ascii() {
                            s.push(b as char);
                        } else {
                            self.push_utf8_char_from(self.pos - 1, &mut s);
                        }
                    }
                }
                Some(b) if b.is_ascii() => s.push(b as char),
                Some(_) => self.push_utf8_char_from(self.pos - 1, &mut s),
                None => return Err(GlrMaskError::GrammarParse("unterminated regex".into())),
            }
        }
    }

    fn lex_ident(&mut self, first: u8) -> String {
        let mut s = String::new();
        s.push(first as char);
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                s.push(b as char);
                self.pos += 1;
            } else {
                break;
            }
        }
        s
    }

    fn lex_number(&mut self, first: u8) -> Result<usize, GlrMaskError> {
        let mut n = (first - b'0') as usize;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                n = n.checked_mul(10).and_then(|n| n.checked_add((b - b'0') as usize))
                    .ok_or_else(|| GlrMaskError::GrammarParse("integer in Lark source is too large".into()))?;
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(n)
    }

    fn lex_special_token(&mut self) -> Result<Token, GlrMaskError> {
        let prefix = b"token(";
        if !self.input[self.pos..].starts_with(prefix) {
            return Err(GlrMaskError::GrammarParse(
                "expected @token(<token-id>)".into(),
            ));
        }
        self.pos += prefix.len();
        let mut token_id = 0u32;
        let mut digits = 0usize;
        while let Some(byte @ b'0'..=b'9') = self.peek() {
            token_id = token_id
                .checked_mul(10)
                .and_then(|value| value.checked_add((byte - b'0') as u32))
                .ok_or_else(|| {
                    GlrMaskError::GrammarParse("special LLM token id does not fit in u32".into())
                })?;
            digits += 1;
            self.pos += 1;
        }
        if digits == 0 || self.peek() != Some(b')') {
            return Err(GlrMaskError::GrammarParse(
                "expected @token(<token-id>)".into(),
            ));
        }
        self.pos += 1;
        Ok(Token::SpecialToken(token_id))
    }

    fn lex_event_set(&mut self) -> Result<Token,GlrMaskError> {
        self.require_llm("exact-token atoms")?;
        self.pos+=1; // '<'
        if self.peek()==Some(b'|') {
            self.pos+=1;let start=self.pos;
            while self.pos+1<self.input.len() && &self.input[self.pos..self.pos+2]!=b"|>" {self.pos+=1;}
            if self.pos==start || self.pos+1>=self.input.len(){return Err(GlrMaskError::GrammarParse("unterminated or empty named special token".into()));}
            let name=std::str::from_utf8(&self.input[start..self.pos]).map_err(|_|GlrMaskError::GrammarParse("invalid token name".into()))?.to_owned();
            self.pos+=2;return Ok(Token::EventSet(LarkTokenSet::Named(name)));
        }
        if self.advance()!=Some(b'['){return Err(GlrMaskError::GrammarParse("expected <[...]> or <|name|>".into()));}
        self.skip_whitespace_inline();
        let negate=if self.peek()==Some(b'^'){self.pos+=1;true}else{false};
        self.skip_whitespace_inline();
        if self.peek()==Some(b'*') {
            self.pos+=1;self.skip_whitespace_inline();
            if negate || self.advance()!=Some(b']') || self.advance()!=Some(b'>'){
                return Err(GlrMaskError::GrammarParse("wildcard syntax is <[*]>; a negated wildcard is not supported".into()));
            }
            return Ok(Token::EventSet(LarkTokenSet::Any));
        }
        let mut ranges=Vec::new();
        while self.peek()!=Some(b']') {
            let number=|lexer:&mut Self|->Result<u32,GlrMaskError>{
                let first=lexer.advance().filter(u8::is_ascii_digit).ok_or_else(||GlrMaskError::GrammarParse("expected numeric token ID".into()))?;
                let id=lexer.lex_number(first)?;
                u32::try_from(id).map_err(|_|GlrMaskError::GrammarParse("token ID does not fit in u32".into()))
            };
            let start=number(self)?;self.skip_whitespace_inline();
            let end=if self.peek()==Some(b'-'){self.pos+=1;self.skip_whitespace_inline();number(self)?}else{start};
            if start>end{return Err(GlrMaskError::GrammarParse("token range endpoints are reversed".into()));}
            ranges.push((start,end));self.skip_whitespace_inline();
            if self.peek()!=Some(b','){break;}self.pos+=1;self.skip_whitespace_inline();
            if self.peek()==Some(b']'){return Err(GlrMaskError::GrammarParse("missing token ID after comma".into()));}
        }
        if self.advance()!=Some(b']') || self.advance()!=Some(b'>'){return Err(GlrMaskError::GrammarParse("unterminated <[...]> token atom".into()));}
        Ok(Token::EventSet(LarkTokenSet::Ranges{ranges,negate}))
    }

    fn lex_json_schema(&mut self)->Result<Token,GlrMaskError> {
        self.require_llm("inline JSON Schema")?;
        while self.peek().is_some_and(|b|b.is_ascii_whitespace()){self.pos+=1;}
        let start=self.pos;
        if self.advance()!=Some(b'{'){return Err(GlrMaskError::GrammarParse("%json requires a JSON object".into()));}
        let mut depth=1usize;let mut quoted=false;let mut escaped=false;
        while let Some(byte)=self.advance() {
            if quoted {
                if escaped {escaped=false;}else if byte==b'\\'{escaped=true;}else if byte==b'"'{quoted=false;}
                continue;
            }
            match byte {
                b'"'=>quoted=true,
                b'{'=>{depth+=1;if depth>512{return Err(GlrMaskError::GrammarParse("inline JSON nesting exceeds 512".into()));}},
                b'}'=>{depth-=1;if depth==0{return Ok(Token::JsonSchema(start..self.pos));}},
                _=>{},
            }
        }
        Err(GlrMaskError::GrammarParse("unterminated inline JSON Schema".into()))
    }

    fn lex_at_reference(&mut self)->Result<Token,GlrMaskError> {
        self.require_llm("external grammar references")?;
        if self.input[self.pos..].starts_with(b"token("){return self.lex_special_token();}
        let first=self.advance().filter(|b|b.is_ascii_alphabetic()||*b==b'_')
            .ok_or_else(||GlrMaskError::GrammarParse("expected external grammar name after @".into()))?;
        let mut name=self.lex_ident(first);
        while self.peek()==Some(b'-') {
            self.pos+=1;name.push('-');
            let first=self.advance().filter(|b|b.is_ascii_alphanumeric()||*b==b'_')
                .ok_or_else(||GlrMaskError::GrammarParse("invalid external grammar name".into()))?;
            name.push_str(&self.lex_ident(first));
        }
        Ok(Token::External(name))
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, GlrMaskError> {
        let mut tokens = Vec::new();
        self.offsets.clear();
        loop {
            self.skip_whitespace_inline();
            let start=self.pos;self.current_start=start;let count=tokens.len();
            match self.peek() {
                None => break,
                Some(b'/') => {
                    self.pos += 1;
                    if self.peek() == Some(b'/') {
                        self.skip_comment();
                    } else {
                        let rx = self.lex_regex()?;
                        let start = self.pos;
                        while self.peek().is_some_and(|b| b.is_ascii_alphabetic()) { self.pos += 1; }
                        let flags = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
                        let rx = checked_pattern(&rx, flags, true).map_err(GlrMaskError::GrammarParse)?;
                        tokens.push(Token::Regex(rx));
                    }
                }
                Some(b'#') => self.skip_comment(),
                Some(b'%') => {
                    self.pos += 1;
                    let start = self.pos;
                    while self.peek().is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_') {
                        self.pos += 1;
                    }
                    let directive = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
                    match directive {
                        "ignore" => tokens.push(Token::PercentIgnore),
                        "import" => tokens.push(Token::PercentImport),
                        "json" => tokens.push(self.lex_json_schema()?),
                        _ => return Err(GlrMaskError::GrammarParse(format!(
                            "unsupported Lark directive %{directive}; directives are never silently ignored"
                        ))),
                    }
                }
                Some(b'\n') => {
                    self.pos += 1;
                    tokens.push(Token::Newline);
                }
                Some(b'"') => {
                    self.pos += 1;
                    let s = self.lex_string(b'"')?;
                    if self.peek() == Some(b'i') {
                        self.pos += 1;
                        let pattern = s.chars().map(|c| format!(r"\x{{{:X}}}", c as u32)).collect::<String>();
                        tokens.push(Token::Regex(checked_pattern(&pattern, "i", true).map_err(GlrMaskError::GrammarParse)?));
                    } else {
                        tokens.push(Token::Literal(s));
                    }
                }
                Some(b'\'') => {
                    self.pos += 1;
                    let s = self.lex_string(b'\'')?;
                    if self.peek() == Some(b'i') {
                        self.pos += 1;
                        let pattern = s.chars().map(|c| format!(r"\x{{{:X}}}", c as u32)).collect::<String>();
                        tokens.push(Token::Regex(checked_pattern(&pattern, "i", true).map_err(GlrMaskError::GrammarParse)?));
                    } else {
                        tokens.push(Token::Literal(s));
                    }
                }
                Some(b'(') => {
                    self.pos += 1;
                    tokens.push(Token::LParen);
                }
                Some(b')') => {
                    self.pos += 1;
                    tokens.push(Token::RParen);
                }
                Some(b'[') => {
                    self.pos += 1;
                    tokens.push(Token::LBracket);
                }
                Some(b']') => {
                    self.pos += 1;
                    tokens.push(Token::RBracket);
                }
                Some(b'|') => {
                    self.pos += 1;
                    tokens.push(Token::Pipe);
                }
                Some(b'*') => {
                    self.pos += 1;
                    tokens.push(Token::Star);
                }
                Some(b'+') => {
                    self.pos += 1;
                    tokens.push(Token::Plus);
                }
                Some(b'?') => {
                    self.pos += 1;
                    tokens.push(Token::Question);
                }
                Some(b'.') => {
                    self.pos += 1;
                    tokens.push(Token::Dot);
                }
                Some(b'@') => {
                    self.pos += 1;
                    tokens.push(self.lex_at_reference()?);
                }
                Some(b'<') => tokens.push(self.lex_event_set()?),
                Some(b'~') => {
                    self.pos += 1;
                    tokens.push(Token::Tilde);
                }
                Some(b',') => {
                    self.pos += 1;
                    tokens.push(Token::Comma);
                }
                Some(b'-') => {
                    self.pos += 1;
                    if self.peek() == Some(b'>') {
                        self.pos += 1;
                        tokens.push(Token::Arrow);
                    } else {
                        tokens.push(Token::Minus);
                    }
                }
                Some(b':') => {
                    self.pos += 1;
                    tokens.push(Token::Colon);
                }
                Some(b'!') => {
                    self.pos += 1;
                    tokens.push(Token::Bang);
                }
                Some(b) if b.is_ascii_alphabetic() || b == b'_' => {
                    self.pos += 1;
                    let ident = self.lex_ident(b);
                    if ident
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
                    {
                        tokens.push(Token::Terminal(ident));
                    } else {
                        tokens.push(Token::Ident(ident));
                    }
                }
                Some(b) if b.is_ascii_digit() => {
                    self.pos += 1;
                    let n = self.lex_number(b)?;
                    tokens.push(Token::Number(n));
                }
                Some(b) => {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "unexpected character '{}' at position {}",
                        b as char, self.pos
                    )));
                }
            }
            self.offsets.extend((count..tokens.len()).map(|_|(start,self.pos)));
        }
        Ok(tokens)
    }
}

fn bounded_repeat_expr(atom: GrammarExpr, min: usize, max: Option<usize>) -> Result<GrammarExpr, GlrMaskError> {
    if max.is_some_and(|max| max < min) {
        return Err(GlrMaskError::GrammarParse(format!(
            "repetition upper bound {} is smaller than lower bound {min}", max.unwrap()
        )));
    }
    Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::Range(min, max)))
}

fn literal_range_expr(start: &str, end: &str) -> Result<GrammarExpr, GlrMaskError> {
    let mut a = start.chars();
    let mut b = end.chars();
    let (Some(a0), Some(b0), None, None) = (a.next(), b.next(), a.next(), b.next()) else {
        return Err(GlrMaskError::GrammarParse("Lark literal ranges require single Unicode-scalar endpoints".into()));
    };
    if a0 > b0 {
        return Err(GlrMaskError::GrammarParse(format!("invalid Lark literal range {start:?}..{end:?}")));
    }
    // Unicode escapes avoid metacharacter and UTF-8 byte ambiguities.
    Ok(GrammarExpr::RawRegex(checked_pattern(&format!(r"[\U{:08X}-\U{:08X}]", a0 as u32, b0 as u32), "", true).map_err(GlrMaskError::GrammarParse)?))
}

// Recognition equivalents of Lark 1.3.1 common.lark (MIT licensed).
// Resolve imports locally: no implicit file or network access. The two pattern
// rewrites below avoid lookbehind/reluctance while retaining delimiter boundaries.
fn common_terminal_pattern(name: &str) -> Option<&'static str> {
    Some(match name {
        "DIGIT" => r"[0-9]",
        "HEXDIGIT" => r"[0-9a-fA-F]",
        "INT" => r"[0-9]+",
        "SIGNED_INT" => r"[+-]?[0-9]+",
        "DECIMAL" => r"([0-9]+\.[0-9]*|\.[0-9]+)",
        "_EXP" => r"[eE][+-]?[0-9]+",
        "FLOAT" => r"([0-9]+[eE][+-]?[0-9]+|([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?)",
        "SIGNED_FLOAT" => r"[+-]?([0-9]+[eE][+-]?[0-9]+|([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?)",
        "NUMBER" => r"([0-9]+([eE][+-]?[0-9]+)?|([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?)",
        "SIGNED_NUMBER" => r"[+-]?([0-9]+([eE][+-]?[0-9]+)?|([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?)",
        "ESCAPED_STRING" => r#""([^"\\\n]|\\[^\n])*""#,
        "LCASE_LETTER" => r"[a-z]",
        "UCASE_LETTER" => r"[A-Z]",
        "LETTER" => r"[a-zA-Z]",
        "WORD" => r"[a-zA-Z]+",
        "CNAME" => r"[_a-zA-Z][_a-zA-Z0-9]*",
        "WS_INLINE" => r"[ \t]+",
        "WS" => r"[ \t\x0c\r\n]+",
        "CR" => r"\r",
        "LF" => r"\n",
        "NEWLINE" => r"(\r?\n)+",
        "SH_COMMENT" => r"#[^\n]*",
        "CPP_COMMENT" => r"//[^\n]*",
        "C_COMMENT" => r"/\*[^*]*\*+([^/*][^*]*\*+)*/",
        "SQL_COMMENT" => r"--[^\n]*",
        _ => return None,
    })
}

struct Parser {
    tokens: Vec<Token>,
    spans:Vec<SourceSpan>,
    pos: usize,
    profile: LarkProfile,
    ignored:Vec<String>,
    events:Vec<(String,LarkTokenSet)>,
    schemas:Vec<(String,std::ops::Range<usize>)>,
    imports:std::collections::BTreeSet<String>,
    source_spans:BTreeMap<String,SourceSpan>,
    expression_depth:usize,
}

fn is_lark_terminal_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

fn lark_start_rule_name(rules: &[NamedRule]) -> Result<String, GlrMaskError> {
    if rules.iter().any(|rule| rule.name == "start") {
        Ok("start".to_string())
    } else {
        Err(GlrMaskError::GrammarParse("Lark grammar must define a rule named 'start'".into()))
    }
}

fn mark_lark_terminal_rules(rules: &mut [NamedRule]) {
    for rule in rules {
        rule.is_terminal = is_lark_terminal_name(&rule.name);
    }
}

fn synthesize_lark_ignore_rules(rules:&mut Vec<NamedRule>,expressions:Vec<GrammarExpr>)->Vec<String> {
    let mut names=Vec::new();
    for (i,expr) in expressions.into_iter().enumerate() {
        let name=format!("$lark_ignore_{i}");
        rules.push(NamedRule{name:name.clone(),expr,is_terminal:true,is_internal:false});
        names.push(name);
    }
    names
}

const MIN_LARK_TERMINAL_GRAPH_RULES: usize = 32;

fn lark_ref_names(expr: &GrammarExpr) -> Vec<&str> {
    let mut refs = Vec::new();
    let mut stack = vec![expr];
    while let Some(expr) = stack.pop() {
        match expr {
            GrammarExpr::Ref(name) => refs.push(name.as_str()),
            GrammarExpr::Grouped(inner) | GrammarExpr::Shortest(inner) | GrammarExpr::Quantified(inner, _) => {
                stack.push(inner);
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                stack.extend(parts.iter());
            }
            GrammarExpr::Exclude { expr, exclude } => {
                stack.push(expr);
                stack.push(exclude);
            }
            GrammarExpr::Intersect { expr, intersect } => {
                stack.push(expr);
                stack.push(intersect);
            }
            GrammarExpr::SeparatedSequence {
                items, separator, ..
            } => {
                for (item, _) in items {
                    stack.push(item);
                }
                stack.push(separator);
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                stack.extend(expr_nfa.symbols.iter());
            }
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_) | GrammarExpr::SpecialTokenRange(_, _)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => {}
        }
    }
    refs
}

fn validate_lark_terminal_refs(
    expr: &GrammarExpr,
    rule_name: &str,
    terminal_names: &HashSet<String>,
    rule_map: &HashMap<String, GrammarExpr>,
) -> Result<(), GlrMaskError> {
    for target in lark_ref_names(expr) {
        if !rule_map.contains_key(target) {
            return Err(GlrMaskError::GrammarParse(format!(
                "terminal rule {rule_name} references undefined rule {target}"
            )));
        }
        if !terminal_names.contains(target) {
            return Err(GlrMaskError::GrammarParse(format!(
                "terminal rule {rule_name} references nonterminal {target}"
            )));
        }
    }
    Ok(())
}

fn validate_lark_parser_refs(
    expr: &GrammarExpr,
    rule_map: &HashMap<String, GrammarExpr>,
    externs:&HashSet<String>,
) -> Result<(), GlrMaskError> {
    for target in lark_ref_names(expr) {
        if !rule_map.contains_key(target) && !externs.contains(target) {
            return Err(GlrMaskError::GrammarParse(format!(
                "unknown Lark rule reference {target}"
            )));
        }
    }
    Ok(())
}

fn validate_lark_terminal_acyclic(
    rule_map: &HashMap<String, GrammarExpr>,
    terminal_names: &HashSet<String>,
) -> Result<(), GlrMaskError> {
    let mut state = HashMap::<String, u8>::new();
    let mut roots = terminal_names.iter().cloned().collect::<Vec<_>>();
    roots.sort_unstable();

    for root in roots {
        if state.get(&root).copied() == Some(2) {
            continue;
        }
        let mut stack = vec![(root, false)];
        while let Some((name, exiting)) = stack.pop() {
            if exiting {
                state.insert(name, 2);
                continue;
            }
            match state.get(&name).copied().unwrap_or(0) {
                2 => continue,
                1 => {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "cyclic Lark terminal definition involving {name}"
                    )));
                }
                _ => {}
            }

            state.insert(name.clone(), 1);
            stack.push((name.clone(), true));
            let Some(expr) = rule_map.get(&name) else {
                continue;
            };
            let mut dependencies = lark_ref_names(expr)
                .into_iter()
                .filter(|target| terminal_names.contains(*target))
                .map(str::to_owned)
                .collect::<Vec<_>>();
            dependencies.sort_unstable();
            dependencies.dedup();
            for dependency in dependencies.into_iter().rev() {
                if state.get(&dependency).copied() == Some(1) {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "cyclic Lark terminal definition involving {dependency}"
                    )));
                }
                if state.get(&dependency).copied() != Some(2) {
                    stack.push((dependency, false));
                }
            }
        }
    }
    Ok(())
}

fn externally_emitted_lark_terminals(
    grammar: &NamedGrammar,
    terminal_names: &HashSet<String>,
) -> HashSet<String> {
    let mut roots = HashSet::new();
    if terminal_names.contains(&grammar.start) {
        roots.insert(grammar.start.clone());
    }
    if let Some(ignore) = &grammar.ignore {
        roots.insert(ignore.clone());
    }
    roots.extend(grammar.lexer_partitions.keys().cloned());

    for rule in grammar.rules.iter().filter(|rule| !rule.is_terminal) {
        for target in lark_ref_names(&rule.expr) {
            if terminal_names.contains(target) {
                roots.insert(target.to_owned());
            }
        }
    }
    roots
}

struct LarkTerminalGraphIndex<'a> {
    rules_by_name: HashMap<&'a str, &'a NamedRule>,
    dependencies_by_name: HashMap<&'a str, Vec<&'a str>>,
}

impl<'a> LarkTerminalGraphIndex<'a> {
    fn new(grammar: &'a NamedGrammar, terminal_names: &HashSet<String>) -> Self {
        let rules_by_name = grammar
            .rules
            .iter()
            .filter(|rule| rule.is_terminal)
            .map(|rule| (rule.name.as_str(), rule))
            .collect::<HashMap<_, _>>();
        let dependencies_by_name = rules_by_name
            .iter()
            .map(|(&name, rule)| {
                let mut dependencies = lark_ref_names(&rule.expr)
                    .into_iter()
                    .filter(|target| terminal_names.contains(*target))
                    .collect::<Vec<_>>();
                dependencies.sort_unstable();
                dependencies.dedup();
                (name, dependencies)
            })
            .collect();
        Self {
            rules_by_name,
            dependencies_by_name,
        }
    }

    fn reachable(&self, root: &str) -> HashSet<String> {
        let mut reachable = HashSet::new();
        let mut stack = vec![root.to_owned()];
        while let Some(name) = stack.pop() {
            if !reachable.insert(name.clone()) {
                continue;
            }
            let Some(dependencies) = self.dependencies_by_name.get(name.as_str()) else {
                continue;
            };
            for &target in dependencies {
                if !reachable.contains(target) {
                    stack.push(target.to_owned());
                }
            }
        }
        reachable
    }
}

fn reachable_lark_terminal_rules(
    index: &LarkTerminalGraphIndex<'_>,
    root: &str,
) -> HashSet<String> {
    index.reachable(root)
}

fn compile_lark_terminal_graph_root(
    grammar: &NamedGrammar,
    index: &LarkTerminalGraphIndex<'_>,
    root: &str,
    min_graph_rules: usize,
) -> Result<Option<Arc<LexerDfa>>, GlrMaskError> {
    let reachable = reachable_lark_terminal_rules(index, root);
    // The shared right-linear compressor counts only dependency-bearing rules
    // as parser states. Fewer than `min_graph_rules` reachable rules can never
    // satisfy that threshold, so avoid cloning and compiling a temporary
    // grammar for ordinary independent/small terminal definitions.
    if reachable.len() < min_graph_rules {
        return Ok(None);
    }

    let mut temporary_rules = Vec::with_capacity(reachable.len());
    let mut reachable_names = reachable.iter().map(String::as_str).collect::<Vec<_>>();
    reachable_names.sort_unstable();
    for name in reachable_names {
        let rule = index
            .rules_by_name
            .get(name)
            .copied()
            .ok_or_else(|| {
                GlrMaskError::GrammarParse(format!(
                    "internal error resolving Lark terminal graph rule {name}"
                ))
            })?;
        let has_terminal_dependency = index
            .dependencies_by_name
            .get(name)
            .is_some_and(|dependencies| !dependencies.is_empty());
        temporary_rules.push(NamedRule {
            name: rule.name.clone(),
            expr: rule.expr.clone(),
            // Leaf byte languages are the alphabet of the temporary regular
            // grammar. Rules with dependencies become right-linear states;
            // small ones are subsequently macro-expanded by the shared pass.
            is_terminal: !has_terminal_dependency,
            is_internal: false,
        });
    }

    let mut temporary = NamedGrammar {
        rules: temporary_rules,
        start: root.to_owned(),
        ignore: None,
        lexer_partitions: Default::default(),
        lexer_literal_partitions: Default::default(),
        default_lexer_partition: None,
    };
    if !crate::grammar::right_linear::compress_right_linear_grammar_unchecked(
        &mut temporary,
        min_graph_rules,
    ) {
        return Ok(None);
    }

    let expr_nfa = temporary
        .rules
        .into_iter()
        .find(|rule| rule.name == root && !rule.is_terminal)
        .and_then(|rule| match rule.expr {
            GrammarExpr::ExprNFA(expr_nfa) => Some(expr_nfa),
            _ => None,
        })
        .ok_or_else(|| {
            GlrMaskError::GrammarParse(format!(
                "internal error compressing Lark terminal graph rooted at {root}"
            ))
        })?;
    let symbols = resolve_terminal_subexpressions(grammar, &expr_nfa.symbols)?;
    let dfa = compile_expression_labeled_nfa(&expr_nfa.nfa, &symbols)
        .map_err(GlrMaskError::GrammarParse)?;
    Ok(Some(Arc::new(dfa)))
}

fn compress_lark_terminal_graphs(grammar: &mut NamedGrammar,additional_ignores:&[String]) -> Result<(), GlrMaskError> {
    let source = grammar.clone();
    let terminal_names = source.terminal_names_set();
    let graph_index = LarkTerminalGraphIndex::new(&source, &terminal_names);
    let mut external = externally_emitted_lark_terminals(&source, &terminal_names);
    external.extend(additional_ignores.iter().cloned());
    let mut roots = external.iter().cloned().collect::<Vec<_>>();
    roots.sort_unstable();

    let mut compiled = HashMap::new();
    for root in roots {
        if let Some(dfa) = compile_lark_terminal_graph_root(
            &source,
            &graph_index,
            &root,
            MIN_LARK_TERMINAL_GRAPH_RULES,
        )? {
            compiled.insert(root, dfa);
        }
    }

    for rule in &mut grammar.rules {
        if !rule.is_terminal {
            continue;
        }
        rule.is_internal = !external.contains(&rule.name);
        if let Some(dfa) = compiled.get(&rule.name) {
            rule.expr = GrammarExpr::LexerDfa(dfa.clone());
        }
    }

    // A compiled root no longer depends on its source helper rules. Keep only
    // helpers reachable from external roots that could not use the compact
    // right-linear path; otherwise later factoring needlessly walks the original
    // deep dependency graph and can overflow its recursion stack.
    let mut needed_helpers = HashSet::new();
    for root in external.iter().filter(|root| !compiled.contains_key(*root)) {
        needed_helpers.extend(reachable_lark_terminal_rules(&graph_index, root));
    }
    grammar.rules.retain(|rule| {
        !rule.is_terminal
            || external.contains(&rule.name)
            || needed_helpers.contains(&rule.name)
    });

    if (!compiled.is_empty())
        && (std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some())
    {
        let states = compiled.values().map(|dfa| dfa.num_states()).sum::<usize>();
        eprintln!(
            "[glrmask/profile][lark_terminal_graph] roots={} compiled={} dfa_states={}",
            external.len(),
            compiled.len(),
            states,
        );
    }
    Ok(())
}

fn normalize_lark_named(grammar:NamedGrammar)->Result<NamedGrammar,GlrMaskError> {
    normalize_lark_named_with_externals(grammar,&HashSet::new(),&[])
}
fn normalize_lark_named_with_externals(grammar: NamedGrammar, externs:&HashSet<String>,additional_ignores:&[String]) -> Result<NamedGrammar, GlrMaskError> {
    let rule_map: HashMap<String, GrammarExpr> = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), rule.expr.clone()))
        .collect();
    let terminal_names: HashSet<String> = grammar.terminal_names_set();

    for rule in &grammar.rules {
        if terminal_names.contains(&rule.name) {
            validate_lark_terminal_refs(&rule.expr, &rule.name, &terminal_names, &rule_map)?;
        } else {
            validate_lark_parser_refs(&rule.expr, &rule_map,externs)?;
        }
    }
    validate_lark_terminal_acyclic(&rule_map, &terminal_names)?;
    crate::grammar::ast::validate_shortest_operands(&grammar)?;

    let start_is_terminal = terminal_names.contains(&grammar.start);
    let output_start = if start_is_terminal {
        "start".to_string()
    } else {
        grammar.start.clone()
    };

    // Keep terminal references as references. The common terminal resolver
    // converts ordinary small graphs into Arc-shared lexer expressions. Large
    // right-linear dependency graphs are recognized below and compiled from
    // their compact state graph, avoiding eager recursive substitution.
    let mut rules = grammar.rules;
    if start_is_terminal {
        rules.insert(
            0,
            NamedRule {
                name: output_start.clone(),
                expr: GrammarExpr::Ref(grammar.start.clone()),
                is_terminal: true,
                is_internal: false,
            },
        );
    }

    let mut normalized = NamedGrammar {
        rules,
        start: output_start,
        ignore: grammar.ignore,
        lexer_partitions: grammar.lexer_partitions,
        lexer_literal_partitions: Default::default(),
        default_lexer_partition: None,
    };
    compress_lark_terminal_graphs(&mut normalized,additional_ignores)?;
    Ok(normalized)
}

impl Parser {
    fn locate(&self,error:GlrMaskError,position:usize)->GlrMaskError {
        match self.spans.get(position).or_else(||self.spans.last()){Some(span)=>error.at(*span),None=>error}
    }
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, spans:Vec::new(), pos: 0,profile:LarkProfile::Llm,ignored:Vec::new(),events:Vec::new(),schemas:Vec::new(),imports:Default::default(),source_spans:BTreeMap::new(),expression_depth:0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn peek_nth(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.pos + n)
    }

    fn advance(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos)?.clone();
        self.pos += 1;
        Some(tok)
    }

    fn expect_token(&mut self, expected: &Token) -> Result<(), GlrMaskError> {
        let position=self.pos;
        let result=match self.advance() {
            Some(ref tok) if tok == expected => Ok(()),
            Some(tok) => Err(GlrMaskError::GrammarParse(format!(
                "expected {:?}, got {:?}",
                expected, tok
            ))),
            None => Err(GlrMaskError::GrammarParse(format!(
                "expected {:?}, got end of input",
                expected
            ))),
        };
        result.map_err(|e|self.locate(e,position))
    }

    fn skip_newlines(&mut self) {
        while self.peek() == Some(&Token::Newline) {
            self.pos += 1;
        }
    }

    fn parse_rule_name(&mut self) -> Result<String, GlrMaskError> {
        while matches!(self.peek(), Some(Token::Question) | Some(Token::Bang)) {
            self.pos += 1;
        }

        match self.advance() {
            Some(Token::Ident(name)) | Some(Token::Terminal(name)) => Ok(name),
            Some(other) => Err(GlrMaskError::GrammarParse(format!(
                "expected rule name, got {:?}",
                other
            ))),
            None => Err(GlrMaskError::GrammarParse(
                "expected rule name, got end of input".into(),
            )),
        }
    }

    fn reject_rule_priority(&mut self) -> Result<(), GlrMaskError> {
        if self.peek() == Some(&Token::Dot) {
            return Err(GlrMaskError::GrammarParse(
                "Lark rule and terminal priorities are not supported".into()
            ));
        }
        Ok(())
    }

    fn import_name(&mut self) -> Result<String, GlrMaskError> {
        match self.advance() {
            Some(Token::Ident(name)) | Some(Token::Terminal(name)) => Ok(name),
            other => Err(GlrMaskError::GrammarParse(format!("expected name in Lark import, got {other:?}"))),
        }
    }

    fn parse_import(&mut self, rules: &mut Vec<NamedRule>) -> Result<bool, GlrMaskError> {
        if self.peek() != Some(&Token::PercentImport) { return Ok(false); }
        self.pos += 1;
        let module = self.import_name()?;
        if module != "common" {
            return Err(GlrMaskError::GrammarParse(format!("unsupported Lark import module {module:?}; only 'common' is supported")));
        }
        let mut imports = Vec::new();
        if self.peek() == Some(&Token::Dot) {
            self.pos += 1;
            let name = self.import_name()?;
            let alias = if self.peek() == Some(&Token::Arrow) {
                self.pos += 1; self.import_name()?
            } else { name.clone() };
            imports.push((name, alias));
        } else {
            self.expect_token(&Token::LParen)?;
            loop {
                self.skip_newlines();
                let name = self.import_name()?;
                imports.push((name.clone(), name));
                self.skip_newlines();
                if self.peek() != Some(&Token::Comma) { break; }
                self.pos += 1;
            }
            self.expect_token(&Token::RParen)?;
        }
        for (name, alias) in imports {
            if !is_lark_terminal_name(&alias) {
                return Err(GlrMaskError::GrammarParse(format!("imported terminal alias {alias:?} must be uppercase")));
            }
            let pattern = common_terminal_pattern(&name).ok_or_else(||
                GlrMaskError::GrammarParse(format!("unknown common terminal {name:?}")))?;
            rules.push(NamedRule { name: alias, expr: GrammarExpr::RawRegex(checked_pattern(pattern, "", true).map_err(GlrMaskError::GrammarParse)?), is_terminal: true, is_internal: false });
        }
        if !matches!(self.peek(), None | Some(Token::Newline)) {
            return Err(GlrMaskError::GrammarParse("unexpected content after Lark import".into()));
        }
        self.skip_newlines();
        Ok(true)
    }

    fn parse_bounded_repeat(&mut self, atom: GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        let min = match self.advance() {
            Some(Token::Number(value)) => value,
            _ => return Err(GlrMaskError::GrammarParse("expected number after ~".into())),
        };

        let max = if self.peek() == Some(&Token::Dot) {
            self.pos += 1;
            self.expect_token(&Token::Dot)?;
            match self.advance() {
                Some(Token::Number(value)) => value,
                _ => return Err(GlrMaskError::GrammarParse("expected upper bound after '..'".into())),
            }
        } else { min };
        bounded_repeat_expr(atom, min, Some(max))
    }

    fn parse_literal_or_range(&mut self, literal: String) -> Result<GrammarExpr, GlrMaskError> {
        if self.peek() == Some(&Token::Dot) && self.peek_nth(1) == Some(&Token::Dot) {
            self.pos += 2;
            return match self.advance() {
                Some(Token::Literal(end)) => literal_range_expr(&literal, &end),
                Some(other) => Err(GlrMaskError::GrammarParse(format!(
                    "expected literal after .. in Lark literal range, got {:?}",
                    other
                ))),
                None => Err(GlrMaskError::GrammarParse(
                    "expected literal after .. in Lark literal range, got end of input".into(),
                )),
            };
        }

        if literal.is_empty() {
            Ok(sequence_or_single(Vec::new()))
        } else {
            Ok(GrammarExpr::Literal(literal.into_bytes()))
        }
    }

    fn parse_ignore_directive(
        &mut self,
        ignore_exprs: &mut Vec<GrammarExpr>,
    ) -> Result<bool, GlrMaskError> {
        if self.peek() != Some(&Token::PercentIgnore) {
            return Ok(false);
        }

        self.pos += 1;
        ignore_exprs.push(self.parse_alternatives()?);
        self.skip_newlines();
        Ok(true)
    }

    fn parse_rule(&mut self) -> Result<NamedRule, GlrMaskError> {
        let name = self.parse_rule_name()?;
        self.reject_rule_priority()?;
        let lazy = if self.peek() == Some(&Token::LBracket) {
            self.pos += 1;
            let attribute = self.import_name()?;
            if self.profile!=LarkProfile::Llm {
                return Err(GlrMaskError::GrammarParse("[lazy] requires the Lark LLM compatibility profile".into()));
            }
            if attribute != "lazy" {
                return Err(GlrMaskError::GrammarParse(format!("unsupported Lark rule attribute {attribute:?}; only [lazy] on regular bodies is supported")));
            }
            self.expect_token(&Token::RBracket)?;
            true
        } else { false };
        self.expect_token(&Token::Colon)?;
        let expr = self.parse_alternatives()?;
        let expr = if lazy { GrammarExpr::Shortest(Box::new(expr)) } else { expr };
        Ok(NamedRule {
            name,
            expr,
            is_terminal: false,
            is_internal: false,
        })
    }

    fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {
        let mut rules: Vec<NamedRule> = Vec::new();
        let mut ignore_exprs: Vec<GrammarExpr> = Vec::new();

        self.skip_newlines();
        while self.pos < self.tokens.len() {
            if self.parse_ignore_directive(&mut ignore_exprs)? || self.parse_import(&mut rules)? {
                continue;
            }

            rules.push(self.parse_rule()?);
            self.skip_newlines();
        }

        if rules.is_empty() {
            return Err(GlrMaskError::GrammarParse("empty grammar".into()));
        }

        let mut seen = HashSet::new();
        for rule in &rules {
            if !seen.insert(&rule.name) {
                return Err(GlrMaskError::GrammarParse(format!("duplicate Lark declaration {:?}", rule.name)));
            }
        }
        let start = lark_start_rule_name(&rules)?;
        mark_lark_terminal_rules(&mut rules);
        self.ignored=synthesize_lark_ignore_rules(&mut rules,ignore_exprs);
        let ignore=self.ignored.first().cloned();

        Ok(NamedGrammar {
            rules,
            start,
            ignore,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        })
    }

    fn parse_alternatives(&mut self)->Result<GrammarExpr,GlrMaskError> {
        if self.expression_depth>=128 {return Err(GlrMaskError::GrammarParse("Lark expression nesting exceeds the supported depth of 128".into()));}
        self.expression_depth+=1;
        let result=self.parse_alternatives_inner();
        self.expression_depth-=1;result
    }

    fn parse_alternatives_inner(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let first = self.parse_sequence()?;
        self.consume_alias_if_present()?;
        let mut alts = vec![first];

        while self.consume_alternative_separator() {
            let alt = self.parse_sequence()?;
            self.consume_alias_if_present()?;
            alts.push(alt);
        }

        Ok(choice_or_single(alts))
    }

    fn consume_alternative_separator(&mut self) -> bool {
        if self.peek() == Some(&Token::Pipe) {
            self.pos += 1;
            return true;
        }

        let saved = self.pos;
        while self.peek() == Some(&Token::Newline) {
            self.pos += 1;
        }
        if self.pos > saved && self.peek() == Some(&Token::Pipe) {
            self.pos += 1;
            return true;
        }

        self.pos = saved;
        false
    }

    fn consume_alias_if_present(&mut self) -> Result<(), GlrMaskError> {
        if self.peek() != Some(&Token::Arrow) {
            return Ok(());
        }

        self.pos += 1;
        match self.advance() {
            Some(Token::Ident(_)) | Some(Token::Terminal(_)) => Ok(()),
            Some(other) => Err(GlrMaskError::GrammarParse(format!(
                "expected alias name after ->, got {:?}",
                other
            ))),
            None => Err(GlrMaskError::GrammarParse(
                "expected alias name after ->, got end of input".into(),
            )),
        }
    }

    fn parse_sequence(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let mut parts = Vec::new();

        while self.is_unit_start() {
            parts.push(self.parse_unit()?);
        }

        Ok(sequence_or_single(parts))
    }

    fn is_unit_start(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Ident(_))
                | Some(Token::Terminal(_))
                | Some(Token::Literal(_))
                | Some(Token::SpecialToken(_))
                | Some(Token::EventSet(_))
                | Some(Token::External(_))
                | Some(Token::JsonSchema(_))
                | Some(Token::Regex(_))
                | Some(Token::LParen)
                | Some(Token::LBracket)
                | Some(Token::Dot)
        )
    }

    fn parse_unit(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let atom = self.parse_atom()?;

        match self.peek() {
            Some(Token::Star) => {
                self.pos += 1;
                Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::ZeroPlus))
            }
            Some(Token::Plus) => {
                self.pos += 1;
                Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::OnePlus))
            }
            Some(Token::Question) => {
                self.pos += 1;
                Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::Optional))
            }
            Some(Token::Tilde) => {
                self.pos += 1;
                self.parse_bounded_repeat(atom)
            }
            _ => Ok(atom),
        }
    }

    fn remember_source_span(&mut self,name:&str,position:usize) {
        if let Some(span)=self.spans.get(position) {
            self.source_spans.entry(name.to_owned()).or_insert(*span);
        }
    }

    fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        let position=self.pos;
        match self.advance() {
            Some(Token::Ident(name)) | Some(Token::Terminal(name)) => Ok(GrammarExpr::Ref(name)),
            Some(Token::Literal(literal)) => self.parse_literal_or_range(literal),
            Some(Token::SpecialToken(token_id)) => {
                let name=format!("$lark_event_{}",self.events.len());
                self.remember_source_span(&name,position);
                self.events.push((name.clone(),LarkTokenSet::Ranges{ranges:vec![(token_id,token_id)],negate:false}));
                Ok(GrammarExpr::Ref(name))
            }
            Some(Token::EventSet(set))=>{
                let name=format!("$lark_event_{}",self.events.len());self.remember_source_span(&name,position);self.events.push((name.clone(),set));Ok(GrammarExpr::Ref(name))
            }
            Some(Token::External(name))=>{self.remember_source_span(&name,position);self.imports.insert(name.clone());Ok(GrammarExpr::Ref(format!("@{name}")))}
            Some(Token::JsonSchema(range))=>{
                let name=format!("$lark_json_{}",self.schemas.len());self.remember_source_span(&name,position);self.schemas.push((name.clone(),range));Ok(GrammarExpr::Ref(name))
            }
            Some(Token::Regex(regex)) => Ok(GrammarExpr::RawRegex(regex)),
            Some(Token::Dot) => Err(GlrMaskError::GrammarParse(
                "bare '.' is not a Lark atom; use /./ for a Unicode character".into()
            )),
            Some(Token::LParen) => {
                let expr = self.parse_alternatives()?;
                self.expect_token(&Token::RParen)?;
                Ok(expr)
            }
            Some(Token::LBracket) => {
                let expr = self.parse_alternatives()?;
                self.expect_token(&Token::RBracket)?;
                Ok(GrammarExpr::Quantified(Box::new(expr), Quantifier::Optional))
            }
            other => Err(GlrMaskError::GrammarParse(format!(
                "expected atom, got {:?}",
                other
            ))),
        }
    }
}

pub fn parse_lark(input: &str) -> Result<GrammarDef, GlrMaskError> {
    let named = parse_lark_to_named(input)?;
    let factored = factor_named_grammar(named);
    lower(&factored)
}

pub fn parse_lark_to_named_uncompressed(input:&str)->Result<NamedGrammar,GlrMaskError> {
    let document=parse_lark_document(input,LarkProfile::Llm)?;
    document.into_plain_grammar()
}

pub fn parse_lark_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {
    let mut named = parse_lark_to_named_uncompressed(input)?;
    optimize_lark_grammar(&mut named);
    Ok(named)
}

fn optimize_lark_grammar(named:&mut NamedGrammar) {
    // The transformation is recognition-preserving and retains the ordinary
    // static backend. Keep an explicit falsey escape hatch for diagnosis and
    // rollback, but use the compressed representation by default for matching
    // large right-linear grammars.
    if std::env::var("GLRMASK_ENABLE_RIGHT_LINEAR_COMPRESSION")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
    {
        crate::grammar::right_linear::compress_large_right_linear_grammar(named);
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_unicode_is_preserved_in_lark_literals_and_regexes() {
        let mut lexer = Lexer::new("start: \" —\" / →/\n");
        let tokens = lexer.tokenize().unwrap();
        assert!(tokens.contains(&Token::Literal(" —".to_string())));
        assert!(tokens.contains(&Token::Regex(checked_pattern(" →", "", true).unwrap())));

        let named = parse_lark_to_named_uncompressed("start: \" —\"\n").unwrap();
        let start = named.rules.iter().find(|rule| rule.name == "start").unwrap();
        assert_eq!(start.expr, GrammarExpr::Literal(" —".as_bytes().to_vec()));

        let named = parse_lark_to_named_uncompressed("start: / →/\n").unwrap();
        let start = named.rules.iter().find(|rule| rule.name == "start").unwrap();
        assert_eq!(start.expr, GrammarExpr::RawRegex(checked_pattern(" →", "", true).unwrap()));


        let named = parse_lark_to_named_uncompressed(r#"start: " \u2014 \U0001F600 \uD83D\uDE00"
"#).unwrap();
        let start = named.rules.iter().find(|rule| rule.name == "start").unwrap();
        assert_eq!(
            start.expr,
            GrammarExpr::Literal(" — 😀 😀".as_bytes().to_vec())
        );
    }

    fn large_right_linear_terminal_grammar(states: usize) -> String {
        assert!(states >= MIN_LARK_TERMINAL_GRAPH_RULES + 16);
        let mut source = String::from("start: ROOT\nROOT: S0\n");
        for index in 0..states - 1 {
            source.push_str(&format!("S{index}: \"a\" S{} | \"z\"\n", index + 1));
        }
        source.push_str(&format!("S{}: \"z\"\n", states - 1));
        source
    }

    fn dfa_accepts(dfa: &LexerDfa, input: &[u8]) -> bool {
        let mut state = 0;
        for &byte in input {
            let Some(next) = dfa.step(state, byte) else {
                return false;
            };
            state = next;
        }
        !dfa.finalizers(state).is_empty()
    }

    fn assert_dfa_language_equivalent(left: &LexerDfa, right: &LexerDfa) {
        use std::collections::{HashSet, VecDeque};

        let mut seen = HashSet::new();
        let mut queue = VecDeque::from([(Some(0u32), Some(0u32))]);
        while let Some((left_state, right_state)) = queue.pop_front() {
            if !seen.insert((left_state, right_state)) {
                continue;
            }
            let left_accepting = left_state
                .is_some_and(|state| !left.finalizers(state).is_empty());
            let right_accepting = right_state
                .is_some_and(|state| !right.finalizers(state).is_empty());
            assert_eq!(
                left_accepting, right_accepting,
                "acceptance differs at product state {left_state:?}/{right_state:?}"
            );

            for byte in 0u8..=255 {
                let next = (
                    left_state.and_then(|state| left.step(state, byte)),
                    right_state.and_then(|state| right.step(state, byte)),
                );
                if next != (None, None) && !seen.contains(&next) {
                    queue.push_back(next);
                }
            }
        }
    }

    #[test]
    fn compact_terminal_graph_matches_direct_expression_compilation() {
        let mut source = String::from("start: ROOT\nROOT: S0\n");
        for index in 0..8 {
            source.push_str(&format!(
                "S{index}: \"a\" S{} | \"b\" S{}\n",
                index + 1,
                index + 1
            ));
        }
        source.push_str("S8: \"z\"\n");

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let raw = parser.parse_grammar().unwrap();
        let terminal_names = raw.terminal_names_set();
        let graph_index = LarkTerminalGraphIndex::new(&raw, &terminal_names);

        let compact = compile_lark_terminal_graph_root(&raw, &graph_index, "ROOT", 1)
            .unwrap()
            .expect("test graph should use compact compilation");
        let direct_expr = resolve_terminal_subexpressions(
            &raw,
            &[GrammarExpr::Ref("ROOT".to_owned())],
        )
        .unwrap()
        .pop()
        .unwrap();
        let direct = crate::automata::lexer::compile::compile_terminal_expr_dfa(&direct_expr);

        assert_dfa_language_equivalent(&compact, &direct);
    }

    #[test]
    fn large_right_linear_terminal_graph_compiles_without_expansion() {
        let named = parse_lark_to_named_uncompressed(&large_right_linear_terminal_grammar(96))
            .expect("large terminal graph should import");

        // The parser start and the externally visible root remain. The 96
        // source helper terminals have been absorbed into the root DFA.
        assert_eq!(named.rules.len(), 2);
        let root = named.rules.iter().find(|rule| rule.name == "ROOT").unwrap();
        let GrammarExpr::LexerDfa(dfa) = &root.expr else {
            panic!("large right-linear terminal root was not compiled to a DFA");
        };

        assert!(dfa_accepts(dfa, b"z"));
        assert!(dfa_accepts(dfa, b"az"));
        assert!(dfa_accepts(dfa, &[b'a'; 95].into_iter().chain([b'z']).collect::<Vec<_>>()));
        assert!(!dfa_accepts(dfa, b"a"));
        assert!(!dfa_accepts(
            dfa,
            &[b'a'; 96].into_iter().chain([b'z']).collect::<Vec<_>>()
        ));
    }

    #[test]
    fn small_terminal_helpers_remain_shared_and_internal() {
        let named = parse_lark_to_named_uncompressed(
            "start: ROOT\nROOT: PREFIX SUFFIX\nPREFIX: \"a\"\nSUFFIX: \"b\"\n",
        )
        .unwrap();
        let root = named.rules.iter().find(|rule| rule.name == "ROOT").unwrap();
        assert!(!root.is_internal);
        assert!(matches!(root.expr, GrammarExpr::Sequence(_)));
        for helper in ["PREFIX", "SUFFIX"] {
            assert!(
                named
                    .rules
                    .iter()
                    .find(|rule| rule.name == helper)
                    .is_some_and(|rule| rule.is_internal),
                "{helper} should be an internal terminal helper"
            );
        }
        lower(&factor_named_grammar(named)).unwrap();
    }

    #[test]
    fn cyclic_terminal_definitions_still_error() {
        let error = parse_lark_to_named_uncompressed("start: A\nA: B\nB: A\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("cyclic Lark terminal definition"), "{error}");
    }
}


#[cfg(test)]
mod compatibility_regressions {
    use super::*;
    #[test]
    fn exact_repetition_and_invalid_ranges() {
        let g = parse_lark_to_named_uncompressed("start: \"a\"~3").unwrap();
        assert!(matches!(g.rules[0].expr, GrammarExpr::Quantified(_, Quantifier::Range(3, Some(3)))));
        assert!(parse_lark_to_named_uncompressed("start: \"a\"~3..2").is_err());
        assert!(parse_lark_to_named_uncompressed("start: \"a\"~99999999999999999999999999999999").is_err());
    }
    #[test]
    fn never_accept_priorities_or_unknown_directives() {
        for s in ["start.0: \"a\"", "start.-2: \"a\"", "start: A\nA.2: \"a\"",
                  "%extend start: \"x\"\nstart: \"a\"", "%unknown thing\nstart: \"a\"", "%import unknown.A\nstart: A"] {
            assert!(parse_lark_to_named_uncompressed(s).is_err(), "{s}");
        }
    }
    #[test]
    fn common_alias_group_and_duplicate_checks() {
        let g = parse_lark_to_named_uncompressed("%import common (INT, WS)\n%ignore WS\nstart: INT").unwrap();
        assert!(g.rules.iter().any(|r| r.name == "INT"));
        assert!(parse_lark_to_named_uncompressed("%import common.INT -> N\nstart: N").is_ok());
        assert!(parse_lark_to_named_uncompressed("%import common.INT\nINT: \"a\"\nstart: INT").is_err());
        assert!(parse_lark_to_named_uncompressed("%import common.BOGUS\nstart: BOGUS").is_err());
    }
    #[test]
    fn unicode_scalar_ranges_and_missing_start() {
        assert!(parse_lark_to_named_uncompressed("start: \"α\"..\"ω\"").is_ok());
        assert!(parse_lark_to_named_uncompressed("start: \"ab\"..\"z\"").is_err());
        assert!(parse_lark_to_named_uncompressed("other: \"a\"").is_err());
        assert!(parse_lark_to_named_uncompressed("start: .").is_err());
        assert!(parse_lark_to_named_uncompressed("start: \"a\"\n%ignore /[ \\t]/+").is_ok());
    }
}

#[cfg(test)]
mod regular_frontend_regressions {
    use super::*;
    #[test]
    fn flags_casefold_and_lazy_have_shared_regular_nodes() {
        for source in [r#"start: /abc/i"#, r#"start: "abc"i"#, r#"start: /./s"#, r#"start: /\p{L}+/"#,
                       "start: word\nword[lazy]: /[a-z]+/"] {
            assert!(parse_lark_to_named_uncompressed(source).is_ok(), "{source}");
        }
        for source in ["start: x\nx[lazy]: y\ny: \"a\"", "start[capture]: /a+/", "start: /a+/q", "start: /a+?/"] {
            assert!(parse_lark_to_named_uncompressed(source).is_err(), "{source}");
        }
    }
}
