//! Checked regular-language regex frontend shared by GLRM and Lark.
//!
//! Match preferences, lookaround and anchors are not language operators in this
//! frontend. Explicit shortest-prefix operation is handled outside regex parsing.
use regex_syntax::{ParserBuilder, hir::{Class, Hir, HirKind}};
use super::{decode_unicode_escape, utf8_range_expr};
use crate::{Expr, U8Set};

/// Resolve external regex flags and produce an unambiguous, self-contained
/// pattern. The returned string can be retained in the shared RawRegex AST.
pub fn checked_pattern(pattern: &str, flags: &str, unicode: bool) -> Result<String, String> {
    let mut seen = std::collections::BTreeSet::new();
    for flag in flags.chars() {
        if !matches!(flag, 'i' | 's' | 'x' | 'u') {
            return Err(format!("unsupported regex flag {flag:?}; supported flags are i, s, x, u"));
        }
        if !seen.insert(flag) { return Err(format!("duplicate regex flag {flag:?}")); }
        if flag == 'u' && !unicode { return Err("Unicode flag is not valid on a byte regex".into()); }
    }
    let mut active = String::new();
    if unicode { active.push('u'); }
    for flag in ['i', 's', 'x'] { if seen.contains(&flag) { active.push(flag); } }
    let mut inactive = String::new();
    if !unicode { inactive.push('u'); }
    // Pin defaults: nested patterns do not inherit an external library mode.
    for flag in ['i', 's', 'x'] { if !seen.contains(&flag) { inactive.push(flag); } }
    let normalized = normalize_scalar_escapes(pattern,unicode,seen.contains(&'x'))?;
    let encoded = if inactive.is_empty() { format!("(?{active}:{normalized})") }
        else { format!("(?{active}-{inactive}:{normalized})") };
    validate_syntax(&encoded,Some(unicode))?;
    parse_checked_regex(&encoded)?;
    Ok(encoded)
}

/// Render a mode-qualified checked pattern without accumulating wrapper groups
/// on a parse/format round trip. Regex AST spans prove this is one whole group;
/// byte/Unicode and i/s/x choices stay explicit in the emitted source literal.
pub fn source_regex_literal(pattern: &str) -> Option<String> {
    use regex_syntax::ast::{Ast, Flag};
    let ast=regex_syntax::ast::parse::Parser::new().parse(pattern).ok()?;
    let Ast::Group(group)=&ast else {return None;};
    if group.span.start.offset!=0 || group.span.end.offset!=pattern.len() {return None;}
    let flags=group.flags()?;
    let unicode=flags.flag_state(Flag::Unicode)?;
    let mut suffix=String::new();
    for (flag,ch) in [(Flag::CaseInsensitive,'i'),(Flag::DotMatchesNewLine,'s'),(Flag::IgnoreWhitespace,'x')] {
        if flags.flag_state(flag)? {suffix.push(ch);}
    }
    // Only our complete source-mode wrapper is peeled. User-local flags remain
    // inside the raw pattern and are parsed independently on the next import.
    if flags.items.iter().any(|f| matches!(&f.kind,
        regex_syntax::ast::FlagsItemKind::Flag(Flag::MultiLine|Flag::SwapGreed|Flag::CRLF))) {return None;}
    let colon=pattern.find(':')?;
    let body=&pattern[colon+1..pattern.len()-1];
    Some(format!("{}/{}/{}",if unicode {""} else {"b"},body.replace('/',r"\/"),suffix))
}

/// Parse a source regex into a byte-language expression without fallback to a
/// literal or silent syntax loss. Unicode classes are encoded as UTF-8 ranges.
pub fn parse_checked_regex(pattern: &str) -> Result<Expr, String> {
    validate_syntax(pattern,None)?;
    let hir = ParserBuilder::new().utf8(false).build().parse(pattern)
        .map_err(|error| error.to_string())?;
    lower(&hir)
}

// Inspect syntax before HIR simplification. An unreachable assertion or a
// zero-count reluctant repetition must not quietly disappear during optimization.
fn validate_syntax(pattern:&str,unicode:Option<bool>)->Result<(),String> {
    use regex_syntax::ast::{Ast,Flag,Flags,FlagsItemKind};
    fn flags(flags:&Flags,unicode:Option<bool>)->Result<(),String> {
        for item in &flags.items {
            if let FlagsItemKind::Flag(flag)=&item.kind {
                if matches!(flag,Flag::MultiLine|Flag::SwapGreed|Flag::CRLF){
                    return Err("inline m, U, and R regex flags are not supported".into());
                }
            }
        }
        if let (Some(expected),Some(actual))=(unicode,flags.flag_state(Flag::Unicode)) {
            if actual!=expected{return Err("regex byte/Unicode alphabet cannot be changed by an inline flag; choose b/.../ or /.../ explicitly".into());}
        }
        Ok(())
    }
    let ast=regex_syntax::ast::parse::Parser::new().parse(pattern).map_err(|e|e.to_string())?;
    let mut pending=vec![&ast];
    while let Some(node)=pending.pop(){match node {
        Ast::Assertion(_)=>return Err("regex anchors and assertions are not supported; patterns describe whole lexemes".into()),
        Ast::Repetition(rep)=>{
            if !rep.greedy{return Err("reluctant regex quantifiers are unsupported; use the whole-terminal shortest operation or [lazy]".into());}
            pending.push(&rep.ast);
        }
        Ast::Group(group)=>{if let Some(f)=group.flags(){flags(f,unicode)?;}pending.push(&group.ast);},
        Ast::Flags(f)=>flags(&f.flags,unicode)?,
        Ast::Concat(concat)=>pending.extend(&concat.asts),
        Ast::Alternation(alt)=>pending.extend(&alt.asts),
        _=>{},
    }}
    Ok(())
}

fn lower(hir: &Hir) -> Result<Expr, String> {
    Ok(match hir.kind() {
        HirKind::Empty => Expr::Epsilon,
        HirKind::Literal(lit) => Expr::U8Seq(lit.0.to_vec()),
        HirKind::Class(Class::Bytes(class)) => {
            let mut set = U8Set::empty();
            for range in class.iter() { for byte in range.start()..=range.end() { set.insert(byte); } }
            Expr::U8Class(set)
        }
        HirKind::Class(Class::Unicode(class)) => {
            let mut parts = class.iter().map(|range| utf8_range_expr(range.start(), range.end())).collect::<Vec<_>>();
            if parts.len() == 1 { parts.pop().unwrap() } else { Expr::Choice(parts) }
        }
        HirKind::Look(_) => return Err("regex anchors and lookaround are not supported; patterns already describe full lexemes".into()),
        HirKind::Capture(capture) => lower(&capture.sub)?,
        HirKind::Repetition(rep) => {
            if !rep.greedy {
                return Err("reluctant regex quantifiers are not silently ignored; use shortest(...) or a regular [lazy] rule for earliest lexeme completion".into());
            }
            Expr::Repeat { expr: Box::new(lower(&rep.sub)?), min: rep.min as usize, max: rep.max.map(|n| n as usize) }
        }
        HirKind::Concat(parts) => Expr::Seq(parts.iter().map(lower).collect::<Result<_,_>>()?),
        HirKind::Alternation(parts) => Expr::Choice(parts.iter().map(lower).collect::<Result<_,_>>()?),
    })
}

/// Preserve escaped scalar literals (including valid UTF-16 pairs) without
/// accidentally interpreting an escaped metacharacter as regex syntax.
fn normalize_scalar_escapes(pattern: &str, unicode: bool, mut extended: bool) -> Result<String, String> {
    let mut out=String::new();let bytes=pattern.as_bytes();let mut pos=0;
    let mut groups=Vec::<bool>::new();let mut classes=Vec::<u8>::new();
    while pos<bytes.len() {
        // Extended-mode comments are not syntax: malformed-looking escapes in
        // them must not be decoded or rejected before the regex parser sees it.
        if extended && bytes[pos]==b'#' {
            let end=bytes[pos..].iter().position(|&b|b==b'\n').map_or(bytes.len(),|n|pos+n+1);
            out.push_str(&pattern[pos..end]);pos=end;continue;
        }
        if bytes[pos]==b'\\' && pos+1<bytes.len() {
            if let Some(first)=classes.last_mut(){*first=2;}
            if matches!(bytes[pos+1],b'u'|b'U') {
                if !unicode {return Err("Unicode scalar escapes are not valid in byte regexes; use byte escapes".into());}
                if bytes.get(pos+2)!=Some(&b'{') {
                    let (scalar,end)=decode_unicode_escape(bytes,pos).ok_or_else(||"invalid Unicode scalar escape in regex".to_string())?;
                    out.push_str(&format!(r"\x{{{:X}}}",scalar as u32));pos=end;continue;
                }
            }
            let ch=pattern[pos+1..].chars().next().unwrap();out.push('\\');out.push(ch);pos+=1+ch.len_utf8();continue;
        }
        let ch=pattern[pos..].chars().next().unwrap();
        if extended && ch.is_ascii_whitespace(){out.push(ch);pos+=ch.len_utf8();continue;}
        if !classes.is_empty() {
            match ch {
                '['=>{*classes.last_mut().unwrap()=2;classes.push(0);}
                ']' if *classes.last().unwrap()<2=>*classes.last_mut().unwrap()=2,
                ']'=>{classes.pop();}
                '^' if *classes.last().unwrap()==0=>*classes.last_mut().unwrap()=1,
                _=>*classes.last_mut().unwrap()=2,
            }
        } else if ch=='[' {classes.push(0);}
        else if ch=='(' {
            // Follow only the flag header, not arbitrary regex subexpressions.
            // Captures and non-capturing groups still restore the enclosing mode.
            let mut end=pos+2;let mut next=extended;let mut enabled=true;
            if bytes.get(pos+1)==Some(&b'?') {
                while let Some(&flag)=bytes.get(end) {
                    if flag==b'-' {enabled=false;}
                    else if b"imsuxUR".contains(&flag) {if flag==b'x'{next=enabled;}}
                    else {break;}
                    end+=1;
                }
                if end>pos+2 && matches!(bytes.get(end),Some(b':'|b')')) {
                    if bytes[end]==b':' {groups.push(extended);}
                    extended=next;out.push_str(&pattern[pos..=end]);pos=end+1;continue;
                }
            }
            groups.push(extended);
        } else if ch==')' {if let Some(previous)=groups.pop(){extended=previous;}}
        out.push(ch);pos+=ch.len_utf8();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Parser result shape tests avoid conflating syntax with execution preference.
    #[test]
    fn flags_and_bad_syntax_are_explicit() {
        assert!(checked_pattern("abc", "i", true).is_ok());
        assert!(checked_pattern(".", "s", true).is_ok());
        assert!(checked_pattern(" a # ignored\n b", "x", true).is_ok());
        assert!(checked_pattern(r"\p{L}+", "", true).is_ok());
        assert!(checked_pattern(r"\xFF", "", false).is_ok());
        for (pat, flags) in [("a", "ii"), ("a", "q"), ("a*?", ""), ("^a$", ""), (r"(a)\1", ""), ("(?=a)", "")] {
            assert!(checked_pattern(pat, flags, true).is_err(), "{pat:?}/{flags}");
        }
    }
    #[test]
    fn eliminated_syntax_still_obeys_the_source_contract() {
        for pattern in ["a{0}?","a{1}?","(?:^){0}","(?U:a)","(?m:a)","(?R:a)","(?-u:.)"] {
            assert!(checked_pattern(pattern,"",true).is_err(),"{pattern}");
        }
        assert!(checked_pattern("(?u:.)","",false).is_err());
        assert!(checked_pattern("a # ^ *? ignored\n b","x",true).is_ok());
        assert!(checked_pattern("(?i:a)","",true).is_ok());
    }

    #[test]
    fn scalar_escapes_and_empty_language() {
        assert!(checked_pattern(r"\uD83D\uDE00", "", true).is_ok());
        assert!(checked_pattern(r"\uD800", "", true).is_err());
        assert_eq!(parse_checked_regex(r"[^\s\S]").unwrap(), Expr::U8Class(U8Set::empty()));
    }

    fn matches(pattern: &str, flags: &str, unicode: bool, text: &[u8]) -> bool {
        use std::sync::Arc;
        use crate::automata::lexer::tokenizer::Lexer as _;
        let encoded = checked_pattern(pattern, flags, unicode).unwrap();
        let expr = parse_checked_regex(&encoded).unwrap();
        let lexer = expr.clone().build().into_tokenizer(1, Some(Arc::from([expr])));
        lexer.execute_from_state(text, lexer.initial_state()).matches.iter()
            .any(|m| m.id == 0 && m.width == text.len())
    }

    #[test]
    fn checked_flags_scalar_and_byte_semantics() {
        assert!(matches("abc", "i", true, b"AbC"));
        assert!(!matches("abc", "", true, b"AbC"));
        assert!(matches(".", "s", true, b"\n"));
        assert!(!matches(".", "", true, b"\n"));
        assert!(matches(r"\p{L}+", "", true, "αβ".as_bytes()));
        assert!(matches(".", "", true, "😀".as_bytes()));
        assert!(!matches(".", "", true, &[0xFF]));
        assert!(matches(".", "s", false, &[0xFF]));
        assert!(!matches(".", "s", false, "α".as_bytes()));
        assert!(matches(r"\uD83D\uDE00", "", true, "😀".as_bytes()));
        assert!(matches(r"\u002e", "", true, b"."));
        assert!(!matches(r"\u002e", "", true, b"x"));
    }
    #[test]
    fn scalar_escape_normalization_respects_extended_comments_and_byte_mode() {
        assert!(checked_pattern("a # \\uD800\\UFFFFFFFF\n b","x",true).is_ok());
        assert!(checked_pattern("(?x:a # \\uD800\n b)","",true).is_ok());
        assert!(checked_pattern(r"\u0041","",false).is_err());
        assert!(checked_pattern(r"\u{41}","",false).is_err());
        assert!(checked_pattern(r"\\u0041","",false).is_ok());
        assert!(checked_pattern("(?x:a # \\uD800\n b)","",false).is_ok());
    }

}
