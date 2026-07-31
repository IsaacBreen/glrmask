use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use super::{choice_or_single, sequence_or_single};
use crate::GlrMaskError;
use crate::automata::lexer::{DFA as LexerDfa, compile::compile_expression_labeled_nfa};
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
}

struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer {
            input: input.as_bytes(),
            pos: 0,
        }
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

    fn lex_string(&mut self, quote: u8) -> Result<String, GlrMaskError> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some(b) if b == quote => return Ok(s),
                Some(b'\\') => match self.advance() {
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
                            GlrMaskError::GrammarParse(format!("invalid \\x escape: \\x{hex_str}"))
                        })?;
                        s.push(byte as char);
                    }
                    Some(c) => {
                        s.push('\\');
                        s.push(c as char);
                    }
                    None => return Err(GlrMaskError::GrammarParse("unterminated escape".into())),
                },
                Some(b) => s.push(b as char),
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
                        s.push(b as char);
                    }
                }
                Some(b) => s.push(b as char),
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

    fn lex_number(&mut self, first: u8) -> usize {
        let mut n = (first - b'0') as usize;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                n = n * 10 + (b - b'0') as usize;
                self.pos += 1;
            } else {
                break;
            }
        }
        n
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

    fn tokenize(&mut self) -> Result<Vec<Token>, GlrMaskError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_inline();
            match self.peek() {
                None => break,
                Some(b'/') => {
                    self.pos += 1;
                    if self.peek() == Some(b'/') {
                        self.skip_comment();
                    } else {
                        let rx = self.lex_regex()?;
                        tokens.push(Token::Regex(rx));
                    }
                }
                Some(b'#') => self.skip_comment(),
                Some(b'%') => {
                    self.pos += 1;
                    // Check for %ignore directive
                    let rest = &self.input[self.pos..];
                    if rest.starts_with(b"ignore") && rest.get(6).map_or(true, |b| !b.is_ascii_alphanumeric() && *b != b'_') {
                        self.pos += 6;
                        tokens.push(Token::PercentIgnore);
                    } else {
                        // Skip other %directives (e.g. %import)
                        self.skip_comment();
                    }
                }
                Some(b'\n') => {
                    self.pos += 1;
                    tokens.push(Token::Newline);
                }
                Some(b'"') => {
                    self.pos += 1;
                    let s = self.lex_string(b'"')?;
                    tokens.push(Token::Literal(s));
                }
                Some(b'\'') => {
                    self.pos += 1;
                    let s = self.lex_string(b'\'')?;
                    tokens.push(Token::Literal(s));
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
                    tokens.push(self.lex_special_token()?);
                }
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
                        return Err(GlrMaskError::GrammarParse("unexpected '-'".into()));
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
                    let n = self.lex_number(b);
                    tokens.push(Token::Number(n));
                }
                Some(b) => {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "unexpected character '{}' at position {}",
                        b as char, self.pos
                    )));
                }
            }
        }
        Ok(tokens)
    }
}

fn bounded_repeat_expr(atom: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {
    if let Some(max) = max {
        assert!(max >= min, "tilde max must be >= min");
    }
    GrammarExpr::Quantified(Box::new(atom), Quantifier::Range(min, max))
}

fn escape_char_class_byte(b: u8) -> String {
    match b {
        b'\\' | b']' | b'^' | b'-' => format!("\\{}", b as char),
        b'\n' => "\\n".into(),
        b'\r' => "\\r".into(),
        b'\t' => "\\t".into(),
        byte if byte.is_ascii_graphic() || byte == b' ' => (byte as char).to_string(),
        byte => format!("\\x{byte:02x}"),
    }
}

fn literal_range_expr(start: &str, end: &str) -> Result<GrammarExpr, GlrMaskError> {
    let start_bytes = start.as_bytes();
    let end_bytes = end.as_bytes();
    if start_bytes.len() != 1 || end_bytes.len() != 1 {
        return Err(GlrMaskError::GrammarParse(
            "Lark literal ranges currently require single-byte endpoints".into(),
        ));
    }

    let start_byte = start_bytes[0];
    let end_byte = end_bytes[0];
    if start_byte > end_byte {
        return Err(GlrMaskError::GrammarParse(format!(
            "invalid Lark literal range {:?}..{:?}",
            start, end
        )));
    }

    Ok(GrammarExpr::CharClass {
        def: format!(
            "{}-{}",
            escape_char_class_byte(start_byte),
            escape_char_class_byte(end_byte)
        ),
        negate: false,
        utf8: true,
    })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

fn is_lark_terminal_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

fn lark_start_rule_name(rules: &[NamedRule]) -> String {
    if rules.iter().any(|rule| rule.name == "start") {
        "start".to_string()
    } else {
        rules[0].name.clone()
    }
}

fn mark_lark_terminal_rules(rules: &mut [NamedRule]) {
    for rule in rules {
        rule.is_terminal = is_lark_terminal_name(&rule.name);
    }
}

fn synthesize_lark_ignore_rule(
    rules: &mut Vec<NamedRule>,
    ignore_exprs: Vec<GrammarExpr>,
) -> Option<String> {
    if ignore_exprs.is_empty() {
        return None;
    }

    let ignore_name = "__IGNORE".to_string();
    rules.push(NamedRule {
        name: ignore_name.clone(),
        expr: choice_or_single(ignore_exprs),
        is_terminal: true,
        is_internal: false,
    });
    Some(ignore_name)
}

const MIN_LARK_TERMINAL_GRAPH_RULES: usize = 32;

fn lark_ref_names(expr: &GrammarExpr) -> Vec<&str> {
    let mut refs = Vec::new();
    let mut stack = vec![expr];
    while let Some(expr) = stack.pop() {
        match expr {
            GrammarExpr::Ref(name) => refs.push(name.as_str()),
            GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
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
            | GrammarExpr::SpecialToken(_)
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
) -> Result<(), GlrMaskError> {
    for target in lark_ref_names(expr) {
        if !rule_map.contains_key(target) {
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

fn compress_lark_terminal_graphs(grammar: &mut NamedGrammar) -> Result<(), GlrMaskError> {
    let source = grammar.clone();
    let terminal_names = source.terminal_names_set();
    let graph_index = LarkTerminalGraphIndex::new(&source, &terminal_names);
    let external = externally_emitted_lark_terminals(&source, &terminal_names);
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

fn normalize_lark_named(grammar: NamedGrammar) -> Result<NamedGrammar, GlrMaskError> {
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
            validate_lark_parser_refs(&rule.expr, &rule_map)?;
        }
    }
    validate_lark_terminal_acyclic(&rule_map, &terminal_names)?;

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
    compress_lark_terminal_graphs(&mut normalized)?;
    Ok(normalized)
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
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
        match self.advance() {
            Some(ref tok) if tok == expected => Ok(()),
            Some(tok) => Err(GlrMaskError::GrammarParse(format!(
                "expected {:?}, got {:?}",
                expected, tok
            ))),
            None => Err(GlrMaskError::GrammarParse(format!(
                "expected {:?}, got end of input",
                expected
            ))),
        }
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

    fn skip_rule_priority(&mut self) {
        if self.peek() == Some(&Token::Dot) && matches!(self.peek_nth(1), Some(Token::Number(_))) {
            self.pos += 2;
        }
    }

    fn parse_bounded_repeat(&mut self, atom: GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        let min = match self.advance() {
            Some(Token::Number(value)) => value,
            _ => return Err(GlrMaskError::GrammarParse("expected number after ~".into())),
        };

        let max = if self.peek() == Some(&Token::Dot) {
            let saved = self.pos;
            self.pos += 1;
            if self.peek() == Some(&Token::Dot) {
                self.pos += 1;
                match self.advance() {
                    Some(Token::Number(value)) => Some(value),
                    _ => {
                        return Err(GlrMaskError::GrammarParse(
                            "expected number after ..".into(),
                        ));
                    }
                }
            } else {
                self.pos = saved;
                None
            }
        } else {
            None
        };

        Ok(bounded_repeat_expr(atom, min, max))
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
        ignore_exprs.push(self.parse_atom()?);
        self.skip_newlines();
        Ok(true)
    }

    fn parse_rule(&mut self) -> Result<NamedRule, GlrMaskError> {
        let name = self.parse_rule_name()?;
        self.skip_rule_priority();
        self.expect_token(&Token::Colon)?;
        let expr = self.parse_alternatives()?;
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
            if self.parse_ignore_directive(&mut ignore_exprs)? {
                continue;
            }

            rules.push(self.parse_rule()?);
            self.skip_newlines();
        }

        if rules.is_empty() {
            return Err(GlrMaskError::GrammarParse("empty grammar".into()));
        }

        let start = lark_start_rule_name(&rules);
        mark_lark_terminal_rules(&mut rules);
        let ignore = synthesize_lark_ignore_rule(&mut rules, ignore_exprs);

        Ok(NamedGrammar {
            rules,
            start,
            ignore,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        })
    }

    fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {
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

    fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {
        match self.advance() {
            Some(Token::Ident(name)) | Some(Token::Terminal(name)) => Ok(GrammarExpr::Ref(name)),
            Some(Token::Literal(literal)) => self.parse_literal_or_range(literal),
            Some(Token::SpecialToken(token_id)) => Ok(GrammarExpr::SpecialToken(token_id)),
            Some(Token::Regex(regex)) => Ok(GrammarExpr::RawRegex(regex)),
            Some(Token::Dot) => Ok(GrammarExpr::AnyByte),
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

pub(crate) fn parse_lark_to_named_uncompressed(
    input: &str,
) -> Result<NamedGrammar, GlrMaskError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let named = parser.parse_grammar()?;
    normalize_lark_named(named)
}

pub fn parse_lark_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {
    let mut named = parse_lark_to_named_uncompressed(input)?;
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
        crate::grammar::right_linear::compress_large_right_linear_grammar(&mut named);
    }
    Ok(named)
}


#[cfg(test)]
mod tests {
    use super::*;

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
