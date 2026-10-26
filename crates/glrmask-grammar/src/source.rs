//! Original UTF-8 source coordinates, independent of transformed grammar nodes.
/// A location in grammar source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SourceSpan {
    /// Byte offset of the first byte.
    pub start: usize,
    /// Byte offset just past the last byte.
    pub end: usize,
    /// Line of `start`, counting from 1.
    pub line: usize,
    /// Column of `start`, counting characters from 1.
    pub column: usize,
}
impl SourceSpan {
    #[doc(hidden)]
    pub const fn new(start: usize, end: usize, line: usize, column: usize) -> Self {
        Self { start, end, line, column }
    }
}
impl std::fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}
impl SourceSpan {
    pub(crate) fn at(source: &str, start: usize, end: usize) -> Self {
        let mut start=start.min(source.len());
        while !source.is_char_boundary(start) {start-=1;}
        let mut end=end.max(start).min(source.len());
        while !source.is_char_boundary(end) {end+=1;}
        let mut line=1;let mut column=1;
        for ch in source[..start].chars() {if ch=='\n' {line+=1;column=1;}else{column+=1;}}
        Self{start,end,line,column}
    }

    /// Token offsets are monotone, so locating every token is one linear scan.
    /// Recounting the prefix per token would be quadratic on generated one-line
    /// grammars. Literal and regex contents still count as original source.
    pub(crate) fn tokens(source:&str, offsets:&[(usize,usize)]) -> Vec<Self> {
        let mut result=Vec::with_capacity(offsets.len());
        let mut cursor=0;let mut line=1;let mut column=1;
        for &(start,end) in offsets {
            debug_assert!(start>=cursor && end>=start && end<=source.len());
            for ch in source[cursor..start].chars() {if ch=='\n' {line+=1;column=1;}else{column+=1;}}
            result.push(Self{start,end,line,column});cursor=start;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_locations_count_scalars_in_one_pass() {
        let source="α\n x";
        let spans=SourceSpan::tokens(source,&[(0,2),(4,5),(5,5)]);
        assert_eq!((spans[1].line,spans[1].column),(2,2));
        assert_eq!((spans[2].line,spans[2].column),(2,3));
        assert_eq!(SourceSpan::at(source,1,2).start,0);
    }
    #[test]
    fn native_and_lark_errors_retain_original_spans() {
        let source="start root\nnt root ::= \"x\"\nnt other ::= )";
        let e=crate::grammar::glrm::from_glrm(source).unwrap_err();
        let span=e.source_span().expect("native parser has a source coordinate");
        assert_eq!(&source[span.start..span.end],")");assert_eq!(span.line,3);
        let source="start: \"x\"\n%bogus hello";
        let e=crate::import::lark::parse_lark_to_named(source).unwrap_err();
        let span=e.source_span().expect("directive error has a source coordinate");
        assert_eq!(&source[span.start..span.end],"%bogus");assert_eq!((span.line,span.column),(2,1));
        let source="start: (\"x\"]";
        let e=crate::import::lark::parse_lark_to_named(source).unwrap_err();
        let span=e.source_span().expect("expected-delimiter error has a source coordinate");
        assert_eq!(&source[span.start..span.end],"]");
    }
}

#[cfg(test)]
mod depth_tests {
    #[test]
    fn deeply_nested_source_is_an_error_not_a_stack_overflow() {
        let native=format!("start root nt root::={}\"x\"{}","(".repeat(1024),")".repeat(1024));
        let lark=format!("start: {}\"x\"{}","(".repeat(1024),")".repeat(1024));
        assert!(crate::grammar::glrm::ParsedComponent::parse_glrm(&native).unwrap_err().to_string().contains("nesting"));
        assert!(crate::parse_lark_document(&lark).unwrap_err().to_string().contains("nesting"));
        let mut scope=String::from("start root nt root::=\"x\"");
        for _ in 0..80 {scope=format!("start root grammar child {{ {scope} }} nt root::=child");}
        assert!(crate::grammar::glrm::ParsedComponent::parse_glrm(&scope).unwrap_err().to_string().contains("nesting"));
    }
}
