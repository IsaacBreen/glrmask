//! String utility functions for grammar construction.
//!
//! This module contains helper functions for escaping strings
//! when building grammar patterns.

/// Escape a character for use in a character class regex pattern.
/// 
/// Characters that have special meaning in character classes are escaped:
/// - `\`, `]`, `^`, `-` are preceded by backslash
/// - Control characters are converted to escape sequences
pub fn escape_char_for_char_class(c: char) -> String {
    match c {
        '\\' | ']' | '^' | '-' => format!("\\{}", c),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        c if c.is_ascii_control() => format!("\\x{:02x}", c as u8),
        c => c.to_string(),
    }
}

/// Escape a string for use in JSON.
///
/// This handles the standard JSON escape sequences:
/// - `"` becomes `\"`
/// - `\` becomes `\\`
/// - Control characters are escaped
pub fn escape_string_for_json(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_char_for_char_class() {
        assert_eq!(escape_char_for_char_class('a'), "a");
        assert_eq!(escape_char_for_char_class('\\'), "\\\\");
        assert_eq!(escape_char_for_char_class(']'), "\\]");
        assert_eq!(escape_char_for_char_class('^'), "\\^");
        assert_eq!(escape_char_for_char_class('-'), "\\-");
        assert_eq!(escape_char_for_char_class('\n'), "\\n");
        assert_eq!(escape_char_for_char_class('\t'), "\\t");
    }

    #[test]
    fn test_escape_string_for_json() {
        assert_eq!(escape_string_for_json("hello"), "hello");
        assert_eq!(escape_string_for_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_string_for_json("path\\to\\file"), "path\\\\to\\\\file");
        assert_eq!(escape_string_for_json("line1\nline2"), "line1\\nline2");
    }
}
