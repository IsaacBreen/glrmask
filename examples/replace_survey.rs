use std::fs;
use std::path::{Path, PathBuf};

use glrmask::{Constraint, Vocab};
use serde_json::Value;

#[derive(Debug)]
struct ExtractedSchema {
    source_file: &'static str,
    case_name: String,
    schema: String,
}

#[derive(Debug)]
struct SkippedCase {
    source_file: &'static str,
    test_name: String,
    reason: String,
}

#[derive(Debug)]
struct SurveyRow {
    source_file: &'static str,
    case_name: String,
    total_gotos: usize,
    replace_gotos: usize,
    total_shifts: usize,
    replace_shifts: usize,
}

fn byte_vocab() -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = (0..=255u32).map(|byte| (byte, vec![byte as u8])).collect();
    Vocab::new(entries, None)
}

fn parse_total_replace_line(line: &str, prefix: &str) -> Option<(usize, usize)> {
    let rest = line.strip_prefix(prefix)?.trim();
    let total_end = rest.find(' ')?;
    let total = rest[..total_end].parse().ok()?;
    let replace_marker = "(replace: ";
    let replace_start = rest.find(replace_marker)? + replace_marker.len();
    let replace_end = rest[replace_start..].find(')')? + replace_start;
    let replace = rest[replace_start..replace_end].parse().ok()?;
    Some((total, replace))
}

fn parse_replace_counts(stats: &str) -> Option<(usize, usize, usize, usize)> {
    let mut total_shifts = None;
    let mut replace_shifts = None;
    let mut total_gotos = None;
    let mut replace_gotos = None;

    for line in stats.lines() {
        if let Some((total, replace)) = parse_total_replace_line(line, "Total shifts:") {
            total_shifts = Some(total);
            replace_shifts = Some(replace);
        }
        if let Some((total, replace)) = parse_total_replace_line(line, "Total gotos:") {
            total_gotos = Some(total);
            replace_gotos = Some(replace);
        }
    }

    Some((
        total_gotos?,
        replace_gotos?,
        total_shifts?,
        replace_shifts?,
    ))
}

fn parse_string_literal(src: &str, start: usize) -> Option<(String, usize)> {
    let bytes = src.as_bytes();
    if start >= bytes.len() {
        return None;
    }

    if bytes[start] == b'r' {
        let mut hash_count = 0usize;
        let mut quote_pos = start + 1;
        while quote_pos < bytes.len() && bytes[quote_pos] == b'#' {
            hash_count += 1;
            quote_pos += 1;
        }
        if quote_pos >= bytes.len() || bytes[quote_pos] != b'"' {
            return None;
        }
        let content_start = quote_pos + 1;
        let terminator = format!("\"{}", "#".repeat(hash_count));
        let tail = &src[content_start..];
        let rel_end = tail.find(&terminator)?;
        let content_end = content_start + rel_end;
        let literal_end = content_end + terminator.len();
        return Some((src[content_start..content_end].to_string(), literal_end));
    }

    if bytes[start] != b'"' {
        return None;
    }
    let mut pos = start + 1;
    let mut escaped = false;
    while pos < bytes.len() {
        let byte = bytes[pos];
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            let literal = &src[start..=pos];
            let parsed: String = serde_json::from_str(literal).ok()?;
            return Some((parsed, pos + 1));
        }
        pos += 1;
    }
    None
}

fn extract_fixture_schema(body: &str) -> Option<String> {
    let path_call = body.find("Path::new(")?;
    let after = &body[path_call + "Path::new(".len()..];
    let literal_start = after.find('"').or_else(|| after.find('r'))?;
    let (path, _) = parse_string_literal(after, literal_start)?;
    let fixture_text = fs::read_to_string(path).ok()?;
    let fixture: Value = serde_json::from_str(&fixture_text).ok()?;
    Some(fixture.get("schema")?.to_string())
}

fn extract_schema_literals(body: &str) -> Vec<String> {
    let mut schemas = Vec::new();
    let mut search_start = 0usize;
    while let Some(rel_pos) = body[search_start..].find("let schema =") {
        let assign_start = search_start + rel_pos + "let schema =".len();
        let mut value_start = assign_start;
        while value_start < body.len() && body.as_bytes()[value_start].is_ascii_whitespace() {
            value_start += 1;
        }
        if let Some((schema, end)) = parse_string_literal(body, value_start) {
            schemas.push(schema);
            search_start = end;
            continue;
        }
        search_start = value_start.saturating_add(1);
    }
    schemas
}

fn extract_test_functions(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    let mut saw_test_attr = false;

    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[test]") {
            saw_test_attr = true;
        } else if saw_test_attr && trimmed.starts_with("fn test_") {
            let name_end = trimmed.find('(').unwrap_or(trimmed.len());
            let name = trimmed[3..name_end].trim().to_string();
            let brace_pos = src[offset..].find('{').map(|rel| offset + rel).unwrap();
            let mut depth = 0i32;
            let mut end = brace_pos;
            for (idx, ch) in src[brace_pos..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = brace_pos + idx + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            out.push((name, src[brace_pos..end].to_string()));
            saw_test_attr = false;
        } else if !trimmed.starts_with("#[ignore") && !trimmed.is_empty() {
            saw_test_attr = false;
        }
        offset += line.len() + 1;
    }

    out
}

fn collect_cases(
    source_file: &'static str,
    body: &str,
    test_name: &str,
    extracted: &mut Vec<ExtractedSchema>,
    skipped: &mut Vec<SkippedCase>,
) {
    let schema_literals = extract_schema_literals(body);
    if !schema_literals.is_empty() {
        for (index, schema) in schema_literals.into_iter().enumerate() {
            let case_name = if index == 0 {
                test_name.to_string()
            } else {
                format!("{}#{}", test_name, index + 1)
            };
            extracted.push(ExtractedSchema {
                source_file,
                case_name,
                schema,
            });
        }
        return;
    }

    if body.contains("read_fixture_schema(") {
        if let Some(schema) = extract_fixture_schema(body) {
            extracted.push(ExtractedSchema {
                source_file,
                case_name: format!("{}#fixture", test_name),
                schema,
            });
        } else {
            skipped.push(SkippedCase {
                source_file,
                test_name: test_name.to_string(),
                reason: "fixture schema not available".to_string(),
            });
        }
        return;
    }

    if body.contains("let schema = format!(") || body.contains("let schema = match ") {
        skipped.push(SkippedCase {
            source_file,
            test_name: test_name.to_string(),
            reason: "dynamic schema construction".to_string(),
        });
    }
}

fn survey_file(
    root: &Path,
    relative: &'static str,
    extracted: &mut Vec<ExtractedSchema>,
    skipped: &mut Vec<SkippedCase>,
) {
    let full_path = root.join(relative);
    let src = fs::read_to_string(&full_path).expect("source file should be readable");
    for (test_name, body) in extract_test_functions(&src) {
        collect_cases(relative, &body, &test_name, extracted, skipped);
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vocab = byte_vocab();

    let mut extracted = Vec::new();
    let mut skipped = Vec::new();
    survey_file(&root, "src/import/test_json_schema.rs", &mut extracted, &mut skipped);
    survey_file(&root, "src/import/json_schema.rs", &mut extracted, &mut skipped);

    let mut rows = Vec::new();
    let mut failures = Vec::new();

    for case in extracted {
        match Constraint::from_json_schema(&case.schema, &vocab) {
            Ok(constraint) => {
                let stats = constraint.debug_table_stats();
                if let Some((total_gotos, replace_gotos, total_shifts, replace_shifts)) =
                    parse_replace_counts(&stats)
                {
                    rows.push(SurveyRow {
                        source_file: case.source_file,
                        case_name: case.case_name,
                        total_gotos,
                        replace_gotos,
                        total_shifts,
                        replace_shifts,
                    });
                } else {
                    failures.push(format!(
                        "{} {} parse_replace_counts failed",
                        case.source_file, case.case_name
                    ));
                }
            }
            Err(error) => failures.push(format!(
                "{} {} compile failed: {}",
                case.source_file, case.case_name, error
            )),
        }
    }

    rows.sort_by(|left, right| {
        right
            .replace_gotos
            .cmp(&left.replace_gotos)
            .then(right.replace_shifts.cmp(&left.replace_shifts))
            .then(right.total_gotos.cmp(&left.total_gotos))
            .then(right.total_shifts.cmp(&left.total_shifts))
            .then(left.case_name.cmp(&right.case_name))
    });

    let nonzero_goto = rows.iter().filter(|row| row.replace_gotos > 0).count();
    let nonzero_shift = rows.iter().filter(|row| row.replace_shifts > 0).count();
    let max_goto = rows.iter().max_by_key(|row| row.replace_gotos);
    let max_shift = rows.iter().max_by_key(|row| row.replace_shifts);

    let mut out = String::new();
    out.push_str("Replace survival survey across extracted JSON-schema test grammars\n");
    out.push_str(&format!("compiled_cases: {}\n", rows.len()));
    out.push_str(&format!("skipped_cases: {}\n", skipped.len()));
    out.push_str(&format!("compile_failures: {}\n", failures.len()));
    out.push_str(&format!("cases_with_nonzero_goto_replace: {}\n", nonzero_goto));
    out.push_str(&format!("cases_with_nonzero_shift_replace: {}\n", nonzero_shift));
    if let Some(row) = max_goto {
        out.push_str(&format!(
            "max_goto_replace: {} {} {}/{}\n",
            row.source_file, row.case_name, row.replace_gotos, row.total_gotos
        ));
    }
    if let Some(row) = max_shift {
        out.push_str(&format!(
            "max_shift_replace: {} {} {}/{}\n",
            row.source_file, row.case_name, row.replace_shifts, row.total_shifts
        ));
    }
    out.push('\n');
    out.push_str("source_file\tcase_name\treplace_gotos\ttotal_gotos\treplace_shifts\ttotal_shifts\n");
    for row in &rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            row.source_file,
            row.case_name,
            row.replace_gotos,
            row.total_gotos,
            row.replace_shifts,
            row.total_shifts,
        ));
    }

    if !skipped.is_empty() {
        out.push('\n');
        out.push_str("Skipped cases\n");
        for case in &skipped {
            out.push_str(&format!(
                "{}\t{}\t{}\n",
                case.source_file, case.test_name, case.reason,
            ));
        }
    }

    if !failures.is_empty() {
        out.push('\n');
        out.push_str("Compile failures\n");
        for failure in &failures {
            out.push_str(failure);
            out.push('\n');
        }
    }

    fs::write("/tmp/replace_survey.txt", &out).expect("survey output should be writable");
    print!("{}", out);
}