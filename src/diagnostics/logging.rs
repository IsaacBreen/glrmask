//! Diagnostic logging boundary. New diagnostic output should be routed here.

pub(crate) fn emit_stderr(message: impl AsRef<str>) { eprintln!("{}", message.as_ref()); }
