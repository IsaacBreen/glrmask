use std::{env, ffi::OsString, sync::Mutex};

use glrmask::{Constraint, Vocab};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = env::var_os(key);
        unsafe {
            env::set_var(key, value);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => unsafe {
                env::set_var(self.key, value);
            },
            None => unsafe {
                env::remove_var(self.key);
            },
        }
    }
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

#[test]
fn chunk16_bounded_service_name_allows_spaces_token_after_open_quote() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _chunk = EnvVarGuard::set("GLRMASK_STRING_REPEAT_CHUNK", "16");

    let schema = r#"{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "serviceName": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100
            }
        },
        "required": ["serviceName"]
    }"#;
    let prefix = br#"{"serviceName": ""#;
    let vocab = Vocab::new(vec![(0, vec![b' '; 24])], None);

    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();

    assert!(token_allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
}