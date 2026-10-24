//! Composition: a tool call whose arguments must match that tool's JSON Schema.
//!
//! Each part is precompiled once, so a new combination of tools compiles quickly.
use glrmask::{Grammar, Module, Vocab};

const WEATHER: &str = r#"{
    "type": "object",
    "properties": { "city": { "type": "string" } },
    "required": ["city"],
    "additionalProperties": false
}"#;

const SEARCH: &str = r#"{
    "type": "object",
    "properties": { "query": { "type": "string" } },
    "required": ["query"],
    "additionalProperties": false
}"#;

fn main() -> glrmask::Result<()> {
    // One token per byte.
    let vocab = Vocab::new((0..=255u32).map(|b| (b, vec![b as u8])).collect());

    let mut weather = Module::new(&Grammar::from_json_schema(WEATHER)?);
    weather.precompile(&vocab)?;
    let mut search = Module::new(&Grammar::from_json_schema(SEARCH)?);
    search.precompile(&vocab)?;

    // The call format, with an extern for each tool's arguments.
    let mut call = Module::new(&Grammar::from_glrm(r#"
        extern grammar WEATHER_ARGS
        extern grammar SEARCH_ARGS
        start call
        nt call = "<weather>" WEATHER_ARGS "</weather>" | "<search>" SEARCH_ARGS "</search>"
    "#)?);
    call.precompile(&vocab)?;
    call.bind("WEATHER_ARGS", &weather)?;
    call.bind("SEARCH_ARGS", &search)?;
    let constraint = call.compile(&vocab)?;

    let mut state = constraint.start();
    assert!(state.commit_bytes(br#"<weather>{"city": "Perth"}</weather>"#));
    assert!(state.is_accepting());
    println!("accepted a weather call");

    // The weather tool takes no "query".
    let mut state = constraint.start();
    assert!(!state.commit_bytes(br#"<weather>{"query""#));
    println!("rejected a weather call with search arguments");
    Ok(())
}
