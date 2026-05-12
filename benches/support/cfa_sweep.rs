#![allow(dead_code)]

use std::{collections::BTreeMap, path::{Path, PathBuf}};

use criterion::{black_box, BenchmarkId, Criterion};
use glrmask::{clear_weight_interners, clear_weight_op_caches, Constraint, Vocab};

pub struct BenchCase {
    pub id: &'static str,
    pub cfa_name: &'static str,
    pub cfa_build_seconds: f64,
    pub schema: &'static str,
}

pub const CASES: &[BenchCase] = &[
    BenchCase { id: "github_trivial_o20469", cfa_name: "Github_trivial---o20469", cfa_build_seconds: 0.005186, schema: r#"{"$schema":"http://json-schema.org/draft-04/schema#","id":"http://localhost:3000/schemas/get-devices-response.json#","title":"Devices","type":"object","additionalProperties":false,"properties":{}}"# },
    BenchCase { id: "bfcl_parallel_88", cfa_name: "BFCL_parallel_88", cfa_build_seconds: 0.012104, schema: r#"{"type":"object","properties":{"calculate_final_speed":{"type":"object","properties":{"initial_velocity":{"type":"integer"},"height":{"type":"integer"},"gravity":{"type":"number"}},"required":["initial_velocity","height"],"additionalProperties":false}},"required":["calculate_final_speed"],"additionalProperties":false}"# },
    BenchCase { id: "github_easy_o37087", cfa_name: "Github_easy---o37087", cfa_build_seconds: 0.021933, schema: r#"{"$schema":"http://json-schema.org/draft-07/schema#","$id":"https://mcda.drugis.org/emptyPerformance.json#","title":"MCDA empty performance for the performance table entry of absolute data","type":"object","required":["type"],"additionalProperties":false,"properties":{"type":{"type":"string","enum":["empty"]},"value":{"type":["string","number"]}}}"# },
    BenchCase { id: "bfcl_multiple_27", cfa_name: "BFCL_multiple_27", cfa_build_seconds: 0.025408, schema: r#"{"anyOf":[{"type":"object","properties":{"maps.route_times":{"type":"object","properties":{"route":{"type":"string"},"mode":{"type":"string"}},"required":["route"],"additionalProperties":false}},"required":["maps.route_times"],"additionalProperties":false},{"type":"object","properties":{"maps.shortest_path":{"type":"object","properties":{"start_location":{"type":"string"},"end_location":{"type":"string"},"mode":{"type":"string"}},"required":["start_location","end_location"],"additionalProperties":false}},"required":["maps.shortest_path"],"additionalProperties":false}]}"# },
    BenchCase { id: "github_trivial_o83308", cfa_name: "Github_trivial---o83308", cfa_build_seconds: 0.030312, schema: r#"{"definitions":{},"description":"API lets you interact with service","links":[{"href":"https://api.example.com","rel":"self"}],"properties":{},"title":"API","type":["object"]}"# },
    BenchCase { id: "bfcl_java_91", cfa_name: "BFCL_java_91", cfa_build_seconds: 0.034667, schema: r#"{"type":"object","properties":{"invokemethod007.runIt":{"type":"object","properties":{"args":{"type":"array","items":{"type":"string"}},"out":{}},"required":["args","out"],"additionalProperties":false}},"required":["invokemethod007.runIt"],"additionalProperties":false}"# },
    BenchCase { id: "glaive_calculate_area_ae21a3ae", cfa_name: "Glaiveai2K---calculate_area_ae21a3ae", cfa_build_seconds: 0.035502, schema: r#"{"properties":{"dimensions":{"description":"The dimensions required for the shape","properties":{"base":{"description":"The base of the triangle","type":"number"},"height":{"description":"The height of the triangle","type":"number"},"radius":{"description":"The radius for circle","type":"number"},"side_length":{"description":"The length of a side for square","type":"number"}},"required":["side_length","radius","base","height"],"type":"object"},"shape":{"description":"The type of shape (e.g. square, circle, triangle)","type":"string"}},"required":["shape","dimensions"],"type":"object"}"# },
    BenchCase { id: "github_easy_o44204", cfa_name: "Github_easy---o44204", cfa_build_seconds: 0.036089, schema: r#"{"$schema":"http://json-schema.org/draft-04/schema#","type":"object","properties":{"timestamp":{"description":"the number of seconds since the Unix epoch","type":"string","minLength":10,"maxLength":10,"pattern":"[0-9]{10,10}"},"agent":{"description":"a free-form string that identifies the build and test runner","type":"string"},"status":{"description":"the final status of a build or test","type":"string","enum":["success","failure"]},"url":{"type":"string"},"v":{"type":"integer","enum":[0]}},"required":["timestamp","agent"]}"# },
    BenchCase { id: "glaive_search_product_02ca757b", cfa_name: "Glaiveai2K---search_product_02ca757b", cfa_build_seconds: 0.036484, schema: r#"{"properties":{"category":{"description":"The category of the product","type":"string"},"price_range":{"properties":{"max_price":{"description":"The maximum price of the product","type":"number"},"min_price":{"description":"The minimum price of the product","type":"number"}},"type":"object"},"product_name":{"description":"The name of the product to search for","type":"string"}},"required":["product_name"],"type":"object"}"# },
    BenchCase { id: "glaive_create_calendar_event_1cc9b5e0", cfa_name: "Glaiveai2K---create_calendar_event_1cc9b5e0", cfa_build_seconds: 0.036948, schema: r#"{"properties":{"end_time":{"description":"The end time of the event","format":"date-time","type":"string"},"event_title":{"description":"The title of the event","type":"string"},"location":{"description":"The location of the event","type":"string"},"start_time":{"description":"The start time of the event","format":"date-time","type":"string"}},"required":["event_title","start_time","end_time"],"type":"object"}"# },
    BenchCase { id: "glaive_calculate_area_ab215361", cfa_name: "Glaiveai2K---calculate_area_ab215361", cfa_build_seconds: 0.037569, schema: r#"{"properties":{"dimensions":{"properties":{"base":{"description":"The base of the triangle (if shape is triangle)","type":"number"},"height":{"description":"The height of the triangle (if shape is triangle)","type":"number"},"length":{"description":"The length of the rectangle (if shape is rectangle)","type":"number"},"radius":{"description":"The radius of the circle (if shape is circle)","type":"number"},"width":{"description":"The width of the rectangle (if shape is rectangle)","type":"number"}},"required":["radius","length","width","base","height"],"type":"object"},"shape":{"description":"The shape for which to calculate area (e.g. circle, rectangle, triangle)","type":"string"}},"required":["shape"],"type":"object"}"# },
    BenchCase { id: "glaive_calculate_area_0f50c849", cfa_name: "Glaiveai2K---calculate_area_0f50c849", cfa_build_seconds: 0.038283, schema: r#"{"properties":{"dimensions":{"properties":{"base":{"description":"The base of the shape","type":"number"},"height":{"description":"The height of the shape","type":"number"},"length":{"description":"The length of the shape","type":"number"},"radius":{"description":"The radius of the shape","type":"number"},"width":{"description":"The width of the shape","type":"number"}},"type":"object"},"shape":{"description":"The type of shape (e.g. rectangle, circle, triangle)","type":"string"}},"required":["shape","dimensions"],"type":"object"}"# },
    BenchCase { id: "glaive_generate_invoice_5e32c363", cfa_name: "Glaiveai2K---generate_invoice_5e32c363", cfa_build_seconds: 0.039207, schema: r#"{"properties":{"client_details":{"properties":{"email":{"description":"The email address of the client","type":"string"},"name":{"description":"The name of the client","type":"string"}},"required":["name","email"],"type":"object"},"items":{"items":{"properties":{"name":{"description":"The name of the item","type":"string"},"price":{"description":"The price of the item","type":"number"},"quantity":{"description":"The quantity of the item","type":"integer"}},"required":["name","quantity","price"],"type":"object"},"type":"array"}},"required":["client_details","items"],"type":"object"}"# },
    BenchCase { id: "github_medium_o13", cfa_name: "Github_medium---o13", cfa_build_seconds: 0.041893, schema: r#"{"items":{"properties":{"address":{"type":"string"},"addressNumber":{"type":"string"},"bikes":{"pattern":"^\\d{1,2}$","type":"string"},"id":{"pattern":"^\\d{1,3}$","type":"string"},"lat":{"pattern":"^\\d{1,3}\\.\\d{2,6}$","type":"string"},"lon":{"pattern":"^\\d{1,3}\\.\\d{2,6}$","type":"string"},"slots":{"pattern":"^\\d{1,2}$","type":"string"},"stationType":{"enum":["BIKE","ELECTRIC_BIKE"],"type":"string"},"status":{"enum":["OPN","CLS"],"type":"string"}},"required":["id","district","lon","lat","bikes","slots","zip","address","nearbyStations","status","name","stationType"],"type":"object"},"type":"array"}"# },
    BenchCase { id: "github_medium_o7381", cfa_name: "Github_medium---o7381", cfa_build_seconds: 0.046726, schema: r##"{"type":"object","additionalProperties":false,"patternProperties":{"^[0-9]+$":{"type":"object","properties":{"timestamp":{"default":"","type":"string"},"versions":{"default":[],"type":"array","items":{"$ref":"#/definitions/ForgeVersion"}},"mcversion":{"default":"","type":"string"}},"required":["mcversion","timestamp","versions"]}},"definitions":{"ForgeVersion":{"type":"object","properties":{"mcversion":{"description":"The minecraft version","type":"string"},"version":{"description":"The forge version (without minecraft version)","type":"string"},"date":{"type":"string"},"installer":{"$ref":"#/definitions/ForgeDownload"},"universal":{"$ref":"#/definitions/ForgeDownload"},"changelog":{"description":"The changelog info","$ref":"#/definitions/ForgeDownload"},"mdk":{"$ref":"#/definitions/ForgeDownload"},"source":{"$ref":"#/definitions/ForgeDownload"},"launcher":{"$ref":"#/definitions/ForgeDownload"},"type":{"description":"The type of the forge release. The `common` means the normal release.","enum":["buggy","common","latest","recommended"],"type":"string"}},"required":["date","installer","mcversion","type","universal","version"]},"ForgeDownload":{"type":"object","properties":{"md5":{"type":"string"},"sha1":{"type":"string"},"path":{"description":"The url path to concat with forge maven","type":"string"}},"required":["path","sha1"]}},"$schema":"http://json-schema.org/draft-07/schema#"}"## },
    BenchCase { id: "github_medium_o81602", cfa_name: "Github_medium---o81602", cfa_build_seconds: 0.065818, schema: r#"{"$schema":"http://json-schema.org/draft-04/schema#","properties":{"orders":{"additionalItems":false,"items":{"properties":{"customerName":{"type":"string"},"date":{"pattern":"^([0][1-9]|[1][0-2])-([0][0-9]|[1][0-9]|[2][0-9]|[3][0-1])-20\\d{2}$","type":"string"},"drink":{"properties":{"drinkType":{"enum":["Latte","Espresso","Cappuccino","Chai","Tea","Steamer","Hot Chocolate"],"type":"string"},"flavor":{"enum":["Carmel","Chocolate","Hazelnut","Vanilla","Peppermint","White Chocolate"],"type":"string"},"milk":{"enum":["Non-Fat","Whole","Breve","Soy","Almond"],"type":"string"},"size":{"enum":["Small","Medium","Large","Extra-Large","Bucket"],"type":"string"}},"type":"object"},"espressoConCard":{"pattern":"^[A-Fa-f0-9]{8}-([A-Fa-f0-9]{4}-){3}[A-Fa-f0-9]{12}$","type":"string"},"id":{"type":"integer"},"muffin":{"enum":["Blueberry","Double Berry Crumb","Carrot Cake","Chocolate Chip","Double Chocolate Chip","Cherry Cheesecake","Cinnamon Cheesecake","Chocolate Cheesecake","Banana Nut"],"type":"string"},"orderId":{"type":"integer"},"time":{"pattern":"^([0-2][0-3]|[0-1][0-9])(:[0-5][0-9]){2}$","type":"string"},"totalPrice":{"pattern":"^\\$[0-9]{1,3}.[0-9]{2}$","type":"string"}},"required":["id","orderId","customerName","drink","muffin","date","time","espressoConCard","totalPrice"],"type":"object"},"minItems":1,"type":"array"}},"required":["orders"],"type":"object"}"# },
    BenchCase { id: "kubernetes_kb_543_normalized", cfa_name: "Kubernetes---kb_543_Normalized", cfa_build_seconds: 0.092518, schema: include_str!("../data/kubernetes_kb_543_normalized.schema.json") },
    BenchCase { id: "github_easy_o53115", cfa_name: "Github_easy---o53115", cfa_build_seconds: 0.147971, schema: r#"{"$schema":"http://json-schema.org/draft-07/schema#","title":"Resource types","type":"object","properties":{"id":{"type":"integer","description":"Resource type ID","minimum":0},"abbrev":{"type":"string","description":"Resource type abbreviation","maxLength":10},"description":{"type":"string","description":"Resource type description","maxLength":50}},"required":["id","abbrev","description"]}"# },
    BenchCase { id: "github_easy_o9857", cfa_name: "Github_easy---o9857", cfa_build_seconds: 0.338005, schema: r#"{"$schema":"http://json-schema.org/draft-04/schema#","additionalProperties":false,"properties":{"email":{"_format":"email","maxLength":1024,"type":"string"},"token":{"minLength":1,"type":"string"}},"required":["email","token"],"type":"object"}"# },
    BenchCase { id: "github_ultra_o62058", cfa_name: "Github_ultra---o62058", cfa_build_seconds: 1.221666, schema: include_str!("../data/github_ultra_o62058.schema.json") },
];

pub fn assert_release_benchmark(bench: &str) {
    if cfg!(debug_assertions) {
        panic!("{bench} must be run in release/bench mode, e.g. `cargo bench --bench {bench}`");
    }
}

pub fn force_single_threaded_compile() {
    unsafe {
        std::env::set_var("GLRMASK_COMPILE_THREADS", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
}

pub fn load_llama3_vocab() -> Vocab {
    let path = std::env::var_os("GLRMASK_LLAMA3_VOCAB_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/vocab_cache/llama3_vocab.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read Llama 3 vocab from {}: {err}", path.display()));
    let id_to_hex: BTreeMap<u32, String> = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse Llama 3 vocab JSON from {}: {err}", path.display()));
    Vocab::new(
        id_to_hex.into_iter().map(|(token_id, hex)| {
            (token_id, hex_to_bytes(&hex).unwrap_or_else(|err| {
                panic!("invalid hex bytes for token {token_id} in {}: {err}", path.display())
            }))
        }).collect(),
        None,
    )
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err(format!("odd hex length {}", hex.len()));
    }
    (0..hex.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).map_err(|err| err.to_string()))
        .collect()
}

pub fn clear_compile_caches() {
    clear_weight_interners();
    clear_weight_op_caches();
}

pub fn build_schema(case: &BenchCase, vocab: &Vocab) {
    clear_compile_caches();
    let constraint = Constraint::from_json_schema(black_box(case.schema), black_box(vocab))
        .unwrap_or_else(|err| panic!("{} schema should compile: {err}", case.cfa_name));
    black_box(constraint);
}

pub fn selected_cases(group_name: &str) -> Vec<&'static BenchCase> {
    let filters = requested_case_filters();
    if filters.is_empty() {
        return CASES.iter().collect();
    }

    let selected: Vec<_> = CASES
        .iter()
        .filter(|case| filters.iter().any(|filter| case_matches_filter(group_name, case, filter)))
        .collect();
    if selected.is_empty() {
        let available = CASES.iter().map(|case| case.id).collect::<Vec<_>>().join(", ");
        panic!(
            "no CFA sweep cases matched filters {:?}; available case ids: {}",
            filters, available
        );
    }
    selected
}

fn requested_case_filters() -> Vec<String> {
    let mut filters = Vec::new();
    for env_name in ["GLRMASK_BENCH_CASE", "GLRMASK_BENCH_FILTER"] {
        if let Ok(raw) = std::env::var(env_name) {
            filters.extend(split_filters(&raw));
        }
    }
    filters.extend(criterion_positional_filters());
    filters.sort();
    filters.dedup();
    filters
}

fn split_filters(raw: &str) -> impl Iterator<Item = String> + '_ {
    raw.split(',')
        .map(str::trim)
        .filter(|filter| !filter.is_empty())
        .map(str::to_owned)
}

fn criterion_positional_filters() -> Vec<String> {
    let mut filters = Vec::new();
    let mut skip_next = false;
    for arg in std::env::args().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if option_takes_value(&arg) {
            skip_next = !arg.contains('=');
            continue;
        }
        if arg.starts_with('-') {
            continue;
        }
        filters.extend(split_cli_filter(&arg));
    }
    filters
}

fn split_cli_filter(arg: &str) -> Vec<String> {
    if let Some((key, value)) = arg.split_once('=') {
        if matches!(key, "CASE" | "case" | "FILTER" | "filter") {
            return split_filters(value).collect();
        }
    }
    vec![arg.to_owned()]
}

fn option_takes_value(arg: &str) -> bool {
    matches!(
        arg,
        "--baseline"
            | "--color"
            | "--confidence-level"
            | "--measurement-time"
            | "--nresamples"
            | "--output-format"
            | "--plotting-backend"
            | "--profile-time"
            | "--sample-size"
            | "--save-baseline"
            | "--significance-level"
            | "--warm-up-time"
    )
}

fn case_matches_filter(group_name: &str, case: &BenchCase, filter: &str) -> bool {
    let benchmark_id = format!("{}/{}", group_name, case.id);
    case.id.contains(filter) || case.cfa_name.contains(filter) || benchmark_id.contains(filter)
}

pub fn profile_single_builds(cases: &[&BenchCase], vocab: &Vocab) {
    unsafe {
        std::env::set_var("GLRMASK_PROFILE_COMPILE", "1");
        std::env::set_var("GLRMASK_PROFILE_COMPILE_SUMMARY", "1");
    }
    for case in cases {
        eprintln!(
            "[bench][cfa_sweep_schema_build] diagnostic_build=1 case={} cfa_build_seconds={:.6}",
            case.id, case.cfa_build_seconds
        );
        build_schema(case, vocab);
    }
    unsafe {
        std::env::remove_var("GLRMASK_PROFILE_COMPILE");
        std::env::remove_var("GLRMASK_PROFILE_COMPILE_SUMMARY");
    }
}

pub fn bench_cases(c: &mut Criterion, group_name: &str, cases: &[&BenchCase], vocab: &Vocab) {
    let mut group = c.benchmark_group(group_name);
    for case in cases {
        group.bench_with_input(BenchmarkId::from_parameter(case.id), case, |b, case| {
            b.iter(|| build_schema(case, vocab));
        });
    }
    group.finish();
}
