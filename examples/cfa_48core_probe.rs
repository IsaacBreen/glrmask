use std::{collections::BTreeMap, fs, path::Path, time::Instant};

use glrmask::{Constraint, Vocab};
use glrmask::__private::{ConstraintExt as _, VocabExt as _};

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct CpuTimespec {
    tv_sec: i64,
    tv_nsec: i64,
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn clock_gettime(clockid: i32, tp: *mut CpuTimespec) -> i32;
}

#[cfg(target_os = "linux")]
fn process_cpu_ms() -> f64 {
    // Sniper's syscall emulation does not implement CLOCK_PROCESS_CPUTIME_ID.
    // CPU-work timing is a native-only metric; simulated runs use Sniper's own stats.
    if std::env::var_os("GLRMASK_SNIPER_ROI").is_some() {
        return 0.0;
    }
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;
    let mut ts = CpuTimespec { tv_sec: 0, tv_nsec: 0 };
    let rc = unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_PROCESS_CPUTIME_ID) failed");
    ts.tv_sec as f64 * 1000.0 + ts.tv_nsec as f64 / 1_000_000.0
}

#[cfg(not(target_os = "linux"))]
fn process_cpu_ms() -> f64 { 0.0 }

fn load_vocab(path: &Path) -> Vocab {
    let raw = fs::read_to_string(path).unwrap();
    let map: BTreeMap<u32, String> = serde_json::from_str(&raw).unwrap();
    Vocab::new(
        map.into_iter()
            .map(|(id, hex)| (id, hex_to_bytes(&hex)))
            .collect(),
    )
}


#[inline(always)]
fn sniper_magic(cmd: u64) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut rax = cmd;
        core::arch::asm!(
            "xchg bx, bx",
            inout("rax") rax,
            options(nostack, preserves_flags),
        );
        std::hint::black_box(rax);
    }
    #[cfg(not(target_arch = "x86_64"))]
    std::hint::black_box(cmd);
}

#[inline(always)]
fn sniper_roi_start() {
    if std::env::var_os("GLRMASK_SNIPER_ROI").is_some() {
        sniper_magic(1);
    }
}

#[inline(always)]
fn sniper_roi_end() {
    if std::env::var_os("GLRMASK_SNIPER_ROI").is_some() {
        sniper_magic(2);
    }
}

fn main() {
    let process_started = Instant::now();
    let schema_path = std::env::args().nth(1).expect("schema path");
    let vocab_path = std::env::args().nth(2).expect("vocab path");
    let label = std::env::args().nth(3).unwrap_or_else(|| "schema".to_string());

    // Match CFA's GlrMaskNativeAdapter defaults before any schema import.
    unsafe {
        std::env::set_var("GLRMASK_LLGUIDANCE_COMPAT", "1");
        std::env::set_var(
            "GLRMASK_ENABLE_PREDECESSOR_SENSITIVE_UNIT_REDUCE_STACK_SHIFTS",
            "1",
        );
    }

    let schema = fs::read_to_string(schema_path).unwrap();
    let vocab = load_vocab(Path::new(&vocab_path));

    // CFA's Python extension warms the dedicated TI certification pool at
    // module import. Do the same before the request ROI so pool construction
    // is not charged to schema build latency.
    Constraint::warm_ti_pool();

    // CFA explicitly treats these as vocabulary-only setup, outside per-schema
    // build latency. Populate exactly those caches before the measured request.
    let vocab_prepare_started = Instant::now();
    vocab.prepare_for_compile();
    eprintln!(
        "[cfa-probe] vocab_prepare_ms={:.3}",
        vocab_prepare_started.elapsed().as_secs_f64() * 1000.0
    );

    // Match the adapter's independent-build cache reset.
    Constraint::clear_weight_op_caches();
    Constraint::clear_stale_weights();

    // Warm only process/thread-pool first-use effects, not schema-dependent caches.
    let warm = Constraint::from_json_schema(r#"{"type":"null"}"#, &vocab).unwrap();
    std::hint::black_box(warm);
    Constraint::clear_weight_op_caches();
    Constraint::clear_stale_weights();

    eprintln!("[cfa-probe] begin {label} offset_ms={:.3} vocab={}", process_started.elapsed().as_secs_f64() * 1000.0, vocab.len());
    sniper_roi_start();
    let cpu_started_ms = process_cpu_ms();
    let started = Instant::now();
    let constraint = Constraint::from_json_schema(&schema, &vocab).expect("schema compile");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cpu_ms = process_cpu_ms() - cpu_started_ms;
    sniper_roi_end();
    eprintln!("[cfa-probe] end {label} wall_ms={elapsed_ms:.3} cpu_ms={cpu_ms:.3} avg_cpus={:.3}", cpu_ms / elapsed_ms.max(1e-9));
    eprintln!(
        "[cfa-probe] tokenizer_states={} parser_states={}",
        constraint.num_tokenizer_states(),
        constraint.num_parser_states(),
    );
    std::hint::black_box(constraint);
}
