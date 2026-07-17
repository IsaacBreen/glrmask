use std::env;
use std::fs;

use glrmask_runtime::{RuntimeArtifact, Session};

fn first_allowed(mask: &[u32]) -> Option<u32> {
    mask.iter().enumerate().find_map(|(word_index, &word)| {
        (word != 0).then(|| word_index as u32 * 32 + word.trailing_zeros())
    })
}

fn main() {
    let path = env::args().nth(1).expect("usage: roundtrip <artifact.glrmaskc>");
    let bytes = fs::read(path).expect("read artifact");
    let artifact = RuntimeArtifact::from_bytes(bytes).expect("parse artifact");
    let mut session = Session::from_artifact(artifact).expect("load compiled constraint");

    for step in 0..12 {
        let mask = session.mask_words();
        let token = first_allowed(&mask).expect("at least one admissible token");
        println!("step={step} words={} token={token}", mask.len());
        session.commit_token(token).expect("commit mask member");
    }
    println!("roundtrip-ok is_finished={}", session.is_finished());
}
