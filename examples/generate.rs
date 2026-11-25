//! A generation loop: mask the model's scores, take the best allowed token,
//! stop if EOS is chosen, otherwise commit it.
//!
//! Random numbers stand in for the model.
use glrmask::{Grammar, Vocab};

fn main() -> glrmask::Result<()> {
    // One token per character, plus a caller-managed EOS ID.
    let chars = b"0123456789+*()";
    let eos = chars.len() as u32;
    let vocab = Vocab::new(chars.iter().enumerate().map(|(id, &c)| (id as u32, vec![c])).collect::<Vec<_>>()).unwrap();

    // Arithmetic grammar.
    let arithmetic = Grammar::from_glrm(r#"
        start sum
        t NUMBER = /[0-9]+/
        nt sum = product | sum "+" product
        nt product = atom | product "*" atom
        nt atom = NUMBER | "(" sum ")"
    "#)?;
    let constraint = arithmetic.compile(&vocab)?;

    let mut model = Model { seed: 7 };
    let mut state = constraint.start();
    let mut mask = vec![0; constraint.mask_len().max(eos as usize / 32 + 1)];
    let mut text = Vec::new();
    loop {
        let mut scores = model.scores(eos as usize + 1);
        // This model grows keener to stop as the text gets longer.
        scores[eos as usize] += 0.1 * text.len() as f32;

        state.fill_mask(&mut mask);
        if state.is_accepting() {
            mask[eos as usize / 32] |= 1 << (eos % 32);
        }
        let allowed = |id: usize| (mask[id / 32] >> (id % 32)) & 1 == 1;
        let token = (0..scores.len())
            .filter(|&id| allowed(id))
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .expect("no allowed token");
        if token == eos as usize {
            break;
        }

        assert!(state.commit_token(token as u32));
        text.extend_from_slice(vocab.get(token as u32).unwrap_or_default());
    }
    assert!(state.is_accepting());
    println!("{}", String::from_utf8_lossy(&text));
    Ok(())
}

/// Stands in for a language model: a score for each token ID.
struct Model {
    seed: u64,
}

impl Model {
    fn scores(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (self.seed >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }
}
