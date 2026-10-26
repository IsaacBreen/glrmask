//! A generation loop: mask the model's scores, take the best allowed token,
//! commit it, and stop when nothing more can be committed.
//!
//! Random numbers stand in for the model.
use glrmask::{Expr, Grammar, TokenSet, Vocab};

fn main() -> glrmask::Result<()> {
    // One token per character, and an end-of-sequence token with no bytes.
    let chars = b"0123456789+*()";
    let eos = chars.len() as u32;
    let vocab = Vocab::new(chars.iter().enumerate().map(|(id, &c)| (id as u32, vec![c])).collect())
        .with_special_tokens([eos]);

    // Arithmetic, then the end token.
    let arithmetic = Grammar::from_glrm(r#"
        start sum
        t NUMBER ::= /[0-9]+/
        nt sum ::= product | sum "+" product
        nt product ::= atom | product "*" atom
        nt atom ::= NUMBER | "(" sum ")"
    "#)?;
    let grammar = Grammar::from_expr(Expr::sequence([Expr::from(arithmetic), Expr::from(TokenSet::from(eos))]))?;
    let constraint = grammar.compile(&vocab)?;

    let mut model = Model { seed: 7 };
    let mut state = constraint.start();
    let mut mask = vec![0; constraint.mask_len()];
    let mut text = Vec::new();
    while !state.is_terminated() {
        let mut scores = model.scores(eos as usize + 1);
        // This model grows keener to stop as the text gets longer.
        scores[eos as usize] += 0.1 * text.len() as f32;

        state.fill_mask(&mut mask);
        let allowed = |id: usize| (mask[id / 32] >> (id % 32)) & 1 == 1;
        let token = (0..scores.len())
            .filter(|&id| allowed(id))
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .expect("the grammar allows a token until it terminates");

        assert!(state.commit_token(token as u32));
        text.extend_from_slice(vocab.get(token as u32).unwrap_or_default());
    }
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
