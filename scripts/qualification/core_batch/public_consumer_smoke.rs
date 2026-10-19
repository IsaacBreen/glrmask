use glrmask::{Grammar, Token, Vocab};

fn main() -> glrmask::Result<()> {
    let vocab = Vocab::new(vec![(0, b"!".to_vec()), (1, b"a".to_vec())]);
    let mut grammar = Grammar::from_glrm("extern token MARK\nstart root\nnt root = MARK");
    let original = grammar.clone();
    grammar.bind("MARK", Token::new(0))?;
    assert_eq!(original.requirements()?.len(), 1);
    assert!(grammar.requirements()?.is_empty());
    let constraint = grammar.compile(&vocab)?;
    let mut state = constraint.start();
    let forced = state.forced_tokens()?;
    assert_eq!(forced, vec![0]);
    let mut mask = vec![0; constraint.mask_len()];
    state.fill_mask(&mut mask);
    state.commit_token(0)?;
    assert!(state.is_accepting());
    let mut prepared = original.prepare(&vocab)?;
    prepared.bind("MARK", Token::new(0))?;
    let closed = prepared.link()?;
    assert_eq!(closed.start().forced_tokens()?, vec![0]);
    let regex = Grammar::from_glrm("start root\nt A = (/a|ab/ - (/a|ab/ b/./s+))\nnt root = A");
    let c = regex.compile(&vocab)?;
    let mut s = c.start();
    s.commit_bytes(b"a")?;
    assert!(s.is_accepting());
    Ok(())
}
