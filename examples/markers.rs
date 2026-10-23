//! Report numeric grammar-boundary observations during token generation.
use glrmask::{Grammar, MarkerEvent, Result, Vocab};

fn main() -> Result<()> {
    let vocab = Vocab::new(vec![(0, b"helloworld".to_vec())]);
    let constraint = Grammar::from_glrm(
        r#"glrm 1; start root; nt root = @mark(10) "hello" @mark(20) "world" @mark(30);"#,
    ).compile(&vocab)?;
    let mut state = constraint.start();
    assert_eq!(state.marker_events(), &[MarkerEvent { id: 10, byte_offset: 0 }]);
    println!("initial: {:?}", state.marker_events());
    assert!(state.commit_token(0));
    assert_eq!(state.marker_events(), &[
        MarkerEvent { id: 20, byte_offset: 5 },
        MarkerEvent { id: 30, byte_offset: 10 },
    ]);
    println!("after token: {:?}", state.marker_events());
    assert!(state.is_accepting());
    Ok(())
}
