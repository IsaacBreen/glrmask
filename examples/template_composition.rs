//! Public precompiled template composition; see docs/template-parser-composition.md.
use glrmask::test_modes::MaskMode;
fn main() -> glrmask::Result<()> {
    use glrmask::{BuildOptions, Constraint, Grammar, MaskStrategy, ParserBackend, Vocab};
    
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"y".to_vec()),
        (3, b"xay".to_vec()), (4, b"xy".to_vec()),
    ]);
    let child = Grammar::from_ebnf(r#"start ::= "a"?"#)?.compile_with_build_options(
        &vocab,
        BuildOptions::default()
            .optimization(MaskStrategy::ParserDwa)
            .parser_backend(ParserBackend::TemplateDfa),
    )?;
    let parent = Grammar::from_glrm(
        r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#,
    )?.prepare(&vocab)?;
    
    for optimization in [MaskMode::C, MaskMode::ParserDwa] {
        let linked = parent.bind_snapshot("child", &child)?.link_with(
            optimization.build_options()
                .parser_backend(ParserBackend::TemplateDfa),
        )?;
        assert_eq!(linked.parser_backend(), ParserBackend::TemplateDfa);
        let mut state = linked.start();
        assert!(state.commit_token(4)); // The child body can be empty.
        assert!(state.is_accepting());
    
        let loaded = Constraint::load(linked.save(), &vocab)?;
        let mut state = loaded.start();
        assert!(state.commit_token(3));
        assert!(state.is_accepting());
    }
    Ok(())
}
