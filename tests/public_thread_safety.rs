//! Downstream auto-trait regression: intentionally no recursion_limit override.
use glrmask::{Extern, Externs, Expr, Grammar, Module, TokenSet, Vocab};

fn require_send_sync<T: Send + Sync>() {}

#[test]
fn source_module_and_expression_are_send_sync_at_the_default_budget() {
    require_send_sync::<Grammar>();
    require_send_sync::<Module>();
    require_send_sync::<Expr>();
    require_send_sync::<Extern>();
    require_send_sync::<Externs>();
    fn require_error<T: Send + Sync + 'static>() {}
    require_error::<glrmask::Error>();
    // Constraint and ConstraintState are covered by owned_constraint_state.rs,
    // whose larger recursion limit accommodates their internal persistent maps.
    require_send_sync::<Vocab>();
    require_send_sync::<TokenSet>();
}

#[test]
fn scoped_threads_share_module_snapshots_without_mutating_the_source() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]).unwrap();
    let source = Grammar::from_glrm("extern grammar item start root nt root=item").unwrap();
    let mut module = Module::new(&source);
    module.precompile(&vocab).unwrap();
    std::thread::scope(|scope| {
        for (text, id) in [("x", 0), ("y", 1)] {
            let module = &module;
            let vocab = &vocab;
            scope.spawn(move || {
                let mut local = module.clone();
                local.bind("item", Grammar::from_ebnf(format!("start ::= {text:?}")).unwrap()).unwrap();
                let constraint = local.compile(vocab).unwrap();
                let mut state = constraint.start();
                assert!(state.commit_token(id));
                assert!(state.is_accepting());
            });
        }
    });
    assert_eq!(module.externs().unbound().len(), 1);
    assert_eq!(Module::new(&source).externs().unbound().len(), 1);
}

#[test]
fn externs_snapshot_outlives_grammar_and_records_binding_at_creation() {
    let (before, after) = {
        let grammar = Grammar::from_glrm("extern grammar item start root nt root=item").unwrap();
        let before = grammar.externs();
        let mut module = Module::new(&grammar);
        module.bind("item", Grammar::from_lark("start: \"x\"").unwrap()).unwrap();
        (before, module.externs())
    };
    assert_eq!(before.unbound().len(), 1);
    assert_eq!(after.unbound().len(), 0);
    assert_eq!(before.get("item").unwrap().kind(), after.get("item").unwrap().kind());
}
