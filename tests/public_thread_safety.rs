//! Downstream auto-trait regression: intentionally no recursion_limit override.
use glrmask::{Expr, Grammar, Module, Vocab};

fn require_send_sync<T: Send + Sync>() {}

#[test]
fn source_module_and_expression_are_send_sync_at_the_default_budget() {
    require_send_sync::<Grammar>();
    require_send_sync::<Module>();
    require_send_sync::<Expr>();
}

#[test]
fn scoped_threads_share_module_snapshots_without_mutating_the_source() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]);
    let source = Grammar::from_glrm("extern grammar item start root nt root=item");
    let mut module = Module::new(&source);
    module.precompile(&vocab).unwrap();
    std::thread::scope(|scope| {
        for (text, id) in [("x", 0), ("y", 1)] {
            let module = &module;
            let vocab = &vocab;
            scope.spawn(move || {
                let mut local = module.clone();
                local.bind("item", Grammar::from_ebnf(format!("start ::= {text:?}"))).unwrap();
                let constraint = local.compile(vocab).unwrap();
                let mut state = constraint.start();
                state.commit_token(id).unwrap();
                assert!(state.is_accepting());
            });
        }
    });
    assert_eq!(module.externs().unbound().len(), 1);
    assert_eq!(Module::new(&source).externs().unbound().len(), 1);
}
