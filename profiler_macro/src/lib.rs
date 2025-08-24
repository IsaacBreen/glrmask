use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream, Parser},
    parse_macro_input, Block, Expr, ItemFn, LitStr, Stmt, Token,
};

#[proc_macro_attribute]
pub fn time_it(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    let timer_name_expr = if attr.is_empty() {
        let fn_name = item_fn.sig.ident.to_string();
        quote! { format!("{}::{}", module_path!(), #fn_name) }
    } else {
        let attr_str = parse_macro_input!(attr as LitStr);
        let name = attr_str.value();
        quote! { String::from(#name) }
    };

    let fn_vis = &item_fn.vis;
    let fn_sig = &item_fn.sig;
    let fn_block = &item_fn.block;
    let fn_attrs = &item_fn.attrs;

    let result = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            let __profiler_name = #timer_name_expr;
            let _guard = crate::profiler::TimedBlockGuard::new(__profiler_name);
            #fn_block
        }
    };

    result.into()
}

enum TimeitArgs {
    Named { name: Expr, expr: Expr },
    Unnamed { expr: Expr },
}

impl Parse for TimeitArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let first_expr = input.parse::<Expr>()?;
        if input.peek(Token![,]) {
            let _: Token![,] = input.parse()?;
            let expr = input.parse::<Expr>()?;
            Ok(TimeitArgs::Named {
                name: first_expr,
                expr,
            })
        } else {
            Ok(TimeitArgs::Unnamed { expr: first_expr })
        }
    }
}

#[proc_macro]
pub fn timeit(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as TimeitArgs);

    let tokens: TokenStream2 = match args {
        TimeitArgs::Named { name, expr } => emit_timeit_for_expr(quote! { (#name).into() }, expr),
        TimeitArgs::Unnamed { expr } => {
            // Default the name to a stringified form of the expression tokens.
            let expr_str = quote! { #expr }.to_string();
            emit_timeit_for_expr(quote! { String::from(#expr_str) }, expr)
        }
    };

    tokens.into()
}

// Generate code for the `timeit!` macro depending on whether the input is a block.
// - If it's a block, "unwrap" the block and inject its statements directly into the
//   surrounding scope so that bindings are visible afterwards. The guard is explicitly
//   dropped right after the user's statements to limit timing to that region.
// - Otherwise, keep expression semantics and return the value of the expression.
fn emit_timeit_for_expr(name_code: TokenStream2, expr: Expr) -> TokenStream2 {
    match expr {
        Expr::Block(expr_block) => {
            let (stmts, trailing_expr) = split_block_stmts(&expr_block.block);

            quote! {
                let __profiler_name = #name_code;
                let __timeit_guard = crate::profiler::TimedBlockGuard::new(__profiler_name);
                #(#stmts)*
                #( #trailing_expr ; )*
                ::core::mem::drop(__timeit_guard);
            }
        }
        other_expr => {
            // Keep expression semantics; ensure the guard is dropped right after evaluating
            // the expression (before yielding the value) to avoid timing beyond the expr.
            quote! {{
                let __profiler_name = #name_code;
                let __timeit_guard = crate::profiler::TimedBlockGuard::new(__profiler_name);
                let __timeit_value = { #other_expr };
                ::core::mem::drop(__timeit_guard);
                __timeit_value
            }}
        }
    }
}

// Helper: split a syn::Block into (all_stmts_except_trailing_expr, optional_trailing_expr_without_semicolon)
fn split_block_stmts(block: &Block) -> (Vec<Stmt>, Option<Expr>) {
    let mut stmts = block.stmts.clone();
    // If the last statement is a bare expression (without semicolon), we want to
    // re-emit it with a semicolon so it becomes a statement in the outer scope.
    let trailing_expr = match stmts.last() {
        Some(Stmt::Expr(expr)) => Some(expr.clone()),
        _ => None,
    };
    if trailing_expr.is_some() {
        stmts.pop();
    }
    (stmts, trailing_expr)
}

/* ----------------------------- timeitblock! -----------------------------
Goal:
- Provide a macro to time a block of code without creating an extra scope, so
  that bindings defined inside are visible afterwards.
- Accept:
    timeitblock!("some description", { ... });
    timeitblock!({ ... });

Note on syntax:
Rust macro invocations cannot use two delimiter groups in a single call,
so forms like `timeitblock!("desc") { ... }` are not valid as a single macro
invocation. The supported forms here are the closest possible on stable Rust:
- With an explicit description: timeitblock!("desc", { ... });
- Without a description:      timeitblock!({ ... });
- Or using braces as the only delimiter (same as above): timeitblock! { ... }
-------------------------------------------------------------------------- */

#[proc_macro]
pub fn timeitblock(input: TokenStream) -> TokenStream {
    // Try to parse as: <name_expr> , <block>
    // Example: timeitblock!("desc", { ... })
    let try_named_block = (|input: ParseStream| -> syn::Result<(Expr, Block)> {
        let name_expr = input.parse::<Expr>()?;
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
        let block = input.parse::<Block>()?;
        if !input.is_empty() {
            // Consume trailing commas or whitespace-only; otherwise error.
            return Err(input.error("unexpected tokens after block"));
        }
        Ok((name_expr, block))
    })
    .parse(input.clone());

    let tokens: TokenStream2 = match try_named_block {
        Ok((name_expr, block)) => {
            let (stmts, trailing_expr) = split_block_stmts(&block);
            let name_code = quote! { (#name_expr).into() };
            quote! {
                let __profiler_name = #name_code;
                let __timeit_guard = crate::profiler::TimedBlockGuard::new(__profiler_name);
                #(#stmts)*
                #( #trailing_expr ; )*
                ::core::mem::drop(__timeit_guard);
            }
        }
        Err(_) => {
            // Fallback: parse the input as a sequence of statements (no explicit name).
            // This covers calls like:
            //   timeitblock! { ... }
            //   timeitblock!({ ... })   // also works
            let stmts_parser = |input: ParseStream| -> syn::Result<Vec<Stmt>> {
                let mut stmts = Vec::new();
                while !input.is_empty() {
                    stmts.push(input.parse()?);
                }
                Ok(stmts)
            };
            let stmts: Vec<Stmt> = match Parser::parse(stmts_parser, input.clone()) {
                Ok(s) => s,
                Err(e) => return e.to_compile_error().into(),
            };

            // If the last statement is a bare expression (no ';'), re-emit with ';'.
            let (stmts_no_tail, trailing_expr) = match stmts.last() {
                Some(Stmt::Expr(expr)) => {
                    let mut v = stmts.clone();
                    v.pop();
                    (v, Some(expr.clone()))
                }
                _ => (stmts.clone(), None),
            };

            // Reasonable default name for unnamed block: module path + line number.
            // Using line!() ensures multiple calls have distinct names.
            let default_name = quote! { format!("{}:{}", module_path!(), line!()) };

            quote! {
                let __profiler_name = #default_name;
                let __timeit_guard = crate::profiler::TimedBlockGuard::new(__profiler_name);
                #(#stmts_no_tail)*
                #( #trailing_expr ; )*
                ::core::mem::drop(__timeit_guard);
            }
        }
    };

    tokens.into()
}
