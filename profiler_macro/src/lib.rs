use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Expr, ItemFn, LitStr, Token,
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
            let _guard = crate::profiler::TimedBlockGuard::new(#timer_name_expr);
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

    let (name_code, expr) = match args {
        TimeitArgs::Named { name, expr } => (quote! { (#name).into() }, expr),
        TimeitArgs::Unnamed { expr } => {
            let expr_str = quote! { #expr }.to_string();
            (quote! { String::from(#expr_str) }, expr)
        }
    };

    let result = quote! {{
        let _guard = crate::profiler::TimedBlockGuard::new(#name_code);
        #expr
    }};

    result.into()
}
