use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Attribute to time a function.
///
/// Wraps the function body in `crate::profiler::time!`, using the function's
/// name as the timer name.
#[proc_macro_attribute]
pub fn timed(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new_spanned(proc_macro2::TokenStream::from(attr), "timed attribute does not take arguments")
            .to_compile_error()
            .into();
    }

    let mut input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = input_fn.sig.ident.to_string();
    let block = &input_fn.block;

    // Note: we use `crate::profiler::time!` to ensure it finds the macro
    // at the root of the crate where this attribute is used.
    // This is consistent with how the `time!` macro itself is implemented,
    // which uses `$crate::profiler::...`.
    let new_body = quote! {
        {
            crate::profiler::time!(#fn_name, #block)
        }
    };

    input_fn.block = syn::parse2(new_body).expect("Failed to parse new function body");

    TokenStream::from(quote! { #input_fn })
}
