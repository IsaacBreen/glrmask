use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitStr};

#[proc_macro_attribute]
pub fn time_it(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);
    let attr_str = parse_macro_input!(attr as LitStr);

    let fn_vis = &item_fn.vis;
    let fn_sig = &item_fn.sig;
    let fn_block = &item_fn.block;
    let fn_attrs = &item_fn.attrs;

    let timer_name = attr_str.value();

    let result = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            let _guard = crate::profiler::TimedBlockGuard::new(String::from(#timer_name));
            #fn_block
        }
    };

    result.into()
}
