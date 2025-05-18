extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(JSONConvertible)]
pub fn json_convertible_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let mut new_generics = ast.generics.clone(); // Clone original generics to modify them
    let where_clause = new_generics.make_where_clause(); // Get or create a mutable WhereClause

    // Add `T: crate::json_serialization::JSONConvertible` bound for each type parameter T
    // from the original generics.
    let original_type_params = ast.generics.type_params().collect::<Vec<_>>();
    if !original_type_params.is_empty() {
        // If there are new bounds to add, and if the existing predicates
        // (from original generics, now in `where_clause.predicates`)
        // are not empty and don't already end with a comma, add one.
        if !where_clause.predicates.is_empty() && !where_clause.predicates.trailing_punct() {
            where_clause.predicates.push_punct(syn::token::Comma::default());
        }

        for (i, param) in original_type_params.iter().enumerate() {
            let ident = &param.ident;
            where_clause.predicates.push_value(syn::parse_quote!(#ident: crate::json_serialization::JSONConvertible));
            // Add a comma if this is not the last new predicate being added.
            if i < original_type_params.len() - 1 {
                where_clause.predicates.push_punct(syn::token::Comma::default());
            }
        }
    }

    // Get the components for the impl block from the modified generics.
    // The `impl_generics` will now include the fully formed where clause.
    let (impl_generics, ty_generics, _ /* where_clause is now part of impl_generics */) = new_generics.split_for_impl();


    let gen = match &ast.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields) => {
                let to_json_fields = fields.named.iter().map(|f| {
                    let field_name = f.ident.as_ref().unwrap();
                    let field_name_str = field_name.to_string();
                    quote! {
                        obj.insert(#field_name_str.to_string(), self.#field_name.to_json());
                    }
                });

                let from_json_fields = fields.named.iter().map(|f| {
                    let field_name = f.ident.as_ref().unwrap();
                    let field_name_str = field_name.to_string();
                    let field_ty = &f.ty;
                    quote! {
                        let #field_name = obj.remove(#field_name_str)
                            .ok_or_else(|| format!("Missing field '{}' for struct {}", #field_name_str, stringify!(#name)))
                            .and_then(|json_value| #field_ty::from_json(json_value, cache))?;
                    }
                });

                let field_names = fields.named.iter().map(|f| {
                    f.ident.as_ref().unwrap()
                });

                quote! {
                    impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics {
                        fn to_json(&self) -> crate::json_serialization::JSONNode {
                            let mut obj = std::collections::BTreeMap::new();
                            #(#to_json_fields)*
                            crate::json_serialization::JSONNode::Object(obj)
                        }

                        fn from_json(node: crate::json_serialization::JSONNode, cache: &mut crate::json_serialization::DeserializationCache) -> Result<Self, String> {
                            match node {
                                crate::json_serialization::JSONNode::Object(mut obj) => {
                                    #(#from_json_fields)*
                                    // Note: Extra fields in the JSON object are currently ignored.
                                    // To make this an error, you could check if obj is empty here:
                                    // if !obj.is_empty() {
                                    //     return Err(format!("Unexpected fields in JSON object for struct {}: {:?}", stringify!(#name), obj.keys().collect::<Vec<_>>()));
                                    // }
                                    Ok(Self {
                                        #(#field_names),*
                                    })
                                }
                                _ => Err(format!("Expected JSONNode::Object for struct {}", stringify!(#name))),
                            }
                        }
                    }
                }
            }
            Fields::Unnamed(_) => {
                syn::Error::new_spanned(
                    &data_struct.fields,
                    "JSONConvertible derive does not support tuple structs yet.",
                )
                .to_compile_error()
            }
            Fields::Unit => {
                // Unit structs serialize to an empty JSON object
                quote! {
                    impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics {
                        fn to_json(&self) -> crate::json_serialization::JSONNode {
                            crate::json_serialization::JSONNode::Object(std::collections::BTreeMap::new())
                        }

                        fn from_json(node: crate::json_serialization::JSONNode, _cache: &mut crate::json_serialization::DeserializationCache) -> Result<Self, String> {
                            match node {
                                crate::json_serialization::JSONNode::Object(obj) if obj.is_empty() => Ok(Self),
                                crate::json_serialization::JSONNode::Object(_) => Err(format!("Expected empty JSONNode::Object for unit struct {}, got one with fields.", stringify!(#name))),
                                _ => Err(format!("Expected JSONNode::Object for unit struct {}", stringify!(#name))),
                            }
                        }
                    }
                }
            }
        },
        Data::Enum(data_enum) => {
            syn::Error::new_spanned(
                data_enum.enum_token,
                "JSONConvertible derive does not support enums yet.",
            )
            .to_compile_error()
        }
        Data::Union(data_union) => {
            syn::Error::new_spanned(
                data_union.union_token,
                "JSONConvertible derive does not support unions.",
            )
            .to_compile_error()
        }
    };

    gen.into()
}

