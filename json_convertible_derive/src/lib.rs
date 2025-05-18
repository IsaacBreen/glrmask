extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(JSONConvertible)]
pub fn json_convertible_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let generics = &ast.generics;

    let (impl_generics, ty_generics, original_where_clause) = generics.split_for_impl();

    // Add `T: crate::json_serialization::JSONConvertible` bound for each type parameter T
    let mut new_where_clause = original_where_clause.cloned().unwrap_or_else(|| {
        // If no original where clause, create one
        syn::parse_quote!(where)
    });
    // Ensure the where clause doesn't end with a comma if it was empty before adding predicates
    if !new_where_clause.predicates.is_empty() && !new_where_clause.predicates.trailing_punct() {
         new_where_clause.predicates.push_punct(Default::default());
    }

    for param in generics.type_params() {
        let ident = &param.ident;
        new_where_clause
            .predicates
            .push(syn::parse_quote!(#ident: crate::json_serialization::JSONConvertible));
        // Add trailing comma for the next predicate if any
        new_where_clause.predicates.push_punct(Default::default());
    }
    // Remove trailing comma if it's the last thing
    if new_where_clause.predicates.trailing_punct() && !new_where_clause.predicates.is_empty() {
        let mut new_punctuated = syn::punctuated::Punctuated::new();
        let pairs = new_where_clause.predicates.clone().into_pairs().collect::<Vec<_>>();
        for (i, pair) in pairs.into_iter().enumerate() {
            new_punctuated.push_value(pair.into_value());
            // Add comma if not the last element
            if i < new_where_clause.predicates.len() -1 {
                 new_punctuated.push_punct(Default::default());
            }
        }
        new_where_clause.predicates = new_punctuated;
    }


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
                            .and_then(#field_ty::from_json)?;
                    }
                });

                let field_names = fields.named.iter().map(|f| {
                    f.ident.as_ref().unwrap()
                });

                quote! {
                    impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics #new_where_clause {
                        fn to_json(&self) -> crate::json_serialization::JSONNode {
                            let mut obj = std::collections::BTreeMap::new();
                            #(#to_json_fields)*
                            crate::json_serialization::JSONNode::Object(obj)
                        }

                        fn from_json(node: crate::json_serialization::JSONNode) -> Result<Self, String> {
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
                    impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics #new_where_clause {
                        fn to_json(&self) -> crate::json_serialization::JSONNode {
                            crate::json_serialization::JSONNode::Object(std::collections::BTreeMap::new())
                        }

                        fn from_json(node: crate::json_serialization::JSONNode) -> Result<Self, String> {
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