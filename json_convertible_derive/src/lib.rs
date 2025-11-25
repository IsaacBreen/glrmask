extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Meta};

/// Converts a PascalCase or camelCase identifier to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

/// Extract rename_all attribute value from #[json_convertible(...)]
fn get_rename_all(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("json_convertible") {
            if let Meta::List(meta_list) = &attr.meta {
                // Parse the tokens to find rename_all = "..."
                let tokens = meta_list.tokens.clone();
                let tokens_str = tokens.to_string();
                // Simple parsing for rename_all = "snake_case"
                if let Some(pos) = tokens_str.find("rename_all") {
                    let rest = &tokens_str[pos..];
                    if let Some(start) = rest.find('"') {
                        if let Some(end) = rest[start+1..].find('"') {
                            return Some(rest[start+1..start+1+end].to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

#[proc_macro_derive(JSONConvertible, attributes(json_convertible))]
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
            where_clause
                .predicates
                .push_punct(syn::token::Comma::default());
        }

        for (i, param) in original_type_params.iter().enumerate() {
            let ident = &param.ident;
            where_clause.predicates.push_value(
                syn::parse_quote!(#ident: crate::json_serialization::JSONConvertible),
            );
            // Add a comma if this is not the last new predicate being added.
            if i < original_type_params.len() - 1 {
                where_clause
                    .predicates
                    .push_punct(syn::token::Comma::default());
            }
        }
    }

    // Get the components for the impl block from the modified generics.
    // The `impl_generics` will now include the fully formed where clause.
    let (impl_generics, ty_generics, _ /* where_clause is now part of impl_generics */) =
        new_generics.split_for_impl();

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
                        let #field_name = obj
                            .remove(#field_name_str)
                            .ok_or_else(|| format!(
                                "Missing field '{}' for struct {}",
                                #field_name_str,
                                stringify!(#name)
                            ))
                            .and_then(|node| <#field_ty as crate::json_serialization::JSONConvertible>::from_json(node))?;
                    }
                });

                let field_names = fields.named.iter().map(|f| f.ident.as_ref().unwrap());

                quote! {
                    impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics {
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
            Fields::Unnamed(fields) => {
                // Tuple structs: serialize as array
                if fields.unnamed.len() == 1 {
                    // Newtype pattern: serialize as the inner type
                    let field_ty = &fields.unnamed[0].ty;
                    quote! {
                        impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics {
                            fn to_json(&self) -> crate::json_serialization::JSONNode {
                                self.0.to_json()
                            }

                            fn from_json(node: crate::json_serialization::JSONNode) -> Result<Self, String> {
                                <#field_ty as crate::json_serialization::JSONConvertible>::from_json(node)
                                    .map(Self)
                            }
                        }
                    }
                } else {
                    // Multi-field tuple struct: serialize as array
                    let field_count = fields.unnamed.len();
                    let to_json_elements = (0..field_count).map(|i| {
                        let idx = syn::Index::from(i);
                        quote! { self.#idx.to_json() }
                    });

                    let from_json_fields = fields.unnamed.iter().enumerate().map(|(i, f)| {
                        let field_ty = &f.ty;
                        quote! {
                            <#field_ty as crate::json_serialization::JSONConvertible>::from_json(
                                arr.get(#i).cloned().ok_or_else(|| format!(
                                    "Missing element {} for tuple struct {}",
                                    #i,
                                    stringify!(#name)
                                ))?
                            )?
                        }
                    });

                    quote! {
                        impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics {
                            fn to_json(&self) -> crate::json_serialization::JSONNode {
                                crate::json_serialization::JSONNode::Array(vec![#(#to_json_elements),*])
                            }

                            fn from_json(node: crate::json_serialization::JSONNode) -> Result<Self, String> {
                                match node {
                                    crate::json_serialization::JSONNode::Array(arr) => {
                                        if arr.len() != #field_count {
                                            return Err(format!(
                                                "Expected array of length {} for tuple struct {}, got {}",
                                                #field_count, stringify!(#name), arr.len()
                                            ));
                                        }
                                        Ok(Self(#(#from_json_fields),*))
                                    }
                                    _ => Err(format!("Expected JSONNode::Array for tuple struct {}", stringify!(#name))),
                                }
                            }
                        }
                    }
                }
            }
            Fields::Unit => {
                // Unit structs serialize to an empty JSON object
                quote! {
                    impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics {
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
            // Check for rename_all attribute
            let rename_all = get_rename_all(&ast.attrs);
            
            // Helper closure to transform variant names based on rename_all
            let transform_name = |name: &str| -> String {
                match rename_all.as_deref() {
                    Some("snake_case") => to_snake_case(name),
                    _ => name.to_string(),
                }
            };
            
            // Check if all variants are unit variants (no fields)
            let all_unit = data_enum.variants.iter().all(|v| matches!(v.fields, Fields::Unit));

            if all_unit {
                // Simple unit-only enum: serialize to string
                let variant_names: Vec<_> = data_enum.variants.iter().map(|v| &v.ident).collect();
                let variant_strs: Vec<String> = variant_names.iter().map(|v| transform_name(&v.to_string())).collect();

                let to_json_arms = variant_names.iter().zip(variant_strs.iter()).map(|(vname, vstr)| {
                    quote! {
                        Self::#vname => crate::json_serialization::JSONNode::String(#vstr.to_string()),
                    }
                });

                let from_json_arms = variant_names.iter().zip(variant_strs.iter()).map(|(vname, vstr)| {
                    quote! {
                        #vstr => Ok(Self::#vname),
                    }
                });

                quote! {
                    impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics {
                        fn to_json(&self) -> crate::json_serialization::JSONNode {
                            match self {
                                #(#to_json_arms)*
                            }
                        }

                        fn from_json(node: crate::json_serialization::JSONNode) -> Result<Self, String> {
                            match node {
                                crate::json_serialization::JSONNode::String(s) => match s.as_str() {
                                    #(#from_json_arms)*
                                    other => Err(format!("Unknown variant '{}' for enum {}", other, stringify!(#name))),
                                },
                                _ => Err(format!("Expected JSONNode::String for unit enum {}", stringify!(#name))),
                            }
                        }
                    }
                }
            } else {
                // Complex enum with data variants: serialize as tagged object
                // Format: { "variant": "VariantName", ...fields } for struct variants
                //         { "variant": "VariantName", "value": ... } for newtype variants
                //         { "variant": "VariantName", "values": [...] } for tuple variants
                //         { "variant": "VariantName" } for unit variants

                let to_json_arms = data_enum.variants.iter().map(|v| {
                    let vname = &v.ident;
                    let vstr = transform_name(&vname.to_string());
                    match &v.fields {
                        Fields::Unit => {
                            quote! {
                                Self::#vname => {
                                    let mut obj = std::collections::BTreeMap::new();
                                    obj.insert("variant".to_string(), crate::json_serialization::JSONNode::String(#vstr.to_string()));
                                    crate::json_serialization::JSONNode::Object(obj)
                                }
                            }
                        }
                        Fields::Unnamed(fields) => {
                            if fields.unnamed.len() == 1 {
                                // Newtype variant
                                quote! {
                                    Self::#vname(inner) => {
                                        let mut obj = std::collections::BTreeMap::new();
                                        obj.insert("variant".to_string(), crate::json_serialization::JSONNode::String(#vstr.to_string()));
                                        obj.insert("value".to_string(), inner.to_json());
                                        crate::json_serialization::JSONNode::Object(obj)
                                    }
                                }
                            } else {
                                // Tuple variant with multiple fields
                                let field_bindings: Vec<_> = (0..fields.unnamed.len())
                                    .map(|i| syn::Ident::new(&format!("f{}", i), proc_macro2::Span::call_site()))
                                    .collect();
                                let to_json_elements = field_bindings.iter().map(|fb| {
                                    quote! { #fb.to_json() }
                                });
                                quote! {
                                    Self::#vname(#(#field_bindings),*) => {
                                        let mut obj = std::collections::BTreeMap::new();
                                        obj.insert("variant".to_string(), crate::json_serialization::JSONNode::String(#vstr.to_string()));
                                        obj.insert("values".to_string(), crate::json_serialization::JSONNode::Array(vec![#(#to_json_elements),*]));
                                        crate::json_serialization::JSONNode::Object(obj)
                                    }
                                }
                            }
                        }
                        Fields::Named(fields) => {
                            let field_names: Vec<_> = fields.named.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                            let field_strs: Vec<_> = field_names.iter().map(|f| f.to_string()).collect();
                            let field_inserts = field_names.iter().zip(field_strs.iter()).map(|(fname, fstr)| {
                                quote! {
                                    obj.insert(#fstr.to_string(), #fname.to_json());
                                }
                            });
                            quote! {
                                Self::#vname { #(#field_names),* } => {
                                    let mut obj = std::collections::BTreeMap::new();
                                    obj.insert("variant".to_string(), crate::json_serialization::JSONNode::String(#vstr.to_string()));
                                    #(#field_inserts)*
                                    crate::json_serialization::JSONNode::Object(obj)
                                }
                            }
                        }
                    }
                });

                let from_json_arms = data_enum.variants.iter().map(|v| {
                    let vname = &v.ident;
                    let vstr = transform_name(&vname.to_string());
                    match &v.fields {
                        Fields::Unit => {
                            quote! {
                                #vstr => Ok(Self::#vname),
                            }
                        }
                        Fields::Unnamed(fields) => {
                            if fields.unnamed.len() == 1 {
                                let field_ty = &fields.unnamed[0].ty;
                                quote! {
                                    #vstr => {
                                        let value_node = obj.remove("value").ok_or_else(|| format!(
                                            "Missing 'value' field for variant {} of enum {}",
                                            #vstr, stringify!(#name)
                                        ))?;
                                        let inner = <#field_ty as crate::json_serialization::JSONConvertible>::from_json(value_node)?;
                                        Ok(Self::#vname(inner))
                                    }
                                }
                            } else {
                                let field_tys: Vec<_> = fields.unnamed.iter().map(|f| &f.ty).collect();
                                let field_count = field_tys.len();
                                let field_extractions = field_tys.iter().enumerate().map(|(i, fty)| {
                                    quote! {
                                        <#fty as crate::json_serialization::JSONConvertible>::from_json(
                                            arr.get(#i).cloned().ok_or_else(|| format!(
                                                "Missing element {} for variant {} of enum {}",
                                                #i, #vstr, stringify!(#name)
                                            ))?
                                        )?
                                    }
                                });
                                quote! {
                                    #vstr => {
                                        let values_node = obj.remove("values").ok_or_else(|| format!(
                                            "Missing 'values' field for variant {} of enum {}",
                                            #vstr, stringify!(#name)
                                        ))?;
                                        match values_node {
                                            crate::json_serialization::JSONNode::Array(arr) => {
                                                if arr.len() != #field_count {
                                                    return Err(format!(
                                                        "Expected {} values for variant {} of enum {}, got {}",
                                                        #field_count, #vstr, stringify!(#name), arr.len()
                                                    ));
                                                }
                                                Ok(Self::#vname(#(#field_extractions),*))
                                            }
                                            _ => Err(format!(
                                                "Expected JSONNode::Array for 'values' of variant {} of enum {}",
                                                #vstr, stringify!(#name)
                                            )),
                                        }
                                    }
                                }
                            }
                        }
                        Fields::Named(fields) => {
                            let field_names: Vec<_> = fields.named.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                            let field_strs: Vec<_> = field_names.iter().map(|f| f.to_string()).collect();
                            let field_tys: Vec<_> = fields.named.iter().map(|f| &f.ty).collect();
                            let field_extractions = field_names.iter().zip(field_strs.iter()).zip(field_tys.iter()).map(|((fname, fstr), fty)| {
                                quote! {
                                    let #fname = obj.remove(#fstr).ok_or_else(|| format!(
                                        "Missing field '{}' for variant {} of enum {}",
                                        #fstr, #vstr, stringify!(#name)
                                    )).and_then(|node| <#fty as crate::json_serialization::JSONConvertible>::from_json(node))?;
                                }
                            });
                            quote! {
                                #vstr => {
                                    #(#field_extractions)*
                                    Ok(Self::#vname { #(#field_names),* })
                                }
                            }
                        }
                    }
                });

                quote! {
                    impl #impl_generics crate::json_serialization::JSONConvertible for #name #ty_generics {
                        fn to_json(&self) -> crate::json_serialization::JSONNode {
                            match self {
                                #(#to_json_arms)*
                            }
                        }

                        fn from_json(node: crate::json_serialization::JSONNode) -> Result<Self, String> {
                            match node {
                                crate::json_serialization::JSONNode::Object(mut obj) => {
                                    let variant = obj.remove("variant").ok_or_else(|| format!(
                                        "Missing 'variant' field for enum {}", stringify!(#name)
                                    ))?;
                                    let variant_str = match variant {
                                        crate::json_serialization::JSONNode::String(s) => s,
                                        _ => return Err(format!(
                                            "Expected string for 'variant' field of enum {}", stringify!(#name)
                                        )),
                                    };
                                    match variant_str.as_str() {
                                        #(#from_json_arms)*
                                        other => Err(format!("Unknown variant '{}' for enum {}", other, stringify!(#name))),
                                    }
                                }
                                _ => Err(format!("Expected JSONNode::Object for enum {}", stringify!(#name))),
                            }
                        }
                    }
                }
            }
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
