import sys

with open("src/import/json_schema.rs") as f:
    content = f.read()

new_block = """        if ordered.len() > 80 {
            eprintln!("--- BUILDING BIG OBJECT ---");
            eprintln!("factored_ordered_object_enabled() = {}", factored_ordered_object_enabled());
            eprintln!("Self::factored_closed_object_enabled() = {}", Self::factored_closed_object_enabled());
            eprintln!("!Self::exact_closed_object_disabled() = {}", !Self::exact_closed_object_disabled());
            eprintln!("!has_additional_properties = {}", !has_additional_properties);
            eprintln!("ordered.iter().any... = {}", ordered.iter().any(|(_, _, required)| !*required));
            eprintln!("pattern_properties.is_empty() = {}", pattern_properties.is_empty());
            eprintln!("property_names.is_none() = {}", property_names.is_none());
            eprintln!("---------------------------");
        }
        if factored_ordered_object_enabled()"""

content = content.replace("        if factored_ordered_object_enabled()", new_block, 1)

with open("src/import/json_schema.rs", "w") as f:
    f.write(content)
print("Patched")
