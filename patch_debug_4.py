import sys

with open("src/import/json_schema.rs") as f:
    content = f.read()

new_block_1 = """            if let Some(expr) = self.try_build_factored_ordered_object("""
new_block_1_rep = """
            std::fs::OpenOptions::new().append(true).create(true).open("/tmp/debug_glrmask.txt").unwrap().write_all(format!("Calling try_build_factored 1 for {} with len {}\\n", base_name, ordered.len()).as_bytes()).unwrap();
            if let Some(expr) = self.try_build_factored_ordered_object("""

new_block_2 = """                if let Some(expr) = self.try_build_factored_ordered_object("""
new_block_2_rep = """
                std::fs::OpenOptions::new().append(true).create(true).open("/tmp/debug_glrmask.txt").unwrap().write_all(format!("Calling try_build_factored 2 for {} with len {}\\n", base_name, ordered.len()).as_bytes()).unwrap();
                if let Some(expr) = self.try_build_factored_ordered_object("""

content = content.replace(new_block_1, new_block_1_rep)
content = content.replace(new_block_2, new_block_2_rep)

with open("src/import/json_schema.rs", "w") as f:
    f.write(content)
print("Patched calls")
