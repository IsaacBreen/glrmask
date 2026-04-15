import sys
with open("src/import/json_schema.rs") as f:
    content = f.read()

content = "use std::io::Write;\n" + content
with open("src/import/json_schema.rs", "w") as f:
    f.write(content)
print("Patched imports")
