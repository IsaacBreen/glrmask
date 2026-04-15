import sys

with open("src/import/json_schema.rs") as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if line.strip() == "fn try_build_factored_ordered_object(":
        start_idx = i
        break

lines.insert(start_idx + 10, """        if ordered.len() > 80 {
            eprintln!("try_build_factored_ordered_object called with {} keys!", ordered.len());
        }
""")

with open("src/import/json_schema.rs", "w") as f:
    f.writelines(lines)
print("Patched")
