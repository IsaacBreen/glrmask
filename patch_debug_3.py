import sys

with open("src/import/json_schema.rs") as f:
    content = f.read()

new_block = """    fn build_object_tree(
        &mut self,
        base_name: &str,
        items: &[(String, GrammarExpr, bool)],
        next_rule_index: &mut usize,
        shape: OrderedObjectShape,
    ) -> Result<(GrammarExpr, bool), GlrMaskError> {
        eprintln!("build_object_tree called with size {}", items.len());
"""

import re
content = re.sub(r"    fn build_object_tree\([\s\S]*?\) \-\> Result\<\(GrammarExpr, bool\), GlrMaskError\> \{", new_block, content, count=1)

with open("src/import/json_schema.rs", "w") as f:
    f.write(content)
print("Patched build_object_tree")
