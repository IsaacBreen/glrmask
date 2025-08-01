--- a/src/glr/analyze.rs
+++ b/src/glr/analyze.rs
@@ -197,16 +197,14 @@
             if let Symbol::NonTerminal(nt) = symbol {
                 if nt == lhs { // Found potential left recursion: A ::= ... A ...
                     // Check if all preceding symbols (if any) are nullable non-terminals
                     let prefix = &rhs[0..i];
-                    if !prefix.is_empty() { // Only check if there's a prefix
-                        let prefix_is_nullable = prefix.iter().all(|sym| match sym {
-                            Symbol::NonTerminal(prefix_nt) => nullable_nonterminals.contains(prefix_nt),
-                            Symbol::Terminal(_) => false, // Terminals are not nullable
-                        });
-
-                        if prefix_is_nullable {
-                            errors.push(format!("Left-nullable left recursion detected in rule '{} ::= {:?}'. The prefix '{:?}' before the recursive non-terminal '{}' is nullable.", lhs.0, rhs, prefix, lhs.0));
-                        }
+                    let prefix_is_nullable = prefix.iter().all(|sym| match sym {
+                        Symbol::NonTerminal(prefix_nt) => nullable_nonterminals.contains(prefix_nt),
+                        Symbol::Terminal(_) => false, // Terminals are not nullable
+                    });
+
+                    if prefix_is_nullable {
+                        errors.push(format!("Left-nullable left recursion detected in rule '{} ::= {:?}'. The prefix '{:?}' before the recursive non-terminal '{}' is nullable.", lhs.0, rhs, prefix, lhs.0));
                     }
                     // We only care about the first instance of recursion in a rule.
                     break;
@@ -900,53 +898,47 @@
 }
 
 pub fn inline_null_productions(productions: &[Production]) -> Vec<Production> {
     let nullable_nonterminals = compute_nullable_nonterminals(productions);
-    let null_nonterminals: BTreeSet<NonTerminal> = compute_null_nonterminals(productions);
-
-    let mut final_productions = Vec::new();
-
-    for original_prod in productions {
-        // For each original production, we generate all possible new productions
-        // by removing any combination of nullable non-terminals from its RHS.
-
-        // A worklist of RHS variants to process.
-        let mut worklist: VecDeque<Vec<Symbol>> = VecDeque::new();
-        let mut generated_rhss: Vec<Vec<Symbol>> = Vec::new();
-
-        // Start with the original RHS.
-        worklist.push_back(original_prod.rhs.clone());
-
-        'worklist: while let Some(current_rhs) = worklist.pop_front() {
-            // Iterate over the symbols of the current RHS variant.
-            for i in 0..current_rhs.len() {
-                if let Symbol::NonTerminal(nt) = &current_rhs[i] {
-                    // If we find a nullable non-terminal...
-                    if nullable_nonterminals.contains(nt) {
-                        // ...create a new RHS variant with it removed.
-                        let mut new_rhs = current_rhs.clone();
-                        new_rhs.remove(i);
-
-                        generated_rhss.push(new_rhs.clone());
-                        worklist.push_back(new_rhs);
-                        if null_nonterminals.contains(nt) {
-                            continue 'worklist;
+
+    let mut new_productions = BTreeSet::new();
+
+    for prod in productions {
+        // Skip original epsilon productions, as they are the ones being inlined.
+        if prod.rhs.is_empty() {
+            continue;
+        }
+
+        // Use a worklist to generate all combinations of RHS by removing nullable non-terminals.
+        let mut worklist: VecDeque<Vec<Symbol>> = VecDeque::new();
+        let mut seen_rhss: BTreeSet<Vec<Symbol>> = BTreeSet::new();
+
+        worklist.push_back(prod.rhs.clone());
+        seen_rhss.insert(prod.rhs.clone());
+
+        let mut queue_idx = 0;
+        while queue_idx < worklist.len() {
+            let current_rhs = worklist[queue_idx].clone();
+            queue_idx += 1;
+
+            for i in 0..current_rhs.len() {
+                if let Symbol::NonTerminal(nt) = &current_rhs[i] {
+                    if nullable_nonterminals.contains(nt) {
+                        let mut new_rhs = current_rhs.clone();
+                        new_rhs.remove(i);
+                        if seen_rhss.insert(new_rhs.clone()) {
+                            worklist.push_back(new_rhs);
                         }
                     }
                 }
             }
-            generated_rhss.push(current_rhs.clone());
-        }
-
-        // Add all generated variants as new productions with the original LHS.
-        for rhs in generated_rhss {
-            final_productions.push(Production {
-                lhs: original_prod.lhs.clone(),
+        }
+
+        for rhs in seen_rhss {
+            new_productions.insert(Production {
+                lhs: prod.lhs.clone(),
                 rhs,
             });
         }
     }
 
-    // Finally, remove all productions that are now null (e.g., A -> ε),
-    // as they have been inlined.
-    final_productions.retain(|p| !(p.rhs.is_empty() && nullable_nonterminals.contains(&p.lhs)));
-    // Remove duplicates
-    final_productions.dedup();
-    final_productions
+    new_productions.into_iter().collect()
 }
 
 pub fn inline_unit_productions(productions: &[Production]) -> Vec<Production> {

