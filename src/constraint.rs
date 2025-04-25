+++ b/chatllm.py
@@ -360,7 +360,7 @@
     """
     output_string = ""
     modified_files: Set[str] = set()
-    created_files: List[str] = []  # Simple apply doesn't create
+    created_files: Set[str] = set()
     deleted_files: List[str] = []  # Simple apply doesn't delete
     has_error = False
 
@@ -426,6 +426,7 @@
     attached_files = state.local_system.attached_files
     available_files = set(attached_files)
     explicitly_handled_files = set()
+    files_to_create = set() # Track files identified for creation
 
     output_string += f"Found {len(unique_matches)} code blocks.\n"
     output_string += "Matching against available files:\n"
@@ -442,8 +443,14 @@
                 available_files.remove(filepath)
                 explicitly_handled_files.add(filepath)
                 output_string += f"Assigned block to existing file {filepath} based on explicit fence path.\n"
+            elif Path(filepath).is_dir():
+                 output_string += f"Warning: Explicit file path '{filepath}' in fence points to an existing directory. Skipping block.\n"
+                 block_info['assigned_file'] = 'SKIPPED_EXPLICIT_MISMATCH' # Mark as skipped
             else:
-                output_string += f"Warning: Explicit file path '{filepath}' in fence not in attached files ({list(available_files)}). Skipping block.\n"
+                # File doesn't exist and isn't in attached files - mark for creation
+                block_info['assigned_file'] = filepath
+                files_to_create.add(filepath)
+                output_string += f"Identified block for new file creation: {filepath} based on explicit fence path.\n"
 
     unassigned_blocks = [b for b in block_info_list if b['assigned_file'] is None]
     output_string += f"Remaining unassigned blocks: {len(unassigned_blocks)}, Available files: {len(available_files)}\n"
@@ -568,10 +575,12 @@
     applied_count = 0
     for block_info in block_info_list:
         filepath = block_info['assigned_file']
-        # Check if assigned and not explicitly skipped
-        if filepath and filepath != 'SKIPPED_EXPLICIT_MISMATCH':
+        is_creation = filepath in files_to_create
+
+        # Check if assigned and not explicitly skipped due to directory conflict etc.
+        if filepath and filepath not in ['SKIPPED_EXPLICIT_MISMATCH']:
             code = block_info['code']
-            try:
+            try: # --- Apply change / Create file ---
                 # Ensure parent directory exists
                 directory = os.path.dirname(filepath)
                 if directory and not os.path.exists(directory):
@@ -586,11 +595,21 @@
                 with open(filepath, "w") as f:
                     f.write(code)
 
-                # Avoid duplicate success messages if already logged during assignment
-                log_identifier = f"to {filepath}"
-                if log_identifier not in output_string[-200:]: # Check recent logs
-                     output_string += f"Successfully updated file: {filepath}\n"
-
+                if is_creation:
+                    output_string += f"Successfully created new file: {filepath}\n"
+                    created_files.add(filepath)
+                    # Add to state tracking
+                    if filepath not in state.local_system.attached_files:
+                        state.local_system.attached_files.append(filepath)
+                    state.local_system.initial_file_contents[filepath] = code # Store initial content
+                    state.local_system.prev_file_contents[filepath] = code # Also set prev content
+                    output_string += f"Added newly created file {filepath} to attached files state.\n"
+                else:
+                    # Avoid duplicate success messages if already logged during assignment
+                    log_identifier = f"to {filepath}"
+                    if log_identifier not in output_string[-200:]: # Check recent logs
+                         output_string += f"Successfully updated existing file: {filepath}\n"
+                # Mark as modified regardless of creation/update for return value consistency? Or only if not created? Let's mark all applied.
                 modified_files.add(filepath)
                 applied_count += 1
             except Exception as e:
@@ -613,7 +632,7 @@
 
     # Return modified files (excluding explicitly handled ones if needed? No, return all modified)
     # Return empty lists for created/deleted as simple apply doesn't handle these.
-    return output_string, list(modified_files), created_files, deleted_files, has_error
+    return output_string, list(modified_files), list(created_files), deleted_files, has_error
 
 
 def apply_changes_with_git(state: State, user_message_content: Optional[str] = None) -> Tuple[str, list[str], list[str], list[str], bool]:

