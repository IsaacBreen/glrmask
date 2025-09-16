import _sep1 as ffi
from .py_parser import GLRParser


def py_commit_bytes(state: ffi.GrammarConstraintState, llm_token_bytes: bytes):
    """
    A Python implementation of the commit_bytes logic, using the exposed Rust components.
    """
    if not llm_token_bytes:
        return

    # Get necessary components from the constraint state
    tokenizer = state.tokenizer()
    parser_table_data = state.parser().export_table()
    py_parser = GLRParser(parser_table_data)

    current_state_gss = state.get_state_gss()

    # 1. Reset LLM tokens on all GSS nodes
    for gss_node in current_state_gss.values():
        ffi.gss_reset_llm_tokens(gss_node)

    # 2. Pre-computation for pruning and mapping based on tokenizer execution
    state_map = {}
    terminals_map = {}
    for tokenizer_state_id in current_state_gss.keys():
        exec_result = tokenizer.execute_from_state(llm_token_bytes, tokenizer_state_id)
        if exec_result.end_state is not None:
            state_map[tokenizer_state_id] = exec_result.end_state

        terminals = ffi.Bitset.zeros()
        for match in exec_result.matches:
            terminals.insert(match.id)
        terminals_map[tokenizer_state_id] = terminals

    # 3. Prune and map GSS nodes based on pre-computed info
    for gss_node in current_state_gss.values():
        ffi.gss_prune_disallowed_terminals(gss_node, terminals_map)
        ffi.gss_map_allowed_terminals_tokenizer_states(gss_node, state_map)

    # 4. Main processing loop
    new_overall_state_gss = {}
    processing_queue = {0: current_state_gss}

    while processing_queue:
        offset, states_to_process = min(processing_queue.items())
        del processing_queue[offset]

        for tokenizer_s_id_at_offset, gss_at_offset in states_to_process.items():
            if offset >= len(llm_token_bytes):
                continue

            exec_result = tokenizer.execute_from_state(llm_token_bytes[offset:], tokenizer_s_id_at_offset)

            for match_info in exec_result.matches:
                # This is where the Python GLRParser.step would be called.
                # Since it's not implemented, we'll just pass the GSS through as a placeholder.
                try:
                    new_gss = py_parser.step(gss_at_offset, match_info.id)
                except NotImplementedError:
                    new_gss = gss_at_offset.clone_node()  # Placeholder behavior

                if new_gss.is_ok():
                    new_offset = offset + match_info.width
                    next_tokenizer_id_for_segment = tokenizer.initial_state_id()

                    q_target = new_overall_state_gss if new_offset == len(llm_token_bytes) else processing_queue.setdefault(new_offset, {})

                    if next_tokenizer_id_for_segment in q_target:
                        q_target[next_tokenizer_id_for_segment] = ffi.gss_merge_many_with_depth(
                            [q_target[next_tokenizer_id_for_segment], new_gss], 1)
                    else:
                        q_target[next_tokenizer_id_for_segment] = new_gss

            if exec_result.end_state is not None:
                final_tokenizer_state = exec_result.end_state
                if final_tokenizer_state in new_overall_state_gss:
                    new_overall_state_gss[final_tokenizer_state] = ffi.gss_merge_many_with_depth(
                        [new_overall_state_gss[final_tokenizer_state], gss_at_offset], 1)
                else:
                    new_overall_state_gss[final_tokenizer_state] = gss_at_offset

    # 5. Final cleanup
    final_state = {k: v for k, v in new_overall_state_gss.items() if v.is_ok()}

    for gss_node in final_state.values():
        ffi.gss_reset_llm_tokens(gss_node)
        ffi.gss_fuse_predecessors(gss_node, 1)

    # 6. Update the main state object
    state.set_state_gss(final_state)
