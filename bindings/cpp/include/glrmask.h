#ifndef GLRMASK_H
#define GLRMASK_H

/* GLRMask C ABI, revision 1.
 * All lengths are element counts, except explicitly named byte lengths.
 * Text is UTF-8, length-delimited, and need not be NUL-terminated. Vocabulary
 * bytes are arbitrary bytes (including NUL and partial UTF-8 sequences).
 *
 * Every returned handle is owning; release it with its matching _free function.
 * Input handles and input memory are borrowed only for the duration of a call.
 * A state independently retains its compiled constraint. Clones have independent
 * mutable state/bindings. Views returned by *_data/get remain valid until the
 * corresponding result handle is freed. Never free a view yourself.
 *
 * C callers must supply live correctly-typed handles and valid, aligned,
 * non-overlapping memory; pointer/length checks cannot detect dangling pointers.
 * A NULL array is allowed only with length zero. Do not concurrently access a
 * mutable handle, or mutate a grammar while another thread reads it. Immutable
 * constraints may be shared across threads with separate states.
 *
 * Fallible functions return a status. Output handles are NULL on failure.
 * glrmask_error_message() is thread-local and valid until the next GLRMask call
 * on that thread. A PANIC poisons the affected state: discard it. Process aborts
 * (including out-of-memory) and invalid C memory cannot be recovered.
 */

#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32) && defined(GLRMASK_SHARED)
# if defined(GLRMASK_BUILDING)
#  define GLRMASK_API __declspec(dllexport)
# else
#  define GLRMASK_API __declspec(dllimport)
# endif
#else
# define GLRMASK_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

/// Function status. GLRMASK_OK means success.
typedef int32_t glrmask_status;
/// Status values. On failure an output handle is NULL.
enum {
    GLRMASK_OK=0,                 ///< Success.
    GLRMASK_INVALID_ARGUMENT=1,   ///< A handle, argument or memory range is invalid.
    GLRMASK_ENGINE_ERROR=2,       ///< Compilation, state or artifact failure.
    GLRMASK_PANIC=3,              ///< A Rust panic was contained; discard the affected state.
    GLRMASK_BUFFER_TOO_SMALL=4,   ///< The supplied buffer cannot hold the result.
    GLRMASK_CLOSED=5              ///< The handle was already freed or moved.
};
/// Mask computation strategies.
enum {
    GLRMASK_MASK_STRATEGY_AUTO=0,              ///< Automatic selection.
    GLRMASK_MASK_STRATEGY_PARSER_DWA=1,        ///< Static masks: slowest build, fastest per-token masks.
    GLRMASK_MASK_STRATEGY_TOKEN_TRIE=2,        ///< Trie walk per mask: fastest build.
    GLRMASK_MASK_STRATEGY_TOKEN_CLASS_TRIE=3   ///< Trie walk over token equivalence classes.
};
/// Alias for GLRMASK_MASK_STRATEGY_TOKEN_TRIE.
#define GLRMASK_MASK_STRATEGY_FAST_BUILD GLRMASK_MASK_STRATEGY_TOKEN_TRIE
/// Alias for GLRMASK_MASK_STRATEGY_PARSER_DWA.
#define GLRMASK_MASK_STRATEGY_FAST_RUNTIME GLRMASK_MASK_STRATEGY_PARSER_DWA
/// Boundary parser selection for crossings between composed grammars.
enum {
    GLRMASK_BOUNDARY_MASK_STRATEGY_DYNAMIC=0,    ///< Segmented dynamic boundary parser; faster build.
    GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA=1  ///< Static boundary parser; faster crossings.
};
/// Alias for GLRMASK_BOUNDARY_MASK_STRATEGY_DYNAMIC.
#define GLRMASK_BOUNDARY_MASK_STRATEGY_FAST_BUILD GLRMASK_BOUNDARY_MASK_STRATEGY_DYNAMIC
/// Alias for GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA.
#define GLRMASK_BOUNDARY_MASK_STRATEGY_FAST_RUNTIME GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA
/// Source formats accepted by glrmask_grammar_from_source.
enum {
    GLRMASK_EBNF=0,         ///< EBNF source.
    GLRMASK_GLRM=1,         ///< Native GLRM source.
    GLRMASK_JSON_SCHEMA=2,  ///< JSON Schema document.
    GLRMASK_LARK=3          ///< Lark source.
};
/// Lark dialects.
enum {
    GLRMASK_LARK_RECOGNITION=0,  ///< Standard Lark recognition.
    GLRMASK_LARK_LLM=1           ///< Lark plus GLRMask's LLM extensions.
};
/// Extern slot kinds.
enum {
    GLRMASK_EXTERN_TOKEN=0,    ///< A token-level slot.
    GLRMASK_EXTERN_GRAMMAR=1   ///< A grammar-level slot.
};

typedef struct glrmask_vocab glrmask_vocab;
typedef struct glrmask_exact_tokens glrmask_exact_tokens;
typedef struct glrmask_grammar glrmask_grammar;
typedef struct glrmask_module glrmask_module;
typedef struct glrmask_expr glrmask_expr;
typedef struct glrmask_constraint glrmask_constraint;
typedef struct glrmask_state glrmask_state;
typedef struct glrmask_bytes glrmask_bytes;
typedef struct glrmask_u32s glrmask_u32s;
typedef struct glrmask_externs glrmask_externs;
typedef struct glrmask_events glrmask_events;

typedef struct { uint32_t id; const uint8_t *bytes; size_t bytes_len; } glrmask_vocab_entry;
/* Inclusive endpoints; adjacent and overlapping ranges are normalized. */
typedef struct { uint32_t first; uint32_t last; } glrmask_token_range;
typedef struct {
    size_t struct_size;
    const uint32_t *end_tokens;
    size_t end_tokens_len;
} glrmask_compile_options;
typedef struct { const char *name; size_t name_len; const uint32_t *ids; size_t ids_len; } glrmask_lark_token;
typedef struct { size_t struct_size; uint32_t profile; const glrmask_lark_token *tokens; size_t tokens_len; } glrmask_lark_options;
typedef struct {
    const char *name; size_t name_len; uint32_t kind; uint32_t has_source_span;
    uint64_t start_byte, end_byte, line, column;
} glrmask_extern;
typedef struct { uint32_t id; uint64_t byte_offset; } glrmask_marker_event;
/* kind: 0 Grammar, 1 Module, 2 Expr, 3 Token, 4 ExactTokens */
typedef struct { uint32_t kind; const void* handle; } glrmask_expr_operand;

GLRMASK_API uint32_t glrmask_abi_version(void);
GLRMASK_API const char *glrmask_version(void);
GLRMASK_API const char *glrmask_error_message(void);
GLRMASK_API glrmask_compile_options glrmask_default_compile_options(void);
GLRMASK_API glrmask_lark_options glrmask_default_lark_options(void);

GLRMASK_API glrmask_status glrmask_vocab_new(const glrmask_vocab_entry*,size_t,const uint32_t*,size_t,glrmask_vocab**);
GLRMASK_API glrmask_status glrmask_vocab_clone(const glrmask_vocab*,glrmask_vocab**);
GLRMASK_API void glrmask_vocab_free(glrmask_vocab*);
GLRMASK_API glrmask_status glrmask_vocab_len(const glrmask_vocab*,size_t*);
GLRMASK_API glrmask_status glrmask_vocab_tokens(const glrmask_vocab*,const uint32_t*,size_t,glrmask_exact_tokens**);
GLRMASK_API glrmask_status glrmask_vocab_token_ranges(const glrmask_vocab*,const glrmask_token_range*,size_t,glrmask_exact_tokens**);

GLRMASK_API glrmask_status glrmask_exact_tokens_clone(const glrmask_exact_tokens*,glrmask_exact_tokens**);
GLRMASK_API void glrmask_exact_tokens_free(glrmask_exact_tokens*);
GLRMASK_API glrmask_status glrmask_exact_tokens_len(const glrmask_exact_tokens*,size_t*);
GLRMASK_API glrmask_status glrmask_exact_tokens_ids(const glrmask_exact_tokens*,glrmask_u32s**);
/* NULL output/capacity zero queries the range count without expanding IDs. */
GLRMASK_API glrmask_status glrmask_exact_tokens_ranges(const glrmask_exact_tokens*,glrmask_token_range*,size_t,size_t*);

GLRMASK_API glrmask_status glrmask_grammar_from_source(uint32_t,const char*,size_t,const glrmask_lark_options*,glrmask_grammar**);
GLRMASK_API glrmask_status glrmask_grammar_clone(const glrmask_grammar*,glrmask_grammar**);
GLRMASK_API void glrmask_grammar_free(glrmask_grammar*);
GLRMASK_API glrmask_status glrmask_grammar_compile(const glrmask_grammar*,const glrmask_vocab*,const glrmask_compile_options*,glrmask_constraint**);

GLRMASK_API glrmask_status glrmask_module_new(const glrmask_grammar*,glrmask_module**);
GLRMASK_API glrmask_status glrmask_module_clone(const glrmask_module*,glrmask_module**);
GLRMASK_API void glrmask_module_free(glrmask_module*);
GLRMASK_API glrmask_status glrmask_module_get_mask_strategy(const glrmask_module*,uint32_t*);
GLRMASK_API glrmask_status glrmask_module_set_mask_strategy(glrmask_module*,uint32_t);
GLRMASK_API glrmask_status glrmask_module_get_boundary_mask_strategy(const glrmask_module*,uint32_t*);
GLRMASK_API glrmask_status glrmask_module_set_boundary_mask_strategy(glrmask_module*,uint32_t);
GLRMASK_API glrmask_status glrmask_module_bind_grammar(glrmask_module*,const char*,size_t,const glrmask_grammar*);
GLRMASK_API glrmask_status glrmask_module_bind_module(glrmask_module*,const char*,size_t,const glrmask_module*);
GLRMASK_API glrmask_status glrmask_module_bind_expr(glrmask_module*,const char*,size_t,const glrmask_expr*);
GLRMASK_API glrmask_status glrmask_module_bind_token(glrmask_module*,const char*,size_t,uint32_t);
GLRMASK_API glrmask_status glrmask_module_bind_tokens(glrmask_module*,const char*,size_t,const glrmask_exact_tokens*);
GLRMASK_API glrmask_status glrmask_module_precompile(glrmask_module*,const glrmask_vocab*);
GLRMASK_API glrmask_status glrmask_module_compile(const glrmask_module*,const glrmask_vocab*,const glrmask_compile_options*,glrmask_constraint**);
GLRMASK_API glrmask_status glrmask_module_externs(const glrmask_module*,glrmask_externs**);
GLRMASK_API glrmask_status glrmask_module_unbound_externs(const glrmask_module*,glrmask_externs**);
GLRMASK_API glrmask_status glrmask_module_save(const glrmask_module*,glrmask_bytes**);
GLRMASK_API glrmask_status glrmask_module_load(const uint8_t*,size_t,const glrmask_vocab*,glrmask_module**);

GLRMASK_API glrmask_status glrmask_expr_choice(const glrmask_expr_operand*,size_t,glrmask_expr**);
GLRMASK_API glrmask_status glrmask_expr_sequence(const glrmask_expr_operand*,size_t,glrmask_expr**);
GLRMASK_API glrmask_status glrmask_expr_literal(const uint8_t*,size_t,glrmask_expr**);
GLRMASK_API glrmask_status glrmask_expr_clone(const glrmask_expr*,glrmask_expr**);
GLRMASK_API void glrmask_expr_free(glrmask_expr*);

GLRMASK_API glrmask_status glrmask_constraint_clone(const glrmask_constraint*,glrmask_constraint**);
GLRMASK_API void glrmask_constraint_free(glrmask_constraint*);
GLRMASK_API glrmask_status glrmask_constraint_start(const glrmask_constraint*,glrmask_state**);
GLRMASK_API glrmask_status glrmask_constraint_mask_len(const glrmask_constraint*,size_t*);
GLRMASK_API glrmask_status glrmask_constraint_marker_ids(const glrmask_constraint*,glrmask_u32s**);
GLRMASK_API glrmask_status glrmask_constraint_save(const glrmask_constraint*,uint32_t,glrmask_bytes**);
GLRMASK_API glrmask_status glrmask_constraint_load(const uint8_t*,size_t,const glrmask_vocab*,glrmask_constraint**);

GLRMASK_API glrmask_status glrmask_state_clone(const glrmask_state*,glrmask_state**);
GLRMASK_API void glrmask_state_free(glrmask_state*);
GLRMASK_API glrmask_status glrmask_state_mask_len(const glrmask_state*,size_t*);
/* Writes packed original-ID words, zeroing excess words. Never allocates a
 * temporary language-bridge mask. An undersized buffer is unchanged. */
GLRMASK_API glrmask_status glrmask_state_fill_mask(const glrmask_state*,uint32_t*,size_t);
GLRMASK_API glrmask_status glrmask_state_commit_token(glrmask_state*,uint32_t);
GLRMASK_API glrmask_status glrmask_state_commit_bytes(glrmask_state*,const uint8_t*,size_t);
GLRMASK_API glrmask_status glrmask_state_is_accepting(const glrmask_state*,uint32_t*);
GLRMASK_API glrmask_status glrmask_state_is_rejected(const glrmask_state*,uint32_t*);
GLRMASK_API glrmask_status glrmask_state_forced_tokens(const glrmask_state*,glrmask_u32s**);
GLRMASK_API glrmask_status glrmask_state_marker_events(const glrmask_state*,glrmask_events**);

GLRMASK_API glrmask_status glrmask_bytes_data(const glrmask_bytes*,const uint8_t**,size_t*);
GLRMASK_API void glrmask_bytes_free(glrmask_bytes*);
GLRMASK_API glrmask_status glrmask_u32s_data(const glrmask_u32s*,const uint32_t**,size_t*);
GLRMASK_API void glrmask_u32s_free(glrmask_u32s*);
GLRMASK_API glrmask_status glrmask_events_data(const glrmask_events*,const glrmask_marker_event**,size_t*);
GLRMASK_API void glrmask_events_free(glrmask_events*);

GLRMASK_API glrmask_status glrmask_externs_len(const glrmask_externs*,size_t*);
GLRMASK_API glrmask_status glrmask_externs_get(const glrmask_externs*,size_t,glrmask_extern*);
GLRMASK_API void glrmask_externs_free(glrmask_externs*);

#ifdef __cplusplus
}
#endif
#endif
