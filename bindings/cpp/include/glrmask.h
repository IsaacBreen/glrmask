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
    GLRMASK_CLOSED=5,             ///< The handle was already freed or moved.
    GLRMASK_REJECTED=6            ///< A commit was not allowed; the state is rejected.
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
    GLRMASK_BOUNDARY_MASK_STRATEGY_AUTO=0,        ///< Automatic selection.
    GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA=1,  ///< Static boundary parser; faster crossings.
    GLRMASK_BOUNDARY_MASK_STRATEGY_TOKEN_TRIE=2   ///< Segmented dynamic boundary token walk; faster build.
};
/// Alias for GLRMASK_BOUNDARY_MASK_STRATEGY_TOKEN_TRIE.
#define GLRMASK_BOUNDARY_MASK_STRATEGY_FAST_BUILD GLRMASK_BOUNDARY_MASK_STRATEGY_TOKEN_TRIE
/// Alias for GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA.
#define GLRMASK_BOUNDARY_MASK_STRATEGY_FAST_RUNTIME GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA
/// Source formats accepted by glrmask_grammar_from_source.
enum {
    GLRMASK_EBNF=0,         ///< EBNF source.
    GLRMASK_GLRM=1,         ///< Native GLRM source.
    GLRMASK_JSON_SCHEMA=2,  ///< JSON Schema document.
    GLRMASK_LARK=3          ///< Lark source.
};
/// Extern slot kinds.
enum {
    GLRMASK_EXTERN_TOKEN=0,    ///< A token-level slot.
    GLRMASK_EXTERN_GRAMMAR=1   ///< A grammar-level slot.
};

typedef struct glrmask_vocab glrmask_vocab;
typedef struct glrmask_token_set glrmask_token_set;
typedef struct glrmask_grammar glrmask_grammar;
typedef struct glrmask_module glrmask_module;
typedef struct glrmask_expr glrmask_expr;
typedef struct glrmask_constraint glrmask_constraint;
typedef struct glrmask_state glrmask_state;
typedef struct glrmask_bytes glrmask_bytes;
typedef struct glrmask_u32s glrmask_u32s;
typedef struct glrmask_externs glrmask_externs;
typedef struct glrmask_events glrmask_events;

/// One vocabulary entry: a model token ID and its exact bytes.
typedef struct { uint32_t id; const uint8_t *bytes; size_t bytes_len; } glrmask_vocab_entry;
/// Inclusive token-ID interval; adjacent and overlapping ranges are normalized.
typedef struct { uint32_t first; uint32_t last; } glrmask_token_range;
/// Root compile options. Set struct_size to sizeof(glrmask_compile_options).
typedef struct {
    size_t struct_size;              ///< Size of this structure in bytes.
    uint32_t mask_strategy;          ///< GLRMASK_MASK_STRATEGY_* value; 0 selects Auto.
    uint32_t boundary_mask_strategy; ///< GLRMASK_BOUNDARY_MASK_STRATEGY_* value; 0 selects Auto.
} glrmask_compile_options;
/// A declared extern slot.
typedef struct {
    const char *name; size_t name_len;  ///< Declared name (borrowed).
    uint32_t kind;                      ///< GLRMASK_EXTERN_TOKEN or GLRMASK_EXTERN_GRAMMAR.
} glrmask_extern;
/// A possible boundary observation.
typedef struct { uint32_t id; uint64_t byte_offset; } glrmask_marker_event;
/// One `Expr` operand. kind: 0 Grammar, 1 Module, 2 Expr, 3 Token, 4 TokenSet.
typedef struct { uint32_t kind; const void* handle; } glrmask_expr_operand;
/// Operand kinds for glrmask_expr_operand.
enum {
    GLRMASK_OPERAND_GRAMMAR=0,  ///< A glrmask_grammar handle.
    GLRMASK_OPERAND_MODULE=1,   ///< A glrmask_module handle.
    GLRMASK_OPERAND_EXPR=2,     ///< A glrmask_expr handle.
    GLRMASK_OPERAND_TOKEN=3,    ///< A token ID encoded in the handle pointer.
    GLRMASK_OPERAND_TOKEN_SET=4 ///< A glrmask_token_set handle.
};

/// C ABI revision.
GLRMASK_API uint32_t glrmask_abi_version(void);
/// Library version string (static storage).
GLRMASK_API const char *glrmask_version(void);
/// Last error for this thread, valid until the next GLRMask call on it.
GLRMASK_API const char *glrmask_error_message(void);
/// Root compile options with no end tokens and automatic strategies.
GLRMASK_API glrmask_compile_options glrmask_default_compile_options(void);

/// Creates a vocabulary from byte entries plus special token IDs.
GLRMASK_API glrmask_status glrmask_vocab_new(const glrmask_vocab_entry* entries,size_t n,const uint32_t* exact,size_t exact_n,glrmask_vocab** out);
/// Clones a vocabulary handle.
GLRMASK_API glrmask_status glrmask_vocab_clone(const glrmask_vocab* p,glrmask_vocab** out);
/// Frees a vocabulary handle.
GLRMASK_API void glrmask_vocab_free(glrmask_vocab* p);
/// Writes the number of byte-backed entries.
GLRMASK_API glrmask_status glrmask_vocab_len(const glrmask_vocab* p,size_t* out);
/// Creates a token set from the given IDs.
GLRMASK_API glrmask_status glrmask_token_set_new(const uint32_t* ids,size_t n,glrmask_token_set** out);
/// Creates a token set from inclusive ID ranges; empty or reversed ranges contribute nothing.
GLRMASK_API glrmask_status glrmask_token_set_from_ranges(const glrmask_token_range* r,size_t n,glrmask_token_set** out);
/// Clones a token set.
GLRMASK_API glrmask_status glrmask_token_set_clone(const glrmask_token_set* p,glrmask_token_set** out);
/// Frees a token set.
GLRMASK_API void glrmask_token_set_free(glrmask_token_set* p);

/// Parses a grammar in the given format.
GLRMASK_API glrmask_status glrmask_grammar_from_source(uint32_t format,const char* source,size_t n,glrmask_grammar** out);
/// Builds a grammar from an expression.
GLRMASK_API glrmask_status glrmask_grammar_from_expr(const glrmask_expr* expr,glrmask_grammar** out);
/// Binds one declared grammar extern to a grammar.
GLRMASK_API glrmask_status glrmask_grammar_bind_grammar(glrmask_grammar* p,const char* name,size_t n,const glrmask_grammar* child);
/// Binds one declared grammar extern to an expression.
GLRMASK_API glrmask_status glrmask_grammar_bind_expr(glrmask_grammar* p,const char* name,size_t n,const glrmask_expr* child);
/// Binds one declared grammar extern to one token ID.
GLRMASK_API glrmask_status glrmask_grammar_bind_token(glrmask_grammar* p,const char* name,size_t n,uint32_t id);
/// Binds one declared grammar extern to a token set.
GLRMASK_API glrmask_status glrmask_grammar_bind_tokens(glrmask_grammar* p,const char* name,size_t n,const glrmask_token_set* child);
/// Clones a grammar handle.
GLRMASK_API glrmask_status glrmask_grammar_clone(const glrmask_grammar* p,glrmask_grammar** out);
/// Frees a grammar handle.
GLRMASK_API void glrmask_grammar_free(glrmask_grammar* p);
/// Returns an owning view of the grammar's declared externs.
GLRMASK_API glrmask_status glrmask_grammar_externs(const glrmask_grammar* p,glrmask_externs** out);
/// Compiles a grammar against a vocabulary with root options.
GLRMASK_API glrmask_status glrmask_grammar_compile(const glrmask_grammar* p,const glrmask_vocab* v,const glrmask_compile_options* o,glrmask_constraint** out);
/// Precompiles a grammar against a vocabulary and returns a module; options may be NULL for defaults.
GLRMASK_API glrmask_status glrmask_grammar_precompile(const glrmask_grammar* p,const glrmask_vocab* v,const glrmask_compile_options* o,glrmask_module** out);

/// Creates a module from a grammar with default settings.
GLRMASK_API glrmask_status glrmask_module_new(const glrmask_grammar* p,glrmask_module** out);
/// Clones a module handle.
GLRMASK_API glrmask_status glrmask_module_clone(const glrmask_module* p,glrmask_module** out);
/// Frees a module handle.
GLRMASK_API void glrmask_module_free(glrmask_module* p);
/// Binds an extern to a grammar.
GLRMASK_API glrmask_status glrmask_module_bind_grammar(glrmask_module* p,const char* name,size_t n,const glrmask_grammar* child);
/// Binds an extern to a module.
GLRMASK_API glrmask_status glrmask_module_bind_module(glrmask_module* p,const char* name,size_t n,const glrmask_module* child);
/// Binds an extern to an expression.
GLRMASK_API glrmask_status glrmask_module_bind_expr(glrmask_module* p,const char* name,size_t n,const glrmask_expr* child);
/// Binds an extern to one token event.
GLRMASK_API glrmask_status glrmask_module_bind_token(glrmask_module* p,const char* name,size_t n,uint32_t id);
/// Binds an extern to a token set.
GLRMASK_API glrmask_status glrmask_module_bind_tokens(glrmask_module* p,const char* name,size_t n,const glrmask_token_set* child);
/// Retains vocabulary-specific compilation artifacts for this module; options may be NULL for defaults.
GLRMASK_API glrmask_status glrmask_module_precompile(glrmask_module* p,const glrmask_vocab* v,const glrmask_compile_options* o);
/// Compiles a module assembly; every extern must be bound.
GLRMASK_API glrmask_status glrmask_module_compile(const glrmask_module* p,const glrmask_vocab* v,const glrmask_compile_options* o,glrmask_constraint** out);
/// Returns an owning view of all declared externs.
GLRMASK_API glrmask_status glrmask_module_externs(const glrmask_module* p,glrmask_externs** out);
/// Returns an owning view of unresolved externs only.
GLRMASK_API glrmask_status glrmask_module_unbound_externs(const glrmask_module* p,glrmask_externs** out);
/// Serializes the module assembly and retained artifacts.
GLRMASK_API glrmask_status glrmask_module_save(const glrmask_module* p,glrmask_bytes** out);
/// Deserializes a module assembly.
GLRMASK_API glrmask_status glrmask_module_load(const uint8_t* bytes,size_t n,glrmask_module** out);

/// Creates a union without ordered-choice priority.
GLRMASK_API glrmask_status glrmask_expr_choice(const glrmask_expr_operand* items,size_t n,glrmask_expr** out);
/// Creates a concatenation; a model token may cross operand boundaries.
GLRMASK_API glrmask_status glrmask_expr_sequence(const glrmask_expr_operand* items,size_t n,glrmask_expr** out);
/// Creates an exact-byte literal, with no whitespace or ignore inserted.
GLRMASK_API glrmask_status glrmask_expr_literal(const uint8_t* bytes,size_t n,glrmask_expr** out);
/// Clones an expression handle.
GLRMASK_API glrmask_status glrmask_expr_clone(const glrmask_expr* p,glrmask_expr** out);
/// Frees an expression handle.
GLRMASK_API void glrmask_expr_free(glrmask_expr* p);

/// Clones a constraint handle.
GLRMASK_API glrmask_status glrmask_constraint_clone(const glrmask_constraint* p,glrmask_constraint** out);
/// Frees a constraint handle.
GLRMASK_API void glrmask_constraint_free(glrmask_constraint* p);
/// Creates independent state for one generated sequence.
GLRMASK_API glrmask_status glrmask_constraint_start(const glrmask_constraint* p,glrmask_state** out);
/// Writes the packed u32 word count of a mask.
GLRMASK_API glrmask_status glrmask_constraint_mask_len(const glrmask_constraint* p,size_t* out);
/// Returns owning sorted application marker IDs; free with glrmask_u32s_free.
GLRMASK_API glrmask_status glrmask_constraint_marker_ids(const glrmask_constraint* p,glrmask_u32s** out);
/// Serializes the constraint without its vocabulary.
GLRMASK_API glrmask_status glrmask_constraint_save(const glrmask_constraint* p,glrmask_bytes** out);
/// Deserializes a constraint with the exact original vocabulary.
GLRMASK_API glrmask_status glrmask_constraint_load(const uint8_t* bytes,size_t n,const glrmask_vocab* v,glrmask_constraint** out);

/// Clones a sequence state.
GLRMASK_API glrmask_status glrmask_state_clone(const glrmask_state* p,glrmask_state** out);
/// Frees a sequence state.
GLRMASK_API void glrmask_state_free(glrmask_state* p);
/// Writes the packed u32 word count of a mask.
GLRMASK_API glrmask_status glrmask_state_mask_len(const glrmask_state* p,size_t* out);
/// Writes nonzero when the input so far is a complete match. With end tokens configured, this becomes true only once an end token is committed.
GLRMASK_API glrmask_status glrmask_state_is_accepting(const glrmask_state* p,uint32_t* out);
/// Writes packed original-ID words, zeroing excess words. Never allocates a
/// temporary language-bridge mask. An undersized buffer is unchanged.
GLRMASK_API glrmask_status glrmask_state_fill_mask(const glrmask_state* p,uint32_t* out,size_t n);
/// Commits one model token event; a disallowed event rejects the state.
GLRMASK_API glrmask_status glrmask_state_commit_token(glrmask_state* p,uint32_t id);
/// Commits model token events in order; a disallowed event rejects the state.
GLRMASK_API glrmask_status glrmask_state_commit_tokens(glrmask_state* p,const uint32_t* ids,size_t n);
/// Commits raw bytes rather than a model token event.
GLRMASK_API glrmask_status glrmask_state_commit_bytes(glrmask_state* p,const uint8_t* bytes,size_t n);
/// Returns an owning read-only prefix of library-canonical forced output.
GLRMASK_API glrmask_status glrmask_state_forced_tokens(const glrmask_state* p,glrmask_u32s** out);
/// Returns an owning batch of marker observations for the initial or most recent commit.
GLRMASK_API glrmask_status glrmask_state_marker_events(const glrmask_state* p,glrmask_events** out);

/// Borrows the bytes of an owned byte buffer, valid until the handle is freed.
GLRMASK_API glrmask_status glrmask_bytes_data(const glrmask_bytes* p,const uint8_t** bytes,size_t* n);
/// Frees a byte buffer.
GLRMASK_API void glrmask_bytes_free(glrmask_bytes* p);
/// Borrows u32 values, valid until the handle is freed.
GLRMASK_API glrmask_status glrmask_u32s_data(const glrmask_u32s* p,const uint32_t** values,size_t* n);
/// Frees a u32 buffer.
GLRMASK_API void glrmask_u32s_free(glrmask_u32s* p);
/// Borrows marker events, valid until the handle is freed.
GLRMASK_API glrmask_status glrmask_events_data(const glrmask_events* p,const glrmask_marker_event** events,size_t* n);
/// Frees a marker-event buffer.
GLRMASK_API void glrmask_events_free(glrmask_events* p);

/// Writes the number of externs in the view.
GLRMASK_API glrmask_status glrmask_externs_len(const glrmask_externs* p,size_t* out);
/// Copies the extern at an index; string fields are borrowed from the result handle.
GLRMASK_API glrmask_status glrmask_externs_get(const glrmask_externs* p,size_t index,glrmask_extern* out);
/// Frees an externs view.
GLRMASK_API void glrmask_externs_free(glrmask_externs* p);

#ifdef __cplusplus
}
#endif
#endif
