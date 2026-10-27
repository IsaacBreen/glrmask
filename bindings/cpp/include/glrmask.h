#ifndef GLRMASK_H
#define GLRMASK_H

/* GLRMask C ABI, revision 3.
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
#include <stdbool.h>
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

/// Result of a fallible call. `GLRMASK_OK` means success; otherwise output handles
/// are set to NULL and `glrmask_error_message()` describes the failure.
typedef int32_t glrmask_status;
/// Status values. On failure an output handle is NULL.
enum {
    GLRMASK_OK=0,                 ///< Success.
    GLRMASK_INVALID_ARGUMENT=1,   ///< A handle or argument is invalid.
    GLRMASK_ENGINE_ERROR=2,       ///< Parsing, compiling or loading failed.
    GLRMASK_PANIC=3,              ///< An internal error was caught. Discard the handle involved.
    GLRMASK_BUFFER_TOO_SMALL=4,   ///< The buffer is too small; nothing was written.
    GLRMASK_CLOSED=5              ///< The handle was already freed.
};
/// How masks are computed.
enum {
    GLRMASK_MASK_STRATEGY_AUTO=0,        ///< Reuses any precompiled work; otherwise GLRMASK_MASK_STRATEGY_PARSER_DWA, or GLRMASK_MASK_STRATEGY_TOKEN_TRIE when that would be too large. The default.
    GLRMASK_MASK_STRATEGY_PARSER_DWA=1,  ///< Builds a parser DWA for the grammar. Slowest to compile, with the fastest masks.
    GLRMASK_MASK_STRATEGY_TOKEN_TRIE=2   ///< Computes each mask by walking the vocabulary. Fastest to compile, with slower masks. Its settings are in `glrmask_compile_options.token_trie`.
};
/// Same as GLRMASK_MASK_STRATEGY_TOKEN_TRIE.
#define GLRMASK_MASK_STRATEGY_FAST_COMPILE GLRMASK_MASK_STRATEGY_TOKEN_TRIE
/// Same as GLRMASK_MASK_STRATEGY_PARSER_DWA.
#define GLRMASK_MASK_STRATEGY_FAST_MASKS GLRMASK_MASK_STRATEGY_PARSER_DWA
/// How masks are computed at grammar boundaries.
enum {
    GLRMASK_BOUNDARY_MASK_STRATEGY_AUTO=0,        ///< GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA, or GLRMASK_BOUNDARY_MASK_STRATEGY_TOKEN_TRIE when that would be too large. The default.
    GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA=1,  ///< Builds a parser DWA across each boundary. Slower to compile, with faster masks. Fails with GLRMASK_ENGINE_ERROR if it would be too large.
    GLRMASK_BOUNDARY_MASK_STRATEGY_TOKEN_TRIE=2   ///< Walks the vocabulary at each boundary as needed. Faster to compile, with slower masks.
};
/// Same as GLRMASK_BOUNDARY_MASK_STRATEGY_TOKEN_TRIE.
#define GLRMASK_BOUNDARY_MASK_STRATEGY_FAST_COMPILE GLRMASK_BOUNDARY_MASK_STRATEGY_TOKEN_TRIE
/// Same as GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA.
#define GLRMASK_BOUNDARY_MASK_STRATEGY_FAST_MASKS GLRMASK_BOUNDARY_MASK_STRATEGY_PARSER_DWA
/// How the parser is represented and run when masks are computed with GLRMASK_MASK_STRATEGY_TOKEN_TRIE.
enum {
    GLRMASK_PARSER_BACKEND_LR_TABLE=0,     ///< An LR table. Faster to compile, with slower masks. The default.
    GLRMASK_PARSER_BACKEND_TEMPLATE_DFA=1  ///< Template DFAs. Slower to compile, with faster masks.
};
/// Grammar source formats for `glrmask_grammar_from_source`.
enum {
    GLRMASK_GLRM=1,         ///< GLRM.
    GLRMASK_JSON_SCHEMA=2,  ///< JSON Schema.
    GLRMASK_LARK=3          ///< Lark.
};
/// What an extern can be bound to.
enum {
    GLRMASK_EXTERN_TOKEN=0,    ///< A token set or token ID.
    GLRMASK_EXTERN_GRAMMAR=1   ///< A grammar, module, expression, token set or token ID.
};

typedef struct glrmask_vocab glrmask_vocab;
typedef struct glrmask_token_set glrmask_token_set;
typedef struct glrmask_grammar glrmask_grammar;
/// A grammar you can compose with others and partly compile before its externs are bound.
typedef struct glrmask_module glrmask_module;
typedef struct glrmask_expr glrmask_expr;
typedef struct glrmask_constraint glrmask_constraint;
typedef struct glrmask_state glrmask_state;
typedef struct glrmask_bytes glrmask_bytes;
typedef struct glrmask_u32s glrmask_u32s;
typedef struct glrmask_externs glrmask_externs;

/// One vocabulary entry: a token ID and its exact bytes.
typedef struct { uint32_t id; const uint8_t *bytes; size_t bytes_len; } glrmask_vocab_entry;
/** A range of token IDs. `start` is included and `end` is not. */
typedef struct {
    uint32_t start; /**< First token ID in the range. */
    uint32_t end;   /**< One past the last token ID. */
} glrmask_token_range;
/// Settings for GLRMASK_MASK_STRATEGY_TOKEN_TRIE.
typedef struct {
    uint32_t parser_backend;          ///< A GLRMASK_PARSER_BACKEND_* value; 0 is an LR table.
    uint32_t merge_equivalent_tokens; ///< Nonzero to merge tokens that the grammar treats the same, so masks walk fewer tokens. Tokens are not merged in a composition that includes precompiled parts.
} glrmask_token_trie_settings;
/// Options for compiling a grammar or module. Set `struct_size` to
/// `sizeof(glrmask_compile_options)`; fields past `struct_size` count as zero.
typedef struct {
    size_t struct_size;               ///< Size of this struct in bytes.
    uint32_t mask_strategy;           ///< A `GLRMASK_MASK_STRATEGY_*` value; 0 is Auto.
    uint32_t boundary_mask_strategy;  ///< A `GLRMASK_BOUNDARY_MASK_STRATEGY_*` value; 0 is Auto.
    glrmask_token_trie_settings token_trie; ///< Settings for GLRMASK_MASK_STRATEGY_TOKEN_TRIE; must be zero for any other strategy.
} glrmask_compile_options;
/// An extern declared by a grammar. Its `name` stays valid until the externs view
/// is freed.
typedef struct {
    const char *name; size_t name_len;  ///< Declared name (borrowed).
    uint32_t kind;                      ///< GLRMASK_EXTERN_TOKEN or GLRMASK_EXTERN_GRAMMAR.
} glrmask_extern;
/// One part of an expression; `kind` is a `GLRMASK_OPERAND_*` value.
typedef struct { uint32_t kind; const void* handle; } glrmask_expr_operand;
/// The kinds of expression part: grammar, module, expression, token ID, token set.
enum {
    GLRMASK_OPERAND_GRAMMAR=0,  ///< A glrmask_grammar handle.
    GLRMASK_OPERAND_MODULE=1,   ///< A glrmask_module handle.
    GLRMASK_OPERAND_EXPR=2,     ///< A glrmask_expr handle.
    GLRMASK_OPERAND_TOKEN=3,    ///< A token ID encoded in the handle pointer.
    GLRMASK_OPERAND_TOKEN_SET=4 ///< A glrmask_token_set handle.
};

/// The C ABI revision.
GLRMASK_API uint32_t glrmask_abi_version(void);
/// The GLRMask version, as a static string.
GLRMASK_API const char *glrmask_version(void);
/// The last error on this thread, valid until the next GLRMask call on it.
GLRMASK_API const char *glrmask_error_message(void);
/// Default compile options: automatic strategies.
GLRMASK_API glrmask_compile_options glrmask_default_compile_options(void);

/// Creates a vocabulary from `entries`.
GLRMASK_API glrmask_status glrmask_vocab_new(const glrmask_vocab_entry* entries,size_t n,glrmask_vocab** out);
/// Creates an independent copy.
GLRMASK_API glrmask_status glrmask_vocab_clone(const glrmask_vocab* p,glrmask_vocab** out);
/// Frees a vocabulary handle.
GLRMASK_API void glrmask_vocab_free(glrmask_vocab* p);
/// Creates a token set from `ids`.
GLRMASK_API glrmask_status glrmask_token_set_new(const uint32_t* ids,size_t n,glrmask_token_set** out);
/// Creates a token set from half-open `ranges`, each including `start` and excluding `end`.
GLRMASK_API glrmask_status glrmask_token_set_from_ranges(const glrmask_token_range* r,size_t n,glrmask_token_set** out);
/// Creates an independent copy.
GLRMASK_API glrmask_status glrmask_token_set_clone(const glrmask_token_set* p,glrmask_token_set** out);
/// Writes whether both sets contain the same IDs.
GLRMASK_API glrmask_status glrmask_token_set_equal(const glrmask_token_set* a,const glrmask_token_set* b,uint32_t* out);
/// Frees a token set.
GLRMASK_API void glrmask_token_set_free(glrmask_token_set* p);

/// Creates a grammar from source text in `format`.
GLRMASK_API glrmask_status glrmask_grammar_from_source(uint32_t format,const char* source,size_t n,glrmask_grammar** out);
/// Creates a grammar from an expression.
GLRMASK_API glrmask_status glrmask_grammar_from_expr(const glrmask_expr* expr,glrmask_grammar** out);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_grammar_bind_grammar(glrmask_grammar* p,const char* name,size_t n,const glrmask_grammar* child);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_grammar_bind_expr(glrmask_grammar* p,const char* name,size_t n,const glrmask_expr* child);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_grammar_bind_token(glrmask_grammar* p,const char* name,size_t n,uint32_t id);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_grammar_bind_tokens(glrmask_grammar* p,const char* name,size_t n,const glrmask_token_set* child);
/// Creates an independent copy.
GLRMASK_API glrmask_status glrmask_grammar_clone(const glrmask_grammar* p,glrmask_grammar** out);
/// Frees a grammar handle.
GLRMASK_API void glrmask_grammar_free(glrmask_grammar* p);
/// Writes the grammar's externs, in declaration order.
GLRMASK_API glrmask_status glrmask_grammar_externs(const glrmask_grammar* p,glrmask_externs** out);
/// Writes the grammar's externs that are not yet bound.
GLRMASK_API glrmask_status glrmask_grammar_unbound_externs(const glrmask_grammar* p,glrmask_externs** out);
/// Compiles the grammar for `vocab`. Fails if any extern is unbound. `options`
/// may be NULL for defaults.
GLRMASK_API glrmask_status glrmask_grammar_compile(const glrmask_grammar* p,const glrmask_vocab* v,const glrmask_compile_options* o,glrmask_constraint** out);
/// Compiles as much as possible for `vocab` now, and returns a module that keeps
/// the work. `options` may be NULL.
GLRMASK_API glrmask_status glrmask_grammar_precompile(const glrmask_grammar* p,const glrmask_vocab* v,const glrmask_compile_options* o,glrmask_module** out);

/// Creates a module from a grammar.
GLRMASK_API glrmask_status glrmask_module_new(const glrmask_grammar* p,glrmask_module** out);
/// Creates an independent copy.
GLRMASK_API glrmask_status glrmask_module_clone(const glrmask_module* p,glrmask_module** out);
/// Frees a module handle.
GLRMASK_API void glrmask_module_free(glrmask_module* p);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_module_bind_grammar(glrmask_module* p,const char* name,size_t n,const glrmask_grammar* child);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_module_bind_module(glrmask_module* p,const char* name,size_t n,const glrmask_module* child);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_module_bind_expr(glrmask_module* p,const char* name,size_t n,const glrmask_expr* child);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_module_bind_token(glrmask_module* p,const char* name,size_t n,uint32_t id);
/// Binds the extern `name`. Each fails if `name` is undeclared or already bound,
/// or the value is the wrong kind for it. A token extern takes only a token ID or
/// token set.
GLRMASK_API glrmask_status glrmask_module_bind_tokens(glrmask_module* p,const char* name,size_t n,const glrmask_token_set* child);
/// Compiles as much as possible for `vocab` now, so later compiles are fast.
/// `options` may be NULL.
GLRMASK_API glrmask_status glrmask_module_precompile(glrmask_module* p,const glrmask_vocab* v,const glrmask_compile_options* o);
/// Compiles the module for `vocab`, reusing any precompiled work. Fails if any
/// extern is unbound or the module was precompiled for another vocabulary.
/// `options` may be NULL.
GLRMASK_API glrmask_status glrmask_module_compile(const glrmask_module* p,const glrmask_vocab* v,const glrmask_compile_options* o,glrmask_constraint** out);
/// Writes the module's externs, in declaration order.
GLRMASK_API glrmask_status glrmask_module_externs(const glrmask_module* p,glrmask_externs** out);
/// Writes the module's externs that are not yet bound.
GLRMASK_API glrmask_status glrmask_module_unbound_externs(const glrmask_module* p,glrmask_externs** out);
/// Saves the module, including precompiled work, as bytes. The vocabulary is not
/// included.
GLRMASK_API glrmask_status glrmask_module_save(const glrmask_module* p,glrmask_bytes** out);
/// Loads a module saved with `glrmask_module_save`.
GLRMASK_API glrmask_status glrmask_module_load(const uint8_t* bytes,size_t n,glrmask_module** out);

/// Creates an expression matching any one of `operands`.
GLRMASK_API glrmask_status glrmask_expr_choice(const glrmask_expr_operand* items,size_t n,glrmask_expr** out);
/// Creates an expression matching `operands` one after another.
GLRMASK_API glrmask_status glrmask_expr_sequence(const glrmask_expr_operand* items,size_t n,glrmask_expr** out);
/// Creates an expression matching exactly `bytes`.
GLRMASK_API glrmask_status glrmask_expr_literal(const uint8_t* bytes,size_t n,glrmask_expr** out);
/// Creates an independent copy.
GLRMASK_API glrmask_status glrmask_expr_clone(const glrmask_expr* p,glrmask_expr** out);
/// Frees an expression handle.
GLRMASK_API void glrmask_expr_free(glrmask_expr* p);

/// Creates an independent copy.
GLRMASK_API glrmask_status glrmask_constraint_clone(const glrmask_constraint* p,glrmask_constraint** out);
/// Frees a constraint handle.
GLRMASK_API void glrmask_constraint_free(glrmask_constraint* p);
/// Creates a state for one generated sequence.
GLRMASK_API glrmask_status glrmask_constraint_start(const glrmask_constraint* p,glrmask_state** out);
/// Writes the number of `uint32_t` words in a mask.
GLRMASK_API glrmask_status glrmask_constraint_mask_len(const glrmask_constraint* p,size_t* out);
/// Saves the constraint as bytes, without the vocabulary.
GLRMASK_API glrmask_status glrmask_constraint_save(const glrmask_constraint* p,glrmask_bytes** out);
/// Loads a constraint saved with `glrmask_constraint_save`. Fails if `vocab`
/// isn't the one it was compiled for.
GLRMASK_API glrmask_status glrmask_constraint_load(const uint8_t* bytes,size_t n,const glrmask_vocab* v,glrmask_constraint** out);

/// Copies the state.
GLRMASK_API glrmask_status glrmask_state_clone(const glrmask_state* p,glrmask_state** out);
/// Frees a sequence state.
GLRMASK_API void glrmask_state_free(glrmask_state* p);
/// Writes the number of `uint32_t` words in a mask.
GLRMASK_API glrmask_status glrmask_state_mask_len(const glrmask_state* p,size_t* out);
/// Whether everything committed so far is a complete match. Writes 1 or 0 to
/// `out`.
GLRMASK_API glrmask_status glrmask_state_is_accepting(const glrmask_state* p,uint32_t* out);
/// Writes the tokens allowed next into `out` as a bitmask, setting extra words to
/// zero. Token `id` is allowed when bit `id % 32` of word `id / 32` is set. Returns
/// `GLRMASK_BUFFER_TOO_SMALL`, writing nothing, if `out_len` is less than the mask
/// length.
GLRMASK_API glrmask_status glrmask_state_fill_mask(const glrmask_state* p,uint32_t* out,size_t n);
/// Whether nothing more can be committed. Writes 1 or 0 to `out`.
GLRMASK_API glrmask_status glrmask_state_is_terminated(const glrmask_state* p,uint32_t* out);
/// Commits a token. If it isn't allowed, returns `false` and terminates the
/// state. Returns false for a NULL state.
GLRMASK_API bool glrmask_state_commit_token(glrmask_state* p,uint32_t id);
/// Commits each token in order. If one isn't allowed, returns `false` and
/// terminates the state. Returns false for a NULL state.
GLRMASK_API bool glrmask_state_commit_tokens(glrmask_state* p,const uint32_t* ids,size_t n);
/// Commits raw bytes. If they aren't allowed, returns `false` and terminates the
/// state. Returns false for a NULL state.
GLRMASK_API bool glrmask_state_commit_bytes(glrmask_state* p,const uint8_t* bytes,size_t n);
/// Commits the tokens for what must come next, taking the longest token at each
/// step, and returns them. Stops before anything that could vary, or after 1,000
/// tokens.
GLRMASK_API glrmask_status glrmask_state_commit_forced_tokens(glrmask_state* p,glrmask_u32s** out);

/// Points `*out_data` at the contents, valid until the buffer is freed.
GLRMASK_API glrmask_status glrmask_bytes_data(const glrmask_bytes* p,const uint8_t** out_data,size_t* out_len);
/// Frees a byte buffer.
GLRMASK_API void glrmask_bytes_free(glrmask_bytes* p);
/// Points `*out_data` at the contents, valid until the buffer is freed.
GLRMASK_API glrmask_status glrmask_u32s_data(const glrmask_u32s* p,const uint32_t** out_data,size_t* out_len);
/// Frees a u32 buffer.
GLRMASK_API void glrmask_u32s_free(glrmask_u32s* p);

/// Writes the number of externs.
GLRMASK_API glrmask_status glrmask_externs_len(const glrmask_externs* p,size_t* out);
/// Copies the extern at `index`; its name stays valid until the view is freed.
GLRMASK_API glrmask_status glrmask_externs_get(const glrmask_externs* p,size_t index,glrmask_extern* out);
/// Frees an externs view.
GLRMASK_API void glrmask_externs_free(glrmask_externs* p);

#ifdef __cplusplus
}
#endif
#endif
