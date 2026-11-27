#ifndef GLRMASK_H
#define GLRMASK_H

/* GLRMask C API, ABI revision 8.
 *
 * Lengths count elements, except where a name says bytes. Text is UTF-8 and
 * length-delimited; it doesn't need a NUL terminator. Vocabulary bytes may be
 * any bytes, including NUL and partial UTF-8 sequences.
 *
 * Every handle a function returns is yours: free it with the matching _free
 * function. Pass NULL for default compile options. Handles and memory you pass in are used only during the call. A
 * state keeps its constraint alive, so you may free the constraint first. A
 * clone is independent of the original. Pointers from *_data and
 * glrmask_externs_name stay valid until you free the handle they came from;
 * don't free them yourself.
 *
 * Pass live handles of the right type and valid, aligned memory; GLRMask can't
 * detect dangling pointers. A NULL array is allowed only with length 0. Don't
 * use a handle from two threads at once, or bind a grammar while another thread
 * reads it. A constraint can be shared across threads, with one state per
 * thread.
 *
 * Fallible functions return a status and set output handles to NULL on
 * failure. glrmask_error_message() describes the last failure on the current
 * thread. After GLRMASK_INTERNAL, discard the handle involved. Aborts, including
 * out-of-memory, can't be recovered.
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
/// Treat an unknown non-zero code as a failure.
enum {
    GLRMASK_OK=0,
    GLRMASK_SYNTAX=1,
    GLRMASK_INVALID_INPUT=2, ///< A handle or argument is invalid.
    GLRMASK_COMPILATION=3,
    GLRMASK_RESOURCE_LIMIT=4,
    GLRMASK_STRATEGY_UNSUPPORTED=5, ///< The chosen strategy does not support the grammar.
    GLRMASK_SERIALIZATION=6, ///< Saved bytes could not be loaded.
    GLRMASK_INTERNAL=7, ///< An internal error; discard the affected handle and please report it.
    GLRMASK_BUFFER_TOO_SMALL=8, ///< The buffer is too small; nothing was written.
    GLRMASK_VOCAB_MISMATCH=10, ///< Artifact or precompiled module is for another vocabulary.
    GLRMASK_CLOSED=9 ///< Closed handle (reported by managed wrappers; raw dangling pointers are invalid).
};
/// Controls how the constraint is compiled.
enum {
    GLRMASK_MASK_STRATEGY_AUTO=0,         ///< The default; currently `GLRMASK_MASK_STRATEGY_FAST_MASKS`.
    GLRMASK_MASK_STRATEGY_FAST_COMPILE=3, ///< Faster compilation, slower masks.
    GLRMASK_MASK_STRATEGY_FAST_MASKS=4    ///< Faster masks, slower compilation.
};
/// Grammar source formats for `glrmask_grammar_from_source`.
enum {
    GLRMASK_GLRM=1,
    GLRMASK_JSON_SCHEMA=2,
    GLRMASK_LARK=3
};
/// What an extern can be bound to.
enum {
    GLRMASK_EXTERN_TOKEN=0,    ///< A token set.
    GLRMASK_EXTERN_GRAMMAR=1   ///< A grammar, module, expression or token set.
};

/// Each token ID and its bytes.
typedef struct glrmask_vocab glrmask_vocab;
/// A set of token IDs.
typedef struct glrmask_token_set glrmask_token_set;

typedef struct glrmask_grammar glrmask_grammar;
/// A partially compiled grammar.
typedef struct glrmask_module glrmask_module;
/// A grammar expression.
typedef struct glrmask_expr glrmask_expr;
typedef struct glrmask_constraint glrmask_constraint;
typedef struct glrmask_state glrmask_state;
typedef struct glrmask_bytes glrmask_bytes;
/// A read-only, snapshot of extern declarations.
typedef struct glrmask_externs glrmask_externs;

/// One vocabulary entry: a token ID and its exact bytes.
typedef struct {
    uint32_t id;
    const uint8_t *bytes;
    size_t bytes_len;
} glrmask_vocab_entry;
/** A range of token IDs. `start` is included and `end` is not. */
typedef struct {
    uint32_t start;
    uint32_t end;
} glrmask_token_range;
/// Options for compiling a grammar or module. Set `struct_size` to
/// `sizeof(glrmask_compile_options)`; fields past `struct_size` count as zero.
typedef struct {
    size_t struct_size;
    uint32_t mask_strategy;           ///< A `GLRMASK_MASK_STRATEGY_*` value; 0 is Auto.
    uint32_t reserved[3];             ///< Reserved; must be zero.
} glrmask_compile_options;
/// One part of an expression; `kind` is a `GLRMASK_OPERAND_*` value.
typedef struct { uint32_t kind; const void* handle; } glrmask_expr_operand;
/// The kinds of expression part: grammar, module, expression, token set.
enum {
    GLRMASK_OPERAND_GRAMMAR=0,
    GLRMASK_OPERAND_MODULE=1,
    GLRMASK_OPERAND_EXPR=2,
    GLRMASK_OPERAND_TOKEN_SET=4
};

GLRMASK_API uint32_t glrmask_abi_version(void);
/// The GLRMask version, as a static string.
GLRMASK_API const char *glrmask_version(void);
/// The last error on this thread (empty string if none). Valid until the next
/// GLRMask call on this thread, except this error-message accessor.
GLRMASK_API const char *glrmask_error_message(void);
/// Initializes the caller-owned options buffer with defaults and its size in bytes.
GLRMASK_API glrmask_status glrmask_compile_options_init(glrmask_compile_options* opts,size_t struct_size);

GLRMASK_API glrmask_status glrmask_vocab_new(const glrmask_vocab_entry* entries,size_t n,glrmask_vocab** out);
GLRMASK_API glrmask_status glrmask_vocab_clone(const glrmask_vocab* vocab,glrmask_vocab** out);
GLRMASK_API void glrmask_vocab_free(glrmask_vocab* vocab);
GLRMASK_API glrmask_status glrmask_token_set_new(const uint32_t* ids,size_t n,glrmask_token_set** out);
GLRMASK_API glrmask_status glrmask_token_set_from_ranges(const glrmask_token_range* ranges,size_t n,glrmask_token_set** out);
GLRMASK_API glrmask_status glrmask_token_set_clone(const glrmask_token_set* token_set,glrmask_token_set** out);
GLRMASK_API glrmask_status glrmask_token_set_equal(const glrmask_token_set* a,const glrmask_token_set* b,uint32_t* out);
GLRMASK_API void glrmask_token_set_free(glrmask_token_set* token_set);

GLRMASK_API glrmask_status glrmask_grammar_from_source(uint32_t format,const char* source,size_t n,glrmask_grammar** out);

/// Builds a grammar from an `Expr`.
GLRMASK_API glrmask_status glrmask_grammar_from_expr(const glrmask_expr* expr,glrmask_grammar** out);
GLRMASK_API glrmask_status glrmask_grammar_clone(const glrmask_grammar* grammar,glrmask_grammar** out);
GLRMASK_API void glrmask_grammar_free(glrmask_grammar* grammar);
GLRMASK_API glrmask_status glrmask_grammar_externs(const glrmask_grammar* grammar,glrmask_externs** out);
GLRMASK_API glrmask_status glrmask_grammar_unbound_externs(const glrmask_grammar* grammar,glrmask_externs** out);
/// Compiles the grammar into a constraint.
GLRMASK_API glrmask_status glrmask_grammar_compile(const glrmask_grammar* grammar,const glrmask_vocab* vocab,const glrmask_compile_options* options,glrmask_constraint** out);

GLRMASK_API glrmask_status glrmask_module_new(const glrmask_grammar* grammar,glrmask_module** out);
GLRMASK_API glrmask_status glrmask_module_clone(const glrmask_module* module,glrmask_module** out);
GLRMASK_API void glrmask_module_free(glrmask_module* module);
/// Fails if the extern is undeclared, already bound or the wrong kind.
GLRMASK_API glrmask_status glrmask_module_bind_grammar(glrmask_module* module,const char* name,size_t name_len,const glrmask_grammar* value);
/// Fails if the extern is undeclared, already bound or the wrong kind.
GLRMASK_API glrmask_status glrmask_module_bind_module(glrmask_module* module,const char* name,size_t name_len,const glrmask_module* value);
/// Fails if the extern is undeclared, already bound or the wrong kind.
GLRMASK_API glrmask_status glrmask_module_bind_expr(glrmask_module* module,const char* name,size_t name_len,const glrmask_expr* value);
/// Fails if the extern is undeclared, already bound or the wrong kind.
GLRMASK_API glrmask_status glrmask_module_bind_tokens(glrmask_module* module,const char* name,size_t name_len,const glrmask_token_set* value);
/// Compiles what it can now, even with externs unbound. Further compilation must use the same `vocab`.
GLRMASK_API glrmask_status glrmask_module_precompile(glrmask_module* module,const glrmask_vocab* vocab,const glrmask_compile_options* options);
/// Finishes compiling the module into a constraint.
GLRMASK_API glrmask_status glrmask_module_compile(const glrmask_module* module,const glrmask_vocab* vocab,const glrmask_compile_options* options,glrmask_constraint** out);
GLRMASK_API glrmask_status glrmask_module_externs(const glrmask_module* module,glrmask_externs** out);
GLRMASK_API glrmask_status glrmask_module_unbound_externs(const glrmask_module* module,glrmask_externs** out);
GLRMASK_API glrmask_status glrmask_module_save(const glrmask_module* module,glrmask_bytes** out);
/// May fail if saved by a different GLRMask version.
GLRMASK_API glrmask_status glrmask_module_load(const uint8_t* bytes,size_t n,glrmask_module** out);

/// Matches any one of `operands`.
GLRMASK_API glrmask_status glrmask_expr_choice(const glrmask_expr_operand* operands,size_t n,glrmask_expr** out);
/// Matches `operands` in order.
GLRMASK_API glrmask_status glrmask_expr_sequence(const glrmask_expr_operand* operands,size_t n,glrmask_expr** out);
/// Matches `bytes` exactly.
GLRMASK_API glrmask_status glrmask_expr_literal(const uint8_t* bytes,size_t n,glrmask_expr** out);
GLRMASK_API glrmask_status glrmask_expr_clone(const glrmask_expr* expr,glrmask_expr** out);
GLRMASK_API void glrmask_expr_free(glrmask_expr* expr);

GLRMASK_API glrmask_status glrmask_constraint_clone(const glrmask_constraint* constraint,glrmask_constraint** out);
GLRMASK_API void glrmask_constraint_free(glrmask_constraint* constraint);
/// Starts a new sequence.
GLRMASK_API glrmask_status glrmask_constraint_start(const glrmask_constraint* constraint,glrmask_state** out);
/// The mask's length in bits (the largest token ID plus one).
GLRMASK_API glrmask_status glrmask_constraint_mask_bit_len(const glrmask_constraint* constraint,size_t* out);

GLRMASK_API glrmask_status glrmask_constraint_save(const glrmask_constraint* constraint,glrmask_bytes** out);
/// May fail if saved by a different GLRMask version. Fails if `vocab` differs from the one it was compiled with.
GLRMASK_API glrmask_status glrmask_constraint_load(const uint8_t* bytes,size_t n,const glrmask_vocab* vocab,glrmask_constraint** out);

GLRMASK_API glrmask_status glrmask_state_clone(const glrmask_state* state,glrmask_state** out);
GLRMASK_API void glrmask_state_free(glrmask_state* state);
/// The mask's length in bits (the largest token ID plus one).
GLRMASK_API glrmask_status glrmask_state_mask_bit_len(const glrmask_state* state,size_t* out);
/// Whether the output so far could end here. Writes 1 or 0 to `out`.
GLRMASK_API glrmask_status glrmask_state_is_accepting(const glrmask_state* state,uint32_t* out);
/// Which tokens are allowed next, as a bitmask: token `i` is bit `i % 32` of word `i / 32`.
/// Writes the words into `out`. Returns
/// `GLRMASK_BUFFER_TOO_SMALL`, writing nothing, if `word_count` is below (N + 31) / 32,
/// where N is the state mask bit length.
GLRMASK_API glrmask_status glrmask_state_fill_mask(const glrmask_state* state,uint32_t* out,size_t word_count);
/// Whether nothing more can be committed. Writes 1 or 0 to `out`.
GLRMASK_API glrmask_status glrmask_state_is_terminated(const glrmask_state* state,uint32_t* out);
/// Advances the state by the token. Writes 1 on acceptance and 0 on ordinary rejection (which terminates the state).
/// Returns GLRMASK_OK for either outcome, or a non-OK status on native failure. On failure, a valid accepted out-parameter is zeroed.
GLRMASK_API glrmask_status glrmask_state_commit_token(glrmask_state* state,uint32_t id,uint32_t* accepted);

/// Advances the state by the bytes, with the same status and accepted contract as commit_token.
GLRMASK_API glrmask_status glrmask_state_commit_bytes(glrmask_state* state,const uint8_t* bytes,size_t n,uint32_t* accepted);
/// The bytes that must come next.
GLRMASK_API glrmask_status glrmask_state_forced_bytes(const glrmask_state* state,glrmask_bytes** out);

/// Points `*out_data` at the contents, valid until the buffer is freed.
GLRMASK_API glrmask_status glrmask_bytes_data(const glrmask_bytes* bytes,const uint8_t** out_data,size_t* out_len);
GLRMASK_API void glrmask_bytes_free(glrmask_bytes* bytes);

GLRMASK_API glrmask_status glrmask_externs_len(const glrmask_externs* externs,size_t* out);
/// Copies the extern at `index`; its name stays valid until the view is freed.
GLRMASK_API glrmask_status glrmask_externs_name(const glrmask_externs* externs,size_t index,const char** out,size_t* n);
GLRMASK_API glrmask_status glrmask_externs_kind(const glrmask_externs* externs,size_t index,uint32_t* out);
GLRMASK_API void glrmask_externs_free(glrmask_externs* externs);

#ifdef __cplusplus
}
#endif
#endif
