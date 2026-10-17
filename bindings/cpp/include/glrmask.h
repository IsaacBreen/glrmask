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

typedef int32_t glrmask_status;
enum { GLRMASK_OK=0, GLRMASK_INVALID_ARGUMENT=1, GLRMASK_ENGINE_ERROR=2,
       GLRMASK_PANIC=3, GLRMASK_BUFFER_TOO_SMALL=4, GLRMASK_CLOSED=5 };
enum { GLRMASK_AUTO=0, GLRMASK_FAST_BUILD=1, GLRMASK_BALANCED=2, GLRMASK_FAST_RUNTIME=3 };
enum { GLRMASK_BOUNDARY_NONE=0, GLRMASK_BOUNDARY_TOKENS=1, GLRMASK_BOUNDARY_EXACT=2 };
#define GLRMASK_BOUNDARY_DEFAULT UINT32_MAX
enum { GLRMASK_EBNF=0, GLRMASK_GLRM=1, GLRMASK_JSON_SCHEMA=2, GLRMASK_LARK=3 };
enum { GLRMASK_LARK_RECOGNITION=0, GLRMASK_LARK_LLM=1 };
enum { GLRMASK_TOKEN_REQUIREMENT=0, GLRMASK_GRAMMAR_REQUIREMENT=1 };

typedef struct glrmask_vocab glrmask_vocab;
typedef struct glrmask_exact_tokens glrmask_exact_tokens;
typedef struct glrmask_grammar glrmask_grammar;
typedef struct glrmask_prepared glrmask_prepared;
typedef struct glrmask_constraint glrmask_constraint;
typedef struct glrmask_state glrmask_state;
typedef struct glrmask_bytes glrmask_bytes;
typedef struct glrmask_u32s glrmask_u32s;
typedef struct glrmask_requirements glrmask_requirements;
typedef struct glrmask_events glrmask_events;

typedef struct { uint32_t id; const uint8_t *bytes; size_t bytes_len; } glrmask_vocab_entry;
/* Inclusive endpoints; adjacent and overlapping ranges are normalized. */
typedef struct { uint32_t first; uint32_t last; } glrmask_token_range;
typedef struct {
    size_t struct_size;
    uint32_t optimization;
    uint32_t boundary_trigger;
    const uint32_t *end_tokens;
    size_t end_tokens_len;
} glrmask_build_options;
typedef struct { const char *name; size_t name_len; const uint32_t *ids; size_t ids_len; } glrmask_lark_token;
typedef struct { size_t struct_size; uint32_t profile; const glrmask_lark_token *tokens; size_t tokens_len; } glrmask_lark_options;
typedef struct {
    const char *name; size_t name_len; uint32_t kind; uint32_t has_source_span;
    uint64_t start_byte, end_byte, line, column;
} glrmask_requirement;
typedef struct { uint32_t id; uint64_t byte_offset; } glrmask_marker_event;

GLRMASK_API uint32_t glrmask_abi_version(void);
GLRMASK_API const char *glrmask_version(void);
GLRMASK_API const char *glrmask_error_message(void);
GLRMASK_API glrmask_build_options glrmask_default_build_options(void);
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
GLRMASK_API glrmask_status glrmask_grammar_exact(const glrmask_exact_tokens*,glrmask_grammar**);
GLRMASK_API glrmask_status glrmask_grammar_clone(const glrmask_grammar*,glrmask_grammar**);
GLRMASK_API void glrmask_grammar_free(glrmask_grammar*);
GLRMASK_API glrmask_status glrmask_grammar_bind_grammar(glrmask_grammar*,const char*,size_t,const glrmask_grammar*);
GLRMASK_API glrmask_status glrmask_grammar_bind_token(glrmask_grammar*,const char*,size_t,uint32_t);
GLRMASK_API glrmask_status glrmask_grammar_bind_tokens(glrmask_grammar*,const char*,size_t,const glrmask_exact_tokens*);
GLRMASK_API glrmask_status glrmask_grammar_requirements(const glrmask_grammar*,glrmask_requirements**);
/* Preparation retains only a boundary metadata request; final build options
 * and root end tokens belong to compile/link, not prepare. */
GLRMASK_API glrmask_status glrmask_grammar_prepare(const glrmask_grammar*,const glrmask_vocab*,uint32_t,glrmask_prepared**);
GLRMASK_API glrmask_status glrmask_grammar_compile(const glrmask_grammar*,const glrmask_vocab*,const glrmask_build_options*,glrmask_constraint**);

GLRMASK_API glrmask_status glrmask_prepared_clone(const glrmask_prepared*,glrmask_prepared**);
GLRMASK_API void glrmask_prepared_free(glrmask_prepared*);
GLRMASK_API glrmask_status glrmask_prepared_bind_prepared(glrmask_prepared*,const char*,size_t,const glrmask_prepared*);
GLRMASK_API glrmask_status glrmask_prepared_bind_token(glrmask_prepared*,const char*,size_t,uint32_t);
GLRMASK_API glrmask_status glrmask_prepared_bind_tokens(glrmask_prepared*,const char*,size_t,const glrmask_exact_tokens*);
GLRMASK_API glrmask_status glrmask_prepared_requirements(const glrmask_prepared*,glrmask_requirements**);
GLRMASK_API glrmask_status glrmask_prepared_link(const glrmask_prepared*,const glrmask_build_options*,glrmask_constraint**);
GLRMASK_API glrmask_status glrmask_prepared_save(const glrmask_prepared*,glrmask_bytes**);
GLRMASK_API glrmask_status glrmask_prepared_load(const uint8_t*,size_t,const glrmask_vocab*,glrmask_prepared**);

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
GLRMASK_API glrmask_status glrmask_requirements_len(const glrmask_requirements*,size_t*);
GLRMASK_API glrmask_status glrmask_requirements_get(const glrmask_requirements*,size_t,glrmask_requirement*);
GLRMASK_API void glrmask_requirements_free(glrmask_requirements*);
GLRMASK_API glrmask_status glrmask_events_data(const glrmask_events*,const glrmask_marker_event**,size_t*);
GLRMASK_API void glrmask_events_free(glrmask_events*);

#ifdef __cplusplus
}
#endif
#endif
