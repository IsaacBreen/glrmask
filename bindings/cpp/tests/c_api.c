#include <glrmask.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"C ABI check failed at line %d: %s; %s\n",__LINE__,#x,glrmask_error_message());return 1;}}while(0)
int main(void) {
    glrmask_vocab* vocab=NULL;glrmask_grammar* grammar=NULL;glrmask_constraint* constraint=NULL;glrmask_state* state=NULL;
    const uint8_t a[]={'a'};const glrmask_vocab_entry entries[]={{0,a,1},{31,a,1}};
    CHECK(glrmask_abi_version()==1);CHECK(strcmp(glrmask_version(),GLRMASK_TEST_EXPECTED_VERSION)==0);
    CHECK(glrmask_vocab_new(NULL,1,NULL,0,&vocab)==GLRMASK_INVALID_ARGUMENT);CHECK(vocab==NULL);
    CHECK(glrmask_vocab_new(entries,2,NULL,0,&vocab)==GLRMASK_OK);
    const char source[]="start ::= \"a\"";
    CHECK(glrmask_grammar_from_source(GLRMASK_EBNF,source,strlen(source),NULL,&grammar)==GLRMASK_OK);
    glrmask_compile_options options=glrmask_default_compile_options();options.struct_size=0;
    CHECK(glrmask_grammar_compile(grammar,vocab,&options,&constraint)==GLRMASK_INVALID_ARGUMENT);CHECK(constraint==NULL);
    CHECK(glrmask_grammar_compile(grammar,vocab,NULL,&constraint)==GLRMASK_OK);
    /* Canonical Module/Expr handles own snapshots across source destruction. */
    glrmask_module *module=NULL,*loaded=NULL;glrmask_expr* expr=NULL;glrmask_bytes* saved=NULL;
    CHECK(glrmask_module_new(grammar,&module)==GLRMASK_OK);
    CHECK(glrmask_module_set_mask_strategy(module,GLRMASK_MASK_STRATEGY_TOKEN_CLASS_TRIE)==GLRMASK_OK);
    CHECK(glrmask_module_precompile(module,vocab)==GLRMASK_OK);
    CHECK(glrmask_module_bind_token(module,"missing",7,0)!=GLRMASK_OK);
    CHECK(glrmask_module_save(module,&saved)==GLRMASK_OK);
    const uint8_t* artifact=NULL;size_t artifact_len=0;CHECK(glrmask_bytes_data(saved,&artifact,&artifact_len)==GLRMASK_OK);
    CHECK(glrmask_module_load(artifact,artifact_len,vocab,&loaded)==GLRMASK_OK);
    glrmask_bytes_free(saved);glrmask_module_free(module);
    uint32_t strategy=0;CHECK(glrmask_module_get_mask_strategy(loaded,&strategy)==GLRMASK_OK);CHECK(strategy==GLRMASK_MASK_STRATEGY_TOKEN_CLASS_TRIE);
    const glrmask_expr_operand operand={1,loaded};CHECK(glrmask_expr_choice(&operand,1,&expr)==GLRMASK_OK);
    glrmask_expr_free(expr);
    glrmask_expr* literal=NULL;CHECK(glrmask_expr_literal((const uint8_t*)"x",1,&literal)==GLRMASK_OK);glrmask_expr_free(literal);
    glrmask_constraint* canonical=NULL;CHECK(glrmask_module_compile(loaded,vocab,NULL,&canonical)==GLRMASK_OK);
    glrmask_module_free(loaded);glrmask_constraint_free(canonical);
    CHECK(glrmask_constraint_start(constraint,&state)==GLRMASK_OK);
    glrmask_constraint_free(constraint);glrmask_grammar_free(grammar);glrmask_vocab_free(vocab);
    uint32_t words[3]={123,123,123};
    CHECK(glrmask_state_fill_mask(state,words,0)==GLRMASK_BUFFER_TOO_SMALL);CHECK(words[0]==123);
    CHECK(glrmask_state_fill_mask(state,(uint32_t*)((uint8_t*)words+1),1)==GLRMASK_INVALID_ARGUMENT);
    CHECK(glrmask_state_fill_mask(state,NULL,1)==GLRMASK_INVALID_ARGUMENT);
    CHECK(glrmask_state_fill_mask(state,words,3)==GLRMASK_OK);
    CHECK(words[0]==(UINT32_C(1)|UINT32_C(0x80000000)) && words[1]==0 && words[2]==0);
    CHECK(glrmask_state_commit_token(state,31)==GLRMASK_OK);
    uint32_t accepting=0;CHECK(glrmask_state_is_accepting(state,&accepting)==GLRMASK_OK);CHECK(accepting==1);
    glrmask_state* copy=NULL;CHECK(glrmask_state_clone(state,&copy)==GLRMASK_OK);
    glrmask_state_free(state);CHECK(glrmask_state_is_accepting(copy,&accepting)==GLRMASK_OK && accepting==1);
    glrmask_state_free(copy);glrmask_state_free(NULL);
    const char bad_utf8[]={(char)255};grammar=(glrmask_grammar*)(uintptr_t)1;
    CHECK(glrmask_grammar_from_source(GLRMASK_EBNF,bad_utf8,1,NULL,&grammar)==GLRMASK_INVALID_ARGUMENT);CHECK(grammar==NULL);
    CHECK(glrmask_constraint_load(NULL,1,NULL,&constraint)==GLRMASK_INVALID_ARGUMENT);CHECK(constraint==NULL);
    puts("C ABI regression checks passed");return 0;
}
