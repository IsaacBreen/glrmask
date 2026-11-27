#include <glrmask.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"C ABI check failed at line %d: %s; %s\n",__LINE__,#x,glrmask_error_message());return 1;}}while(0)
int main(void) {
    uint32_t accepted=0;
    glrmask_vocab* vocab=NULL;glrmask_grammar* grammar=NULL;glrmask_constraint* constraint=NULL;glrmask_state* state=NULL;
    const uint8_t a[]={'a'};const glrmask_vocab_entry entries[]={{0,a,1},{31,a,1}};
    CHECK(glrmask_abi_version()==8);CHECK(strcmp(glrmask_version(),GLRMASK_TEST_EXPECTED_VERSION)==0);
    CHECK(glrmask_vocab_new(NULL,1,&vocab)==GLRMASK_INVALID_INPUT);CHECK(vocab==NULL);
    CHECK(glrmask_vocab_new(entries,2,&vocab)==GLRMASK_OK);
    {
        const char* rejected[]={"start: <[1]>","start: <[1,3-8]>","start: <[]>","start: <[^0]>","start: <[*]>","start: @token(7)"};
        for (size_t i=0;i<sizeof(rejected)/sizeof(rejected[0]);i++){
            CHECK(glrmask_grammar_from_source(GLRMASK_LARK,rejected[i],strlen(rejected[i]),&grammar)==GLRMASK_SYNTAX);
            CHECK(grammar==NULL);
            CHECK(strstr(glrmask_error_message(),"token IDs can't appear in grammar source")!=NULL);
        }
    }
    const char source[]="start: \"a\"";
    CHECK(glrmask_grammar_from_source(GLRMASK_LARK,source,strlen(source),&grammar)==GLRMASK_OK);
    glrmask_compile_options options; CHECK(glrmask_compile_options_init(&options,sizeof(options))==GLRMASK_OK);
    CHECK(options.mask_strategy==GLRMASK_MASK_STRATEGY_AUTO);
    CHECK(options.reserved[0]==0 && options.reserved[1]==0 && options.reserved[2]==0);
    {
        glrmask_compile_options wrong; CHECK(glrmask_compile_options_init(&wrong,sizeof(wrong))==GLRMASK_OK);
        wrong.reserved[1]=1;
        CHECK(glrmask_grammar_compile(grammar,vocab,&wrong,&constraint)==GLRMASK_INVALID_INPUT);
        CHECK(strcmp(glrmask_error_message(),"reserved compile-options fields must be zero")==0);
    }
    {
        /* The old strict codes are rejected, not silently reinterpreted. */
        for (uint32_t old=1;old<=2;old++){
            glrmask_compile_options wrong; CHECK(glrmask_compile_options_init(&wrong,sizeof(wrong))==GLRMASK_OK);
            wrong.mask_strategy=old;
            CHECK(glrmask_grammar_compile(grammar,vocab,&wrong,&constraint)==GLRMASK_INVALID_INPUT && constraint==NULL);
        }
        const uint32_t modes[]={GLRMASK_MASK_STRATEGY_AUTO,GLRMASK_MASK_STRATEGY_FAST_COMPILE,GLRMASK_MASK_STRATEGY_FAST_MASKS};
        for(size_t i=0;i<3;i++){
            glrmask_compile_options requested; CHECK(glrmask_compile_options_init(&requested,sizeof(requested))==GLRMASK_OK);
            requested.mask_strategy=modes[i];
            CHECK(glrmask_grammar_compile(grammar,vocab,&requested,&constraint)==GLRMASK_OK);
            size_t n=0,m=0;glrmask_state* probe=NULL;uint32_t words[2]={0xffffffff,0xffffffff};
            CHECK(glrmask_constraint_mask_bit_len(constraint,&n)==GLRMASK_OK && n==32);
            CHECK(glrmask_constraint_start(constraint,&probe)==GLRMASK_OK);
            CHECK(glrmask_state_mask_bit_len(probe,&m)==GLRMASK_OK && n==m);
            CHECK(glrmask_state_fill_mask(probe,words,(n+31)/32)==GLRMASK_OK && words[0]==0x80000001);
            CHECK(glrmask_state_fill_mask(probe,words,0)==GLRMASK_BUFFER_TOO_SMALL);
            CHECK(glrmask_state_fill_mask(probe,words,2)==GLRMASK_OK && words[1]==0);
            glrmask_bytes* forced=NULL;const uint8_t* data=NULL;size_t count=0;
            CHECK(glrmask_state_forced_bytes(probe,&forced)==GLRMASK_OK);
            CHECK(glrmask_bytes_data(forced,&data,&count)==GLRMASK_OK && count==1 && data[0]=='a');
            CHECK(glrmask_state_fill_mask(probe,words,2)==GLRMASK_OK && words[0]==0x80000001 && words[1]==0);
            CHECK(glrmask_state_commit_bytes(probe,data,count,&accepted)==GLRMASK_OK && accepted==1);glrmask_bytes_free(forced);forced=NULL;
            CHECK(glrmask_state_forced_bytes(probe,&forced)==GLRMASK_OK);
            CHECK(glrmask_bytes_data(forced,&data,&count)==GLRMASK_OK && count==0);
            glrmask_bytes_free(forced);
            glrmask_state_free(probe);
            for(size_t rejection=0;rejection<3;rejection++){
                uint32_t terminated=0,accepting=0;
                CHECK(glrmask_constraint_start(constraint,&probe)==GLRMASK_OK);
                CHECK(glrmask_state_is_terminated(probe,&terminated)==GLRMASK_OK && terminated==0);
                if(rejection==0){CHECK(glrmask_state_commit_token(probe,1,&accepted)==GLRMASK_OK && accepted==0);}
                else if(rejection==1){CHECK(glrmask_state_commit_token(probe,65,&accepted)==GLRMASK_OK && accepted==0);}
                else{CHECK(glrmask_state_commit_bytes(probe,(const uint8_t*)"z",1,&accepted)==GLRMASK_OK && accepted==0);}
                CHECK(glrmask_state_is_terminated(probe,&terminated)==GLRMASK_OK && terminated==1);
                CHECK(glrmask_state_is_accepting(probe,&accepting)==GLRMASK_OK && accepting==0);
                CHECK(glrmask_state_fill_mask(probe,words,2)==GLRMASK_OK && words[0]==0 && words[1]==0);
                CHECK(glrmask_state_forced_bytes(probe,&forced)==GLRMASK_OK);
                CHECK(glrmask_bytes_data(forced,&data,&count)==GLRMASK_OK && count==0);
                glrmask_bytes_free(forced); forced=NULL;
                CHECK(glrmask_state_commit_token(probe,0,&accepted)==GLRMASK_OK && accepted==0);
                glrmask_state_free(probe);
            }
            glrmask_constraint_free(constraint);constraint=NULL;
        }
    }
    {
        struct old_options { size_t struct_size; uint32_t mask_strategy, reserved; };
        struct { struct old_options opts; uint64_t guard; } old;
        memset(&old,0xff,sizeof(old));
        CHECK(glrmask_compile_options_init((glrmask_compile_options*)&old.opts,sizeof(old.opts))==GLRMASK_OK);
        CHECK(old.opts.struct_size==sizeof(old.opts) && old.opts.mask_strategy==0 && old.opts.reserved==0);
        CHECK(old.guard==UINT64_MAX);
        CHECK(glrmask_grammar_compile(grammar,vocab,(const glrmask_compile_options*)&old.opts,&constraint)==GLRMASK_OK);
        glrmask_constraint_free(constraint); constraint=NULL;
        CHECK(glrmask_compile_options_init(NULL,sizeof(options))==GLRMASK_INVALID_INPUT);
    }
    {
        struct { glrmask_compile_options opts; uint64_t trailing; } future;
        CHECK(glrmask_compile_options_init(&future.opts,sizeof(future))==GLRMASK_OK);
        CHECK(glrmask_grammar_compile(grammar,vocab,&future.opts,&constraint)==GLRMASK_OK);
        glrmask_constraint_free(constraint);constraint=NULL;
        future.trailing=1;
        CHECK(glrmask_grammar_compile(grammar,vocab,&future.opts,&constraint)==GLRMASK_INVALID_INPUT && constraint==NULL);

        CHECK(strlen(glrmask_error_message())>0);
        accepted=99; CHECK(glrmask_state_commit_token(NULL,0,&accepted)==GLRMASK_INVALID_INPUT && accepted==0);

        CHECK(glrmask_abi_version()==8);
        CHECK(!strlen(glrmask_error_message()));
        size_t zero=99;
        CHECK(glrmask_constraint_mask_bit_len(NULL,&zero)==GLRMASK_INVALID_INPUT && zero==0);
    }
    options.struct_size=0;
    CHECK(glrmask_grammar_compile(grammar,vocab,&options,&constraint)==GLRMASK_INVALID_INPUT);CHECK(constraint==NULL);
    CHECK(glrmask_grammar_compile(grammar,vocab,NULL,&constraint)==GLRMASK_OK);
    /* Canonical Module/Expr handles own snapshots across source destruction. */
    glrmask_module *module=NULL,*loaded=NULL;glrmask_expr* expr=NULL;glrmask_bytes* saved=NULL;
    CHECK(glrmask_module_new(grammar,&module)==GLRMASK_OK);
    {
        glrmask_constraint* sentinel=(glrmask_constraint*)(uintptr_t)1;
        glrmask_externs* ext=(glrmask_externs*)(uintptr_t)1;
        CHECK(glrmask_module_compile(NULL,vocab,NULL,&sentinel)==GLRMASK_INVALID_INPUT && sentinel==NULL);
        CHECK(glrmask_grammar_externs(NULL,&ext)==GLRMASK_INVALID_INPUT && ext==NULL);
        ext=(glrmask_externs*)(uintptr_t)1;
        CHECK(glrmask_grammar_unbound_externs(NULL,&ext)==GLRMASK_INVALID_INPUT && ext==NULL);
        ext=(glrmask_externs*)(uintptr_t)1;
        CHECK(glrmask_module_externs(NULL,&ext)==GLRMASK_INVALID_INPUT && ext==NULL);
        ext=(glrmask_externs*)(uintptr_t)1;
        CHECK(glrmask_module_unbound_externs(NULL,&ext)==GLRMASK_INVALID_INPUT && ext==NULL);
        const char open[]="start s extern grammar child nt s = child";
        glrmask_grammar* open_g=NULL; glrmask_module* open_m=NULL;
        CHECK(glrmask_grammar_from_source(GLRMASK_GLRM,open,strlen(open),&open_g)==GLRMASK_OK);
        CHECK(glrmask_module_new(open_g,&open_m)==GLRMASK_OK);
        sentinel=(glrmask_constraint*)(uintptr_t)1;
        CHECK(glrmask_module_compile(open_m,vocab,NULL,&sentinel)!=GLRMASK_OK && sentinel==NULL);
        glrmask_module_free(open_m); glrmask_grammar_free(open_g);
    }
    glrmask_compile_options module_options; CHECK(glrmask_compile_options_init(&module_options,sizeof(module_options))==GLRMASK_OK);
    module_options.mask_strategy=GLRMASK_MASK_STRATEGY_FAST_MASKS;
    CHECK(glrmask_module_precompile(module,vocab,&module_options)==GLRMASK_OK);
    {
        const uint8_t z[]={'z'}; const glrmask_vocab_entry wrong_entries[]={{0,z,1}};
        glrmask_vocab* wrong=NULL; glrmask_constraint* bad=NULL;
        CHECK(glrmask_vocab_new(wrong_entries,1,&wrong)==GLRMASK_OK);
        CHECK(glrmask_module_precompile(module,wrong,&module_options)==GLRMASK_VOCAB_MISMATCH);
        CHECK(strstr(glrmask_error_message(),"vocabular")!=NULL);
        CHECK(glrmask_module_compile(module,wrong,&module_options,&bad)==GLRMASK_VOCAB_MISMATCH && bad==NULL);
        CHECK(strstr(glrmask_error_message(),"vocabular")!=NULL);
        glrmask_vocab_free(wrong);
    }
    CHECK(glrmask_module_bind_tokens(module,"missing",7,NULL)!=GLRMASK_OK);
    CHECK(glrmask_module_save(module,&saved)==GLRMASK_OK);
    const uint8_t* artifact=NULL;size_t artifact_len=0;CHECK(glrmask_bytes_data(saved,&artifact,&artifact_len)==GLRMASK_OK);
    CHECK(glrmask_module_load(artifact,artifact_len,&loaded)==GLRMASK_OK);
    glrmask_bytes_free(saved);glrmask_module_free(module);
    const glrmask_expr_operand operand={1,loaded};CHECK(glrmask_expr_choice(&operand,1,&expr)==GLRMASK_OK);
    {
        const char open[]="start s extern grammar item nt s = item";
        glrmask_grammar* parent=NULL; glrmask_module* outer=NULL; glrmask_expr* nested=NULL;
        CHECK(glrmask_grammar_from_source(GLRMASK_GLRM,open,strlen(open),&parent)==GLRMASK_OK);
        const glrmask_expr_operand inner={GLRMASK_OPERAND_EXPR,expr};
        CHECK(glrmask_expr_sequence(&inner,1,&nested)==GLRMASK_OK);
        glrmask_expr* cases[]={expr,nested};
        for (size_t i=0;i<2;i++) {
            glrmask_grammar* bad=(glrmask_grammar*)(uintptr_t)1;
            CHECK(glrmask_grammar_from_expr(cases[i],&bad)==GLRMASK_INVALID_INPUT && bad==NULL);
            CHECK(strcmp(glrmask_error_message(),"a Grammar expression cannot contain a Module")==0);
            CHECK(glrmask_module_new(parent,&outer)==GLRMASK_OK);
            CHECK(glrmask_module_bind_expr(outer,"item",4,cases[i])==GLRMASK_OK);
            glrmask_module_free(outer);
        }
        glrmask_expr_free(nested);
        const uint32_t id=5; glrmask_token_set* tokens=NULL; glrmask_expr* leaf=NULL;
        CHECK(glrmask_token_set_new(&id,1,&tokens)==GLRMASK_OK);
        const glrmask_expr_operand token_operand={GLRMASK_OPERAND_TOKEN_SET,tokens};
        CHECK(glrmask_expr_sequence(&token_operand,1,&leaf)==GLRMASK_OK);
        glrmask_grammar* bad=NULL;
        CHECK(glrmask_grammar_from_expr(leaf,&bad)==GLRMASK_INVALID_INPUT && bad==NULL);
        CHECK(glrmask_module_new(parent,&outer)==GLRMASK_OK);
        CHECK(glrmask_module_bind_expr(outer,"item",4,leaf)==GLRMASK_OK);
        glrmask_module_free(outer);glrmask_expr_free(leaf);glrmask_token_set_free(tokens);
        glrmask_grammar_free(parent);
    }
    glrmask_expr_free(expr);
    glrmask_expr* literal=NULL;CHECK(glrmask_expr_literal((const uint8_t*)"x",1,&literal)==GLRMASK_OK);glrmask_expr_free(literal);
    glrmask_constraint* canonical=NULL;CHECK(glrmask_module_compile(loaded,vocab,NULL,&canonical)==GLRMASK_OK);
    glrmask_module_free(loaded);glrmask_constraint_free(canonical);
    CHECK(glrmask_constraint_start(constraint,&state)==GLRMASK_OK);
    glrmask_constraint_free(constraint);glrmask_grammar_free(grammar);glrmask_vocab_free(vocab);
    uint32_t words[3]={123,123,123};
    CHECK(glrmask_state_fill_mask(state,words,0)==GLRMASK_BUFFER_TOO_SMALL);CHECK(words[0]==123);
    CHECK(glrmask_state_fill_mask(state,(uint32_t*)((uint8_t*)words+1),1)==GLRMASK_INVALID_INPUT);
    CHECK(glrmask_state_fill_mask(state,NULL,1)==GLRMASK_INVALID_INPUT);
    CHECK(glrmask_state_fill_mask(state,words,3)==GLRMASK_OK);
    CHECK(words[0]==(UINT32_C(1)|UINT32_C(0x80000000)) && words[1]==0 && words[2]==0);
    uint32_t terminated=0;
    CHECK(glrmask_state_is_terminated(state,&terminated)==GLRMASK_OK);CHECK(terminated==0);
    CHECK(glrmask_state_commit_token(state,31,&accepted)==GLRMASK_OK && accepted==1);
    uint32_t accepting=0;CHECK(glrmask_state_is_accepting(state,&accepting)==GLRMASK_OK);CHECK(accepting==1);
    CHECK(glrmask_state_is_terminated(state,&terminated)==GLRMASK_OK);CHECK(terminated==1);
    glrmask_state* copy=NULL;CHECK(glrmask_state_clone(state,&copy)==GLRMASK_OK);
    glrmask_state_free(state);CHECK(glrmask_state_is_accepting(copy,&accepting)==GLRMASK_OK && accepting==1);
    CHECK(glrmask_state_is_terminated(copy,&terminated)==GLRMASK_OK && terminated==1);
    accepted=99; CHECK(glrmask_state_commit_token(NULL,0,&accepted)==GLRMASK_INVALID_INPUT && accepted==0);CHECK(strlen(glrmask_error_message())>0);
    CHECK(glrmask_state_commit_token(copy,0,&accepted)==GLRMASK_OK && accepted==0);CHECK(strlen(glrmask_error_message())==0);
    accepted=99; CHECK(glrmask_state_commit_bytes(NULL,NULL,0,&accepted)==GLRMASK_INVALID_INPUT && accepted==0);CHECK(strlen(glrmask_error_message())>0);
    glrmask_state_free(copy);glrmask_state_free(NULL);
    const char bad_utf8[]={(char)255};grammar=(glrmask_grammar*)(uintptr_t)1;
    CHECK(glrmask_grammar_from_source(GLRMASK_LARK,bad_utf8,1,&grammar)==GLRMASK_INVALID_INPUT);CHECK(grammar==NULL);
    CHECK(glrmask_constraint_load(NULL,1,NULL,&constraint)==GLRMASK_INVALID_INPUT);CHECK(constraint==NULL);
    const uint8_t* invalid_data=(const uint8_t*)(uintptr_t)1; size_t invalid_len=123;
    CHECK(glrmask_bytes_data(NULL,&invalid_data,&invalid_len)==GLRMASK_INVALID_INPUT);
    CHECK(invalid_data==NULL && invalid_len==0);
    {
        const size_t lengths[]={0,33,63,64};
        for (size_t i=0;i<4;i++){
            size_t n=lengths[i],w=(n+31)/32,cn=99,sn=99;
            glrmask_vocab_entry e={(uint32_t)(n?n-1:0),a,1};
            CHECK(glrmask_vocab_new(n?&e:NULL,n?1:0,&vocab)==GLRMASK_OK);
            CHECK(glrmask_grammar_from_source(GLRMASK_LARK,source,strlen(source),&grammar)==GLRMASK_OK);
            CHECK(glrmask_grammar_compile(grammar,vocab,NULL,&constraint)==GLRMASK_OK);
            CHECK(glrmask_constraint_mask_bit_len(constraint,&cn)==GLRMASK_OK && cn==n);
            CHECK(glrmask_constraint_start(constraint,&state)==GLRMASK_OK);
            CHECK(glrmask_state_mask_bit_len(state,&sn)==GLRMASK_OK && sn==n);
            uint32_t out[4]={123,123,123,123};
            if(w){CHECK(glrmask_state_fill_mask(state,out,w-1)==GLRMASK_BUFFER_TOO_SMALL);CHECK(out[0]==123);}
            CHECK(glrmask_state_fill_mask(state,out,w)==GLRMASK_OK);
            if(n) CHECK(out[(n-1)/32] & (UINT32_C(1)<<((n-1)%32)));
            CHECK(glrmask_state_fill_mask(state,out,w+2)==GLRMASK_OK);
            CHECK(out[w]==0 && out[w+1]==0);
            if(n){CHECK(glrmask_state_commit_token(state,(uint32_t)(n-1),&accepted)==GLRMASK_OK && accepted==1);CHECK(glrmask_state_fill_mask(state,out,w)==GLRMASK_OK);for(size_t j=0;j<w;j++)CHECK(out[j]==0);}
            glrmask_state_free(state);glrmask_constraint_free(constraint);glrmask_grammar_free(grammar);glrmask_vocab_free(vocab);
        }
    }
    puts("C ABI regression checks passed");return 0;
}
