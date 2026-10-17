#include <glrmask.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    glrmask_vocab *v=NULL;glrmask_grammar *g=NULL;glrmask_constraint *c=NULL;glrmask_state *s=NULL;
    const uint8_t a[]={'a'};const glrmask_vocab_entry e={0,a,1};const char *source="start ::= \"a\"";
    uint32_t mask=0,accepting=0;
    if(glrmask_vocab_new(&e,1,NULL,0,&v) || glrmask_grammar_from_source(0,source,strlen(source),NULL,&g)
      || glrmask_grammar_compile(g,v,NULL,&c) || glrmask_constraint_start(c,&s)
      || glrmask_state_fill_mask(s,&mask,1) || mask!=1 || glrmask_state_commit_token(s,0)
      || glrmask_state_is_accepting(s,&accepting) || !accepting) {
        fprintf(stderr,"Installed C consumer failed: %s\n",glrmask_error_message());return 1;
    }
    glrmask_state_free(s);glrmask_constraint_free(c);glrmask_grammar_free(g);glrmask_vocab_free(v);
    puts("Installed C consumer passed");return 0;
}
