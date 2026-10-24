import api from './index.cjs';
export const {
  Vocab, TokenSet, Grammar, Module, CompileOptions, Expr, Constraint, ConstraintState,
  MaskStrategy, BoundaryMaskStrategy, ParserBackend, GlrMaskError, version
} = api;
export default api;
