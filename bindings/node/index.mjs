import api from './index.cjs';
export const {
  Vocab, TokenSet, Grammar, Module, CompileOptions, Expr, Constraint, ConstraintState,
  MaskStrategy, BoundaryMaskStrategy, ParserBackend, GLRMaskError, version
} = api;
export default api;
