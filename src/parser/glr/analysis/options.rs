// Grammar-analysis profiling policy.
//
// Kept separate from normalization code so the analysis transformations can be
// read as language-preserving rewrites rather than as environment-variable
// dispatch.

pub(crate) fn analysis_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}
