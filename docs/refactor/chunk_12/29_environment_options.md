# Commit environment options

Commit-local environment switches remain isolated in `options.rs`. This keeps operational code readable and makes it clear which flags affect semantics-preserving implementation choices.

The most important rule is that an environment flag may select an implementation strategy, diagnostic assertion, or validation mode. It must not silently change the accepted language.

Future publication cleanup should document every environment variable in one public configuration document and hide unstable flags behind a clearly marked diagnostics feature.
