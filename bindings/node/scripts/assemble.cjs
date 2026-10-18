'use strict';
// This script used to consolidate per-platform CI artifacts into one
// self-contained fat npm tarball. That distribution is retired: the thin
// @glrmask/node root no longer embeds native binaries, and a combined
// native-manifest.json with binaries inside the root is rejected by
// scripts/check-package.cjs.
//
// Use `node scripts/stage-platforms.cjs --input ARTIFACT_DIR --output STAGE_DIR`
// to split CI artifacts into staged per-platform packages plus a combined
// platform-manifest.json for the thin root. See bindings/node/README.md.
throw new Error(
  'assemble.cjs no longer builds a fat all-platform tarball. ' +
  'Run node scripts/stage-platforms.cjs --input ARTIFACT_DIR --output STAGE_DIR instead.');
