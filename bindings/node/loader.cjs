'use strict';
// Resolve the platform native binary for the thin @glrmask/node package.
//
// Resolution order:
//   1. A developer-built `glrmask.<tag>.node` next to this file (source
//      checkout after `npm run build`). Local files always win so a source
//      build is testable without publishing platform packages.
//   2. The installed optional platform package `@glrmask/node-<tag>`,
//      which npm installs automatically on a matching OS/arch/libc. The
//      binary is required by path so a version drift between the thin root
//      and the platform package is caught by the version check below.
//
// There is no download, compile or subprocess fallback. A missing binary is a
// hard error that names the expected optional package and the likely cause
// (unsupported platform, `--no-optional`/`--omit=optional` install, or a
// version skew between root and platform package).
const fs = require('node:fs');
const path = require('node:path');
const {currentTarget, platformPackageFor} = require('./platform.cjs');
const pkg = require('./package.json');
const target = currentTarget();
const entry = platformPackageFor(target);
const filename = `glrmask.${target}.node`;
function tryLocal() {
  const location = path.join(__dirname, filename);
  if (fs.existsSync(location)) return location;
  return null;
}
function tryPlatformPackage() {
  try {
    return require.resolve(`${entry.package}/${filename}`, {paths: [__dirname]});
  } catch (cause) {
    if (cause && cause.code === 'MODULE_NOT_FOUND') return null;
    throw cause;
  }
}
let location = tryLocal();
let origin = 'local build';
if (!location) {
  location = tryPlatformPackage();
  origin = `optional platform package ${entry.package}`;
}
if (!location) {
  throw new Error(
    `GLRMask native binary ${filename} was not found. ` +
    `Install the optional platform package ${entry.package}@${pkg.version} for ${target} ` +
    `(npm installs it automatically with @glrmask/node@${pkg.version} on a matching platform). ` +
    `If installation used --no-optional or --omit=optional, reinstall without that flag. ` +
    `If your platform has no entry in the supported list, build from the complete GLRMask ` +
    `source checkout with npm run build in bindings/node. ` +
    `No Python or subprocess fallback is used.`);
}
let addon;
try { addon = require(location); }
catch (cause) {
  throw new Error(
    `Could not load GLRMask native binary for ${target} from ${origin} (${location}): ${cause.message}. ` +
    `Reinstall ${origin === 'local build' ? 'by rebuilding' : `${entry.package}@${pkg.version}`} for ${target}.`,
    {cause});
}
if (addon.version() !== pkg.version) {
  throw new Error(
    `GLRMask JavaScript/native version mismatch: root ${pkg.version} versus native binary ${addon.version()} ` +
    `from ${origin} (${location}). Reinstall matching versions of @glrmask/node and ${entry.package}.`);
}
if (origin !== 'local build') {
  const platformPkgPath = require.resolve(`${entry.package}/package.json`, {paths: [__dirname]});
  const platformPkg = require(platformPkgPath);
  if (platformPkg.version !== pkg.version) {
    throw new Error(
      `GLRMask platform package version mismatch: ${entry.package}@${platformPkg.version} ` +
      `versus @glrmask/node@${pkg.version}. Reinstall matching versions.`);
  }
}
// Local development build without the platform package installed: the
// binary version check above is sufficient.
module.exports = addon;
