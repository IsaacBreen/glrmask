'use strict';
// Validate the thin @glrmask/node root or a staged per-platform package.
//
// Root (default, run from `prepack`): the thin package must contain no
// native binaries and no legacy combined native-manifest.json. Its
// optionalDependencies must pin exactly the eight platform packages from
// platform.cjs at the root version, and a bundled platform-manifest.json, if
// present, must describe all eight checksummed targets. A present native
// binary or legacy manifest fails: the fat-tarball format is retired.
//
//   node scripts/check-package.cjs [--all]
//
// `--all` additionally requires platform-manifest.json with every release
// target (CI thin-root qualification). Without it, a missing manifest is a
// warning for local development packs; the release workflow always stages one.
//
// Platform directory:
//
//   node scripts/check-package.cjs --platform-dir DIR --tag TAG
//
// validates a staged `@glrmask/node-<tag>` directory: package.json identity,
// version alignment with the root, npm os/cpu/libc routing, file list, no
// dependencies or install scripts, and the single binary against its
// single-target native-manifest.json fragment (name, release profile,
// size, sha256).
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {targets, releaseTargets, PLATFORM_PACKAGES, platformPackageFor} = require('../platform.cjs');
const pkg = require('../package.json');
const root = path.resolve(__dirname, '..');
const cli = process.argv.slice(2);
function sha256(data) { return crypto.createHash('sha256').update(data).digest('hex'); }
function fail(message) { throw new Error(message); }
function checkDeclarations() {
  const canonical = "export * from './index.cjs';\n";
  for (const file of ['index.d.mts', 'index.d.ts']) {
    if (fs.readFileSync(path.join(root, file), 'utf8') !== canonical) fail(`Stale declaration copy ${file}`);
  }
  for (const file of ['README.md', 'LICENSE-MIT', 'LICENSE-APACHE', 'NOTICE-LARK', 'index.mjs', 'index.cjs', 'loader.cjs', 'platform.cjs']) {
    if (fs.statSync(path.join(root, file)).size === 0) fail(`Empty required package file ${file}`);
  }
}
function checkBinaryAbsent() {
  // A developer-built glrmask.<tag>.node may sit next to the sources (it is
  // gitignored and matched by no `files` entry, so `npm pack` excludes it).
  // What must never happen is the pack including a binary or the retired fat
  // manifest, so the `files` allowlist itself is the enforcement point.
  for (const pattern of pkg.files) {
    if (/\.node$/.test(pattern) || pattern === 'native-manifest.json' || pattern === 'glrmask.*.node') {
      fail(`Thin root files must not embed native binaries: ${pattern}`);
    }
  }
  if (pkg.files.includes('native-manifest.json')) fail('Legacy combined native-manifest.json is retired');
  if (fs.existsSync(path.join(root, 'native-manifest.json'))) {
    // A local `npm run build` records its single-target fragment here for CI
    // upload. It is gitignored, matched by no `files` entry, and never
    // packed. Only a combined/multi-target legacy manifest is fatal.
    const legacy = JSON.parse(fs.readFileSync(path.join(root, 'native-manifest.json'), 'utf8'));
    const legacyTags = Object.keys(legacy.targets || {});
    if (legacy.kind === 'platform-manifest' || legacyTags.length !== 1) {
      fail('Legacy combined native-manifest.json is retired; the thin root ships platform-manifest.json instead');
    }
    console.warn('Note: local single-target native-manifest.json present (dev build record, excluded from pack).');
  }
}
function checkOptionalDependencies() {
  const optional = pkg.optionalDependencies || {};
  const names = Object.keys(optional).sort();
  const expected = releaseTargets.map(t => PLATFORM_PACKAGES[t].package).sort();
  if (JSON.stringify(names) !== JSON.stringify(expected)) {
    fail(`optionalDependencies must be exactly the platform packages: ${expected.join(', ')}`);
  }
  for (const tag of releaseTargets) {
    if (optional[PLATFORM_PACKAGES[tag].package] !== pkg.version) {
      fail(`Optional platform package ${PLATFORM_PACKAGES[tag].package} must be pinned to ${pkg.version}`);
    }
  }
  if (pkg.dependencies && Object.keys(pkg.dependencies).length) fail('Thin root must have no runtime dependencies');
  for (const section of ['scripts']) {
    for (const hook of ['install', 'postinstall', 'preinstall', 'prepublish']) {
      if (pkg[section] && pkg[section][hook]) fail(`Refusing installer hook scripts.${hook}: no install scripts or compiler`);
    }
  }
}
function checkManifest(requireAll) {
  const manifestPath = path.join(root, 'platform-manifest.json');
  if (!fs.existsSync(manifestPath)) {
    if (requireAll || cli.includes('--all')) fail('Missing platform-manifest.json for all-platform qualification');
    console.warn('Warning: no platform-manifest.json staged; local development pack only. CI stages one.');
    return;
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  if (manifest.schema !== 1 || manifest.kind !== 'platform-manifest' || manifest.version !== pkg.version || manifest.napi !== 8) {
    fail('Invalid or stale platform-manifest.json');
  }
  const tags = Object.keys(manifest.targets).sort();
  if (JSON.stringify(tags) !== JSON.stringify([...releaseTargets].sort())) {
    fail(`platform-manifest.json must list all release targets: ${releaseTargets.join(', ')}`);
  }
  for (const tag of releaseTargets) {
    const item = manifest.targets[tag];
    const routing = platformPackageFor(tag);
    if (item.package !== routing.package || item.file !== `glrmask.${tag}.node` || item.profile !== 'release') {
      fail(`Invalid platform-manifest entry for ${tag}`);
    }
    if (typeof item.size !== 'number' || !/^[0-9a-f]{64}$/.test(item.sha256)) fail(`Invalid provenance digest for ${tag}`);
    const rustTargets = Object.keys(targets).filter(t => targets[t] === tag);
    if (rustTargets.length !== 1 || item.rustTarget !== rustTargets[0]) fail(`Invalid rustTarget for ${tag}`);
  }
  console.log(`Validated platform-manifest.json: ${releaseTargets.join(', ')}`);
}
function checkPlatformDir(dir, tag) {
  const routing = platformPackageFor(tag);
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
  if (manifest.name !== routing.package) fail(`Platform package name must be ${routing.package}`);
  if (manifest.version !== pkg.version) {
    fail(`Platform package version ${manifest.version} must align with root ${pkg.version}`);
  }
  if (manifest.license !== 'MIT OR Apache-2.0') fail(`Unexpected platform license for ${tag}`);
  for (const [field, expected] of [['os', routing.os], ['cpu', routing.cpu]]) {
    if (JSON.stringify(manifest[field]) !== JSON.stringify(expected)) fail(`Platform ${tag} must declare ${field} ${JSON.stringify(expected)}`);
  }
  const expectedLibc = routing.libc || null;
  if (JSON.stringify(manifest.libc || null) !== JSON.stringify(expectedLibc)) {
    fail(`Platform ${tag} must declare libc ${JSON.stringify(expectedLibc)}`);
  }
  if (manifest.dependencies || manifest.optionalDependencies || manifest.scripts) {
    fail(`Platform ${tag} must have no dependencies or scripts`);
  }
  const fragment = JSON.parse(fs.readFileSync(path.join(dir, 'native-manifest.json'), 'utf8'));
  if (fragment.schema !== 1 || fragment.version !== pkg.version || fragment.napi !== 8) fail(`Invalid manifest fragment for ${tag}`);
  const entries = Object.keys(fragment.targets);
  if (entries.length !== 1 || entries[0] !== tag) fail(`Manifest fragment for ${tag} must list only ${tag}`);
  const item = fragment.targets[tag];
  if (item.file !== `glrmask.${tag}.node` || item.profile !== 'release') fail(`Invalid release artifact entry for ${tag}`);
  const data = fs.readFileSync(path.join(dir, item.file));
  if (data.length !== item.size || sha256(data) !== item.sha256) fail(`Native artifact checksum mismatch for ${tag}`);
  for (const file of fs.readdirSync(dir).filter(f => /^glrmask\..+\.node$/.test(f))) {
    if (file !== item.file) fail(`Unrecorded native binary ${file} in ${tag}`);
  }
  for (const file of [item.file, 'native-manifest.json', 'package.json', 'README.md', 'LICENSE-MIT', 'LICENSE-APACHE', 'NOTICE-LARK']) {
    if (!fs.existsSync(path.join(dir, file))) fail(`Missing ${file} in platform ${tag}`);
  }
  console.log(`Validated platform package ${routing.package}@${pkg.version} (${item.size} bytes, sha256 ${item.sha256.slice(0, 16)}…)`);
}
const platformDirIndex = cli.indexOf('--platform-dir');
if (platformDirIndex !== -1) {
  const tagIndex = cli.indexOf('--tag');
  if (tagIndex === -1 || !cli[tagIndex + 1]) fail('Usage: node scripts/check-package.cjs --platform-dir DIR --tag TAG');
  checkPlatformDir(path.resolve(cli[platformDirIndex + 1]), cli[tagIndex + 1]);
} else {
  checkOptionalDependencies();
  checkBinaryAbsent();
  checkDeclarations();
  checkManifest(false);
  // The loader resolves the local binary when developing; requiring it here
  // would fail on a clean checkout before `npm run build`. Native execution
  // is covered by `npm test` and the installed-tarball test instead.
  try {
    require.resolve('../loader.cjs');
    require('../platform.cjs');
  } catch (cause) { fail(`Unparseable loader/platform module: ${cause.message}`); }
  console.log(`Validated thin @glrmask/node@${pkg.version}: no binaries, ${releaseTargets.length} optional platform packages`);
}
