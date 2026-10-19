'use strict';
// Split per-platform CI artifacts into staged per-platform npm package
// directories plus one combined provenance manifest.
//
// Input: a directory tree containing one `native-manifest.json` fragment per
// CI platform job, each next to its `glrmask.<tag>.node` binary (this matches
// `actions/download-artifact` with `pattern: node-native-*`, which places
// each artifact in its own subdirectory).
//
// Output (`--output STAGE`):
//   STAGE/<tag>/                         staged platform package directory
//     package.json                       generated `@glrmask/node-<tag>`
//     glrmask.<tag>.node                 release binary (checksummed)
//     native-manifest.json               single-target fragment (provenance)
//     README.md, LICENSE-MIT, LICENSE-APACHE, NOTICE-LARK
//   STAGE/platform-manifest.json         combined manifest for the thin root
//
// Every fragment must be schema 1, match the root version, carry a `release`
// profile, and pass size/sha256 verification. All eight releaseTargets must
// be present exactly once. The old fat-tarball layout (all binaries inside
// the root package) is not produced anymore.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {targets, releaseTargets, PLATFORM_PACKAGES} = require('../platform.cjs');
const pkg = require('../package.json');
const rootDir = path.resolve(__dirname, '..');
const repoRoot = path.resolve(rootDir, '..', '..');
function usage() {
  console.error('Usage: node scripts/stage-platforms.cjs --input ARTIFACT_DIR --output STAGE_DIR [--tag SINGLE_TEST_TARGET]');
  process.exit(2);
}
const cli = process.argv.slice(2);
let input, output, selectedTag;
for (let i = 0; i < cli.length; ++i) {
  if (cli[i] === '--input' && cli[i + 1]) input = path.resolve(cli[++i]);
  else if (cli[i] === '--output' && cli[i + 1]) output = path.resolve(cli[++i]);
  else if (cli[i] === '--tag' && cli[i + 1]) selectedTag = cli[++i];
  else usage();
}
if (!input || !output) usage();
if (selectedTag && !releaseTargets.includes(selectedTag)) throw new Error(`Unknown platform ${selectedTag}`);
// Partial staging exists only for per-runner installation tests. Production
// aggregation omits --tag and still requires every target; --all validation
// rejects a partial manifest if someone attempts to release it as the root.
const wanted = selectedTag ? [selectedTag] : releaseTargets;
function sha256(data) { return crypto.createHash('sha256').update(data).digest('hex'); }
const collected = new Map();
function walk(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const location = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(location);
    else if (entry.name === 'native-manifest.json') {
      const manifest = JSON.parse(fs.readFileSync(location, 'utf8'));
      if (manifest.schema !== 1 || manifest.version !== pkg.version || manifest.napi !== 8) {
        throw new Error(`Mismatched manifest ${location}: expected schema 1, version ${pkg.version}, napi 8`);
      }
      for (const [tag, item] of Object.entries(manifest.targets)) {
        if (!releaseTargets.includes(tag)) throw new Error(`Unexpected CI target ${tag} in ${location}`);
        if (collected.has(tag)) throw new Error(`Duplicate CI target ${tag}`);
        if (item.file !== `glrmask.${tag}.node` || item.profile !== 'release') {
          throw new Error(`Invalid release artifact entry for ${tag} in ${location}`);
        }
        const data = fs.readFileSync(path.join(path.dirname(location), item.file));
        if (data.length !== item.size || sha256(data) !== item.sha256) {
          throw new Error(`Corrupt CI artifact ${tag}: size/sha256 mismatch`);
        }
        collected.set(tag, {data, item});
      }
    }
  }
}
walk(input);
const missing = wanted.filter(t => !collected.has(t));
if (missing.length) throw new Error(`Missing CI targets: ${missing.join(', ')}`);
fs.mkdirSync(output, {recursive: true});
const combined = {schema: 1, kind: 'platform-manifest', version: pkg.version, napi: 8, targets: {}};
for (const tag of wanted) {
  const {data, item} = collected.get(tag);
  const routing = PLATFORM_PACKAGES[tag];
  const rustTarget = Object.keys(targets).find(t => targets[t] === tag);
  const dir = path.join(output, tag);
  fs.mkdirSync(dir, {recursive: true});
  fs.writeFileSync(path.join(dir, item.file), data);
  const fragment = {schema: 1, version: pkg.version, napi: 8, targets: {[tag]: item}};
  fs.writeFileSync(path.join(dir, 'native-manifest.json'), JSON.stringify(fragment, null, 2) + '\n');
  const manifest = {
    name: routing.package,
    version: pkg.version,
    description: `Prebuilt native binary for @glrmask/node on ${tag} (${rustTarget})`,
    license: 'MIT OR Apache-2.0',
    os: routing.os,
    cpu: routing.cpu,
    ...(routing.libc ? {libc: routing.libc} : {}),
    files: [item.file, 'native-manifest.json', 'README.md', 'LICENSE-MIT', 'LICENSE-APACHE', 'NOTICE-LARK'],
    engines: {node: '>=22'},
    repository: {type: 'git', url: 'git+https://github.com/IsaacBreen/glrmask.git', directory: 'bindings/node'},
    homepage: 'https://glrmask.com/api/javascript/',
    bugs: {url: 'https://github.com/IsaacBreen/glrmask/issues'},
  };
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify(manifest, null, 2) + '\n');
  for (const license of ['LICENSE-MIT', 'LICENSE-APACHE']) {
    fs.copyFileSync(path.join(repoRoot, license), path.join(dir, license));
  }
  fs.copyFileSync(path.join(repoRoot, 'crates/glrmask-grammar/NOTICE-LARK'), path.join(dir, 'NOTICE-LARK'));
  fs.writeFileSync(path.join(dir, 'README.md'),
    `# ${routing.package}\n\nPrebuilt native binary for ` +
    `[@glrmask/node](https://glrmask.com/api/javascript/) version ${pkg.version} on ` +
    `\`${tag}\` (${rustTarget}). This package is installed automatically as an ` +
    `optional dependency of \`@glrmask/node\`; do not depend on it directly.\n` +
    `It contains no JavaScript runtime and runs no install scripts.\n`);
  combined.targets[tag] = {
    package: routing.package, file: item.file, rustTarget,
    profile: item.profile, size: item.size, sha256: item.sha256,
  };
}
fs.writeFileSync(path.join(output, 'platform-manifest.json'), JSON.stringify(combined, null, 2) + '\n');
console.log(`Staged ${wanted.length} platform packages in ${output}`);
