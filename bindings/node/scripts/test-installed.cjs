'use strict';
// Install the thin root tarball plus its platform tarball in an unrelated
// directory, with lifecycle scripts disabled, then run both module systems,
// real TypeScript consumers, and the missing/wrong-platform error paths.
//
// Usage:
//   node scripts/test-installed.cjs ROOT_TGZ [PLATFORM_TGZ] [WORK_DIRECTORY]
//   NPM_OFFLINE=1 appends --offline to the consumer install.
//
// With only ROOT_TGZ, the install must succeed but requiring the package must
// fail with the explicit missing-native-binary error (the negative case: a
// consumer that skipped the optional platform package gets a hard error, not
// a silent fallback). With both tarballs, ESM+CJS+tsc must pass and the
// wrong-platform/unsupported-platform overrides must fail explicitly.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
let [rootTarball, second, third] = process.argv.slice(2);
if (!rootTarball) throw new Error('Usage: node scripts/test-installed.cjs ROOT_TGZ [PLATFORM_TGZ] [WORK_DIRECTORY]');
let platformTarball = null, destination;
if (second && /\.tgz$/i.test(second)) { platformTarball = second; destination = third; }
else destination = second;
const rootArtifact = path.resolve(rootTarball);
if (!fs.statSync(rootArtifact).isFile()) throw new Error('Root package tarball is missing');
let platformArtifact = null;
if (platformTarball) {
  platformArtifact = path.resolve(platformTarball);
  if (!fs.statSync(platformArtifact).isFile()) throw new Error('Platform package tarball is missing');
}
const work = destination ? path.resolve(destination) : fs.mkdtempSync(path.join(os.tmpdir(), 'glrmask-installed-'));
if (destination) {
  if (fs.existsSync(work)) throw new Error('Use a fresh installation-test directory');
  fs.mkdirSync(work, {recursive: true});
}
const candidates = [process.env.npm_execpath,
  path.join(path.dirname(process.execPath), 'node_modules/npm/bin/npm-cli.js'),
  path.resolve(path.dirname(process.execPath), '../lib/node_modules/npm/bin/npm-cli.js')].filter(Boolean);
const npm = candidates.find(p => fs.existsSync(p));
if (!npm) throw new Error('Cannot locate npm-cli.js; run this script through npm or an ordinary Node installation');
const offline = process.env.NPM_OFFLINE === '1' ? ['--offline'] : [];
function run(program, args, options = {}) {
  console.log('+', program, ...args);
  const r = spawnSync(program, args, {cwd: work, stdio: 'inherit', timeout: 240000, ...options});
  if (r.error) throw r.error;
  if (r.status !== 0) throw new Error(`Installed consumer failed with exit ${r.status}`);
  return r;
}
function runCapture(program, args, options = {}) {
  const r = spawnSync(program, args, {cwd: work, stdio: 'pipe', encoding: 'utf8', timeout: 120000, ...options});
  if (r.error) throw r.error;
  return r;
}
fs.writeFileSync(path.join(work, 'package.json'), JSON.stringify({name: 'glrmask-installed-consumer', private: true, type: 'module'}, null, 2) + '\n');
const toInstall = platformArtifact ? [rootArtifact, platformArtifact] : [rootArtifact];
run(process.execPath, [npm, 'install', '--ignore-scripts', '--no-audit', '--no-fund', ...offline, ...toInstall]);
const installed = path.join(work, 'node_modules/@glrmask/node');
if (!fs.realpathSync(installed).startsWith(fs.realpathSync(work) + path.sep)) {
  throw new Error('Installation resolved outside the isolated consumer');
}
const runtime = {...process.env};
delete runtime.NODE_PATH; delete runtime.NODE_OPTIONS;
runtime.PATH = [path.dirname(process.execPath), process.env.SystemRoot ? path.join(process.env.SystemRoot, 'System32') : ''].filter(Boolean).join(path.delimiter);
if (!platformArtifact) {
  // Negative case: thin root without its platform package must fail loudly.
  const probe = runCapture(process.execPath, ['--input-type=module', '-e', "await import('@glrmask/node')"], {env: runtime});
  if (probe.status === 0) throw new Error('Requiring the thin root without a platform package unexpectedly succeeded');
  const stderr = `${probe.stderr}${probe.stdout}`;
  if (!/native binary glrmask\..+\.node was not found/.test(stderr) || !/@glrmask\/node-/.test(stderr)) {
    throw new Error(`Missing-platform error is not explicit:\n${stderr}`);
  }
  console.log('Missing-platform negative case passed: explicit error names the optional platform package');
  fs.writeFileSync(path.join(work, 'verification.json'), JSON.stringify({package: rootArtifact, missingPlatformError: true, noLifecycleScripts: true}, null, 2) + '\n');
  console.log(`Isolated installed package passed: ${work}`);
  return;
}
for (const name of ['decode.mjs', 'json.cjs']) {
  const source = path.join(installed, 'examples', name);
  fs.copyFileSync(source, path.join(work, name));
}
run(process.execPath, ['decode.mjs'], {env: runtime});
run(process.execPath, ['json.cjs'], {env: runtime});
// Type declarations shipped in the thin root are checked with the repository's
// locked TypeScript compiler and @types/node copies, so this step needs no
// registry access.
for (const file of ['types-consumer.cts', 'types-consumer.mts', 'tsconfig.json']) {
  fs.copyFileSync(path.join(root, 'tests', file), path.join(work, file));
}
fs.mkdirSync(path.join(work, 'node_modules', '@types'), {recursive: true});
fs.cpSync(path.join(root, 'node_modules', '@types', 'node'), path.join(work, 'node_modules', '@types', 'node'), {recursive: true});
fs.cpSync(path.join(root, 'node_modules', 'undici-types'), path.join(work, 'node_modules', 'undici-types'), {recursive: true});
const tsc = path.join(root, 'node_modules/typescript/bin/tsc');
if (!fs.existsSync(tsc)) throw new Error('Locked TypeScript compiler is missing; run npm ci in bindings/node first');
run(process.execPath, [tsc, '-p', 'tsconfig.json'], {env: runtime});
// Wrong-platform override must name the other platform package explicitly.
const {currentTarget} = require('../platform.cjs');
const current = currentTarget();
const other = current.startsWith('darwin-') ? 'linux-x64-gnu' : 'darwin-arm64';
const wrong = runCapture(process.execPath,
  ['--input-type=module', '-e', "await import('@glrmask/node')"],
  {env: {...runtime, GLRMASK_TEST_TARGET: other}});
if (wrong.status === 0) throw new Error('Wrong-platform import unexpectedly succeeded');
if (!new RegExp(`@glrmask/node-${other}`).test(`${wrong.stderr}${wrong.stdout}`)) {
  throw new Error(`Wrong-platform error does not name @glrmask/node-${other}:\n${wrong.stderr}${wrong.stdout}`);
}
// Unsupported target must list the supported set.
const bogus = runCapture(process.execPath,
  ['--input-type=module', '-e', "await import('@glrmask/node')"],
  {env: {...runtime, GLRMASK_TEST_TARGET: 'plan9-mips-unknown'}});
if (bogus.status === 0) throw new Error('Unsupported-platform import unexpectedly succeeded');
if (!/Unsupported native platform/.test(`${bogus.stderr}${bogus.stdout}`)) {
  throw new Error(`Unsupported-platform error is not explicit:\n${bogus.stderr}${bogus.stdout}`);
}
fs.writeFileSync(path.join(work, 'verification.json'), JSON.stringify(
  {package: rootArtifact, platform: platformArtifact, cjs: true, esm: true, typescript: 'locked-local', wrongPlatformError: true, unsupportedPlatformError: true, noLifecycleScripts: true, noCompilerOnRuntimePath: true}, null, 2) + '\n');
console.log(`Isolated installed package passed: ${work}`);
