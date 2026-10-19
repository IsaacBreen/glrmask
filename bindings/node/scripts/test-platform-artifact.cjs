'use strict';
// Qualify a real single-platform native artifact without downloading binaries
// or repacking a universal package. Full-release aggregation still requires all
// eight targets; this helper exercises the installed loader on each real OS.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const {currentTarget} = require('../platform.cjs');
const pkg = require('../package.json');
const tag = currentTarget();
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'glrmask-single-platform-'));
const npm = [process.env.npm_execpath,
  path.join(path.dirname(process.execPath), 'node_modules/npm/bin/npm-cli.js'),
  path.resolve(path.dirname(process.execPath), '../lib/node_modules/npm/bin/npm-cli.js')]
  .filter(Boolean).find(p => fs.existsSync(p));
if (!npm) throw new Error('Cannot locate npm-cli.js');
function run(args, cwd = root, env = process.env) {
  const result = spawnSync(process.execPath, args, {cwd, env, stdio: 'inherit', timeout: 240000});
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`Platform install qualification failed: exit ${result.status}`);
}
try {
  const input = path.join(work, 'input');
  const stage = path.join(work, 'stage');
  const packs = path.join(work, 'packs');
  fs.mkdirSync(input); fs.mkdirSync(packs);
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'native-manifest.json'), 'utf8'));
  if (!manifest.targets[tag]) throw new Error(`Missing actual host artifact ${tag}`);
  const binary = `glrmask.${tag}.node`;
  fs.copyFileSync(path.join(root, binary), path.join(input, binary));
  fs.writeFileSync(path.join(input, 'native-manifest.json'), JSON.stringify({
    ...manifest, targets: {[tag]: manifest.targets[tag]},
  }));
  run([path.join(__dirname, 'stage-platforms.cjs'), '--input', input, '--output', stage, '--tag', tag]);
  run([path.join(__dirname, 'check-package.cjs'), '--platform-dir', path.join(stage, tag), '--tag', tag]);
  run([npm, 'pack', '--pack-destination', packs]);
  run([npm, 'pack', '--pack-destination', packs], path.join(stage, tag));
  const rootTgz = path.join(packs, `glrmask-node-${pkg.version}.tgz`);
  const platformTgz = path.join(packs, `glrmask-node-${tag}-${pkg.version}.tgz`);
  // Seed each install with the local matching-platform tarball. All other
  // optional platform packages remain absent; no registry access is required.
  run([path.join(__dirname, 'test-installed.cjs'), rootTgz, platformTgz,
    path.join(work, 'consumer')], root, {...process.env, NPM_OFFLINE: '1'});
  run([path.join(__dirname, 'test-installed.cjs'), rootTgz,
    path.join(work, 'missing-consumer')], root, {...process.env, NPM_OFFLINE: '1'});
  const report = {platform: tag, version: pkg.version, nativeBytes: manifest.targets[tag].size,
    rootTarballBytes: fs.statSync(rootTgz).size, platformTarballBytes: fs.statSync(platformTgz).size,
    installedConsumer: true, missingOptionalNegative: true, noInstallScripts: true};
  console.log(JSON.stringify(report));
  if (process.env.RUNNER_TEMP) fs.writeFileSync(path.join(process.env.RUNNER_TEMP,
    'platform-install-report.json'), JSON.stringify(report, null, 2) + '\n');
} finally { fs.rmSync(work, {recursive: true, force: true}); }