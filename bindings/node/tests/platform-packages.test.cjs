'use strict';
// Pure-JavaScript qualification for the thin-root plus platform-package
// distribution. No native binary is required: binaries are synthesized as
// random bytes so manifest/sha/size/provenance logic is exercised exactly,
// and loader error paths run in child processes with GLRMASK_TEST_TARGET.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');
const platform = require('../platform.cjs');
const pkg = require('../package.json');
const root = path.resolve(__dirname, '..');

const EXPECTED = {
  'win32-x64-msvc': {package: '@glrmask/node-win32-x64-msvc', os: ['win32'], cpu: ['x64']},
  'win32-arm64-msvc': {package: '@glrmask/node-win32-arm64-msvc', os: ['win32'], cpu: ['arm64']},
  'darwin-x64': {package: '@glrmask/node-darwin-x64', os: ['darwin'], cpu: ['x64']},
  'darwin-arm64': {package: '@glrmask/node-darwin-arm64', os: ['darwin'], cpu: ['arm64']},
  'linux-x64-gnu': {package: '@glrmask/node-linux-x64-gnu', os: ['linux'], cpu: ['x64'], libc: ['glibc']},
  'linux-arm64-gnu': {package: '@glrmask/node-linux-arm64-gnu', os: ['linux'], cpu: ['arm64'], libc: ['glibc']},
  'linux-x64-musl': {package: '@glrmask/node-linux-x64-musl', os: ['linux'], cpu: ['x64'], libc: ['musl']},
  'linux-arm64-musl': {package: '@glrmask/node-linux-arm64-musl', os: ['linux'], cpu: ['arm64'], libc: ['musl']},
};
function npmPackDryRun(dir) {
  const candidates = [process.env.npm_execpath,
    path.join(path.dirname(process.execPath), 'node_modules/npm/bin/npm-cli.js'),
    path.resolve(path.dirname(process.execPath), '../lib/node_modules/npm/bin/npm-cli.js')].filter(Boolean);
  const cli = candidates.find(p => fs.existsSync(p));
  if (cli) return spawnSync(process.execPath, [cli, 'pack', '--dry-run'], {cwd: dir, encoding: 'utf8'});
  return spawnSync('npm', ['pack', '--dry-run'], {cwd: dir, encoding: 'utf8'});
}

test('platform taxonomy covers all supported OS/arch/libc combinations', () => {
  assert.deepEqual(Object.keys(EXPECTED).sort(), [...platform.releaseTargets].sort());
  assert.equal(platform.releaseTargets.length, 8);
  for (const [tag, routing] of Object.entries(EXPECTED)) {
    assert.deepEqual(platform.platformPackageFor(tag), routing);
    assert.deepEqual(platform.PLATFORM_PACKAGES[tag], routing);
  }
  assert.throws(() => platform.platformPackageFor('plan9-mips'), /Unsupported GLRMask native target/);
  const rustTargets = Object.keys(platform.targets);
  assert.equal(rustTargets.length, 8);
  for (const tag of platform.releaseTargets) assert.ok(Object.values(platform.targets).includes(tag), tag);
  const rustFor = tag => rustTargets.find(t => platform.targets[t] === tag);
  assert.equal(rustFor('win32-x64-msvc'), 'x86_64-pc-windows-msvc');
  assert.equal(rustFor('win32-arm64-msvc'), 'aarch64-pc-windows-msvc');
  assert.equal(rustFor('linux-x64-musl'), 'x86_64-unknown-linux-musl');
  assert.equal(rustFor('linux-arm64-musl'), 'aarch64-unknown-linux-musl');
});

test('currentTarget honors the test override and rejects unknown platforms', () => {
  assert.ok(platform.releaseTargets.includes(platform.currentTarget()));
  const run = target => spawnSync(process.execPath, ['-e', 'console.log(require("./platform.cjs").currentTarget())'],
    {cwd: root, env: {...process.env, GLRMASK_TEST_TARGET: target}, encoding: 'utf8'});
  const ok = run('darwin-arm64');
  assert.equal(ok.status, 0, ok.stderr);
  assert.equal(ok.stdout.trim(), 'darwin-arm64');
  const bad = run('plan9-mips-unknown');
  assert.notEqual(bad.status, 0);
  assert.match(`${bad.stderr}`, /Unsupported native platform/);
});

test('thin root declares only pinned platform optionalDependencies, no scripts or binaries', () => {
  assert.equal(pkg.name, '@glrmask/node');
  const optional = pkg.optionalDependencies || {};
  assert.deepEqual(Object.keys(optional).sort(), Object.values(EXPECTED).map(r => r.package).sort());
  for (const routing of Object.values(EXPECTED)) assert.equal(optional[routing.package], pkg.version);
  assert.ok(!pkg.dependencies || Object.keys(pkg.dependencies).length === 0);
  for (const hook of ['install', 'postinstall', 'preinstall', 'prepublish']) {
    assert.equal(pkg.scripts && pkg.scripts[hook], undefined, hook);
  }
  assert.ok(!pkg.files.some(f => f.includes('.node')), 'thin root files must not embed binaries');
  assert.ok(!pkg.files.includes('native-manifest.json'), 'legacy fat manifest must not ship');
  assert.ok(pkg.files.includes('platform-manifest.json'));
  assert.ok(pkg.files.includes('loader.cjs') && pkg.files.includes('platform.cjs'));
  assert.equal(pkg.main, 'index.cjs');
  assert.equal(pkg.module, 'index.mjs');
});

test('loader reports the exact missing optional package without a local binary', () => {
  const tag = platform.currentTarget().startsWith('linux-') ? 'darwin-arm64' : 'linux-x64-musl';
  const probe = spawnSync(process.execPath, ['--input-type=module', '-e', "await import('./index.cjs')"],
    {cwd: root, env: {...process.env, GLRMASK_TEST_TARGET: tag}, encoding: 'utf8'});
  assert.notEqual(probe.status, 0, 'thin root without a platform package must fail loudly');
  const output = `${probe.stderr}${probe.stdout}`;
  assert.match(output, new RegExp(`native binary glrmask\\.${tag}\\.node was not found`));
  assert.match(output, new RegExp(EXPECTED[tag].package.replace(/[/@-]/g, m => `\\${m}`)));
});

test('stage, validate and dry-pack round-trip with synthesized binaries', () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'glrmask-platforms-'));
  try {
    const input = path.join(work, 'artifacts');
    for (const tag of platform.releaseTargets) {
      const dir = path.join(input, `node-native-${tag}`);
      fs.mkdirSync(dir, {recursive: true});
      const bytes = crypto.randomBytes(1024 + tag.length);
      const file = `glrmask.${tag}.node`;
      fs.writeFileSync(path.join(dir, file), bytes);
      const rustTarget = Object.keys(platform.targets).find(t => platform.targets[t] === tag);
      fs.writeFileSync(path.join(dir, 'native-manifest.json'), JSON.stringify({
        schema: 1, version: pkg.version, napi: 8,
        targets: {[tag]: {file, rustTarget, profile: 'release', size: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex')}},
      }, null, 2) + '\n');
    }
    const stageDir = path.join(work, 'staged');
    let r = spawnSync(process.execPath, [path.join(root, 'scripts/stage-platforms.cjs'), '--input', input, '--output', stageDir],
      {encoding: 'utf8'});
    assert.equal(r.status, 0, r.stderr);
    const combined = JSON.parse(fs.readFileSync(path.join(stageDir, 'platform-manifest.json'), 'utf8'));
    assert.equal(combined.kind, 'platform-manifest');
    assert.equal(combined.version, pkg.version);
    assert.deepEqual(Object.keys(combined.targets).sort(), [...platform.releaseTargets].sort());
    for (const tag of platform.releaseTargets) {
      const dir = path.join(stageDir, tag);
      r = spawnSync(process.execPath, [path.join(root, 'scripts/check-package.cjs'), '--platform-dir', dir, '--tag', tag], {encoding: 'utf8'});
      assert.equal(r.status, 0, r.stdout + r.stderr);
      const staged = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
      assert.equal(staged.name, EXPECTED[tag].package);
      assert.equal(staged.version, pkg.version);
      assert.deepEqual(staged.os, EXPECTED[tag].os);
      assert.deepEqual(staged.cpu, EXPECTED[tag].cpu);
      assert.deepEqual(staged.libc || undefined, EXPECTED[tag].libc);
      assert.equal(staged.scripts, undefined);
      assert.equal(staged.dependencies, undefined);
      r = npmPackDryRun(dir);
      assert.equal(r.status, 0, r.stderr);
      assert.match(`${r.stdout}${r.stderr}`, new RegExp(`glrmask\\.${tag}\\.node`));
    }
    // Corruption and version skew are refused.
    const victim = path.join(stageDir, platform.releaseTargets[0], `glrmask.${platform.releaseTargets[0]}.node`);
    fs.appendFileSync(victim, Buffer.from([0]));
    const corrupt = spawnSync(process.execPath,
      [path.join(root, 'scripts/check-package.cjs'), '--platform-dir', path.join(stageDir, platform.releaseTargets[0]), '--tag', platform.releaseTargets[0]], {encoding: 'utf8'});
    assert.notEqual(corrupt.status, 0);
    assert.match(`${corrupt.stderr}`, /checksum mismatch/);
  } finally {
    fs.rmSync(work, {recursive: true, force: true});
  }
});

test('thin root dry-pack contains no binaries', () => {
  const r = npmPackDryRun(root);
  assert.equal(r.status, 0, r.stderr);
  assert.doesNotMatch(`${r.stdout}${r.stderr}`, /\.node/);
});
