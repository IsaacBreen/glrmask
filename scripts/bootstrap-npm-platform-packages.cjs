#!/usr/bin/env node
// Prepares npm for trusted publishing of every @glrmask/node-<platform> package:
// publishes a 0.0.1-placeholder placeholder for each one that doesn't exist on npm yet,
// then links each to this repository's release workflow with `npm trust github`.
// Run once, logged in to npm (`npm login`), before the first release that
// publishes platform packages. The first trust call asks for two-factor
// authentication; choose "skip for the next 5 minutes" and the rest go through.
// Pass --dry-run to only list what it would do.
'use strict';

const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const dryRun = process.argv.includes('--dry-run');
const root = require('../bindings/node/package.json');
const names = Object.keys(root.optionalDependencies);
const trust = ['--file', 'release.yml', '--repository', 'IsaacBreen/glrmask', '--environment', 'release'];

function exists(name) {
  try {
    execFileSync('npm', ['view', name, 'name'], { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

function trusted(name) {
  try {
    return execFileSync('npm', ['trust', 'list', name], { encoding: 'utf8', stdio: 'pipe' }).includes('release.yml');
  } catch {
    return false;
  }
}

for (const name of names) {
  if (exists(name)) {
    console.log(`${name}: already on npm`);
    continue;
  }
  if (dryRun) {
    console.log(`${name}: would publish 0.0.1-placeholder`);
    continue;
  }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glrmask-npm-bootstrap-'));
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({
    name,
    version: '0.0.1-placeholder',
    description: `Placeholder for the ${name} native binary. Install ${root.name} instead.`,
    license: root.license,
    repository: root.repository,
  }, null, 2) + '\n');
  try {
    execFileSync('npm', ['publish', '--access', 'public', '--tag', 'placeholder'], { cwd: dir, encoding: 'utf8', stdio: 'pipe' });
    console.log(`${name}: published 0.0.1-placeholder`);
  } catch (error) {
    // npm may hold the publish for approval; a held placeholder is enough.
    if (!String(error.stderr).includes('previously staged')) throw error;
    console.log(`${name}: 0.0.1-placeholder already staged`);
  }
  fs.rmSync(dir, { recursive: true, force: true });
}

for (const name of [root.name, ...names]) {
  if (!dryRun && trusted(name)) {
    console.log(`${name}: trusted publisher already set`);
    continue;
  }
  if (dryRun) {
    console.log(`${name}: would run npm trust github ${name} ${trust.join(' ')}`);
    continue;
  }
  try {
    execFileSync('npm', ['trust', 'github', name, ...trust, '--yes'], { stdio: 'inherit' });
    console.log(`${name}: trusted publisher set`);
  } catch {
    console.error(`${name}: npm trust github failed`);
    process.exitCode = 1;
  }
  execFileSync('sleep', ['2']);
}
