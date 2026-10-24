#!/usr/bin/env node
// Publishes a 0.0.0-stage placeholder for each @glrmask/node-<platform> package
// that doesn't exist on npm yet, so its trusted publisher can be configured.
// Run once, logged in to npm (`npm login`), before the first release that
// publishes platform packages. Pass --dry-run to only list what it would do.
'use strict';

const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const dryRun = process.argv.includes('--dry-run');
const root = require('../bindings/node/package.json');
const names = Object.keys(root.optionalDependencies);

function exists(name) {
  try {
    execFileSync('npm', ['view', name, 'name'], { stdio: 'pipe' });
    return true;
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
    console.log(`${name}: would publish 0.0.0-stage`);
    continue;
  }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glrmask-npm-bootstrap-'));
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({
    name,
    version: '0.0.0-stage',
    description: `Placeholder for the ${name} native binary. Install ${root.name} instead.`,
    license: root.license,
    repository: root.repository,
  }, null, 2) + '\n');
  execFileSync('npm', ['publish', '--access', 'public', '--tag', 'stage'], { cwd: dir, stdio: 'inherit' });
  fs.rmSync(dir, { recursive: true, force: true });
  console.log(`${name}: published 0.0.0-stage`);
}
