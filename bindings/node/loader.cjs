'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {currentTarget} = require('./platform.cjs');
const pkg = require('./package.json');
const target = currentTarget();
const filename = `glrmask.${target}.node`;
const location = path.join(__dirname, filename);
if (!fs.existsSync(location)) {
  throw new Error(`GLRMask native binary ${filename} is not included in this package. ` +
    'Install a release artifact containing your platform, or build from the complete GLRMask source checkout with npm run build in bindings/node. ' +
    'No Python or subprocess fallback is used.');
}
let addon;
try { addon = require(location); }
catch (cause) { throw new Error(`Could not load GLRMask native binary for ${target}: ${cause.message}`, {cause}); }
if (addon.version() !== pkg.version) {
  throw new Error(`GLRMask JavaScript/native version mismatch: ${pkg.version} versus ${addon.version()}. Rebuild or reinstall the matching package.`);
}
module.exports = addon;
