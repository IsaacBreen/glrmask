'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {currentTarget,releaseTargets} = require('../platform.cjs');
const pkg = require('../package.json');
const root=path.resolve(__dirname,'..');
const manifestPath=path.join(root,'native-manifest.json');
if (!fs.existsSync(manifestPath)) throw new Error('Missing native build manifest. Build the addon before packing; empty packages are not distributable.');
const manifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
if (manifest.schema!==1 || manifest.version!==pkg.version || manifest.napi!==8) throw new Error('Invalid or stale native manifest');
const selected=process.argv.includes('--all') ? releaseTargets : [currentTarget()];
for (const tag of selected) if (!manifest.targets[tag]) throw new Error(`Missing required native target ${tag}`);
for (const [tag,item] of Object.entries(manifest.targets)) {
  if (item.file!==`glrmask.${tag}.node`) throw new Error(`Invalid artifact filename for ${tag}`);
  if (item.profile!=='release' && process.env.GLRMASK_ALLOW_DEV_ARTIFACTS!=='1') throw new Error(`Refusing to pack a non-release artifact for ${tag}`);
  const data=fs.readFileSync(path.join(root,item.file));
  if (data.length!==item.size || crypto.createHash('sha256').update(data).digest('hex')!==item.sha256) throw new Error(`Native artifact checksum mismatch for ${tag}`);
}
for (const file of fs.readdirSync(root).filter(f=>/^glrmask\..+\.node$/.test(f))) {
  if (!Object.values(manifest.targets).some(t=>t.file===file)) throw new Error(`Unrecorded native binary ${file}`);
}
const canonical="export * from './index.cjs';\n";
for (const file of ['index.d.mts','index.d.ts']) if (fs.readFileSync(path.join(root,file),'utf8')!==canonical) throw new Error(`Stale declaration copy ${file}`);
for (const file of ['README.md','LICENSE-MIT','LICENSE-APACHE','NOTICE-LARK','index.mjs','index.cjs']) {
  if (fs.statSync(path.join(root,file)).size===0) throw new Error(`Empty required package file ${file}`);
}
const addon=require('../loader.cjs');
if (addon.version()!==pkg.version) throw new Error('Native version mismatch');
console.log(`Validated GLRMask ${pkg.version}: ${Object.keys(manifest.targets).join(', ')}`);
