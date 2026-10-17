'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');
const {targets} = require('../platform.cjs');
const pkg = require('../package.json');
const packageDir = path.resolve(__dirname,'..');
const root = path.resolve(packageDir,'../..');
const cli = process.argv.slice(2);
let profile = 'release', target;
for (let i=0; i<cli.length; ++i) {
  if (cli[i] === '--profile' && cli[i+1]) profile=cli[++i];
  else if (cli[i] === '--target' && cli[i+1]) target=cli[++i];
  else throw new Error(`Unknown or incomplete option: ${cli[i]}`);
}
if (!/^[a-zA-Z0-9_-]+$/.test(profile)) throw new Error('Invalid Cargo profile');
if (!fs.existsSync(path.join(root,'Cargo.toml')) || !fs.existsSync(path.join(root,'src','lib.rs'))) {
  throw new Error('Building requires the complete GLRMask source checkout. Run this script in its bindings/node directory, not an installed binary-only package.');
}
if (!target) {
  const probe=spawnSync('rustc',['-vV'],{encoding:'utf8'});
  if (probe.error || probe.status !== 0) throw new Error('A Rust toolchain is required to build GLRMask from source', {cause:probe.error});
  target=/^host: (.+)$/m.exec(probe.stdout)?.[1];
}
const tag=targets[target];
if (!tag) throw new Error(`Unsupported Rust target ${target}`);
const flags=[process.env.RUSTFLAGS || '',process.env.CARGO_ENCODED_RUSTFLAGS || ''].join(' ');
if (profile==='release' && /target-(?:cpu|feature)/.test(flags)) {
  throw new Error('Release package builds require portable target defaults. Remove target-cpu/target-feature overrides from RUSTFLAGS.');
}
const args=['build','--locked','--manifest-path',path.join(packageDir,'Cargo.toml'),'--profile',profile,'--message-format=json-render-diagnostics'];
if (cli.includes('--target')) args.push('--target',target);
console.log(`Building GLRMask ${pkg.version} for ${target} (${profile})`);
const run=spawnSync('cargo',args,{cwd:root,encoding:'utf8',stdio:['inherit','pipe','inherit'],maxBuffer:64*1024*1024});
if (run.error) throw run.error;
if (run.status !== 0) process.exit(run.status || 1);
let artifact;
for (const line of run.stdout.split(/\r?\n/)) {
  let item; try {item=JSON.parse(line);} catch {continue;}
  if (item.reason==='compiler-artifact' && item.target.name==='glrmask_node') {
    artifact=item.filenames.find(f=>/\.(dll|so|dylib)$/.test(f));
  }
}
if (!artifact || !fs.existsSync(artifact)) throw new Error('Cargo did not report a native GLRMask shared-library artifact');
const filename=`glrmask.${tag}.node`;
const output=path.join(packageDir,filename);
fs.copyFileSync(artifact,output);
const bytes=fs.readFileSync(output);
const manifestPath=path.join(packageDir,'native-manifest.json');
let manifest={schema:1,version:pkg.version,napi:8,targets:{}};
if (fs.existsSync(manifestPath)) {
  const old=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
  if (old.schema===1 && old.version===pkg.version) manifest=old;
}
manifest.targets[tag]={file:filename,rustTarget:target,profile,size:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex')};
fs.writeFileSync(manifestPath,JSON.stringify(manifest,null,2)+'\n');
for (const license of ['LICENSE-MIT','LICENSE-APACHE']) fs.copyFileSync(path.join(root,license),path.join(packageDir,license));
fs.copyFileSync(path.join(root,'crates/glrmask-grammar/NOTICE-LARK'),path.join(packageDir,'NOTICE-LARK'));
console.log(`Created ${filename} (${bytes.length} bytes)`);
