'use strict';
// Consolidate checksummed per-platform CI artifacts into one self-contained
// npm tarball input. Installation requires no compiler or postinstall network.
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {releaseTargets}=require('../platform.cjs');
const pkg=require('../package.json');
const root=path.resolve(__dirname,'..');
const source=process.argv[2];
if (!source) throw new Error('Usage: node scripts/assemble.cjs ARTIFACT_DIRECTORY');
const result={schema:1,version:pkg.version,napi:8,targets:{}};
function walk(dir) {
  for (const entry of fs.readdirSync(dir,{withFileTypes:true})) {
    const location=path.join(dir,entry.name);
    if (entry.isDirectory()) walk(location);
    else if (entry.name==='native-manifest.json') {
      const m=JSON.parse(fs.readFileSync(location,'utf8'));
      if (m.schema!==1 || m.version!==pkg.version || m.napi!==8) throw new Error(`Mismatched manifest ${location}`);
      for (const [tag,item] of Object.entries(m.targets)) {
        if (!releaseTargets.includes(tag)) throw new Error(`Unexpected CI target ${tag}`);
        if (result.targets[tag]) throw new Error(`Duplicate CI target ${tag}`);
        if (item.file!==`glrmask.${tag}.node` || item.profile!=='release') throw new Error(`Invalid release artifact ${tag}`);
        const data=fs.readFileSync(path.join(dir,item.file));
        if (data.length!==item.size || crypto.createHash('sha256').update(data).digest('hex')!==item.sha256) throw new Error(`Corrupt CI artifact ${tag}`);
        fs.writeFileSync(path.join(root,item.file),data); result.targets[tag]=item;
      }
    }
  }
}
walk(path.resolve(source));
for (const tag of releaseTargets) if (!result.targets[tag]) throw new Error(`Missing CI target ${tag}`);
fs.writeFileSync(path.join(root,'native-manifest.json'),JSON.stringify(result,null,2)+'\n');
console.log(`Assembled ${releaseTargets.length} native targets`);
