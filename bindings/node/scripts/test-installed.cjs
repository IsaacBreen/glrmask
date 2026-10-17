'use strict';
// Install the actual tarball in an unrelated directory, with lifecycle scripts
// disabled, then run both module systems and real TypeScript consumers there.
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'..');
const tarball=process.argv[2],destination=process.argv[3];
if (!tarball) throw new Error('Usage: node scripts/test-installed.cjs PACKAGE.tgz [NEW_WORK_DIRECTORY]');
const artifact=path.resolve(tarball);
if (!fs.statSync(artifact).isFile()) throw new Error('Package tarball is missing');
const work=destination ? path.resolve(destination) : fs.mkdtempSync(path.join(os.tmpdir(),'glrmask-installed-'));
if (destination) {
  if (fs.existsSync(work)) throw new Error('Use a fresh installation-test directory');
  fs.mkdirSync(work,{recursive:true});
}
const candidates=[process.env.npm_execpath,
  path.join(path.dirname(process.execPath),'node_modules/npm/bin/npm-cli.js'),
  path.resolve(path.dirname(process.execPath),'../lib/node_modules/npm/bin/npm-cli.js')].filter(Boolean);
const npm=candidates.find(p=>fs.existsSync(p));
if (!npm) throw new Error('Cannot locate npm-cli.js; run this script through npm or an ordinary Node installation');
function run(program,args,options={}) {
  console.log('+',program,...args);
  const r=spawnSync(program,args,{cwd:work,stdio:'inherit',timeout:240000,...options});
  if (r.error) throw r.error;
  if (r.status!==0) throw new Error(`Installed consumer failed with exit ${r.status}`);
}
fs.writeFileSync(path.join(work,'package.json'),JSON.stringify({name:'glrmask-installed-consumer',private:true,type:'module'},null,2)+'\n');
run(process.execPath,[npm,'install','--ignore-scripts','--no-audit','--no-fund',artifact]);
const installed=path.join(work,'node_modules/@glrmask/node');
if (!fs.realpathSync(installed).startsWith(fs.realpathSync(work)+path.sep)) throw new Error('Installation resolved outside the isolated consumer');
for (const name of ['decode.mjs','json.cjs']) {
  const source=path.join(installed,'examples',name);
  fs.copyFileSync(source,path.join(work,name));
}
const runtime={...process.env};
delete runtime.NODE_PATH;delete runtime.NODE_OPTIONS;
runtime.PATH=[path.dirname(process.execPath),process.env.SystemRoot?path.join(process.env.SystemRoot,'System32'):''].filter(Boolean).join(path.delimiter);
run(process.execPath,['decode.mjs'],{env:runtime});
run(process.execPath,['json.cjs'],{env:runtime});
// Use exactly the locked development compiler/type versions, not a floating
// online latest. Type checking verifies declarations shipped in the tarball.
const lock=JSON.parse(fs.readFileSync(path.join(root,'package-lock.json'),'utf8'));
const ts=lock.packages['node_modules/typescript'].version;
const nodeTypes=lock.packages['node_modules/@types/node'].version;
run(process.execPath,[npm,'install','--ignore-scripts','--no-audit','--no-fund','--save-dev',`typescript@${ts}`,`@types/node@${nodeTypes}`]);
for (const file of ['types-consumer.cts','types-consumer.mts','tsconfig.json']) {
  fs.copyFileSync(path.join(root,'tests',file),path.join(work,file));
}
run(process.execPath,[path.join(work,'node_modules/typescript/bin/tsc'),'-p','tsconfig.json']);
fs.writeFileSync(path.join(work,'verification.json'),JSON.stringify({package:artifact,installed,cjs:true,esm:true,typescript:ts,noLifecycleScripts:true,noCompilerOnRuntimePath:true},null,2)+'\n');
console.log(`Isolated installed package passed: ${work}`);
