'use strict';
const targets = Object.freeze({
  'x86_64-pc-windows-msvc': 'win32-x64-msvc',
  'aarch64-pc-windows-msvc': 'win32-arm64-msvc',
  'x86_64-apple-darwin': 'darwin-x64',
  'aarch64-apple-darwin': 'darwin-arm64',
  'x86_64-unknown-linux-gnu': 'linux-x64-gnu',
  'aarch64-unknown-linux-gnu': 'linux-arm64-gnu',
  'x86_64-unknown-linux-musl': 'linux-x64-musl',
  'aarch64-unknown-linux-musl': 'linux-arm64-musl',
});
const releaseTargets = Object.freeze([
  'win32-x64-msvc', 'darwin-x64', 'darwin-arm64', 'linux-x64-gnu', 'linux-arm64-gnu',
]);
function currentTarget() {
  const {platform, arch} = process;
  if (platform === 'win32') return `${platform}-${arch}-msvc`;
  if (platform === 'darwin') return `${platform}-${arch}`;
  if (platform === 'linux') {
    const report = process.report.getReport();
    return `${platform}-${arch}-${report.header.glibcVersionRuntime ? 'gnu' : 'musl'}`;
  }
  throw new Error(`Unsupported native platform: ${platform}/${arch}. See https://glrmask.com/api/javascript/`);
}
module.exports = {targets, releaseTargets, currentTarget};
