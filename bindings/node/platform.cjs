'use strict';
// Canonical platform taxonomy for @glrmask/node distribution.
//
// `targets` maps Rust target triples to short distribution tags. `tag` is
// also the optional-dependency package suffix: the native binary for `tag`
// ships in `@glrmask/node-<tag>` and is installed automatically through the
// thin root's optionalDependencies on supporting package managers.
//
// The thin root package contains no .node binaries. Artifacts must stay
// aligned: root version, platform package version and manifest digests are
// cross-checked by scripts/check-package.cjs before packing.
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
// Every prebuilt release target. CI must produce all of these; the thin root
// lists each corresponding platform package as an optionalDependency.
const releaseTargets = Object.freeze([
  'win32-x64-msvc', 'win32-arm64-msvc', 'darwin-x64', 'darwin-arm64',
  'linux-x64-gnu', 'linux-arm64-gnu', 'linux-x64-musl', 'linux-arm64-musl',
]);
// npm install-time routing for each platform package. `os`/`cpu` follow npm's
// package.json semantics (https://docs.npmjs.com/cli/v11/configuring-npm/package-json#os
// and #cpu); `libc` restricts glibc/musl variants on Linux. These fields are
// enforced by check-package.cjs and must match the generated platform
// package.json files byte for byte in meaning.
const PLATFORM_PACKAGES = Object.freeze({
  'win32-x64-msvc': Object.freeze({package: '@glrmask/node-win32-x64-msvc', os: ['win32'], cpu: ['x64']}),
  'win32-arm64-msvc': Object.freeze({package: '@glrmask/node-win32-arm64-msvc', os: ['win32'], cpu: ['arm64']}),
  'darwin-x64': Object.freeze({package: '@glrmask/node-darwin-x64', os: ['darwin'], cpu: ['x64']}),
  'darwin-arm64': Object.freeze({package: '@glrmask/node-darwin-arm64', os: ['darwin'], cpu: ['arm64']}),
  'linux-x64-gnu': Object.freeze({package: '@glrmask/node-linux-x64-gnu', os: ['linux'], cpu: ['x64'], libc: ['glibc']}),
  'linux-arm64-gnu': Object.freeze({package: '@glrmask/node-linux-arm64-gnu', os: ['linux'], cpu: ['arm64'], libc: ['glibc']}),
  'linux-x64-musl': Object.freeze({package: '@glrmask/node-linux-x64-musl', os: ['linux'], cpu: ['x64'], libc: ['musl']}),
  'linux-arm64-musl': Object.freeze({package: '@glrmask/node-linux-arm64-musl', os: ['linux'], cpu: ['arm64'], libc: ['musl']}),
});
function platformPackageFor(tag) {
  const entry = PLATFORM_PACKAGES[tag];
  if (!entry) throw new Error(`Unsupported GLRMask native target: ${tag}. Supported: ${releaseTargets.join(', ')}`);
  return entry;
}
function currentTarget() {
  // Hidden test hook: scripts and unit tests simulate another platform or a
  // corrupt configuration without mocking process.platform/report.
  const override = process.env.GLRMASK_TEST_TARGET;
  if (override) {
    if (!Object.values(targets).includes(override) && !releaseTargets.includes(override)) {
      throw new Error(`Unsupported native platform: ${override}. Supported: ${releaseTargets.join(', ')}. See https://glrmask.com/api/javascript/`);
    }
    return override;
  }
  const {platform, arch} = process;
  if (platform === 'win32') return `${platform}-${arch}-msvc`;
  if (platform === 'darwin') return `${platform}-${arch}`;
  if (platform === 'linux') {
    const report = process.report.getReport();
    return `${platform}-${arch}-${report.header.glibcVersionRuntime ? 'gnu' : 'musl'}`;
  }
  throw new Error(`Unsupported native platform: ${platform}/${arch}. Supported: ${releaseTargets.join(', ')}. See https://glrmask.com/api/javascript/`);
}
module.exports = {targets, releaseTargets, PLATFORM_PACKAGES, platformPackageFor, currentTarget};
