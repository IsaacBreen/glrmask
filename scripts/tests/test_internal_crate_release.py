"""Read-only internal Rust crate release preflight regression checks."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
import urllib.error

SCRIPT = Path(__file__).resolve().parents[1] / 'check-internal-crate-release.py'
spec = importlib.util.spec_from_file_location('glrmask_internal_release_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class InternalCrateReleaseTests(unittest.TestCase):
    def test_current_workspace_pins_and_acyclic_graph(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout.split('INTERNAL_CRATE_RELEASE_PREFLIGHT_OK')[0])
        self.assertEqual(len(payload['internal_versions']), 13)
        self.assertEqual(payload['errors'], [])
        manifest = audit.load(SCRIPT.parent.parent / 'Cargo.toml')
        self.assertEqual(payload['root'], 'glrmask@' + manifest['package']['version'])

    def test_published_exact_version_is_accepted(self):
        body = io.BytesIO(json.dumps({'version': {'num': '0.3.0', 'yanked': False}}).encode())
        with mock.patch.object(audit.urllib.request, 'urlopen', return_value=body):
            self.assertEqual(audit.check_registry('glrmask-vocab', '0.3.0'), 'PUBLISHED')

    def test_yanked_version_is_rejected(self):
        body = io.BytesIO(json.dumps({'version': {'num': '0.3.0', 'yanked': True}}).encode())
        with mock.patch.object(audit.urllib.request, 'urlopen', return_value=body):
            self.assertEqual(audit.check_registry('glrmask-vocab', '0.3.0'), 'YANKED')

    def test_missing_version_is_rejected(self):
        error = urllib.error.HTTPError('https://crates.io', 404, 'Not Found', {}, None)
        with mock.patch.object(audit.urllib.request, 'urlopen', side_effect=error):
            self.assertEqual(audit.check_registry('glrmask-vocab', '0.3.0'), 'MISSING')

    def test_unexpected_version_is_rejected(self):
        body = io.BytesIO(json.dumps({'version': {'num': '0.2.0', 'yanked': False}}).encode())
        with mock.patch.object(audit.urllib.request, 'urlopen', return_value=body):
            with self.assertRaises(ValueError):
                audit.check_registry('glrmask-vocab', '0.3.0')

    def test_authentication_failure_does_not_pass(self):
        error = urllib.error.HTTPError('https://crates.io', 403, 'Forbidden', {}, None)
        with mock.patch.object(audit.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaises(urllib.error.HTTPError):
                audit.check_registry('glrmask-vocab', '0.3.0')


if __name__ == '__main__':
    unittest.main()