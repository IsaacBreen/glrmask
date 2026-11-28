"""Regression checks for generated website survival plots."""
import importlib.util
import json
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("benchmarks", HERE/"sync-benchmarks.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
NS = {"svg": "http://www.w3.org/2000/svg"}

class PlotTests(unittest.TestCase):
    def test_curves_are_monotone_and_no_leaders(self):
        chart = json.loads((bench.CONTENT/"tail-chart.json").read_text(encoding="utf8"))
        for record, title in zip(bench.website_measurements(), ["JSON Schema", "JavaScript"]):
            for dark in (False, True):
                for guide in (False, True):
                    with self.subTest(title=title, dark=dark, guide=guide):
                        svg = ET.fromstring(bench.website_svg(record, chart, dark, title, guide))
                        paths = svg.findall(".//svg:path", NS)
                        curves = [p for p in paths if p.get("clip-path") == "url(#plot)"]
                        self.assertEqual(len(curves), 4 if guide else 3)
                        for p in curves:
                            pts = [(float(x), float(y)) for x,y in re.findall(r"[ML]([0-9.-]+),([0-9.-]+)", p.get("d"))]
                            self.assertEqual(len(pts), len(record["tail"]["exceedance_pct"]))
                            self.assertTrue(all(a[0]>=b[0] and a[1]<=b[1] for a,b in zip(pts,pts[1:])))
                        # Non-series paths may only be the original axes and ticks.
                        for p in paths:
                            if p in curves:
                                continue
                            self.assertIsNone(p.get("stroke-width"))
                            self.assertRegex(p.get("d"), r"^M[0-9.]+,[0-9.]+ (?:V[0-9.]+ H[0-9.]+|H[0-9.]+|v4)$")
                        self.assertFalse(svg.findall(".//svg:line", NS))

if __name__ == "__main__":
    unittest.main()
