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
                        labels = [t for t in svg.findall(".//svg:text", NS) if t.get("font-weight") == "600"]
                        self.assertEqual(len(labels), len(curves))
                        points = {p.get("stroke"): [(float(x), float(y)) for x,y in re.findall(r"[ML]([0-9.-]+),([0-9.-]+)", p.get("d"))] for p in curves}
                        bounds = svg.find(".//svg:clipPath/svg:rect", NS)
                        left, top = float(bounds.get("x")), float(bounds.get("y"))
                        right, bottom = left+float(bounds.get("width")), top+float(bounds.get("height"))
                        boxes = []
                        for label in labels:
                            box = bench.label_box(label.text, float(label.get("x")), float(label.get("y")))
                            x0,y0,x1,y1 = box
                            self.assertTrue(left <= x0 < x1 <= right and top <= y0 < y1 <= bottom)
                            own = points[label.get("fill")]
                            end_y = bench.curve_span(own,right,right)[0]
                            allowance = 32 if title == "JSON Schema" and guide and label.text in ("llguidance", "GLRMask dynamic") else 18
                            self.assertLessEqual(max(y0-end_y,end_y-y1,0),allowance+.01)
                            margin = chart["line_width"]/2
                            for curve in points.values():
                                low,high = bench.curve_span(curve,x0-margin,x1+margin)
                                self.assertTrue(y1+margin < low or y0-margin > high, label.text)
                            self.assertFalse(any(bench.boxes_overlap(box,b,0) for b in boxes))
                            boxes.append(box)
                        if dark:
                            self.assertEqual({t.text:t.get("fill") for t in labels if t.text != "GLRMask dynamic"},
                                             {"GLRMask":"#ff6b7d","llguidance":"#75bfff","XGrammar":"#69d3ae"})


if __name__ == "__main__":
    unittest.main()
