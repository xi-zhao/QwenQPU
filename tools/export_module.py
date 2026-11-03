"""Construct the 48 fixed gate coefficients. No language weights, no downloads."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwenqpu.weights import export_bundle

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
print(json.dumps(export_bundle(args.out), indent=2))
