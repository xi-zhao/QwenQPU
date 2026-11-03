"""Execute the real fixed network on CPU; no language checkpoint required."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwenqpu import execute_packet

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--qubits", type=int, default=16)
ap.add_argument("--shots", type=int, default=1024)
ap.add_argument("--seed", type=int, default=123)
args = ap.parse_args()
packet = {"n_qubits": args.qubits, "prefix": [],
          "circuit": ["H 0"] + [f"CNOT {i} {i+1}" for i in range(args.qubits-1)],
          "repeat": 1, "shots": args.shots}
print(json.dumps(execute_packet(packet, seed=args.seed), indent=2))
