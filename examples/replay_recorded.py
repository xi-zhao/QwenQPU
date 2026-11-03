"""Print a selected recorded Qwen -> network -> counts interaction. No inference."""
from pathlib import Path
import json

p = Path(__file__).resolve().parents[1] / "evidence/public/ghz16.json"
r = json.loads(p.read_text())
print("RECORDED INTERACTION - this command does not run a language model.\n")
print("USER:", r["user"])
print("\nQWEN CIRCUIT:\n", r["Qwen_packet"])
print("\nOBSERVED IDEAL-SIMULATION COUNTS:\n", json.dumps(r["visible_result"], indent=2))
print("\nRECORDED QWEN ANSWER:\n", r["Qwen_counts_answer"])
