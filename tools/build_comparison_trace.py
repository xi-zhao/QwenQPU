"""Replay the frozen C16-4 program with the actual fixed quantum network."""
from pathlib import Path
import hashlib
import json
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwenqpu import LocalGateStatevector, encode

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
source = json.loads((ROOT / "evidence/public/benchmark_examples.json").read_text())
case = next(c for c in source["examples"] if c["question"]["id"] == "C16-4")
q = case["question"]
n = q["n_qubits"]
model = torch.jit.script(LocalGateStatevector(n))
ids = torch.arange(2 ** n, dtype=torch.int64)
selectors = []
for obs in q["observables"]:
    parity = torch.zeros(2 ** n, dtype=torch.int64)
    for bit in obs["qubits"]:
        parity ^= (ids >> bit) & 1
    selectors.append(parity == obs["value"])


def snapshot(step, operation):
    p = model.probabilities()
    return {"step": step, "operation": operation,
            "probabilities": [float(p[s].sum()) for s in selectors]}


frames = [snapshot(0, "RESET")]
for step, op in enumerate(q["circuit"], 1):
    assert bool(model.run(encode([op], n), False).all())
    frames.append(snapshot(step, op))
final = model.state.clone()
segmented = torch.jit.script(LocalGateStatevector(n))
assert bool(segmented.run(encode(q["circuit"][:80], n), True).all())
assert segmented.gate_count == 80
assert bool(segmented.run(encode(q["circuit"][80:], n), False).all())
assert segmented.gate_count == 161 and torch.equal(segmented.state, final)
u = torch.rand(q["shots"], generator=torch.Generator().manual_seed(q["sampling_seed"]), dtype=torch.float64)
bits, accepted = model.sample_terminal(u)
assert bool(accepted.all())
event_bits = []
counts = {}
for obs in q["observables"]:
    cols = [n - 1 - i for i in obs["qubits"]]
    event = (bits[:, cols].to(torch.int64).sum(1) % 2 == obs["value"]).to(torch.int64)
    event_bits.append(event.tolist())
    ones = int(event.sum())
    counts[obs["id"]] = {"0": q["shots"] - ones, "1": ones}
assert counts == case["arms"]["enhanced"]["processor_result"]["event_counts"]
trace = {"question_id": q["id"], "n_qubits": n, "operations": q["circuit"],
         "observables": q["observables"], "shots": q["shots"], "seed": q["sampling_seed"],
         "frames": frames, "event_bits": event_bits, "event_counts": counts,
         "plain_answer": case["arms"]["plain"]["final_answer_object_excerpt"]["answers"],
         "qwenqpu_answer": case["arms"]["enhanced"]["final_answer_object_excerpt"]["answers"],
         "reference_probabilities": case["reference"]["reference_probabilities"],
         "segmented_80_plus_81_state_identical": True,
         "core_sha256": hashlib.sha256((ROOT / "src/qwenqpu/core.py").read_bytes()).hexdigest(),
         "source_record_sha256": {a: case["arms"][a]["source_record_sha256"] for a in ["plain", "enhanced"]},
         "new_language_generation": False, "reference_source": "retained independent Qiskit complex128"}
(ROOT / "assets/demo-trace.json").write_text(json.dumps(trace, separators=(",", ":")) + "\n")
print(json.dumps({"case": q["id"], "gate_operations": len(q["circuit"]),
                  "counts_match_record": True, "segmented_state_identical": True,
                  "final_probabilities": frames[-1]["probabilities"]}))
