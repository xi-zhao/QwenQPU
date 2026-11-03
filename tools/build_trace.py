"""Re-execute the unchanged fixed core for a recorded circuit, without an LLM."""
from pathlib import Path
import collections
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from qwenqpu import LocalGateStatevector, encode

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
record = json.loads((ROOT / "evidence/public/ghz16.json").read_text())
packet = record["packet"]
assert packet["prefix"] == [] and packet["repeat"] == 1
network = torch.jit.script(LocalGateStatevector(16).eval())
frames = []


def snapshot(step, operation):
    p = network.probabilities()
    active = torch.where(p > 1e-12)[0].tolist()
    return {"step": step, "operation": operation, "raw_norm": float(network.raw_norm()),
            "sparse_state": [{"index": i, "bitstring": f"{i:016b}",
                              "real": float(network.state[i, 0]), "imag": float(network.state[i, 1]),
                              "probability": float(p[i])} for i in active]}


frames.append(snapshot(0, "RESET"))
with torch.inference_mode():
    for i, op in enumerate(packet["circuit"], 1):
        code = encode([op], 16)[0]
        assert network.step(int(code[0]), int(code[1]), int(code[2]))
        frames.append(snapshot(i, op))
    u = torch.rand(packet["shots"], generator=torch.Generator().manual_seed(record["visible_result"]["seed"]), dtype=torch.float64)
    bits, ok = network.sample_terminal(u)
assert bool(ok.all())
samples = ["".join(map(str, row)) for row in bits.tolist()]
counts = dict(sorted(collections.Counter(samples).items()))
assert counts == record["visible_result"]["counts"], "Local replay differs from recorded counts; do not render misleading animation"
trace = {
    "mode": "recorded_Qwen_circuit_with_verified_local_state_replay",
    "language_inference_rerun": False, "historical_language_and_network_device": "A100-SXM4-80GB",
    "local_state_replay_device": "CPU", "local_torch": torch.__version__,
    "animation_timing": "illustrative, not elapsed execution time",
    "state_source": "Unchanged validated core replayed gate-by-gate locally; not a historical per-gate GPU recording",
    "model_input": "counts only; state/probability visualization is a diagnostic viewer",
    "unitary_evolutions": 1, "physical_QPU_runs": 0,
    "packet": packet, "frames": frames, "seed": record["visible_result"]["seed"],
    "samples": samples, "counts": counts, "shots": len(samples), "bit_order": "q15..q0",
    "source_record_sha256": hashlib.sha256((ROOT / "evidence/public/ghz16.json").read_bytes()).hexdigest(),
    "core_sha256": hashlib.sha256((ROOT / "src/qwenqpu/core.py").read_bytes()).hexdigest(),
}
(ROOT / "assets/demo-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
print(json.dumps({"frames": len(frames), "counts": counts, "source_matches_recorded_counts": True, "language_inference_rerun": False}))
