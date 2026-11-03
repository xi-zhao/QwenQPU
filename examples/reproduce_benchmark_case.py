"""Execute one published benchmark circuit and optionally check it with Qiskit."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwenqpu import execute_packet

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="C16-4")
parser.add_argument("--qiskit", action="store_true", help="Use an already installed independent Qiskit reference")
args = parser.parse_args()
items = json.loads((ROOT / "evidence/public/benchmark_examples.json").read_text())["examples"]
matches = [c for c in items if c["question"]["id"] == args.case]
if not matches: parser.error("Available cases: " + ", ".join(c["question"]["id"] for c in items))
case = matches[0]; q = case["question"]
ops = q["circuit"] + q["readout"]
packet = {"n_qubits": q["n_qubits"], "prefix": [], "circuit": ops, "repeat": 1, "shots": q["shots"]}
result = execute_packet(packet, seed=q["sampling_seed"])
events = {}
for obs in q["observables"]:
    one = sum(count for bits, count in result["counts"].items()
              if sum(int(bits[q["n_qubits"] - 1 - i]) for i in obs["qubits"]) % 2 == obs["value"])
    events[obs["id"]] = {"0": q["shots"] - one, "1": one}
assert events == case["arms"]["enhanced"]["processor_result"]["event_counts"]
output = {"case": args.case, "n_qubits": q["n_qubits"], "gate_operations": len(ops),
          "shots": q["shots"], "seed": q["sampling_seed"], "event_counts": events,
          "sample_estimates": {k: v["1"] / q["shots"] for k, v in events.items()},
          "reference_probabilities": case["reference"]["reference_probabilities"],
          "published_answers": {a: case["arms"][a]["final_answer_object_excerpt"] for a in ["plain", "enhanced"]},
          "answers_source": "frozen eval40 v4", "new_language_generation": False}
if args.qiskit:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    circuit = QuantumCircuit(q["n_qubits"])
    for operation in ops:
        parts = operation.split(); ids = [int(v) for v in parts[1:]]
        if parts[0] == "CNOT": circuit.cx(*ids)
        else: getattr(circuit, parts[0].lower())(*ids)
    probabilities = Statevector.from_instruction(circuit).probabilities()
    independent = {}
    for obs in q["observables"]:
        independent[obs["id"]] = float(sum(p for basis, p in enumerate(probabilities)
            if sum((basis >> bit) & 1 for bit in obs["qubits"]) % 2 == obs["value"]))
    assert all(abs(independent[k] - case["reference"]["reference_probabilities"][k]) < 1e-10 for k in independent)
    output["fresh_Qiskit_reference"] = independent
print(json.dumps(output, indent=2))
