"""Small fixed-parameter bundle; never contains or downloads language weights."""
from pathlib import Path
import hashlib
import json
import torch
from .core import LocalGateStatevector


def export_bundle(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weight = directory / "quantum_gates.pt"
    config = directory / "qwenqpu_module.json"
    if weight.exists() or config.exists():
        raise FileExistsError("Use a fresh module-bundle directory; existing files preserved")
    gates = LocalGateStatevector(1).gates.detach().cpu().contiguous()
    torch.save({"gates": gates}, weight)
    meta = {
        "format": "qwenqpu-fixed-gates-v1", "parameter_count": 48,
        "shape": [3, 4, 4], "dtype": "float32", "trained": False,
        "gate_names": ["H", "X", "T"], "CNOT": "controlled X with graph mask",
        "n_qubits": [1, 16], "max_sequential_gates_between_resets": 1000,
        "weights_file": weight.name,
        "weights_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        "core_sha256": hashlib.sha256(Path(__file__).with_name("core.py").read_bytes()).hexdigest(),
        "language_checkpoint_included": False,
        "loading": "Installed qwenqpu package; torch.load(weights_only=True); no remote Python",
    }
    config.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_bundle(directory, n_qubits=16, device="cpu"):
    directory = Path(directory)
    meta = json.loads((directory / "qwenqpu_module.json").read_text())
    if meta.get("format") != "qwenqpu-fixed-gates-v1" or meta.get("weights_file") != "quantum_gates.pt":
        raise ValueError("Unsupported fixed-module bundle")
    path = directory / "quantum_gates.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["weights_sha256"]:
        raise ValueError("Fixed-module weight checksum mismatch")
    source_sha = hashlib.sha256(Path(__file__).with_name("core.py").read_bytes()).hexdigest()
    if meta["core_sha256"] != source_sha:
        raise ValueError("Module was exported with a different core implementation")
    # weights_only restricts unpickling. Exact comparison also rejects arbitrary
    # substituted/learned gates; the released mathematical gates remain fixed.
    values = torch.load(path, map_location="cpu", weights_only=True)
    model = LocalGateStatevector(n_qubits)
    if set(values) != {"gates"} or values["gates"].dtype != torch.float32 or not torch.equal(values["gates"], model.gates):
        raise ValueError("Bundle does not match the fixed H/X/T gate parameters")
    return torch.jit.script(model.to(device).eval())
