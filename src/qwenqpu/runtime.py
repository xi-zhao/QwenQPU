"""Host scheduling and counts aggregation; quantum arithmetic stays in core.py."""
import collections
import torch
from torch import nn
from .core import LocalGateStatevector
from .packet import parse_packet


class QwenQPU(nn.Module):
    """Composite.forward invokes the quantum child; language.generate is separate."""
    def __init__(self, n_qubits=16, language=None, device="cpu"):
        super().__init__()
        self.language = language
        self.quantum = torch.jit.script(LocalGateStatevector(n_qubits).to(device).eval())

    def forward(self, instructions, uniforms):
        accepted = self.quantum.run(instructions, True)
        if not bool(accepted.all()):
            raise ValueError("Fixed quantum graph rejected the circuit")
        return self.quantum.sample_terminal(uniforms)


def execute_packet(packet, seed=123, device="cpu", language=None):
    """One ideal evolution followed by terminal draws; returns counts only."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    operations, codes = parse_packet(packet)
    model = QwenQPU(packet["n_qubits"], language=language, device=device)
    uniforms = torch.rand(packet["shots"], generator=torch.Generator().manual_seed(seed), dtype=torch.float64).to(device)
    with torch.inference_mode():
        bits, accepted = model(codes, uniforms)
    if not bool(accepted.all()):
        raise RuntimeError("Terminal sampling rejected random inputs")
    counts = dict(sorted(collections.Counter("".join(map(str, row)) for row in bits.cpu().tolist()).items()))
    return {
        "mode": "ideal_simulated_terminal_counts", "n_qubits": packet["n_qubits"],
        "gate_operations": len(operations), "shots": packet["shots"],
        "counts": counts, "seed": seed, "bit_order": f"q{packet['n_qubits']-1}..q0",
    }
