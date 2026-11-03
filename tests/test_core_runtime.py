"""Small local checks for packaging/runtime; the large A100 evidence is retained."""
from pathlib import Path
import json
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from qwenqpu import execute_packet, LocalGateStatevector, encode
from qwenqpu.weights import export_bundle, load_bundle

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "review").mkdir(exist_ok=True)
torch.set_num_threads(1)


class RuntimeChecks(unittest.TestCase):
    def test_historical_counts_reproduced(self):
        for name in ("ghz16", "phase_hth_q15"):
            record = json.loads((ROOT / "evidence/public" / (name + ".json")).read_text())
            self.assertEqual(execute_packet(record["packet"], seed=record["visible_result"]["seed"]), record["visible_result"])

    def test_sessions_and_segments(self):
        a = torch.jit.script(LocalGateStatevector(3))
        b = torch.jit.script(LocalGateStatevector(3))
        self.assertTrue(bool(a.run(encode(["H 0"], 3), True).all()))
        self.assertTrue(bool(a.run(encode(["CNOT 0 2"], 3)).all()))
        self.assertEqual(b.state[0, 0], 1)
        self.assertEqual(a.gate_count, 2)
        a.reset_()
        self.assertEqual(a.gate_count, 0)
        self.assertEqual(a.state[0, 0], 1)

    def test_limits_and_illegal_input(self):
        for n, ops in [(17,["H 0"]),(2,["CNOT 1 1"]),(1,["RY 0"]),(1,["X 0"]*1001)]:
            with self.assertRaises(ValueError):
                execute_packet({"n_qubits":n,"prefix":[],"circuit":ops,"repeat":1,"shots":32})
        packet = {"n_qubits":1,"prefix":[],"circuit":["X 0"],"repeat":1000,"shots":32}
        self.assertEqual(execute_packet(packet)["counts"], {"0":32})

    def test_fixed_weight_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "review") as tmp:
            meta = export_bundle(tmp)
            self.assertEqual(meta["parameter_count"], 48)
            q = load_bundle(tmp, n_qubits=3)
            self.assertTrue(bool(q.run(encode(["H 0","CNOT 0 2"],3),True).all()))
            self.assertAlmostEqual(float(q.probabilities()[5]), .5, places=7)
            with self.assertRaises(FileExistsError):
                export_bundle(tmp)


if __name__ == "__main__":
    unittest.main()
