---
library_name: pytorch
base_model: divagr1925/Qwen3.5-4B-Qwenputer
tags:
  - qwenqpu
  - quantum-simulation
  - fixed-weights
  - experimental
---

# QwenQPU — fixed quantum module (release draft)

**HF model-card draft. No Hugging Face model repository is currently published.** The
owner must finalize the target, visibility and new-module license before upload.

## What these weights are

48 fixed FP32 coefficients implement local H/X/T gate maps. CNOT uses the same
X map with a control mask. There is no training or task-specific learned solver.
The validated module supports 1–16 qubits, at most 1,000 sequential operations
between resets, and ideal terminal sampling. It is classical simulation.

The sidecar does not include Qwen's language checkpoint, tokenizer, decoder
or KV state. The existing experiments use official upstream revision
`2d9068508f2b680badb7abcded96fdd107e9ba5d`. Its metadata declares Apache-2.0;
upstream weights must not be represented as newly trained QwenQPU parameters.

## Loading contract

Install the reviewed `qwenqpu` Python package, then use its `load_bundle` function.
The loader reads the JSON manifest, validates the SHA-256, uses PyTorch
`weights_only=True`, and checks that the values are exactly the declared gates.
It does not download or execute repository Python. A standard HF
`AutoModelForCausalLM.from_pretrained` call alone does not load this composite
runtime. Language `generate` and quantum `forward` are host-scheduled separately.

## Evidence and limitations

Existing core evidence uses independent Qiskit complex128 references, including
1,000-operation random circuits at 1, 3, 8, 12 and 16 qubits. The selected GHZ16 replay
has 1024 simulated draws: 534 all-zero and 490 all-one with seed 123. This is one ideal
state evolution with repeated terminal draws, not physical QPU execution.

The 40-question language comparison has a known repeated-JSON parser defect.
For 20 calculation tasks, the original affected automatic score is 1/20 vs 19/20;
the narrow post-hoc duplicate-object diagnostic is 8/20 vs 19/20, and additionally
requiring natural finish gives 7/20 vs 19/20. It is not a complete semantic audit,
new blind test or quantum-advantage result. Original planning joint scores are
1/20 in both arms. Preserve those qualifications in any future model card.

No noise, mid-circuit measurement, unbounded state, standard-decoder fusion or
quantum-hardware backend is implemented. The module's state memory scales
exponentially. See the repository's architecture and evidence documentation.
