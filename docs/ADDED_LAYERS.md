# QPU in standard Qwen layers

The current implementation puts quantum state storage, gate arithmetic and
probability readout into four added `Qwen3_5ForCausalLM` layers. The complete
model has 32 inherited language layers and four QPU layers. Its forward method
is the stock Transformers implementation.

## Execution

Qwen turns a task into a circuit and indexed measurement events. The host parses
that request and loads instruction tokens. Gated DeltaNet states store amplitude
digits, instructions and the program counter. Fixed SwiGLU projections implement
the shared H/X/T arithmetic; CNOT selects the addressed X operation. The model
emits numerical tokens, which are fed back to write the next state.

Born probabilities, event sums and readout correction also execute in the added
layers. The host schedules event queries, transports emitted digits, supplies
random proposals for sampling and decodes the final number. Correctness is
checked externally with Qiskit. There is no second language-model answer after
the numerical readout.

The fixed small-gate source is
[`quantum_gates.pt`](../weight-release/module-bundle/quantum_gates.pt).
The compiler copies its coefficients into standard model projections. The old
composite state-vector module is not invoked by this execution path.

| Source | Role |
|---|---|
| [compact_checkpoint.py](../research/standard_layers/compact_checkpoint.py) | Shared arithmetic, addressed digit memory, instruction ROM and native-token driver |
| [merge_language_checkpoint.py](../research/standard_layers/merge_language_checkpoint.py) | Maps the inherited language tensors and QPU tensors into one standard model |
| [task_interface.py](../research/standard_layers/task_interface.py) | Parses circuits/events and decodes numerical output |
| [compact_normalized_readout.py](../research/standard_layers/experiments/compact_normalized_readout.py) | Near-unit probability correction in fixed layers |
| [compact_sampling_rejection.py](../research/standard_layers/experiments/compact_sampling_rejection.py) | Native outcome/rejection selection |

## Completed full-model examples

Both examples started from `|000000>` and ran on an A100. The language model
generated the requests once; the QPU layers produced these final answers.

| Circuit | Event | QwenQPU | Qiskit |
|---|---|---:|---:|
| H3, T3, H3 | q3 = 1 | 0.1464464664 | 0.1464466094 |
| H3, T3, H3 | all zero | 0.8535535336 | 0.8535533906 |
| H0, CNOT 0→1→2→3→4→5 | all zero | 0.5 | 0.5 |
| Same GHZ circuit | all one | 0.5 | 0.5 |

[Prompts, generated requests, answers and timing](../evidence/added-layers/completed-demonstrations.json)
are included. The phase task took 161.171 seconds and the GHZ task 230.806
seconds, including language generation and quantum execution, excluding model
assembly. Their peak allocated memory was 37.16 and 37.34 GB respectively.

## Build and run

The execution environment used PyTorch 2.6.0+cu124 and Transformers 5.19.0.
The code uses ordinary Transformers classes, safetensors, NumPy and Qiskit
(the latter for external scoring). Keep all arithmetic in FP32 and TF32 off.

From the repository root, construct a six-qubit prefix:

```bash
python research/standard_layers/build_prefix.py --out ./models/qpu-prefix
```

This constructs about 1.42 GB of weights without training or downloading a
language model. It preserves the checked compiler and patch order. The tested
prefix hashes are in the [weight manifest](../evidence/added-layers/weight-manifest.json).

With the pinned Qwenputer checkpoint already available locally:

```bash
python research/standard_layers/demo_repaired_full_model.py \
  --base ./models/Qwen3.5-4B-Qwenputer \
  --prefix ./models/qpu-prefix \
  --out ./reproduced/added-layer-demo
```

Use a new output directory. Assembly retains the 32 inherited language layers,
omits the 11 CPU layers and inserts the four QPU layers. The resulting FP32
model contains 9,167,926,528 parameters (36,671,706,112 weight bytes). The builder
requires space for those weights plus a 5 GiB GPU reserve; actual execution
needs additional cache memory.

The complete model has been assembled and executed in memory. A serialized
full-model checkpoint has not yet been published. The current Hugging Face
root weights remain the earlier composite release at
[`78dfa7b`](https://huggingface.co/xi-zhao/QwenQPU/tree/78dfa7bcc58f6a66da3ba78a5b86aa5eb917aadf).

## Numerical checks

Amplitude components use two base-4096 digits with resolution `2^-22`.
The near-unit readout emits `E*(2-T)` for event mass E and total mass T, guarded
by `abs(T-1) <= 5e-4`. Within this domain its algebraic difference from E/T is
at most `2.5e-7`; measured floating-point error is checked separately.

The retained six-qubit 1,000-gate raw-readout run missed the `5e-5` criterion
(all-outcomes error about `1.21e-4`). Corrected readout of its saved final state
passed nine reference comparisons, maximum error `1.072e-5`. This checks the
readout of that saved state; the whole 1,000-gate circuit was not rerun with the
combined patch. The benchmark uses fresh 12/18/24-gate circuits.

Sampling uses model-selected outcomes or a REJECT token for independent uniform
proposals on `[0,2)`. The host retries rejected proposals, up to 64 per shot,
and counts accepted bitstrings. The six-qubit boundary/reload test passed all
18 endpoint probes and 18 proposal decisions. Eight accepted GHZ outcomes at
seed 7640346 gave four zeros and four ones; ten proposals were rejected.

For a small CPU check of the actual standard layers:

```bash
python research/standard_layers/experiments/test_normalized_readout.py \
  --out ./reproduced/readout-check
```

## Attribution

The inherited checkpoint is `divagr1925/Qwen3.5-4B-Qwenputer` at revision
`2d9068508f2b680badb7abcded96fdd107e9ba5d` (Apache-2.0).
The source-to-standard-layer approach builds on Qwenputer; its
[MIT notice](../research/standard_layers/upstream/LICENSE.qwenputer) is retained.
Original QwenQPU material keeps the [existing component terms](LICENSES.md).
