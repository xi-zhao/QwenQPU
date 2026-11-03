<p align="center">
  <img src="assets/banner.png" alt="QwenQPU: Qwen with a fixed quantum state-vector network." width="100%">
</p>

QwenQPU weights: publication in progress  
Language checkpoint: [Hugging Face](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer)  
[简体中文](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Evidence](docs/EVIDENCE.md)

**QwenQPU gives Qwen a fixed quantum execution module: a state-vector network for
1–16 qubits, built from H, X, T and CNOT gates.** Qwen writes a circuit, the network
computes its state and samples measurement outcomes, and Qwen continues from the
counts. The gate coefficients are constructed analytically. Nothing is trained.

The quantum register holds real and imaginary amplitudes. Small fixed linear
maps update pairs of amplitudes, controlled masks implement CNOT, and the final
state supplies terminal measurement samples. Quantum arithmetic runs in the
scripted quantum child called by a composite model's `forward`; a host driver schedules the
language calls, instructions and readout. This is ideal classical simulation.

The current module accepts up to 1,000 sequential gate operations between resets.
Its execution has been checked against independent Qiskit references, including
16-qubit non-Clifford circuits.

<p align="center"><img src="assets/computation-evidence.png" alt="Qwen and QwenQPU on twenty calculation tasks, with original parser-affected scores and labelled post-hoc diagnostics." width="100%"></p>

## Benchmarks

Greedy decoding, the same language checkpoint in both conditions, and a total
16,384-generated-token cap per condition. Qwen gets no execution module;
QwenQPU gets the fixed quantum network and 4,096 terminal samples. The comparison
contains 20 supplied-circuit calculation tasks and 20 auxiliary planning tasks.
It is a same-question re-evaluation, not a new blind test.

| Task and scoring view | Qwen | **QwenQPU** |
|---|---:|---:|
| Calculation, post-hoc identical-JSON diagnostic (20) | 8/20 | **19/20** |
| Calculation, same diagnostic and natural completion (20) | 7/20 | **19/20** |
| Planning, original joint score (20) | 1/20 | 1/20 |

The calculation rows are explicitly **post-hoc diagnostics**. The frozen parser
rejected repeated copies of the same correct JSON; its affected original score
was 1/20 versus 19/20. The diagnostic merges identical objects but is not a full
natural-language semantic audit. In that diagnostic there are 12 QwenQPU-only
wins, 1 Qwen-only win and 7 both-correct pairs; the descriptive paired-bootstrap
difference is 55 percentage points (95% interval 30–80). Planning has not improved.
[Scoring definitions, paired data and failure limits](docs/EVIDENCE.md).

All 38 actual quantum calls in the completed comparison passed an independent
execution and sampling audit. The largest component error was 4.531e-7; the
largest probability error was 7.308e-8. An execution can be correct while Qwen's
final answer is wrong. This small synthetic set does not establish quantum
advantage or superiority over an external simulator.

## Quickstart

With Python and PyTorch already installed:

```bash
git clone https://github.com/xi-zhao/QwenQPU && cd QwenQPU
python examples/ghz.py --qubits 16 --shots 1024 --seed 123
```

This executes the actual fixed network on CPU. The result is
`{"0000000000000000": 534, "1111111111111111": 490}`.
To read the selected recorded Qwen interaction without loading language weights:

```bash
python examples/replay_recorded.py
```

For a new language-to-circuit-to-counts dialogue with an existing checkpoint:

```bash
python examples/qwen_local.py \
  --model-path /path/to/an/existing/Qwen3.5-4B-Qwenputer \
  --output ./review/dialogue.json
```

The driver asks Qwen for circuit JSON, validates and encodes it, calls the fixed
quantum module, and sends only the observed counts back to Qwen. Language
`generate` and quantum `forward` are separate calls in this host loop. The
inherited language model uses the standard HF `Qwen3_5ForCausalLM`; no quantum
operation is dispatched to the upstream Qwenputer CPU. Its language weights are
unchanged, and the new quantum module is not fused into the original decoder/KV
state. The optional dialogue has a bounded demo budget and no automatic repairs.

Historical full-model runs used an A100 80 GB. The small core, parameter loading
and examples are checked on CPU; the optional full-checkpoint entry was not
rerun for the GitHub presentation. [Environment and commands](docs/REPRODUCING.md).

<p align="center"><img src="assets/demo.gif" alt="Recorded Qwen GHZ16 circuit and reply with verified state evolution and sampled counts." width="100%"></p>

*Recorded Qwen circuit and reply, with the unchanged core replaying the state.
Playback timing is illustrative; the samples are ideal classical simulation.
[MP4](assets/demo.mp4) · [Trace](assets/demo-trace.json) · [Render source](tools/render_assets.py).*

The implementation is in [`src/qwenqpu/`](src/qwenqpu); the
[architecture notes](docs/ARCHITECTURE.md) describe its graph, state and host boundary.

## Quantum state as memory

<p align="center"><img src="assets/architecture.png" alt="Qwen generation, host validation, fixed quantum forward, sampled bits and counts reply." width="100%"></p>

The register is an FP32 tensor of shape `[2**n, 2]`: one real and one imaginary
component per computational-basis amplitude. A new session starts in the all-zero
basis state. For each instruction, dynamic address indices gather the affected
pairs, a fixed 4×4 real map acts on them, and masking writes the result back.
CNOT applies the X update only where the control bit is one.

This uses local gate maps rather than a dense `2**n × 2**n` operator or a copy of
the state for every possible gate. The state vector still grows exponentially.
TorchScript integer control runs on CPU; the address and state tensor operations
run on the selected CPU/CUDA device. Gate selection, amplitude updates and
terminal bitstring selection remain in the scripted quantum module. The host
handles JSON parsing, repeat expansion, random inputs and counts aggregation.

The register is separate from Qwen's language state. It does not repurpose Qwen's
Gated DeltaNet heads as quantum RAM. Qiskit is an offline reference and is not
called by production execution.

## Fixed-gate execution (experimental)

H, X and T use 48 fixed FP32 coefficients in three 4×4 real maps; CNOT reuses X
with a control mask. The included [parameter bundle](weight-release/module-bundle/qwenqpu_module.json)
is small because it contains those mathematical constants, not a trained
language model. A checked loader verifies the file checksum, core identity and
exact gate values.

| Register | Gates | Sequential bound | Readout |
|---|---|---:|---|
| 1–16 qubits | H, X, T, CNOT | 1,000 operations between resets | 1–10,000 terminal samples |

The 1,000-operation limit is a gate count, not parallel circuit depth. The
calculation evaluation itself contains 5–161 operations. There is no training,
noise model or mid-circuit measurement in this release.

## Stateful execution (experimental)

A scripted `LocalGateStatevector` can receive several instruction segments while
preserving its register and cumulative gate count. Separate module instances
have separate state. `reset_()` returns the register to the all-zero basis state
and clears the instruction count. State must be reset before a new circuit after
terminal sampling.

One ideal circuit evolution produces the final distribution. The driver supplies
one seeded uniform value per shot; the graph computes the CDF and selects each
bitstring. The host then builds counts. `q0` is the least significant bit, so a
16-qubit result is written `q15..q0`. These draws are not repeated physical-QPU
executions. Exact probabilities are diagnostic output and are not sent to Qwen
as if they were sampled counts.

## Version 1 (three qubits)

The direct-network prototype used three qubits and 16 real state components.
The current implementation extends the same fixed-gate approach to 1–16 qubits;
the three-qubit case is still available through the same core:

<details><summary>Three-qubit example and local checks</summary>

```bash
python examples/ghz.py --qubits 3 --shots 1024 --seed 123
python -m unittest discover -s tests -v
```

The checks cover recorded GHZ16 and phase-interference counts, segmented
execution/reset and state isolation, invalid instructions, the gate bound, and
fixed-parameter export/reload. They are small runtime checks, separate from the
historical independent A100/Qiskit and language evaluations.

</details>

More in [`docs/`](docs): the [architecture](docs/ARCHITECTURE.md),
[evaluation](docs/EVIDENCE.md), [reproduction](docs/REPRODUCING.md) and
[media provenance](docs/MEDIA.md). Language weights, fixed quantum parameters and
the host runtime remain distinct components; a standalone HF language-model
load does not attach the quantum module.

## Citing

```bibtex
@software{qwenqpu2026,
  author = {xi-zhao},
  title  = {{QwenQPU}: Qwen with a fixed quantum state-vector network},
  year   = {2026},
  url    = {https://github.com/xi-zhao/QwenQPU}
}
```

The project builds on Qwen and [Qwenputer](https://github.com/divagr18/qwenputer).
Their authors retain credit for the language checkpoint and the broader
compiled-computation idea.

## License

Upstream Qwenputer code is MIT, and the inherited language weights declare
Apache-2.0. No project-wide open-source license has been selected for the
original QwenQPU code and artwork; publication does not add a license grant.
[Component terms and attribution](docs/LICENSES.md).
