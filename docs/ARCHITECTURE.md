# Implementation, not the future architecture

`qwenqpu.core.LocalGateStatevector` is the previously validated 1–16-qubit core,
copied byte-for-byte. Its SHA-256 is
`f941293e3a90ab7afeb39c2347325d77989fc23b649356482b461626bfb392b3`.

1. Qwen's language `generate` proposes a bounded JSON program for planning tasks.
   Given circuits can be dispatched without a planning call.
2. Host code validates the packet and expands `prefix + circuit * repeat`.
3. `QwenQPU.forward` invokes the scripted quantum child's `run` and
   `sample_terminal` methods.
4. For each gate, graph control selects target/control indices. Tensor operations
   gather amplitude pairs, apply a fixed local 4×4 real linear map, mask controlled
   X, and write updated pairs back with `index_copy_`.
5. The state is `[2**n, 2]` in FP32, with real/imaginary columns. H/X/T use 48
   fixed coefficients; CNOT uses X with a control mask. State is a buffer,
   separate from Qwen's decoder and KV cache. There is no training.
6. FP64 probabilities and CDF map host-supplied seeded uniforms to measured
   bitstrings. The host aggregates counts. Qwen sees counts, not exact probabilities.
7. A separate language `generate` reads those counts and produces a reply.

TorchScript integer control and compact instructions reside on CPU. Tensor
state operations run on CPU in the local quick demo, or CUDA in the historical
A100 runs. This is an auxiliary module in a composite `nn.Module`, under a host
driver; it is not a completed integration into Qwen's standard decoder layers.

## Sessions and measurement

`run(codes, reset=False)` supports continued segments; gate counters persist.
`reset_()` returns to all-zero state and clears counters/readout state. Separate
module instances have separate state. Up to1,000 unitary operations are allowed
between resets. Terminal sampling sets a readout flag; further gates require reset.

The current ideal terminal-measurement model evolves a circuit once, then draws
all samples from its final distribution. It does not model noisy or mid-circuit
measurement trajectories, and the state viewer in the animation is diagnostic.

## Integration example boundary

`examples/qwen_local.py` is a local-only convenience entry, not the frozen
evaluation runner. Its budget is shared across its planner and reply. It has
no model download, arbitrary remote code or automatic answer repair. Its packet
checks are protocol checks, not an independent certificate of a model-designed
GHZ state. The selected historical transcript has an independent offline audit.
