# Animation and artwork provenance

The reference [Qwenputer GIF](https://divagr.com/qwenputer/demo.gif) was inspected
locally: 1280×720, 270 frames, 13.5 seconds. Its story pairs a user request and answer
with a changing RAM grid. The new media borrows only the idea of making the
execution visible. No reference frame, logo, grid asset, animation code or brand
artwork is included in QwenQPU's deliverable.

QwenQPU's original composition uses a dark instrument panel, an explicit
gate chain, changing state support, sampled counts and the recorded reply.
The new GIF/MP4 is 24 seconds at 12 fps. The MP4 is 1280×720; GIF is 1024×576.
All playback timing is illustrative. The concise on-screen labels are
`CLASSICAL SIMULATION` and `DEMO REPLAY`; detailed provenance is kept here.
Production notes, diagnostic disclaimers and long explanatory footers have been
removed from the public animation without changing any circuit or result.

Source interaction: the selected historical GHZ16 goal asks Qwen for H followed
by a CNOT chain. Its circuit, 1024-shot seed 123 counts and exact JSON reply are
retained in[`ghz16.json`](../evidence/public/ghz16.json). Counts are 534 all-zero,
490 all-one. Intermediate states are a new local deterministic replay of the
unchanged validated core, not a claim of historical per-gate GPU recording.

The state viewer is diagnostic and is not model input. The language model sees
counts. One ideal final state supplies all terminal draws. The animation does
not depict physical QPU jobs, hardware noise or mid-circuit measurement.

[`demo-trace.json`](../assets/demo-trace.json) contains every displayed gate-state
snapshot and sampled bitstring. The render source and animation-provenance
JSON make data lineage explicit. Original private logs, internal account URLs,
user identifiers, credentials and large model weights are omitted.
