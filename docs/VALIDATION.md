# Publication revision checks

This publication revision changes presentation and documentation. It does not
change the fixed quantum core, model weights, evaluation questions or scores.

## Small runtime checks

Run locally with existing Python 3.13.2 and PyTorch 2.9.1 on CPU:

- The recorded-interaction CLI completes without a model download.
- The actual 16-qubit GHZ CLI produces 534 all-zero and 490 all-one counts
  for 1,024 terminal samples with seed 123.
- All four test methods pass: historical GHZ/HTH counts, state isolation and
  segmented execution/reset, illegal instructions and the 1,000-operation cap,
  and fixed-parameter export/reload.
- All Python files parse, the preview JavaScript passes syntax checking, and
  README/document/media relative links resolve in the publication tree.

The unchanged core SHA-256 is
`f941293e3a90ab7afeb39c2347325d77989fc23b649356482b461626bfb392b3`.

## Media revision

The GIF and MP4 use the same unchanged circuit/state/sample trace. Both contain
288 frames over 24 seconds at 12 fps; the MP4 uses H.264/yuv420p at 1280×720,
and the GIF is 1024×576. The opening, gate sequence, terminal samples and final
reply were visually checked in the actual encoded output.

Removed on-screen production notes include the diagnostic-only warnings,
packet-folding note, full-record note and long replay/provenance footer.
The short `CLASSICAL SIMULATION` and `DEMO REPLAY` labels remain. Detailed
provenance is in [MEDIA.md](MEDIA.md); evaluation caveats remain in
[EVIDENCE.md](EVIDENCE.md) and the README.

No new full-checkpoint language inference, A100 evaluation, training or
dependency installation was performed for this revision. The optional local
Qwen entry remains untested with the large checkpoint in this publication step.
The earlier full-model and independent physical-validation evidence is
reported separately; it is not replaced by these small publication checks.

Only selected synthetic examples and aggregate evidence are included. Full
language weights, private raw logs, credentials, personal paths, font files,
environments and reference animation assets are excluded. Original QwenQPU
code/artwork has no selected project-wide open-source license; see
[LICENSES.md](LICENSES.md).
