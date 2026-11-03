# Runtime and media verification

The quantum core is unchanged, SHA-256
`f941293e3a90ab7afeb39c2347325d77989fc23b649356482b461626bfb392b3`.

The small CPU checks use existing Python 3.13.2 and PyTorch 2.9.1. All four
core/runtime test methods pass: historical GHZ/HTH counts, segmented state/reset
and isolation, illegal instructions and the gate cap, fixed-weight export/load.

The current C16-4 trace executes all 161 gates and generates 4,096 terminal
samples with seed 8311126. Its event counts equal the retained A100 evaluation:
4096 ones for o1, and 2020 ones / 2076 zeros for o2. Executing 80 operations
then continuing with 81 produces the same final tensor as uninterrupted
execution. The independent Qiskit references are 1 and 0.5.

The example CLI also reproduces the C01-4 phase circuit's 2,659 ones out of
4,096 draws. Recorded language answers are preserved as excerpts in the
public benchmark examples; language generation is not rerun for these checks.

Current media: 30 seconds, 360 frames, 12 fps. MP4 is H.264/yuv420p 1280×720;
GIF is 1024×576. The actual encoded opening, gate sequence, continuation,
sampling and final comparison frames are visually inspected. The subtitle is
“QPU inside a language model”. No clock/throughput display is used.

Source code, Markdown/media links, preview JavaScript and the public-data scope
are checked before publication. Full model weights and private raw evaluation
logs are excluded from the GitHub tree.
