# Weight release plan — pending publication

GitHub publication is separate from this future model release. HF registration, repository creation,
tokens and uploads are paused. No download badge points to a nonexistent model.

## Recommended layout

The lightweight candidate contains:

```text
README.md                 # model-card draft, finalized before publication
quantum_gates.pt           # 48 fixed coefficients, no language state/weights
qwenqpu_module.json        # core identity, shape, checksum, loading contract
LICENSE / notices         # selected and finalized by the owner
```

The installed `qwenqpu` runtime safely loads the gate tensor with
`torch.load(weights_only=True)`, verifies checksums and exact fixed-gate values,
then constructs the state for 1–16 qubits. The existing local roundtrip test passes.
There is no custom remote Python download or `trust_remote_code=True` path.

Language generation separately needs the pinned official upstream checkpoint
and the explicit host runtime. This small sidecar is not a complete HF causal
language model and cannot attach itself through a plain AutoModel call.

## Before any Hugging Face weight release

- The GitHub showcase is approved for `xi-zhao/QwenQPU`. That approval does not
  create a project-wide open-source license or authorize a future HF upload.
- Preserve the selected synthetic replay and aggregate evidence;
  private raw evaluation logs remain excluded.
- Decide the HF account/organization, repository name and public/private visibility.
  Account registration is outside this preparation task.
- Finalize the model card and retain applicable upstream license/NOTICE and provenance.
- Prefer the small module + official base reference; if a full composite distribution
  is chosen, explicitly describe its required runtime and unchanged upstream language
  weights. It is not a new trained language checkpoint or decoder-fused QPU.
- Verify the exact 8 shard identities before full redistribution. This draft stores
  only public metadata; no 18.7 GB checkpoint was downloaded or copied.
- Use a supported authenticated upload after the owner approves the exact content
  and destination. No persistent credential is created by these scripts.

`upstream-checkpoint.json` records the verified pinned metadata. Current upstream
weights total 18,715,279,016 bytes. That number excludes tokenizer/config files.
The `module-bundle/` is a small, actually load-tested specimen included with the
GitHub source. It has not been published to a Hugging Face model repository.
