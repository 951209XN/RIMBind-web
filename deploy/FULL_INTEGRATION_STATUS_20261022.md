# Full-model integration status — 2026-10-22

This is implementation/portability acceptance, not a new benchmark or manuscript
performance analysis. Existing cases and existing probability references were
used. No query labels, new AP calculations, checkpoint selection or retraining.

| Check | Runtime/device | Maximum probability error | Fixed 1e-5 gate |
|---|---|---:|---|
| DNA 6JHE, prepared features | PyTorch 2.1.1, RTX 4080 SUPER | 4.869756715930595e-10 | Pass |
| DNA 6JHE, prepared features | PyTorch 2.10, RTX 4080 SUPER | 6.007046504663549e-08 | Pass |
| RNA 9T7T, raw PDB → CR | PyTorch 2.10, RTX 3090 | 4.142958656305762e-06 | Pass |
| DNA 9N3Y, raw PDB → FULL | PyTorch 2.10, RTX 4080 SUPER | 0.0029225050835494926 | Fail |
| DNA 9N3Y, raw PDB → FULL | PyTorch 2.10, RTX 3090 | 1.0341269149782839e-05 | Fail |
| DNA 9N3Y, raw PDB → FULL, isolated Transformers 4.43.3 | PyTorch 2.10, RTX 3090 | 1.0341269149782839e-05 | Fail |

Residue mapping and all three memory vectors matched the existing reference
exactly in the raw cases. RNA ESM3 half tensors matched exactly. DNA ESM3 half
tensors matched on the RTX 3090, but not on the RTX 4080 SUPER (maximum embedding
absolute difference 64). SaProt half tensors had smaller discrepancies across
these runtimes. These observations localize a portability issue; they do not
by themselves prove which GPU kernel or preprocessing operation caused it.

A separate structure-token probe found identical parsed atom coordinates and
sequences before/after normalization for both cases, and zero token differences
between the two GPUs. Thus the observed DNA discrepancy is not explained by a
changed sequence, atom-coordinate input or structure-token identity in this
probe. The Transformer-version alignment in a new isolated environment did not
remove the DNA failure. Original installed environments were preserved (4.36.2
remains in the ESM3 environment; only the new virtual environment has 4.43.3).

Neither raw test was executed on the Hugging Face Blackwell target. Its
hardware-only success is not full-model acceptance. The DNA failure must be
resolved or explicitly addressed through a justified, independently assessed
portability protocol; the existing threshold was not relaxed here.

## Implemented candidate

- Frozen source SHA256 gate and strict task checkpoint loading.
- A local-only asset lock, original Foldseek binary checks, frozen memory hit
  filtering/projection, top-20 selection and original 3Di generation path.
- SaProt fp32 and ESM3 bfloat16 processing with original half-tensor storage,
  exact length checks and explicit historical ESM3 fallback reporting.
- No automatic upstream download or unpinned ESM3 `main` lookup.
- Probability CSV with author residue/chain/insertion-code mapping, probability
  B-field PDB and label-free provenance JSON. No fabricated accuracy metrics.
- Full Gradio candidate, bounded queue and single prediction concurrency,
  upload consent, 2 MiB / 600-residue checks, temporary job isolation and result
  cleanup. No anonymous uploaded PyTorch deserialization.
- A flat source-only packaging tool; it excludes weights and study data.

## Not completed

1. DNA raw-PDB probability equivalence in a portable single-GPU environment.
2. Full target-Space dependency resolution and DNA/RNA numerical acceptance.
3. Authorized complete asset installation on the external host. No laboratory
   server is used as a public backend.
4. Live external browser, latency/quota, malformed request, download and
   privacy/cleanup acceptance. No formal public-service claim is justified yet.

The public Pages interface and the owner's working GPU preflight are preserved.
Do not replace them with the full candidate merely because its source exists.
