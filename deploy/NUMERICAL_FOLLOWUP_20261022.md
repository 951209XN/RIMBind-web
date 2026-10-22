# Controlled numerical follow-up (2026-10-22)

The earlier historical probability gates remain failed. No tolerance, weights,
fusion parameters, case selection or manuscript results were changed.

For the same two pre-existing cases, direct replay of the frozen feature
operations and original fold-inference ordering matched the web adapter exactly
on the same runtime/device. With both features regenerated on PyTorch 2.10 / RTX
4080 SUPER, direct replay and adapter had zero final probability difference, but
both differed from the old references (DNA 0.00292250508; RNA 0.00377273523).

Regenerating SaProt in the original PyTorch 2.1.1 / RTX 4080 SUPER environment,
ESM3 on the original RTX 3090, and predicting with the original C runtime gave
historical errors below 5e-10 for both cases. These are component-replay results,
not target-Space acceptance or a new performance evaluation.

Profiler traces showed FlashAttention in the main ESM3 attention on both GPUs.
Disabling reduced BF16 accumulation did not remove the discrepancy. The exact
low-level numerical cause is not fully established; do not assert a different
Flash/cuDNN selection caused it.

The owner explicitly approved separating **same-runtime implementation
equivalence** from **historical cross-runtime portability** on 2026-10-22:
“同意按同硬件原代码对照验收，并保留跨硬件差异”. The tolerance remains 1e-5.
Neither approval nor an equivalence pass alone authorizes public release.
The original 1e-5 historical comparison is retained independently. The target
gate requires both fixed DNA/RNA cases, unchanged reference-source hashes,
matching source/asset locks and a matching assigned GPU/runtime fingerprint.
The unchanged embedding/C0 scripts run in fresh workspaces; only the documented
unused Mamba import and local foundation-snapshot resolution are adapted.

The source now includes a PRIVATE core-asset packager with checkpoint and
authoritative database-lock checks. It does not upload files or grant public
redistribution rights. Encoders and the third-party Foldseek binary are not in
that bundle. No model assets are committed to Git.
