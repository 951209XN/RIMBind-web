# Research-beta acceptance status: 2026-10-23

This record separates owner-provided target evidence from local checks. It is not an assertion of complete production acceptance. Private model assets, query labels, owner secrets, raw query probability tables and teacher-server details are not published here.

## Observed evidence

The owner supplied a numerical JSON certificate and downloaded prediction files from the target Space. These are reported artifacts, not independently signed proofs or an agent-operated recording of the external UI.

| Check | DNA | RNA |
|---|---|---|
| Fixed case | 9N3Y_6_F_DNA | 9T7T_72_TB_RNA |
| Residues | 241 | 264 |
| Same-target RIMBind max absolute error | 0 | 0 |
| Same-target SSCP max absolute error | 0 | 0 |
| Frozen memory match in certificate | true | true |
| Historical cross-hardware max error | 0.0025442541851654066 | 0.0024916533642272976 |
| Historical 1e-5 criterion | failed, retained | failed, retained |
| Ordinary downloaded files checked locally | CSV/PDB/JSON consistent | CSV/PDB/JSON consistent |
| Task rule recorded | FULL | NO_TEMPLATE_COUNT (CR) |
| Memory hits in ordinary output | 0 | 3 |

Both certificate runtime fingerprints match: Python 3.12.12, Torch 2.8.0+cu128, Transformers 4.43.3, ESM 3.1.1, Hub 0.36.2, NumPy 1.26.4, CUDA 12.8, NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb. Scientific serving-source digest is `1aba2aba2a5d0fbfde4e0f38015bec85a1151dc48e689be042006705464165e5`; protocol SHA256 is `6cca1fcb10fe0d6faf4c20fdcd7cd138d9592a926e61f20ac2d8999954a2e8cf`. The source/protocol digests match the local frozen implementation. The hosted asset-lock file itself was not independently downloaded; its hash agrees across the owner-provided records.

No experimental query labels or new performance metrics were used for deployment acceptance. A zero same-target difference does not erase cross-hardware drift or prove accuracy on arbitrary proteins. RNA probabilities were also consistent with the CR formula using the existing checksum-verified fixed-case memory vectors and exported SSCP, to CSV rounding precision; per-query remote memory vectors were not exported and therefore not independently observed.

## Local checks, not target-Space observations

The release safety tests exercise the unchanged PDB parser, serving class, UI handler and worker transport with explicit CPU/mocked dependency boundaries. Coverage includes wrong task/chain/sequence/accession rejection before GPU calls, oversized/nonfinite/missing-Cα/duplicate inputs, first-model semantics, original residue numbering, consent checks, sanitized model failures, no result export on simulated failure, temporary-work cleanup, scoped expiration cleanup, and simulated worker timeout/owned process-group termination. No model is loaded by these tests; no actual GPU quota is consumed or exhausted.

The frontend remains static with an explicit external predictor link and no automatic file forwarding. Local explorer behavior and public artifact publication are checked separately from model inference.

## Still requiring real operational evidence

- Invalid-input errors through the actual hosted browser UI, not just local function tests.
- Platform queue/quota exhaustion, cancellation/preemption and restart recovery.
- Live result/cache expiry and hosting-platform privacy behavior, including abrupt interruption.
- General availability from other networks/users, accessibility and sustained service monitoring.
- Owner confirmation that the credential previously exposed in chat has been revoked and replaced. No credential value is stored in this repository.

The existing numerical certificate's `public_service_acceptance_completed:false` and per-case `release_authorized:false` are diagnostic limits, not fields to flip manually. This frontend/documentation release does not edit the frozen runtime or invalidate its source certificate. A new runtime, rule or asset change requires the appropriate revalidation.

## Official operational references

Checked 2026-10-23: [ZeroGPU quota/shared allocation](https://huggingface.co/docs/hub/spaces-zerogpu), [ephemeral Space storage](https://huggingface.co/docs/hub/spaces-storage). Actual user messages and account eligibility remain authoritative for a given request; free access is not unlimited inference or a permanent hosting commitment.
