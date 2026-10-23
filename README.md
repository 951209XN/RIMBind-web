# RIMBind Web

Structure-based DNA/RNA-binding residue prediction using frozen SSCP model C and task-specific interface-memory fusion.

## Deployment status

**Research beta: real DNA and RNA prediction workflows have been exercised. Broader operational acceptance is not yet complete.**

- [Website and local result explorer](https://951209xn.github.io/RIMBind-web/)
- [External GPU predictor on Hugging Face](https://huggingface.co/spaces/XuNing985/RIMBind-web)
- [Prediction tutorial, privacy and limits](deploy/USER_GUIDE.md)
- [Dated acceptance evidence and remaining checks](deploy/RESEARCH_BETA_STATUS_20261023.md)

On 2026-10-23, owner-provided target-Space numerical records showed **zero adapter-versus-original probability difference** for both fixed acceptance cases, on the same GPU/runtime, at the unchanged 1e-5 tolerance. Both ordinary prediction workflows subsequently produced CSV, probability-mapped PDB and provenance JSON files, whose residue mappings, probabilities and exported frozen-model identifiers were checked locally. This is evidence for these tested workflows, not a claim about every input or continuous availability.

Historical cross-hardware differences remain **0.0025442541851654066 (DNA)** and **0.0024916533642272976 (RNA)** for the target-Space fixed cases; both fail the old 1e-5 historical comparator. The owner separately approved same-target comparison before these checks. No historical failure was relabeled a pass, and no weights, fusion parameters or acceptance tolerances were changed. Earlier development failures are preserved in [the historical record](HISTORICAL_DEVELOPMENT_STATUS_20261022.md).

GitHub Pages is a static local explorer and link to the external predictor. It does not upload structures or call the laboratory server. Select the external link and submit your file separately in the Space. The public frontend's `live_prediction_enabled:false` means there is no in-page inference client; it is not the external Space's model gate.

The Space retains its source/protocol/asset-bound numerical gate. A restart or GPU/runtime change may require owner revalidation. The frontend does not bypass this gate or claim that the Space is currently healthy.

## Quick start

1. Open the external GPU predictor.
2. Choose a PDB file, at most 2 MiB. Select one alphanumeric chain containing at most 600 observed standard amino acids, each with a Cα coordinate. Only the first PDB model is used.
3. Choose DNA or RNA. If providing FASTA, it must match the observed chain exactly. If known, supply the original four-character PDB accession for same-accession template exclusion; filenames are not trusted as identity.
4. Read and confirm the upload/privacy notice, then select **Predict binding residues**. Wait for the shared GPU allocation; do not submit repeatedly.
5. Download all three outputs promptly. CSV retains the higher-precision probabilities and author residue mapping; PDB B-fields store RIMBind probability 0–1 rounded to two decimals, not pLDDT or temperature factors. JSON records task, rule, weight identifiers and execution details.

This version does not generate structures, support sequence-only prediction, process multiple chains as one query, or calculate AP/AUROC/F1/MCC for unlabeled submissions.

## Privacy and availability

Local inspection and CSV visualization remain in the browser. Space submissions are processed on Hugging Face infrastructure after consent. Do not upload confidential, personal or sensitive unpublished structures. App result directories and Gradio caches are configured to become eligible for cleanup after one hour, checked every five minutes while running. This is not a platform-wide deletion guarantee. Platform logs, backups and exceptional cancellation behavior are outside this application's complete control.

ZeroGPU has shared queues, per-user quotas and interruptions; availability is not guaranteed. Free access is not unlimited inference. Follow the platform's quota/reset message rather than paying for an upgrade as a workaround. Ephemeral container files may disappear on restart. See the [official ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) and [Space storage documentation](https://huggingface.co/docs/hub/spaces-storage), checked 2026-10-23.

## Reproducibility contract

`SingleNABindModel("C")`; sigmoid each of five fold logits, then average probabilities. DNA uses FULL (count-based, alpha=0.2, tau=1). RNA uses CR / `NO_TEMPLATE_COUNT` (n_eff=1 for covered residues and 0 otherwise, alpha=0.8, tau=4). Prior=0.1; TM exponent=2. A displayed threshold of 0.5 is not a calibrated confidence guarantee.

The C-only loader omits only an unused E/F Mamba import from the execution AST; frozen source bytes, layers and checkpoints remain unchanged. Current UI and frozen model dependencies run in separate environments. The acceptance certificate binds scientific source, protocol, assets and target runtime. Numerical equivalence is not model accuracy or journal certification.

## Source layout

| Directory | Purpose |
|---|---|
| `site/` | Static Pages explorer and external predictor link. No credentials or direct uploads. |
| `backend/`, `vendor/`, `configs/` | Frozen inference, input/output contracts and SHA256 manifests; no weights. |
| `deploy/hf-full/` | Full gated Space app and deployment controls. |
| `deploy/hf-space/` | Earlier hardware/input preflight only; not the full predictor. |
| `tests/`, `tools/` | Contract checks, packaging and acceptance utilities. |

## Local checks

```sh
python -m pip install numpy==1.26.4
python -m unittest discover -s tests -v
node --test tests/frontend.test.mjs
python -m http.server 8765 --bind 127.0.0.1 --directory site
```

Contract tests do not invoke a model or reproduce target-Space runtime behavior. See the dated summary for observed tests versus simulated boundaries. Only serve `site/`, not the parent workspace, private assets or audit files.

## License, contact and citation

Original RIMBind source is Apache-2.0; see [LICENSE](LICENSE). This does not relicense third-party encoders or training-memory assets. Weights and private assets are not in this repository. The 3Dmol viewer retains its upstream BSD-3-Clause notice. The manuscript citation and permanent asset archive remain pending; do not invent a DOI or publication reference. For support, use this repository's Issues without posting credentials or sensitive structures.
