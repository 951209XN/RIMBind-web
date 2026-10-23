# RIMBind prediction tutorial

## Run the model

Open [RIMBind on Hugging Face](https://huggingface.co/spaces/XuNing985/RIMBind-web). The [GitHub Pages website](https://951209xn.github.io/RIMBind-web/) is a separate local explorer: choosing a file there does not submit it to the model. Opening the Space link also does not transfer your selected file; choose it again in the Space.

Required input is a protein PDB file ≤2 MiB and one selected alphanumeric chain ID. The chain must contain ≤600 observed standard amino acids, each with Cα coordinates. The first model is used. Missing residues are not reconstructed, ambiguous atoms are rejected, and MSE is not silently converted. Use DNA and RNA as separate tasks. Optional FASTA must exactly match the observed chain, not a full sequence with unobserved residues. Provide the original PDB accession if known; four characters, first a digit. It enables same-accession template exclusion. A filename alone does not identify the query.

Confirm permission to upload and the privacy notice. Select **Predict binding residues** once. No owner key or Hugging Face API token should be entered for ordinary predictions. An owner-validation panel is for administration only, not an alternative prediction interface. If the predictor is awaiting acceptance after restart, report this in repository Issues without sending credentials.

## Interpret and download

| Output | Meaning |
|---|---|
| `residue_probabilities.csv` | One row per observed residue; model position, amino acid, original chain/author number/insertion code, SSCP and RIMBind probabilities, and 0.5 display-threshold flag. |
| `rimbind_probability_Bfactor.pdb` | Protein structure with probability in B-fields, 0–1 rounded to two decimals. It no longer holds original B-factors or pLDDT. Use CSV for precise values. |
| `provenance.json` | Input hash, residue mapping, frozen task/rule and checkpoint identifiers, reported runtime, memory-hit count and job ID. |

DNA uses FULL; RNA uses CR (`NO_TEMPLATE_COUNT`). A lack of qualified templates can leave RIMBind equal to SSCP. A memory-driven probability change is not proof of improved correctness. AP, AUROC, F1 and MCC require experimental labels and are not calculated for anonymous unlabeled predictions. Existing probability tables loaded into the local explorer are visualization only, not rerun or independently authenticated predictions. Retain original Space downloads as the authoritative provenance/mapping record.

Download all outputs promptly, before navigating away. The server does not promise persistent job history or recovery of expired downloads. The local explorer's 6JHE example is precomputed, not a live request or a new accuracy benchmark. No automatic structure prediction is provided.

## Privacy and storage

Submitting in the Space transfers your structure to Hugging Face infrastructure. Use only data you have permission to upload; do not submit confidential, personal or sensitive unpublished structures. Ordinary application results use randomized job paths, not authenticated private accounts; do not share result URLs as if they were access-controlled storage.

Normal working directories are temporary. App result directories and Gradio caches are eligible for removal after one hour and checked every five minutes while the app runs. This is configured behavior and has local contract coverage, not independent verification of all live deletion paths. Abrupt cancellation/container failure may interrupt normal cleanup. Platform logs/backups have their own policy; no platform-wide erasure guarantee is made. Ephemeral files can disappear on restart. See [official storage documentation](https://huggingface.co/docs/hub/spaces-storage).

## Queue, quota and errors

The app queues jobs with one frozen-model operation at a time and an eight-job queue limit. Its GPU functions request at most 120 seconds; the internal model worker has a 110-second timeout. These are limits, not guaranteed response times. CPU preprocessing, queueing and downloads can add time. GPU availability and daily quota depend on Hugging Face; see [official ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu).

- Invalid input: correct the selected chain/sequence/file rather than repeatedly submitting.
- Quota or queue error: follow the platform's reset/wait message. Do not repeatedly request GPU allocation.
- Generic inference error or timeout: no partial prediction should be treated as success. Record the time and non-sensitive error text for the maintainer. Do not post the structure, token or owner key in a public Issue.
- Restart: results and the numerical certificate are held on ephemeral storage by default. Owner revalidation may be needed before prediction resumes. The static website link does not guarantee the Space is ready.

There is no uptime SLA, unlimited GPU promise, clinical-use claim or journal certification. The current evidence covers two fixed numerical cases and their ordinary output downloads; see [acceptance status](RESEARCH_BETA_STATUS_20261023.md).
