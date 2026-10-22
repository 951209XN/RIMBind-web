# GitHub + external free compute

## GitHub Pages

1. Upload **only this repository directory**, never its parent `private_assets/`, `audit/`, manuscript folder, credentials or `.pt` files.
2. Repository Settings → Pages → Source: GitHub Actions.
3. Run “Scientific contracts”. Resolve actual failures.
4. “Publish pre-release interface” runs on changes to `site/` or its workflow on `main`, or can be run manually. Use the URL returned by the deployment, not an assumed URL.
5. Confirm the page displays “inference validation pending”. The initial page is not a validated live WebServer.

## Hugging Face

The account is `XuNing985`; an email address is not the repository namespace. Public metadata shows creation on 2025-10-09 and no PRO subscription. This satisfies only the published age condition; it does not prove email verification or ZeroGPU eligibility.

Current official ZeroGPU documentation lists free personal hosting for eligible accounts, Gradio-only integration, PyTorch 2.8+ compatibility and 48/96GB shared GPU options. Daily quotas and queues apply. Do not start paid GPU Spaces or Jobs as a workaround for missing free access.

The connected research integration has read-repository and Jobs scopes, not repository-write scope. It can inspect identity but cannot by itself create/upload a Space. Use the owner's browser or explicit repository-write authorization. Do not paste access tokens into chat or source code.

### Preflight Space package

`deploy/hf-space/` is a small Gradio diagnostic app, **not full inference**. It verifies input parsing and documents live-release gates. Upload the complete repository with the Space README at its root and app entry `app.py`; `tools/prepare_space.py` builds the upload folder without weights. Choose free ZeroGPU only if the account UI explicitly confirms it is free and eligible. A paid hardware selection is outside this task.

### What must exist before live inference

- Authorized private task checkpoint bundle, all ten original SHA256 matches.
- Original DNA/RNA training-memory label files and the **original predicted memory structures**, with stable template IDs and per-file checksums. Do not replace them with bound experimental chains.
- Exact SaProt and ESM3 weights, tokenizer/config files and validated loading paths. The upstream ESM3 repository has moved; a new namespace/license does not imply the old pinned checkpoint is unchanged.
- Linux-compatible original Foldseek binary/DB with a validated checksum and homolog-filter protocol.
- Separate target-GPU checks for raw PDB → 3Di → SaProt/ESM3 → five-fold C → memory projection → DNA FULL/RNA CR. Compare original same-input probabilities, not just output lengths.
- Isolated jobs, queued execution, failures, size/chain validation, private result links, downloadable residue mapping and explicit cleanup policy.

The present repository intentionally does **not** enable a partially implemented backend or silently return SSCP-only scores as full RIMBind. After numerical/full-pipeline acceptance, connect a real backend and update status truthfully.

## Official references checked on 2026-10-21

- https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://huggingface.co/docs/hub/spaces-overview
- https://academic.oup.com/nar/pages/Submission_Webserver

NAR's dedicated Web Server criteria are an aspirational reference, not an assertion of certification or of eligibility for every journal. Free shared GPU quotas may be inadequate for a long-term no-login service; this must be tested, not promised.
