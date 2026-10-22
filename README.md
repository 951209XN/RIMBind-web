# RIMBind Web

A pre-release scientific web interface and external-compute deployment workspace.

## Deployment status

**Not yet a validated live prediction server.** GitHub Pages hosts the interface only. The application currently offers a documented precomputed DNA example, browser-local structure inspection, visualization of existing probability tables, and CSV/PDB/provenance downloads. New-protein inference is disabled. No laboratory server is exposed or contacted by this project.

Public interface: https://951209xn.github.io/RIMBind-web/ . Pages deployment succeeded on 2026-10-22 at source commit `b8af380915cba74902bc0418185f4ce746a0cff1`. All ten public files returned HTTP 200 and matched the local source byte-for-byte. This is static-publication acceptance, not model-serving acceptance. [Deployment run](https://github.com/951209XN/RIMBind-web/actions/runs/39714389887).

### Completed locally

- Frozen DNA FULL / RNA CR formula routing and five-fold probability aggregation contracts.
- Strict PDB chain/sequence intake; model positions are distinguished from original residue keys.
- Sixteen Python contract tests (including six Space-package checks) and seven browser-data/JavaScript contract tests.
- Gradio 6.30.0 / Python 3.12 preflight UI constructed and served locally; the 53-residue input check and wrong-chain/sequence rejection passed. This is not a hosted-Space or model test.
- All ten task-specific checkpoint and two training-memory-label checksums verified against the original release manifest, without publishing those files.
- Prepared-feature model-C test on the 53-residue 6JHE A example. CPU PyTorch 2.8 differed from the original output by **1.3380931289663778e-05**, above the preselected **1e-05** tolerance. Numerical acceptance therefore **did not pass**; the threshold was not relaxed. This is not a raw-PDB or GPU test.

### Remaining release gates

1. Confirmed free ZeroGPU hardware and repository write access for the owner's newly created [Space](https://huggingface.co/spaces/XuNing985/RIMBind-web). Its initial repository exists but contains no application yet. A read-only research connection or paid Jobs API is not a substitute.
2. A private, authorized frozen asset bundle including the original predicted training-memory structures and exact foundation encoders/tokenizers. The local checkpoint archive alone is insufficient.
3. End-to-end raw-PDB processing and numerical acceptance on the target GPU environment, including separate DNA FULL and RNA CR examples.
4. Upload privacy/retention, queue, failure, rate-limit and outside-network browser tests. Public browser navigation did not complete during this session; HTTP resource checks are not a replacement for browser interaction tests.

Do not describe the service as available for online inference in a manuscript before all applicable gates pass. Do not treat an example explorer as a prediction server.

## Layout

| Directory | Purpose |
|---|---|
| `site/` | Buildless GitHub Pages interface. No backend tokens or uploads. |
| `backend/` | Input contracts, verified fusion math, frozen-C loader and deployment diagnostics. |
| `vendor/` | Original source snapshots; original Apache-2.0 notice applies. |
| `configs/` | Task rules and checkpoint SHA256 manifest, not weights. |
| `tests/` | Contract checks. |
| `deploy/` | Hugging Face preflight package and deployment guide. |
| `tools/` | Core numerical acceptance and asset inspection. |

## Local preview

From this repository:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8765/`. This exposes only the `site/` directory on your own machine, not model assets or manuscript files. No Python model runs from the page.

## Tests

```sh
python -m pip install numpy==1.26.4
python -m unittest discover -s tests -v
node --test tests/frontend.test.mjs
```

`tools/core_acceptance.py` additionally requires trusted checkpoints, the prepared-feature fixture and PyTorch. It refuses to call a failed numerical comparison a pass. Never accept `.pt`/pickle files uploaded by anonymous website visitors.

## Model-version contract

The query-intrinsic model is `SingleNABindModel("C")`. Convert each of five fold logits to probabilities, then average those probabilities. DNA uses `BAYES_RIM` (count-based; alpha 0.2, tau 1). RNA uses `NO_TEMPLATE_COUNT` (CR; alpha 0.8, tau 4). Prior 0.1, TM exponent 2. The B-field visualization uses probabilities 0–1, not percentages. AP, AUROC, F1 and MCC require true query labels and are not calculated for anonymous prediction requests.

The unchanged frozen model file contains an unconditional import of Mamba although variant C never instantiates E/F-only Mamba. The C-only loader removes only that unused import from its execution AST. It does not edit source bytes, replace model layers, change checkpoint keys or implement a substitute Mamba. Numerical acceptance remains a separate mandatory check.

## Hosting

See [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md). The Pages workflow supports manual dispatch and publishes changes to `site/` or its workflow on `main`, following the owner's Pages activation. Publication does not enable model inference. The root CI only tests contracts; it does not invoke a GPU or train a model. Keep all credentials in platform-managed authorization/secrets, never JavaScript, Git, logs or Issues.

## License and assets

Original RIMBind source is Apache-2.0; see `LICENSE`. This does not relicense third-party encoders or study-derived training labels. Weights are not in this repository. The 3Dmol browser component, if included, retains its upstream license. Author approval and applicable upstream terms are required before uploading any model assets to a public host. The manuscript citation and permanent asset archive remain pending.
