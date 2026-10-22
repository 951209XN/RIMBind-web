---
title: RIMBind
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# RIMBind full inference candidate

**Not yet formally released or target-Space numerically accepted.** This is a
real offline-model backend candidate, separate from the input/GPU preflight.
Installing this source alone is insufficient: ten frozen C checkpoints, two
training-memory files, the frozen Foldseek binary and databases, SaProt and the
locked ESM3 snapshot must be installed by an authorized administrator. No model
assets, credentials, query labels or laboratory addresses are part of this
source package. By default no asset is downloaded. The optional, explicit
owner-authorized installer reads two checksum-pinned PRIVATE bundles and exact
upstream encoder revisions; it never uploads files or changes any account.

## Dependency isolation (build-error repair)

Gradio 6.30.0 requires Hub >=1.16 while frozen Transformers 4.43.3 requires Hub
<1.0. They cannot share one dependency environment. The UI retains current
Gradio and Hub 1.16.1; the frozen model runs in a **clean separate venv** with
Transformers 4.43.3, esm 3.1.1 and Hub 0.36.2. No `--no-deps`, metadata edits,
UI security downgrade or encoder upgrade is used. The private asset ZIPs,
frozen weights, fusion parameters and reference-source bytes are unchanged.

On first authorized startup, the installer creates this model venv from official
PyPI wheels, runs `pip check` and imports the actual model factories. It can take
several minutes. A failure leaves prediction disabled. Once GPU allocation is
granted, an internal JSON/NPZ call invokes that interpreter; no visitor pickle,
shell command or Hub/owner Secret is forwarded. The model process is killed on
timeout or Linux parent death to avoid keeping GPU work alive after preemption.
The latest certificate is passed explicitly, not read from stale ZeroGPU fork
globals. CUDA fingerprint checks occur in the actual model process.

The two-environment Linux/Python 3.12 dependency resolutions and lab GPU
subprocess test are separate evidence. **Neither establishes actual target
ZeroGPU child-process compatibility**, a successful clean target installation
or public-service acceptance. Those gates must still be tested on the Space.

The five-fold model, DNA FULL and RNA CR rules are unchanged. Never substitute
base-only predictions for RIMBind when the memory assets are missing.

Set `RIMBIND_ASSET_CONFIG` to the administrator's local SHA256 asset lock. Use
`tools/create_asset_config.py` only on trusted assets, not uploaded PyTorch files.
Do not publish that configuration if it contains private paths. After actual
target-Space numerical testing, `RIMBIND_ENABLE_PREDICTION=1` permits prediction
only when the local certificate has BOTH passing fixed cases. The environment
flag alone cannot open the gate. Each allocated prediction checks that the
assigned GPU/runtime still matches the certificate; changes fail closed.

For owner validation, set `RIMBIND_VALIDATION_CASES` to the private six-file QA
directory and create `RIMBIND_VALIDATION_KEY` as a Space Secret. Never put that
key in source or chat. The owner-only panel requires the key before CPU retrieval
or GPU allocation. Run DNA and RNA separately; download the resulting certificate.
The optional `RIMBIND_CERTIFICATE` points to the administrator's local certificate,
not an anonymous upload. Ephemeral storage may require revalidation after restart.
The two original PDBs, historical probabilities and memory-only QA vectors remain
private; they are not part of this source archive.

## Optional authorized hosted installation

Keep `XuNing985/RIMBind-assets` PRIVATE. Upload `RIMBind_PRIVATE_core_assets.zip`
and `RIMBind_PRIVATE_runtime_support.zip` there, NOT to the public Space or GitHub.
The supplement contains only the two fixed label-free QA cases, their historical
probabilities, memory-reference vectors and the frozen Foldseek binary/license.
The owner supplies `HF_TOKEN` as a Space Secret with the minimum necessary read
access. Set `RIMBIND_INSTALL_ASSETS=1` and, after reviewing the applicable upstream
terms, `RIMBIND_THIRD_PARTY_TERMS_CONFIRMED=1` as variables. SaProt/ESM3 are fetched
from their original upstream repositories at exact historical revisions and
checked file by file. No arbitrary latest-weight fallback is permitted. A gated,
removed or renamed revision that cannot be resolved must fail rather than change
the model. Upstream access/license issues can require owner action.

The installer checks that the RIMBind Model repository is still private. It
rejects changed archives, unexpected files, traversal, symlinks in archives and
oversized contents. Model pickles are read only after the normal asset-lock
verification. Private paths, tokens and training labels are not served as query
results. Do not put tokens in Variables: [Space Secrets guidance](https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets-and-environment-variables).

The private bootstrap has not yet been executed on the real target Space; it
does not constitute evidence of successful deployment. A lab offline replay or
mocked downloader test cannot certify Hub network access, gated-model access or
the target Python 3.12 dependency build.

Before manuscript release: verify both tasks on the assigned target GPU,
compare raw-input probabilities against the frozen original computations on the
SAME assigned hardware at the unchanged 1e-5 tolerance, retain historical
cross-hardware comparisons separately, and test unseen-input mapping/error paths, cold/warm latency, quota,
downloads, cleanup and an unauthenticated external browser. Do not change the
paper's weights, parameters or tolerance to make a deployment pass.

The frozen server's original package versions and this single-environment
deployment candidate differ. A successful test on the laboratory GPU does not
replace the target Space test. The dependency resolver and esm source package
must also be checked in the actual Python 3.12 build.

Non-commercial scientific use only as authorized by the project owner;
third-party model/code licenses apply separately. No continuous-availability or
unlimited-free-GPU promise is made.
