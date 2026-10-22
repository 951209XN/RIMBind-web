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
source package. No asset is downloaded automatically.

The five-fold model, DNA FULL and RNA CR rules are unchanged. Never substitute
base-only predictions for RIMBind when the memory assets are missing.

Set `RIMBIND_ASSET_CONFIG` to the administrator's local SHA256 asset lock. Use
`tools/create_asset_config.py` only on trusted assets, not uploaded PyTorch files.
Do not publish that configuration if it contains private paths. After actual
target-Space numerical testing, `RIMBIND_ENABLE_PREDICTION=1` enables prediction;
that flag alone is **not** evidence that the model passed acceptance.

Before manuscript release: verify both tasks on the assigned target GPU,
compare raw-input probabilities to existing same-input references at the fixed
1e-5 tolerance, test unseen-input mapping/error paths, cold/warm latency, quota,
downloads, cleanup and an unauthenticated external browser. Do not change the
paper's weights, parameters or tolerance to make a deployment pass.

The frozen server's original package versions and this single-environment
deployment candidate differ. A successful test on the laboratory GPU does not
replace the target Space test. The dependency resolver and esm source package
must also be checked in the actual Python 3.12 build.

Non-commercial scientific use only as authorized by the project owner;
third-party model/code licenses apply separately. No continuous-availability or
unlimited-free-GPU promise is made.
