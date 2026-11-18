# Data and code availability

The sources of the public training and benchmark test sets are described in the manuscript. Additional methodological details and evaluation results are provided in the Supplementary Material. RIMBind predictions are accessible through the web service linked at https://951209xn.github.io/RIMBind-web/.

The trained, task-specific SSCP model checkpoints used by RIMBind, their SHA256 manifests, and scripts for checkpoint verification and residue-level performance evaluation are available in this repository under `checkpoints/` and `evaluation/` (release ID: `20261118-frozen-C`). Source code for the web interface and service integration is also provided. The evaluation scripts assess supplied residue-level probabilities against aligned experimental labels. Training and ablation workflows, training-memory assets, and third-party encoder weights are outside the scope of this release; it is not a complete end-to-end reproduction package.
