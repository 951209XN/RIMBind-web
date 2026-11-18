# Frozen weights and residue-level evaluation

Release ID: `20261118-frozen-C`.

This release provides the ten trained SSCP model-C checkpoints used by RIMBind (five DNA folds and five RNA folds), checkpoint verification, and evaluation of externally supplied residue-level probabilities. The checkpoint files are byte-identical to the frozen files identified in the original benchmark prediction records. DNA and RNA ensembles are separate; average the five fold probabilities after applying sigmoid to each fold's logits.

The manuscript configuration is DNA FULL and RNA CR. These identifiers describe the evaluated versions; they are not alternative settings to select using test performance. The release does not contain training scripts, ablation experiments, training commands, training-memory assets, third-party encoder weights, or benchmark structures/features. The evaluation script calculates metrics from supplied predictions; it does not generate predictions or provide a complete end-to-end reproduction package. The existing web predictor remains linked from the [project website](https://951209xn.github.io/RIMBind-web/).

## Verify checkpoint identity

From the repository root:

```sh
python evaluation/verify_weights.py
```

This checks all ten file sizes and SHA256 digests using the Python standard library. The manifest is [checkpoints/manifest.json](../checkpoints/manifest.json), with a second checksum listing in [checkpoints/SHA256SUMS](../checkpoints/SHA256SUMS). With PyTorch installed, also check safe deserialization, finite tensors, model variant, tensor names, shapes and dtypes:

```sh
python evaluation/verify_weights.py --check-tensors
```

The original PyTorch checkpoint format is retained. Model parameters are in `state_dict` and the model identifier is `variant: C`. Existing checkpoint metadata is preserved so that the original file hashes remain verifiable. No optimizer state or training/ablation command is included.

## Evaluate labeled predictions

```sh
python -m pip install -r evaluation/requirements.txt
python evaluation/evaluate.py --labels labels.csv --predictions predictions.csv --output metrics.json
```

The script requires two CSV or TSV files. The shared residue key is `(dataset, protein_id, residue_index)`. `residue_index` is the positive, 1-based sequence index used consistently in both files, not an author PDB residue number. Protein/chain identifiers must distinguish different chains. Map server output explicitly to these keys before scoring; do not assume that equal row counts establish correspondence.

`labels.csv` contains the reference labels and, optionally, a fixed evaluation mask:

```csv
dataset,protein_id,residue_index,y_true,eval_mask
example,protein_A,1,1,1
example,protein_A,2,0,1
```

`predictions.csv` contains probabilities for the corresponding residues:

```csv
dataset,protein_id,residue_index,probability
example,protein_A,1,0.8
example,protein_A,2,0.2
```

If `eval_mask` is omitted, all labeled rows are evaluated. A value of `0` excludes a residue according to the supplied evaluation definition; missing experimental labels must not be converted to negatives. Masks must be established independently of prediction performance. Every evaluable key must have exactly one prediction. Duplicates, unknown prediction keys, missing evaluable predictions, invalid masks, nonbinary labels, and nonfinite/out-of-range probabilities abort scoring. Input order may differ because alignment uses the complete residue key.

AP is non-interpolated average precision with tied scores grouped; it is not trapezoidal PR-AUC. AUROC uses all evaluable scores. F1, MCC, precision and recall use the fixed decision rule `probability >= 0.5`. Metrics pool residues within each dataset. AUROC is `null` for a single-class dataset; AP is zero if there are no positives, and undefined precision, recall, F1 or MCC are reported as zero. Output includes sample counts, confusion counts and input file hashes. Coverage is checked against the supplied label manifest; this cannot establish that a user-supplied manifest is itself a complete benchmark.

An unlabeled server submission alone cannot support AP, AUROC, F1 or MCC calculation. Obtain the relevant experimental labels and fix the evaluation scope before running this script.

## Release validation

Before publication, all ten checkpoints passed original-manifest SHA256 checks, CPU tensor checks, strict loading into the unchanged model-C architecture, and comparison to the checkpoint hashes stored in the original five-fold benchmark prediction records. The evaluator was checked against scikit-learn for tied-score and decision-boundary cases and run on the frozen canonical residue predictions (150,118 residues across the three benchmark sets). The resulting RIMBind metrics match the manuscript values:

| Dataset | Residues | AP | AUROC | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| DNA-129 | 37,515 | 0.6188 | 0.9522 | 0.5491 | 0.5464 |
| DNA-181 | 75,258 | 0.4620 | 0.9338 | 0.4082 | 0.4225 |
| RNA-117 | 37,345 | 0.3822 | 0.8987 | 0.4033 | 0.3921 |

These checks verify released checkpoint identity and the calculation of metrics from existing frozen predictions. They are not a new GPU inference run, a retraining result, or a claim that all required prediction assets are publicly distributed.
