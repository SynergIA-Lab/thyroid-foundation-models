# Foundation-model encoders for goiter classification on planar thyroid scintigraphy

Analysis code for the study *Foundation-model encoders for goiter classification on
planar thyroid scintigraphy: a controlled factorial design*. The experiment crosses
the pre-training domain of five frozen image encoders with their learning objective,
and crosses that grid with two region-of-interest strategies, to test whether
pre-training on medical images transfers better to a functional nuclear-medicine
modality once the learning objective is held constant.

Everything runs on CPU. No encoder is fine-tuned, no GPU is required at any stage,
and the complete study, explainability included, reproduces in approximately seven
to eight hours on an eight-core laptop with 16 GB of memory.

The paper is the reference for the methodology, the results and their
interpretation. This file documents only what is needed to install the environment,
run the pipeline and locate the outputs.

## Contents

- [Data](#data)
- [Encoders](#encoders)
- [Installation](#installation)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Design facts that matter for reuse](#design-facts-that-matter-for-reuse)
- [Configuration](#configuration)
- [Citation and license](#citation-and-license)

## Data

The study is a secondary analysis of a public dataset of 641 planar thyroid
scintigraphy images from 235 adult patients, acquired between 2016 and 2024 at the
Nuclear Medicine Service of the Instituto de Investigaciones en Ciencias de la Salud,
Universidad Nacional de Asunción (IICS-UNA), and released on Zenodo under the
CC BY-NC 4.0 license:

> DOI: [10.5281/zenodo.13900963](https://zenodo.org/records/13900964)

The analysis cohort is the binary differential between diffuse and multinodular
goiter, retaining every patient of those two classes: **178 patients and 496 images**
(74 patients / 200 images diffuse, 104 patients / 296 images multinodular). The
remaining diagnostic categories of the source are not used. All but 19 patients
contribute three projections (anterior plus two obliques); the rest contribute the
anterior view only.

The data article declares the acquisition protocol: intravenous
<sup>99m</sup>Tc-pertechnetate at 3–5 mCi, imaging 15 minutes after injection on a
dual-head MEDISO AnyScan SCP gamma camera with a low-energy high-resolution
collimator, a ±20% energy window centered on the 140 keV photopeak, a 256 × 256
matrix, acquired to 400–500 kcounts or at most 5 minutes, with the patient supine
and the neck hyperextended.

Labels are the diagnoses of the reporting physicians, transcribed by the source
center. No independent reference standard such as histopathology, thyroid function
or follow-up was used, so a model trained on these labels reproduces the physician's
reading rather than diagnostic truth.

The original collection was approved by the Research Ethics Committee of the
IICS-UNA under code M06/2024, with written informed consent, and the released images
are anonymized.

### Expected layout

Download the dataset from Zenodo and place it so that each image is addressable by
class and view:

```
data/
  diffuse_goiter/
    ANTERIOR/   LAO/   RAO/
  multinodular_goiter/
    ANTERIOR/   LAO/   RAO/
```

The demographic spreadsheet distributed with the dataset supplies age, sex, weight,
height and administered activity, which the clinical-metadata control arm uses as
its only features.

## Encoders

All five are frozen. Representation width is equalized at **3072 dimensions**: for
the vision transformers by concatenating the class token of the last four blocks,
and for ResNet-50 by concatenating the globally pooled feature maps of the last two
stages, whose 1024 and 2048 channels sum to the same width.

| Encoder | Pre-training domain | Objective | Checkpoint |
|---|---|---|---|
| ResNet-50 | natural | supervised | `torchvision` `ResNet50_Weights.IMAGENET1K_V2` |
| DINOv2 ViT-B/14 | natural | self-supervised | `torch.hub` `facebookresearch/dinov2` → `dinov2_vitb14` |
| CLIP ViT-B/16 | natural | image–text | `timm` `vit_base_patch16_clip_224.openai` |
| BiomedCLIP ViT-B/16 | medical | image–text | `open_clip` `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| RAD-DINO ViT-B/14 | medical | self-supervised | `transformers` `microsoft/rad-dino` |

DINOv2 is the reference encoder: every declared contrast is taken against it, and it
is the encoder whose attention map defines the attention-guided crop.

Weights are downloaded on first use and are not redistributed here. The first run of
Phase 2 therefore requires network access; later runs read the cached checkpoints.

## Installation

Python 3.10 with pinned dependencies.

```bash
conda env create -f envs/environment.yml
conda activate thyroid_env
```

The pinned versions are `torch==2.2.1`, `torchvision==0.17.1`,
`scikit-learn==1.3.2`, `pandas==2.1.4`, `numpy==1.26.4`, `scipy==1.11.4`,
`opencv-python==4.8.1.78` and `matplotlib==3.8.2`, plus `timm`,
`transformers<4.45.0` and `open_clip_torch`, which supply the encoder backbones.
The `transformers` upper bound is required by the RAD-DINO loader.

## Running the pipeline

Six scripts, in order. Each writes to its own subdirectory of `outputs/` and logs to
`outputs/phaseN/phaseN_run.log`.

```bash
python phases/phase0_evidence.py          # dataset audit: inventory, demographics, confounders
python phases/phase1_data_prep.py         # cohort table and the nested cross-validation splits
python phases/phase2_features_baseline.py # ROI crops and frozen features for five encoders
python phases/phase3_experiments.py       # image-level classification, 22 configurations
python phases/phase4_patient_level_fusion.py  # patient-level aggregation, 16 configurations
python phases/phase5_xai_analysis.py      # saliency faithfulness against matched controls
```

The dependencies are strictly sequential: Phase 1 produces the splits that every
later phase reads, and Phase 2 produces the feature tensors that Phases 3 to 5
consume. Phase 2 dominates the runtime; feature extraction for all five encoders on
both crop tracks takes 863 seconds in total, and the per-image latency on CPU ranges
from 70 ms for ResNet-50 to 242 ms for RAD-DINO.

**Changing `STR_COHORT_STRATEGY` or `INT_NUM_REPEATS` in `phases/phase1_data_prep.py`
invalidates every downstream output.** The splits are serialized once and all later
phases address samples by identifier, so a change there requires re-running Phases 1
to 5 in full.

## Outputs

`outputs/` is committed, so every reported value can be recomputed without re-running
encoder inference.

| Path | Contents |
|---|---|
| `phase0/phase0_evidence.json` | inventory, per-class demographics, confounder tests |
| `phase1/phase1_master_table.csv` | one row per image with class, view and clinical covariates |
| `phase1/phase1_nested_cv_splits.json` | the 50 outer folds and their inner calibration partitions |
| `phase1/phase1_confounder_tests.json` | age, sex and anthropometry tested between classes |
| `phase2/*_features_*.npz` | frozen 3072-dimensional features, one row per image |
| `phase2/phase2_roi_crops.npz` | the fixed and attention-guided crops, shared by all encoders |
| `phase2/phase2_roi_quantitative_metrics.csv` | per-image crop geometry and texture statistics |
| `phase2/phase2_encoder_extraction_metrics.csv` | latency, memory and embedding norms per encoder |
| `phase3/phase3_metrics_table.csv` | the 22 image-level configurations, four metrics each |
| `phase3/phase3_per_fold_metrics.csv` | the same, one row per fold |
| `phase3/phase3_statistical_comparisons.csv` | 22 contrasts × 4 metrics = 88 rows, with intervals and adjusted *p* |
| `phase4/phase4_metrics_table.csv` | the 16 patient-level configurations |
| `phase4/phase4_statistical_comparisons.csv` | 15 contrasts × 4 metrics = 60 rows |
| `phase4/phase4_attention_weights.csv` | the learned per-view weights of the gated attention module |
| `phase4/phase4_view_importance.csv` | those weights aggregated by crop track and view |
| `phase5/phase5_xai_quantitative.csv` | per-image saliency concentration and faithfulness |

Tables are written as `.csv` for analysis and mirrored as `.md` and `.tex` where they
appear in the manuscript. Figures are written as `.png` and `.pdf`.

The per-fold tables are the right entry point for re-deriving any aggregate: the
reported means are means over the ten repetition-level means, not over the fifty
folds, and both dispersions are recoverable from these files.

## Design facts that matter for reuse

**Partitions.** Ten-times-repeated five-fold nested cross-validation, grouped by
patient and stratified by class, giving 50 outer folds. Grouping by patient is not
optional: a patient contributes up to three images, so an image-level split would
place views of the same patient on both sides and the model would recognize the
patient rather than the condition.

**Leakage gate.** `assert_folds_patient_disjoint` runs over all 50 folds before any
model is fitted, on both the outer and the inner splits. A violation aborts the run.

**Decision thresholds.** Youden's J, fitted on each of the four inner calibration
partitions and averaged, never on the validation block. The alternative equal-error
criterion is computed and stored per fold in `threshold_alternative` but is not the
operating point used for the reported metrics.

**Shallow head.** `StandardScaler` → `PCA(n_components=100, whiten=True)` →
`SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')`, fitted inside the
fold so the projection never sees validation data. Ranking metrics are computed from
`decision_function` rather than `predict_proba`, because `SVC(probability=True)`
refits Platt scaling on an internal split and its probabilities are not guaranteed to
be a monotone transformation of the final model's decision function.

**Clinical control.** A logistic regression on age, sex, weight, height and
administered activity, standardized and without dimensionality reduction. It bounds
what the demographics alone predict and is evaluated at image level only; there is no
patient-level clinical arm.

**Attention-guided crop.** The 16 × 16 head-averaged class-token attention map of the
last DINOv2 block, thresholded at its 70th percentile, cropped to the enclosing
rectangle of the resulting mask padded by a tenth of its side. It is computed **once,
by the reference encoder, and reused by all five**, so that the crop axis and the
encoder axis stay separable and ResNet-50, which has no attention map, can
participate.

**Statistics.** Paired *t*-tests over the ten repetition-level means, with
Benjamini–Hochberg correction applied within each family and metric. The repetitions
are independent draws of the partition, not independent samples: the resulting
intervals describe variability over partitions of this one cohort, not over cohorts.

## Configuration

The constants that change what is run live at the top of each phase script.

| Script | Constant | Effect |
|---|---|---|
| `phase1_data_prep.py` | `INT_NUM_REPEATS` | number of cross-validation repetitions (10) |
| `phase1_data_prep.py` | `STR_COHORT_STRATEGY` | complete cohort with class weighting, or majority-class subsampling |
| `phase1_data_prep.py` | `INT_GLOBAL_SEED` | base seed; repetition *i* uses `seed + i` |
| `phase2_features_baseline.py` | `LST_ENCODERS_TO_RUN` | which encoders to extract |
| `phase2_features_baseline.py` | `LST_ROI_MODES` | which crop tracks to extract |
| `phase3_experiments.py` | `INT_PCA_COMPONENTS` | width of the whitened projection (100) |
| `phase3_experiments.py` | `DICT_FIXED_SVM_PARAMS` | the fixed `C` and `gamma` of the shallow head |

An encoder whose weights cannot be downloaded is recorded as skipped rather than
aborting the run, so a multi-hour execution is not lost to one unavailable backbone.

## Citation and license

This paper is under review. We would appreciate citations to the following paper:

The code in this repository is released for review and reuse under the terms stated
in `LICENSE`.
