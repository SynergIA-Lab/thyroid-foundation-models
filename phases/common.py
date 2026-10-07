"""
Shared Utilities for the Thyroid Scintigraphy Classification Pipeline
====================================================================

This module centralises every operation that must behave identically across
phases: table loading, feature alignment, cross-validation split handling,
patient-level leakage assertions, threshold calibration and metric reporting.

Design rules enforced here
--------------------------
1. NEVER index a feature matrix or a metadata table with positional integers
   taken from a different table. All alignment is performed through the
   immutable 'sample_id' key.
2. NEVER derive a decision threshold from the outer validation fold. The
   threshold is a hyper-parameter and is calibrated exclusively on the inner
   calibration split produced by Phase 1.
3. Every phase that consumes cross-validation splits calls
   assert_folds_patient_disjoint() before training anything. A patient whose
   views appear on both sides of a split raises an exception instead of
   silently inflating the reported metrics.

Usage:
    from common import (load_balanced_table, load_features, load_splits,
                        assert_folds_patient_disjoint, select_threshold_youden,
                        calculate_full_metrics, summarise_across_repeats, TeeLogger)
"""

import json
import random
import sys
from pathlib import Path

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             recall_score, roc_auc_score, roc_curve)

# --------------------------------------------------------------------------
# Canonical paths (all relative to the repository root)
# --------------------------------------------------------------------------
pathDataDir = Path("data")
pathOutputsDir = Path("outputs")
pathPhase0Dir = pathOutputsDir / "phase0"
pathPhase1Dir = pathOutputsDir / "phase1"
pathPhase2Dir = pathOutputsDir / "phase2"
pathPhase3Dir = pathOutputsDir / "phase3"
pathPhase4Dir = pathOutputsDir / "phase4"
pathPhase5Dir = pathOutputsDir / "phase5"

pathMasterCsv = pathPhase1Dir / "phase1_master_table.csv"
pathBalancedCsv = pathPhase1Dir / "phase1_balanced_table.csv"
pathSplitsJson = pathPhase1Dir / "phase1_nested_cv_splits.json"

# --------------------------------------------------------------------------
# Study-wide constants
# --------------------------------------------------------------------------
LST_TARGET_CLASSES = ["diffuse_goiter", "multinodular_goiter"]

# Explicit label mapping. Never rely on LabelEncoder's alphabetical ordering:
# the positive class must be documented in the manuscript.
DICT_CLASS_TO_LABEL = {"diffuse_goiter": 0, "multinodular_goiter": 1}
STR_POSITIVE_CLASS = "multinodular_goiter"

LST_METRIC_KEYS = ["AUC", "PR-AUC", "MCC", "F1", "Sensitivity", "Specificity",
                   "Accuracy", "Balanced Accuracy"]

# Clinical covariates used for the metadata-only reference model. This arm
# quantifies how much of the discrimination is attributable to demographics
# alone, and is a mandatory comparison for any imaging claim.
LST_CLINICAL_COLUMNS = ["AGE", "WEIGHT (Kg)", "HEIGHT (cm)",
                        "DOSE ADMINISTERED (mCi)"]

INT_GLOBAL_SEED = 42

# --------------------------------------------------------------------------
# Encoder registry
# --------------------------------------------------------------------------
# The study compares five frozen encoders that differ along TWO axes, so that
# the two axes can be separated instead of confounded:
#
#   axis 1  PRE-TRAINING DOMAIN     natural images  vs  medical images
#   axis 2  PRE-TRAINING OBJECTIVE  supervised / self-supervised / vision-language
#
# None of them is trained on nuclear medicine, so the question this design
# answers is not "does a medical encoder win" but "which axis matters more for
# planar scintigraphy". The contrasts that fall out of it are:
#
#   DINOv2 vs CLIP        objective effect, domain held constant
#   DINOv2 vs RAD-DINO    domain effect, objective held constant
#   CLIP   vs BiomedCLIP  domain effect replicated under the other objective
#   DINOv2 vs ResNet-50   does the foundation model earn its place at all
#
# The last one is the FLOOR. Without it no reviewer can tell whether the
# foundation model contributes anything specific or whether any reasonable
# representation would do, and the paper's central claim stops being falsifiable.
#
# 'loader' names the function in phase2_features_baseline.py that builds it.
# 'patch_grid' is the number of patch tokens (14x14=196 for /16 backbones at
# 224px, 16x16=256 for /14). Phase 3's convolutional head reshapes patch tokens
# to a square grid, so it must adapt per encoder rather than assume 256.
DICT_ENCODERS = {
    "dinov2": {
        "display_name": "DINOv2 ViT-B/14",
        "domain": "natural",
        "objective": "self-supervised",
        "family": "vit",
        "patch_grid": 16,
        "embed_dim": 768,
        "n_layers_concat": 4,
        "loader": "load_dinov2",
        "role": "reference foundation model (defines the dynamic ROI crops)",
    },
    "clip": {
        "display_name": "CLIP ViT-B/16",
        "domain": "natural",
        "objective": "vision-language",
        "family": "vit",
        "patch_grid": 14,
        "embed_dim": 768,
        "n_layers_concat": 4,
        "loader": "load_clip",
        "role": "same domain as DINOv2, different objective",
    },
    "biomedclip": {
        "display_name": "BiomedCLIP ViT-B/16",
        "domain": "medical",
        "objective": "vision-language",
        "family": "vit",
        "patch_grid": 14,
        "embed_dim": 768,
        "n_layers_concat": 4,
        "loader": "load_biomedclip",
        "role": "same objective as CLIP, medical domain",
    },
    "raddino": {
        "display_name": "RAD-DINO ViT-B/14",
        "domain": "medical",
        "objective": "self-supervised",
        "family": "vit",
        "patch_grid": 16,
        "embed_dim": 768,
        "n_layers_concat": 4,
        "loader": "load_raddino",
        "role": "same objective as DINOv2, medical domain",
    },
    "resnet50": {
        "display_name": "ResNet-50 (ImageNet)",
        "domain": "natural",
        "objective": "supervised",
        "family": "cnn",
        "patch_grid": 7,
        "embed_dim": 2048,
        "n_layers_concat": 2,
        "loader": "load_resnet50",
        "role": "CONVENTIONAL TRANSFER-LEARNING FLOOR",
    },
}

# Evaluation order used in every table and figure, weakest expected first so the
# floor is visually adjacent to the models it is meant to bound.
LST_ENCODER_ORDER = ["resnet50", "dinov2", "clip", "biomedclip", "raddino"]

# The encoder that defines the attention-guided crops. Every other encoder
# receives THE SAME cropped images, which is what keeps the ROI axis and the
# encoder axis separable: if each encoder cropped with its own attention, a
# difference between encoders would confound representation with field of view.
# It is also the only way ResNet-50 can participate, since it has no attention.
STR_REFERENCE_ENCODER = "dinov2"


def set_global_seed(intSeed=INT_GLOBAL_SEED):
    """
    Seeds every random number generator that influences the pipeline.
    Torch is seeded lazily so that this module stays importable without torch.
    """
    random.seed(intSeed)
    np.random.seed(intSeed)
    try:
        import torch
        torch.manual_seed(intSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(intSeed)
    except ImportError:
        pass


def get_torch_device():
    """
    Resolves the best available compute device without assuming a discrete GPU.
    Falls back through CUDA -> Metal -> CPU so the pipeline runs unmodified on
    any workstation.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# Metadata tables
# --------------------------------------------------------------------------
def _load_table(pathCsv):
    if not pathCsv.exists():
        raise FileNotFoundError(
            f"{pathCsv} not found. Run 'python phases/phase1_data_prep.py' first."
        )
    dfTable = pd.read_csv(pathCsv)
    dfTable = dfTable[dfTable["class"].isin(LST_TARGET_CLASSES)].reset_index(drop=True)
    if dfTable["sample_id"].duplicated().any():
        raise ValueError(f"Duplicated sample_id values found in {pathCsv}.")
    return dfTable


def load_master_table():
    """Full unbalanced inventory: every parsed image with its clinical record."""
    return _load_table(pathMasterCsv)


def load_balanced_table():
    """
    Patient-level class-balanced cohort. THIS is the analysis cohort: the
    cross-validation splits in phase1_nested_cv_splits.json are defined over
    the patients of this table and over no other.
    """
    return _load_table(pathBalancedCsv)


def get_labels(dfTable):
    """Integer label vector aligned row-by-row with dfTable."""
    return dfTable["class"].map(DICT_CLASS_TO_LABEL).astype(int).values


def build_clinical_matrix(dfTable):
    """
    Numeric design matrix for the metadata-only reference model.
    Missing values are imputed with the column median; SEX is one-hot encoded
    as a single indicator because the cohort contains exactly two categories.
    """
    dfClinical = dfTable[LST_CLINICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    dfClinical = dfClinical.fillna(dfClinical.median())
    dfClinical = dfClinical.assign(SEX_FEMALE=(dfTable["SEX"] == "FEMALE").astype(int))
    return dfClinical.values.astype(np.float64), list(dfClinical.columns)


# --------------------------------------------------------------------------
# Feature tensors
# --------------------------------------------------------------------------
def feature_file(strRoiMode, strEncoder=STR_REFERENCE_ENCODER):
    """
    Canonical path of the feature file for one (encoder, ROI mode) pair.

    The legacy naming 'dinov2_features_<roi>.npz' is preserved for the reference
    encoder so that existing files keep working; every other encoder uses
    '<encoder>_features_<roi>.npz'.
    """
    if strEncoder == STR_REFERENCE_ENCODER:
        return pathPhase2Dir / f"dinov2_features_{strRoiMode}.npz"
    return pathPhase2Dir / f"{strEncoder}_features_{strRoiMode}.npz"


def available_encoders(lstRoiModes=("fixed", "dynamic")):
    """
    Encoders whose feature files exist on disk for ALL requested ROI modes.

    Phases 3 and 4 call this so that a partially completed Phase 2 degrades to
    "evaluate what is available" instead of crashing, while still reporting
    exactly what was found.
    """
    lstFound = []
    for strEncoder in LST_ENCODER_ORDER:
        if all(feature_file(m, strEncoder).exists() for m in lstRoiModes):
            lstFound.append(strEncoder)
    return lstFound


def load_features(strRoiMode, dfTable, boolLoadPatch=False,
                  strEncoder=STR_REFERENCE_ENCODER):
    """
    Loads the Phase 2 embeddings and REORDERS them to match dfTable row order
    by 'sample_id'.

    Phase 2 writes a 'sample_ids' array alongside the tensors precisely so that
    this alignment is explicit rather than an assumption about row order. Any
    sample present in dfTable but absent from the tensor file raises an error.

    Parameters
    ----------
    strEncoder : key of DICT_ENCODERS. Defaults to the reference encoder so that
                 every existing call site keeps its previous behaviour.

    Returns
    -------
    arrCls   : (n_rows, D) multi-layer CLS embeddings, aligned to dfTable
    arrPatch : (n_rows, P, d) patch tokens aligned to dfTable, or None
    """
    pathFeatures = feature_file(strRoiMode, strEncoder)
    if not pathFeatures.exists():
        raise FileNotFoundError(
            f"{pathFeatures} not found (encoder '{strEncoder}', ROI '{strRoiMode}'). "
            "Run 'python phases/phase2_features_baseline.py' first."
        )

    dictNpz = np.load(pathFeatures, allow_pickle=False)
    if "sample_ids" not in dictNpz:
        raise KeyError(
            f"{pathFeatures} was produced by an outdated Phase 2 that did not store "
            "'sample_ids'. Re-run Phase 2 so that features can be aligned by key "
            "instead of by row position."
        )

    arrFeatureIds = np.asarray(dictNpz["sample_ids"], dtype=str)
    dictIdToRow = {strId: intRow for intRow, strId in enumerate(arrFeatureIds)}

    lstMissing = [s for s in dfTable["sample_id"] if s not in dictIdToRow]
    if lstMissing:
        raise KeyError(
            f"{len(lstMissing)} sample_id values from the metadata table are absent "
            f"from {pathFeatures.name} (first missing: {lstMissing[0]}). "
            "Re-run Phase 2."
        )

    arrOrder = np.array([dictIdToRow[s] for s in dfTable["sample_id"]], dtype=int)
    arrCls = dictNpz["cls"][arrOrder]
    arrPatch = dictNpz["patch"][arrOrder] if boolLoadPatch else None
    return arrCls, arrPatch


# --------------------------------------------------------------------------
# Cross-validation splits
# --------------------------------------------------------------------------
def load_splits():
    """
    Reads phase1_nested_cv_splits.json and returns (dictMeta, dictFolds).

    Folds are expressed exclusively as lists of 'sample_id' and 'patient_id'
    strings. The legacy schema stored positional integers ('train_indices')
    computed on the balanced table while later phases indexed the master table
    with them, which silently destroyed the patient grouping. Encountering that
    schema is a hard error, not a fallback.
    """
    if not pathSplitsJson.exists():
        raise FileNotFoundError(
            f"{pathSplitsJson} not found. Run 'python phases/phase1_data_prep.py' first."
        )

    with open(pathSplitsJson, "r", encoding="utf-8") as objFile:
        dictRaw = json.load(objFile)

    if "folds" not in dictRaw:
        raise ValueError(
            f"{pathSplitsJson} uses the deprecated flat schema with positional "
            "'train_indices'/'val_indices'. Those indices referred to the balanced "
            "table while downstream phases applied them to the master table. "
            "Re-run Phase 1 to regenerate the splits in the key-based schema."
        )

    dictMeta = dictRaw["meta"]
    dictFolds = dictRaw["folds"]

    lstRequired = ["train", "val", "train_patients", "val_patients",
                   "inner_train", "inner_cal"]
    for strFoldName, dictFold in dictFolds.items():
        lstAbsent = [k for k in lstRequired if k not in dictFold]
        if lstAbsent:
            raise ValueError(f"Fold '{strFoldName}' is missing keys: {lstAbsent}")
        if "train_indices" in dictFold or "val_indices" in dictFold:
            raise ValueError(
                f"Fold '{strFoldName}' still carries positional indices. "
                "Re-run Phase 1."
            )
    return dictMeta, dictFolds


def get_inner_folds(dictFold):
    """
    Returns the list of inner train/calibration partitions of one outer fold.

    Phase 1 writes an 'inner_folds' list with all INT_NUM_INNER_FOLDS partitions.
    Phases 3 and 4 iterate over them to average the decision threshold, which
    halves its standard error relative to using a single calibration block.

    Falls back to wrapping the flat 'inner_train'/'inner_cal' keys in a
    single-element list when the split file predates that change, so an older
    phase1_nested_cv_splits.json still runs (with the old, noisier behaviour)
    instead of raising. The console output of Phases 3 and 4 states which of the
    two situations applies, so a single-partition run is never silent.

    Returns
    -------
    list of dicts, each with keys 'inner_train' and 'inner_cal' (sample_id lists).
    """
    lstInnerFolds = dictFold.get("inner_folds")
    if lstInnerFolds:
        return lstInnerFolds
    return [{"inner_train": dictFold["inner_train"],
             "inner_cal": dictFold["inner_cal"]}]


def get_fold_row_positions(dfTable, lstSampleIds):
    """
    Translates a list of sample_id strings into row positions of dfTable.
    This is the ONLY sanctioned way to convert a split into array indices.
    """
    dictIdToRow = {strId: intRow for intRow, strId in enumerate(dfTable["sample_id"])}
    lstAbsent = [s for s in lstSampleIds if s not in dictIdToRow]
    if lstAbsent:
        raise KeyError(
            f"{len(lstAbsent)} sample_id values from the split are absent from the "
            f"metadata table (first: {lstAbsent[0]}). The split file and the table "
            "are out of sync; re-run Phase 1."
        )
    return np.array([dictIdToRow[s] for s in lstSampleIds], dtype=int)


def assert_folds_patient_disjoint(dfTable, dictFolds, boolVerbose=True):
    """
    Hard gate against patient-level data leakage.

    Raises AssertionError if any patient contributes images to both sides of a
    split, at the outer level (train vs val) or at the inner level
    (inner_train vs inner_cal). Every phase calls this before training.
    """
    dictSampleToPatient = dict(zip(dfTable["sample_id"], dfTable["patient_id"]))
    lstViolations = []

    for strFoldName, dictFold in dictFolds.items():
        for strKeyA, strKeyB in [("train", "val"), ("inner_train", "inner_cal")]:
            setA = {dictSampleToPatient[s] for s in dictFold[strKeyA]
                    if s in dictSampleToPatient}
            setB = {dictSampleToPatient[s] for s in dictFold[strKeyB]
                    if s in dictSampleToPatient}
            setShared = setA & setB
            if setShared:
                lstViolations.append(
                    f"{strFoldName}: {len(setShared)} patient(s) shared between "
                    f"'{strKeyA}' and '{strKeyB}' (e.g. {sorted(setShared)[:3]})"
                )
        # The inner splits must be a partition of the outer training block.
        setInner = set(dictFold["inner_train"]) | set(dictFold["inner_cal"])
        if not setInner.issubset(set(dictFold["train"])):
            lstViolations.append(
                f"{strFoldName}: inner split contains samples outside the outer "
                "training block"
            )
        if set(dictFold["val"]) & setInner:
            lstViolations.append(
                f"{strFoldName}: outer validation samples leaked into the inner split"
            )

    if lstViolations:
        raise AssertionError(
            "PATIENT-LEVEL LEAKAGE DETECTED in the cross-validation splits:\n  - "
            + "\n  - ".join(lstViolations)
        )

    if boolVerbose:
        print(f"[integrity] {len(dictFolds)} folds verified: no patient appears on "
              "both sides of any outer or inner split.")
    return True


# --------------------------------------------------------------------------
# Thresholding and metrics
# --------------------------------------------------------------------------
def select_threshold_youden(arrYTrue, arrYProb):
    """
    Youden's J optimal cut-off.

    MUST be called with calibration data (inner_cal), never with the outer
    validation fold on which the metrics are subsequently reported. If the
    calibration set happens to be single-class, falls back to 0.5.
    """
    arrYTrue = np.asarray(arrYTrue)
    if len(np.unique(arrYTrue)) < 2:
        return 0.5
    arrFpr, arrTpr, arrThresholds = roc_curve(arrYTrue, arrYProb)
    return float(arrThresholds[np.argmax(arrTpr - arrFpr)])


def select_threshold_equal_error(arrYTrue, arrYProb):
    """
    Cut-off where sensitivity equals specificity (equal-error-rate point).

    WHY THIS EXISTS ALONGSIDE YOUDEN'S J
    ------------------------------------
    Youden's J maximises (TPR - FPR) over an empirical, step-shaped ROC curve.
    With 28-31 calibration patients that curve has few steps and the argmax
    jumps between adjacent, nearly-equivalent operating points: the selected
    threshold had sd ~0.15 and range 0.52-0.65 across the 25 folds, and in 3
    folds specificity saturated at 1.0 while in 3 others sensitivity fell below
    0.60. The equal-error criterion targets the crossing of two monotone curves
    instead of the maximum of a difference, which is markedly less sensitive to
    single-sample changes in a small calibration block.

    This is offered as the STABLE alternative; which of the two is primary is a
    reporting decision that must be declared in the manuscript, with the other
    quoted as a sensitivity analysis. Both are computed from calibration data
    only.

    MUST be called with calibration data, never with the outer validation fold.
    Falls back to 0.5 on a single-class calibration set.
    """
    arrYTrue = np.asarray(arrYTrue)
    if len(np.unique(arrYTrue)) < 2:
        return 0.5
    arrFpr, arrTpr, arrThresholds = roc_curve(arrYTrue, arrYProb)
    # |sensitivity - specificity| = |TPR - (1 - FPR)|; minimised at the crossing.
    arrGap = np.abs(arrTpr - (1.0 - arrFpr))
    return float(arrThresholds[np.argmin(arrGap)])


# Threshold criteria available to the phases. The manuscript must state which
# one produced the headline numbers.
DICT_THRESHOLD_CRITERIA = {
    "youden": select_threshold_youden,
    "equal_error": select_threshold_equal_error,
}


def select_threshold_averaged(lstCalPairs, strCriterion="youden"):
    """
    Averages the decision threshold over several calibration blocks.

    Each element of lstCalPairs is an (arrYTrue, arrYProb) tuple obtained from
    ONE inner calibration partition. Phase 1 materialises four such partitions
    per outer fold; averaging the per-partition thresholds reduces the standard
    error of the threshold by approximately 1/sqrt(k).

    This is the single change with the largest expected effect on the
    threshold-dependent metrics (MCC, balanced accuracy, F1), because the ~0.10
    gap observed between AUC and balanced accuracy is driven by threshold noise
    rather than by the ranking quality of the model.

    The median is used rather than the mean: it is robust to a degenerate
    partition whose calibration block happens to be single-class (for which the
    criterion functions return the 0.5 fallback).

    Parameters
    ----------
    lstCalPairs : list of (arrYTrue, arrYProb)
    strCriterion : key of DICT_THRESHOLD_CRITERIA

    Returns
    -------
    (floatThreshold, lstPerPartitionThresholds)
        The aggregated threshold and the individual values, so callers can log
        the spread and the manuscript can report it.
    """
    if strCriterion not in DICT_THRESHOLD_CRITERIA:
        raise KeyError(
            f"Unknown threshold criterion '{strCriterion}'. "
            f"Available: {sorted(DICT_THRESHOLD_CRITERIA)}"
        )
    funCriterion = DICT_THRESHOLD_CRITERIA[strCriterion]

    lstThresholds = []
    for arrYTrue, arrYProb in lstCalPairs:
        if arrYTrue is None or len(arrYTrue) == 0:
            continue
        lstThresholds.append(funCriterion(arrYTrue, arrYProb))

    if not lstThresholds:
        return 0.5, []
    return float(np.median(lstThresholds)), lstThresholds


def calculate_full_metrics(arrYTrue, arrYProb, floatThreshold):
    """
    Computes the reported metric panel.

    The threshold is an explicit argument, deliberately: the previous
    implementation recomputed Youden's J from arrYTrue inside this function,
    which fitted the operating point on the very data being scored.
    """
    arrYTrue = np.asarray(arrYTrue).astype(int)
    arrYProb = np.asarray(arrYProb, dtype=float)
    arrYPred = (arrYProb >= floatThreshold).astype(int)

    intTn, intFp, intFn, intTp = confusion_matrix(
        arrYTrue, arrYPred, labels=[0, 1]).ravel()
    floatSpecificity = intTn / (intTn + intFp) if (intTn + intFp) > 0 else 0.0
    floatSensitivity = recall_score(arrYTrue, arrYPred, zero_division=0)

    boolBothClasses = len(np.unique(arrYTrue)) == 2
    return {
        "AUC": roc_auc_score(arrYTrue, arrYProb) if boolBothClasses else np.nan,
        "PR-AUC": average_precision_score(arrYTrue, arrYProb) if boolBothClasses else np.nan,
        "MCC": matthews_corrcoef(arrYTrue, arrYPred),
        "F1": f1_score(arrYTrue, arrYPred, zero_division=0),
        "Sensitivity": floatSensitivity,
        "Specificity": floatSpecificity,
        "Accuracy": accuracy_score(arrYTrue, arrYPred),
        "Balanced Accuracy": 0.5 * (floatSensitivity + floatSpecificity),
    }


def parse_fold_name(strFoldName):
    """'repeat_3_fold_1' -> (3, 1)"""
    lstParts = strFoldName.split("_")
    return int(lstParts[1]), int(lstParts[3])


def summarise_across_repeats(dfPerFold, lstGroupCols, lstMetricKeys=None):
    """
    Aggregates per-fold metrics into the values reported in the manuscript.

    Folds inside the same repeat share training data, so they are not
    independent and their pooled standard deviation is not a valid dispersion
    estimate. The reported dispersion is therefore the standard deviation
    ACROSS the 5 repeat-level means, and the number of independent replicates
    (n_repeats) is reported alongside it.
    """
    lstMetricKeys = lstMetricKeys or LST_METRIC_KEYS
    dfRepeatMeans = (dfPerFold
                     .groupby(lstGroupCols + ["repeat"], as_index=False)[lstMetricKeys]
                     .mean())

    lstRows = []
    for tupKey, dfGroup in dfRepeatMeans.groupby(lstGroupCols):
        tupKey = tupKey if isinstance(tupKey, tuple) else (tupKey,)
        dictRow = dict(zip(lstGroupCols, tupKey))
        dictRow["n_repeats"] = int(dfGroup["repeat"].nunique())
        for strMetric in lstMetricKeys:
            floatMean = dfGroup[strMetric].mean()
            floatStd = dfGroup[strMetric].std(ddof=1)
            dictRow[strMetric] = f"{floatMean:.4f} ± {floatStd:.4f}"
            dictRow[f"{strMetric}_mean"] = floatMean
            dictRow[f"{strMetric}_sd_across_repeats"] = floatStd
        lstRows.append(dictRow)
    return pd.DataFrame(lstRows)


def format_mean_sd(arrValues):
    arrValues = np.asarray(arrValues, dtype=float)
    return f"{np.nanmean(arrValues):.4f} ± {np.nanstd(arrValues, ddof=1):.4f}"


# --------------------------------------------------------------------------
# Paired comparison over repeat-level means
# --------------------------------------------------------------------------
def compare_configs(dfPerFold, strMetric, strModelA, strModelB):
    """
    Paired comparison between two configurations evaluated on identical folds.

    The primary test is a paired t-test over the five repeat-level means, which
    are independent replicates of the whole cross-validation procedure. A
    fold-level Wilcoxon test is reported as a secondary, ANTICONSERVATIVE
    figure: folds inside a repeat share training data, so its p-value
    understates the true uncertainty and must never be quoted as the headline.

    Returns None when the two configurations share no folds, so callers can
    request comparisons that may not exist in a given run.
    """
    from scipy.stats import t as objTDist, wilcoxon

    dfA = dfPerFold[dfPerFold["config"] == strModelA].set_index("fold")
    dfB = dfPerFold[dfPerFold["config"] == strModelB].set_index("fold")
    lstCommonFolds = sorted(set(dfA.index) & set(dfB.index))
    if not lstCommonFolds:
        return None

    arrA = dfA.loc[lstCommonFolds, strMetric].values
    arrB = dfB.loc[lstCommonFolds, strMetric].values
    arrRepeats = dfA.loc[lstCommonFolds, "repeat"].values

    srRepeatDiff = (pd.DataFrame({"repeat": arrRepeats, "diff": arrA - arrB})
                    .groupby("repeat")["diff"].mean())

    floatMeanDiff = float(srRepeatDiff.mean())
    floatSdDiff = float(srRepeatDiff.std(ddof=1))
    intN = int(len(srRepeatDiff))

    if intN > 1 and floatSdDiff > 0:
        floatTStat = floatMeanDiff / (floatSdDiff / np.sqrt(intN))
        floatPRepeat = float(2.0 * objTDist.sf(abs(floatTStat), df=intN - 1))
        floatHalfWidth = float(objTDist.ppf(0.975, df=intN - 1) * floatSdDiff / np.sqrt(intN))
    else:
        floatPRepeat, floatHalfWidth = np.nan, np.nan

    try:
        floatPFold = float(wilcoxon(arrA, arrB).pvalue)
    except ValueError:
        floatPFold = np.nan

    return {
        "metric": strMetric,
        "model_a": strModelA,
        "model_b": strModelB,
        "mean_a": float(np.mean(arrA)),
        "mean_b": float(np.mean(arrB)),
        "mean_difference": floatMeanDiff,
        "sd_difference_across_repeats": floatSdDiff,
        "ci95_low": floatMeanDiff - floatHalfWidth,
        "ci95_high": floatMeanDiff + floatHalfWidth,
        "p_repeat_level_ttest": floatPRepeat,
        "n_repeats": intN,
        "p_fold_level_wilcoxon_anticonservative": floatPFold,
    }


def add_multiplicity_correction(dfComparisons, strFamilyColumn="family",
                                strPColumn="p_repeat_level_ttest"):
    """
    Adds Benjamini-Hochberg adjusted p-values WITHIN each family of contrasts.

    WHY THIS IS NECESSARY
    ---------------------
    The comparison grid grows fast once several encoders and several aggregation
    strategies are on the table, while the primary test has only n=5 repeats and
    therefore little power. With enough contrasts, some will reach p<0.05 by
    chance alone, and a reviewer will say so.

    Correction is applied per FAMILY rather than across the whole table on
    purpose: pooling unrelated questions (encoder choice, ROI definition,
    aggregation operator) into one correction penalises each of them for the
    others' existence. Callers must label each row with the family it belongs
    to, and the manuscript must declare which contrasts were pre-specified as
    primary and which are exploratory.

    Benjamini-Hochberg controls the false discovery rate, which is the
    appropriate target for a screening table; Bonferroni would be needlessly
    severe here.

    Returns a copy of dfComparisons with two extra columns:
      p_adjusted_bh          adjusted p-value inside the family
      significant_adjusted   boolean at the 0.05 FDR level
    """
    dfOut = dfComparisons.copy()
    if strFamilyColumn not in dfOut.columns:
        dfOut[strFamilyColumn] = "all"

    dfOut["p_adjusted_bh"] = np.nan
    for strFamily, dfGroup in dfOut.groupby(strFamilyColumn):
        srP = dfGroup[strPColumn]
        arrValid = srP.notna().values
        if not arrValid.any():
            continue
        arrP = srP.values[arrValid]
        intM = len(arrP)
        arrOrder = np.argsort(arrP)
        arrRanked = arrP[arrOrder]
        # Benjamini-Hochberg step-up, then enforce monotonicity.
        arrAdj = arrRanked * intM / (np.arange(1, intM + 1))
        arrAdj = np.minimum.accumulate(arrAdj[::-1])[::-1]
        arrAdj = np.clip(arrAdj, 0.0, 1.0)
        arrRestored = np.empty(intM, dtype=float)
        arrRestored[arrOrder] = arrAdj
        arrTarget = np.full(len(srP), np.nan)
        arrTarget[arrValid] = arrRestored
        dfOut.loc[dfGroup.index, "p_adjusted_bh"] = arrTarget

    dfOut["significant_adjusted"] = dfOut["p_adjusted_bh"] < 0.05
    return dfOut


# --------------------------------------------------------------------------
# Manuscript output helpers
# --------------------------------------------------------------------------
# Every figure and table produced by the pipeline is a potential manuscript
# asset, so all of them are written in ENGLISH with publication defaults. These
# helpers centralise that so no phase has to remember the settings.

DICT_FIGURE_STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
}

# Colour-blind safe palette (Okabe-Ito). Journals increasingly require figures
# to remain readable in greyscale and for deuteranopia.
LST_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def apply_manuscript_style():
    """Applies publication defaults to matplotlib. Call once per phase."""
    import matplotlib
    matplotlib.rcParams.update(DICT_FIGURE_STYLE)


def save_figure(objFig, pathOut, lstFormats=("png", "pdf")):
    """
    Writes a figure in several formats.

    PDF is included because Elsevier and IEEE both prefer vector artwork for
    line figures; PNG is kept for quick inspection and for the .txt reports.
    """
    pathOut = Path(pathOut)
    pathOut.parent.mkdir(parents=True, exist_ok=True)
    lstWritten = []
    for strFormat in lstFormats:
        pathFile = pathOut.with_suffix(f".{strFormat}")
        objFig.savefig(pathFile, format=strFormat)
        lstWritten.append(pathFile)
    print(f"[figure] {', '.join(p.name for p in lstWritten)}")
    return lstWritten


def export_manuscript_table(dfTable, pathOutBase, strCaption="", strLabel=""):
    """
    Writes one table as CSV, Markdown and LaTeX (booktabs).

    The LaTeX file is directly \input-able in a manuscript, which removes the
    manual copy step that is the usual source of transcription errors between a
    results file and the text of a paper.
    """
    pathOutBase = Path(pathOutBase)
    pathOutBase.parent.mkdir(parents=True, exist_ok=True)

    dfTable.to_csv(pathOutBase.with_suffix(".csv"), index=False)
    try:
        with open(pathOutBase.with_suffix(".md"), "w", encoding="utf-8") as objFile:
            objFile.write(dfTable.to_markdown(index=False))
    except Exception:
        # to_markdown needs the optional 'tabulate' dependency; its absence must
        # not abort a run that has already produced the CSV.
        pass

    try:
        strLatex = dfTable.to_latex(index=False, escape=True, longtable=False,
                                    caption=strCaption or None,
                                    label=strLabel or None)
        with open(pathOutBase.with_suffix(".tex"), "w", encoding="utf-8") as objFile:
            objFile.write(strLatex)
    except Exception:
        pass

    print(f"[table] {pathOutBase.name}.{{csv,md,tex}}")
    return pathOutBase


def encoder_metadata_frame():
    """Registry as a DataFrame, for the manuscript's encoder description table."""
    lstRows = []
    for strKey in LST_ENCODER_ORDER:
        dictEnc = DICT_ENCODERS[strKey]
        lstRows.append({
            "encoder_key": strKey,
            "encoder": dictEnc["display_name"],
            "pre_training_domain": dictEnc["domain"],
            "pre_training_objective": dictEnc["objective"],
            "architecture_family": dictEnc["family"],
            "patch_grid": dictEnc["patch_grid"],
            "embedding_dim": dictEnc["embed_dim"],
            "role_in_study": dictEnc["role"],
        })
    return pd.DataFrame(lstRows)


def parse_config_label(strConfig):
    """
    Splits a configuration label into its experimental factors.

    Labels follow '<encoder> | <roi> ROI | <head>'. Older two-part labels
    ('<roi> ROI | <head>') are still parsed, with the encoder defaulting to the
    reference one, so tables from previous runs remain readable.
    """
    lstParts = [s.strip() for s in str(strConfig).split("|")]
    dictOut = {"encoder": STR_REFERENCE_ENCODER, "roi": "n/a", "head": strConfig}
    lstRoiTokens = [p for p in lstParts if p.lower().endswith("roi")]
    if lstRoiTokens:
        strRoi = lstRoiTokens[0]
        dictOut["roi"] = strRoi.split()[0].lower()
        intIdx = lstParts.index(strRoi)
        if intIdx > 0:
            dictOut["encoder"] = lstParts[0]
        dictOut["head"] = lstParts[intIdx + 1] if intIdx + 1 < len(lstParts) else "n/a"
    return dictOut
