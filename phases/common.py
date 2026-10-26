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

LST_TARGET_CLASSES = ["diffuse_goiter", "multinodular_goiter"]
DICT_CLASS_TO_LABEL = {"diffuse_goiter": 0, "multinodular_goiter": 1}
STR_POSITIVE_CLASS = "multinodular_goiter"

LST_METRIC_KEYS = ["AUC", "PR-AUC", "MCC", "F1", "Sensitivity", "Specificity",
                   "Accuracy", "Balanced Accuracy"]

LST_CLINICAL_COLUMNS = ["AGE", "WEIGHT (Kg)", "HEIGHT (cm)",
                        "DOSE ADMINISTERED (mCi)"]

INT_GLOBAL_SEED = 42

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

LST_ENCODER_ORDER = ["resnet50", "dinov2", "clip", "biomedclip", "raddino"]
STR_REFERENCE_ENCODER = "dinov2"


def set_global_seed(intSeed=INT_GLOBAL_SEED):
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
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

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
    return _load_table(pathMasterCsv)


def load_balanced_table():
    return _load_table(pathBalancedCsv)


def get_labels(dfTable):
    return dfTable["class"].map(DICT_CLASS_TO_LABEL).astype(int).values


def build_clinical_matrix(dfTable):
    dfClinical = dfTable[LST_CLINICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    dfClinical = dfClinical.fillna(dfClinical.median())
    dfClinical = dfClinical.assign(SEX_FEMALE=(dfTable["SEX"] == "FEMALE").astype(int))
    return dfClinical.values.astype(np.float64), list(dfClinical.columns)

def feature_file(strRoiMode, strEncoder=STR_REFERENCE_ENCODER):
    if strEncoder == STR_REFERENCE_ENCODER:
        return pathPhase2Dir / f"dinov2_features_{strRoiMode}.npz"
    return pathPhase2Dir / f"{strEncoder}_features_{strRoiMode}.npz"


def available_encoders(lstRoiModes=("fixed", "dynamic")):
    lstFound = []
    for strEncoder in LST_ENCODER_ORDER:
        if all(feature_file(m, strEncoder).exists() for m in lstRoiModes):
            lstFound.append(strEncoder)
    return lstFound


def load_features(strRoiMode, dfTable, boolLoadPatch=False,
                  strEncoder=STR_REFERENCE_ENCODER):
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

def load_splits():
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
    lstInnerFolds = dictFold.get("inner_folds")
    if lstInnerFolds:
        return lstInnerFolds
    return [{"inner_train": dictFold["inner_train"],
             "inner_cal": dictFold["inner_cal"]}]


def get_fold_row_positions(dfTable, lstSampleIds):
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

def select_threshold_youden(arrYTrue, arrYProb):
    arrYTrue = np.asarray(arrYTrue)
    if len(np.unique(arrYTrue)) < 2:
        return 0.5
    arrFpr, arrTpr, arrThresholds = roc_curve(arrYTrue, arrYProb)
    return float(arrThresholds[np.argmax(arrTpr - arrFpr)])


def select_threshold_equal_error(arrYTrue, arrYProb):
    arrYTrue = np.asarray(arrYTrue)
    if len(np.unique(arrYTrue)) < 2:
        return 0.5
    arrFpr, arrTpr, arrThresholds = roc_curve(arrYTrue, arrYProb)
    arrGap = np.abs(arrTpr - (1.0 - arrFpr))
    return float(arrThresholds[np.argmin(arrGap)])

DICT_THRESHOLD_CRITERIA = {
    "youden": select_threshold_youden,
    "equal_error": select_threshold_equal_error,
}


def select_threshold_averaged(lstCalPairs, strCriterion="youden"):
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
    lstParts = strFoldName.split("_")
    return int(lstParts[1]), int(lstParts[3])


def summarise_across_repeats(dfPerFold, lstGroupCols, lstMetricKeys=None):
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

def compare_configs(dfPerFold, strMetric, strModelA, strModelB):
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

LST_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def apply_manuscript_style():
    import matplotlib
    matplotlib.rcParams.update(DICT_FIGURE_STYLE)


def save_figure(objFig, pathOut, lstFormats=("png", "pdf")):
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
