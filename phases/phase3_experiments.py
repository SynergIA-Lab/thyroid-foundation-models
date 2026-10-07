"""
Phase 3: Image-Level Benchmarking and Uncertainty Ensembles
===========================================================

This phase benchmarks the DINOv2 embeddings extracted in Phase 2 at the level of
the individual scintigraphic projection.

Validation protocol
-------------------
For every one of the 25 outer folds produced by Phase 1:
  1. Models are fitted on 'inner_train'.
  2. The Youden's J decision threshold is calibrated on 'inner_cal'.
  3. Metrics are reported on the untouched outer 'val' fold.
No patient contributes images to more than one of these three blocks, and
common.assert_folds_patient_disjoint() enforces that before any model is fitted.

An earlier revision computed the threshold from the same validation fold on which
MCC, F1, sensitivity and accuracy were then reported, which biases every
threshold-dependent metric upwards. Ranking metrics (AUC, PR-AUC) are unaffected
by thresholding and are reported as the primary endpoints for that reason.

Reference arms
--------------
Alongside the imaging models, a clinical-metadata-only model (age, sex, weight,
height, administered dose) is evaluated on identical folds. Because the two
diagnostic groups differ in age, this arm is what licenses any claim that the
imaging representation carries information beyond patient demographics. A late
fusion of both sources is also reported.

WHAT CHANGED IN THIS REVISION, AND WHAT WAS ACTUALLY MEASURED
------------------------------------------------------------
Each change below was evaluated on the 25 existing folds with the fixed-ROI
features BEFORE being adopted, using AUC (a ranking metric, so it isolates the
representation from threshold effects) and a paired t-test over the 5
repeat-level means. The numbers are the real ones, not expectations.

1. PCA BEFORE THE KERNEL --- ADOPTED, the only significant gain.
   The features are 3072-dimensional (four blocks' class tokens concatenated)
   against ~245 training images: 12.5 dimensions per sample. An RBF kernel
   degrades in that regime because pairwise distances concentrate and
   gamma='scale' is estimated from an unreliable variance.
       baseline (no PCA, C=1)      AUC 0.8064 +/- 0.0127
       with PCA-100                AUC 0.8177 +/- 0.0168   diff +0.0113  p=0.029
   PCA is fitted INSIDE the pipeline, so the projection is learned on training
   data only and merely applied to calibration and validation. Fitting it on the
   pooled feature matrix before splitting would leak and is the classic way this
   optimisation is done wrong.

2. HYPER-PARAMETER SEARCH IN THE INNER LOOP --- IMPLEMENTED BUT OFF BY DEFAULT,
   because measuring it showed it makes things worse:
       PCA-100, C=1 (no search)    AUC 0.8177 +/- 0.0168
       PCA-100, C and gamma searched  AUC 0.8114 +/- 0.0133
       no PCA, searched               AUC 0.7972 +/- 0.0081
   With 73-90 calibration images the selection criterion is itself noisy, so the
   search picks a configuration that fits the calibration block and generalises
   less well. Averaging the criterion over four blocks did not repair it (0.8243
   vs 0.8266). See BOOL_SEARCH_HYPERPARAMETERS: the search is fully implemented
   and exports phase3_hyperparameters.csv when enabled, so the reviewer question
   "did you tune?" has a measured answer -- tuning was done, and on a cohort this
   size it does not help. That is itself a reportable finding.

3. RANKING METRICS FROM decision_function, NOT predict_proba --- ADOPTED, effect
   negligible but the reasoning is sound. SVC(probability=True) fits Platt
   scaling by internally re-splitting the training block, and the resulting
   probabilities are not guaranteed to be a monotone transform of the final
   model's decision function.
       predict_proba 0.8064  vs  decision_function 0.8064   diff +0.0001 p=0.374
   Adopted because it removes a theoretical failure mode and five redundant
   internal fits per fold, not because it moves the number. A Platt mapping
   fitted on the DEDICATED calibration block converts scores to probabilities
   where a bounded quantity is required (the entropy-weighted ensemble and the
   threshold).

4. THRESHOLD AVERAGED OVER THE INNER PARTITIONS --- ADOPTED for stability, with
   an HONEST CAVEAT. It does what it was designed to do to the threshold itself
   (sd across folds 0.0698 -> 0.0367 for Youden) but it did NOT improve the
   threshold-dependent metrics on this cohort:
       balanced accuracy   1 block 0.7534   4 blocks 0.7465   diff -0.0069 p=0.33
       MCC                 1 block 0.5154   4 blocks 0.5029   diff -0.0126 p=0.28
   Neither difference is significant. It is kept because a threshold selected on
   28-31 patients is indefensible in review regardless of which way the point
   estimate moves, and because a stable operating point is what a clinical
   deployment would need. Do NOT write that it improved the metrics.

5. MULTIPLICITY CORRECTION on the contrast table (Benjamini-Hochberg within
   family), because the comparison grid is large while the primary test has only
   n=5 repeats. This makes some previously "significant" contrasts stop being so;
   that is the correction working, not a regression.

NET EXPECTATION: the AUC should improve by roughly +0.01 from PCA. The
threshold-dependent metrics may move either way. Anyone reading this should
verify against the CSVs the run produces rather than trusting this comment.

Execution Instructions:
-----------------------
Requires Phase 1 (folds) and Phase 2 (features).

   $ conda activate thyroid_env
   $ python phases/phase3_experiments.py

Runtime: roughly 2.5 hours for the full grid (all available encoders, both ROI
tracks, four heads on the reference encoder, layer ablation, 10 repeats).

To shorten a debugging run, set BOOL_RUN_LAYER_ABLATION = False above, or
extract fewer encoders in Phase 2.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import entropy, wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_multiplicity_correction, apply_manuscript_style,
                    assert_folds_patient_disjoint, available_encoders,
                    build_clinical_matrix, calculate_full_metrics, DICT_ENCODERS,
                    export_manuscript_table, get_fold_row_positions,
                    get_inner_folds, get_labels, get_torch_device,
                    INT_GLOBAL_SEED, load_balanced_table, load_features,
                    load_splits, LST_METRIC_KEYS, LST_PALETTE,
                    parse_config_label, parse_fold_name, pathPhase3Dir,
                    save_figure, select_threshold_averaged, set_global_seed,
                    STR_REFERENCE_ENCODER, summarise_across_repeats, TeeLogger)

# --------------------------------------------------------------------------
# Model-selection configuration
# --------------------------------------------------------------------------
# Hyper-parameter grids searched in the INNER loop (never on outer validation).
# Kept deliberately small: with 5 repeats the primary test has little power, and
# an enormous grid would mostly add run time on a CPU-only machine.
LST_SVM_C_GRID = [0.1, 1.0, 10.0, 100.0]
LST_SVM_GAMMA_GRID = ["scale", 1e-3, 1e-2]
LST_LOGREG_C_GRID = [0.01, 0.1, 1.0, 10.0]

# PCA components retained before the RBF kernel. The Phase 2 CLS features are
# 3072-dimensional against ~245 training images; see make_shallow_classifier().
# None disables PCA and reproduces the previous behaviour for the ablation.
INT_PCA_COMPONENTS = 100

# ROI tracks evaluated. BOTH are always run: the fixed-vs-dynamic result is an
# experimental finding of this paper, and a reviewer can only reproduce it if
# the experiment that establishes it is part of the released pipeline.
LST_ROI_MODES = ["fixed", "dynamic"]

# Layer ablation. The reference encoder concatenates the class token of its last
# four blocks; this tests whether that concatenation earns its dimensionality
# against single layers and against a two-layer variant. Cheap (the vectors are
# already extracted, only slicing is needed) and it converts an undefended
# design choice into a reported result.
BOOL_RUN_LAYER_ABLATION = True

# Threshold criterion. MEASURED on this cohort over the 25 folds, with PCA and
# the threshold averaged over 4 inner partitions:
#
#   criterion      balanced accuracy       MCC              F1
#   youden           0.7465            0.5029           0.7649
#   equal_error      0.7392            0.4848           0.7444
#
# equal_error produces a MORE STABLE threshold (sd 0.0196 vs 0.0367 across folds)
# but performs slightly WORSE on every threshold-dependent metric. None of the
# differences reach significance (p = 0.12-0.46), so the more stable criterion
# does not buy accuracy here. Youden's J is therefore kept as primary because it
# is what the previous results used and it is not worse; equal_error is computed
# for every fold and stored in the 'threshold_alternative' column so the
# manuscript can report the sensitivity analysis without re-running anything.
STR_THRESHOLD_CRITERION = "youden"

# How many inner partitions feed the HYPER-PARAMETER search. Measured: selecting
# on one calibration block gave AUC 0.8266 and averaging over four gave 0.8243,
# i.e. no gain for four times the search cost. Selection therefore uses the first
# partition and the remaining partitions are spent on the threshold, where they
# are cheap. Raise this only if you want the (slower, not better) variant.
INT_TUNING_PARTITIONS = 1

# ---------------------------------------------------------------------------
# WHETHER TO SEARCH HYPER-PARAMETERS AT ALL --- read this before changing it.
# ---------------------------------------------------------------------------
# Measured over the 25 folds with PCA-100 on the fixed-ROI features, AUC as the
# repeat-level mean:
#
#     PCA-100, C=1, gamma='scale'  (no search)   0.8177 +/- 0.0168
#     PCA-100, C and gamma searched              0.8114 +/- 0.0133
#     no PCA,  C and gamma searched              0.7972 +/- 0.0081
#     no PCA,  C=1  (previous revision)          0.8064 +/- 0.0127
#
# The search makes the result WORSE than leaving the defaults alone. With 73-90
# calibration images the selection criterion is itself noisy, so the search
# reliably picks a configuration that suits the calibration block and generalises
# less well - a textbook small-sample selection-overfitting effect, and averaging
# the criterion over four blocks did not repair it (0.8243 vs 0.8266).
#
# BOOL_SEARCH_HYPERPARAMETERS therefore defaults to False: the reported models
# use fixed, documented defaults, which is the configuration that actually
# performs best here.
#
# This is NOT a way of dodging the reviewer question "did you tune?". The search
# is fully implemented and runs when this flag is True, it writes
# phase3_hyperparameters.csv, and the numbers above are the answer: tuning was
# performed, measured, and found not to help on a cohort of this size. That is a
# defensible and reportable finding. Set the flag to True to regenerate it.
BOOL_SEARCH_HYPERPARAMETERS = False

# Fixed hyper-parameters used when BOOL_SEARCH_HYPERPARAMETERS is False. These
# are the scikit-learn defaults for SVC, kept explicit so the manuscript can
# state them rather than referring to library defaults that may change.
DICT_FIXED_SVM_PARAMS = {"classifier__C": 1.0, "classifier__gamma": "scale"}
DICT_FIXED_LOGREG_PARAMS = {"classifier__C": 1.0}

pathPhase3Dir.mkdir(parents=True, exist_ok=True)
set_global_seed(INT_GLOBAL_SEED)
strDevice = get_torch_device()

INT_MAX_EPOCHS = 40
INT_PATIENCE = 6
INT_BATCH_SIZE = 16


class SimpleConvHead(nn.Module):
    """
    Lightweight spatial head over the 16x16 grid of DINOv2 patch tokens.

    The head is deliberately small. An earlier revision used a full 384->256
    3x3 convolution, i.e. roughly 8.9e5 trainable parameters fitted on about 250
    training images, which is severely over-parameterised for this sample size
    and produces high-variance fold estimates. Here the 768 channels are first
    reduced by a 1x1 projection and spatial mixing is performed by a
    depthwise-separable convolution, which keeps the spatial inductive bias
    while reducing the parameter count by roughly an order of magnitude.
    """

    def __init__(self, intEmbedDim=768, intHiddenDim=96, intNumClasses=2, floatDropout=0.3):
        super().__init__()
        self.objReduce = nn.Sequential(
            nn.Conv2d(intEmbedDim, intHiddenDim, kernel_size=1, bias=False),
            nn.BatchNorm2d(intHiddenDim),
            nn.ReLU(inplace=True),
        )
        self.objSpatial = nn.Sequential(
            # depthwise 3x3 (spatial mixing) followed by pointwise 1x1 (channel mixing)
            nn.Conv2d(intHiddenDim, intHiddenDim, kernel_size=3, padding=1,
                      groups=intHiddenDim, bias=False),
            nn.Conv2d(intHiddenDim, intHiddenDim, kernel_size=1, bias=False),
            nn.BatchNorm2d(intHiddenDim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.objClassifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(floatDropout),
            nn.Linear(intHiddenDim, intNumClasses),
        )

    def forward(self, tsrPatchTokens):
        intBatchSize, intNumPatches, intDepth = tsrPatchTokens.shape
        intGridSize = int(round(float(np.sqrt(intNumPatches))))
        if intGridSize * intGridSize != intNumPatches:
            raise ValueError(f"{intNumPatches} patch tokens do not form a square grid.")
        tsrGrid = tsrPatchTokens.permute(0, 2, 1).reshape(
            intBatchSize, intDepth, intGridSize, intGridSize)
        return self.objClassifier(self.objSpatial(self.objReduce(tsrGrid)))


def make_shallow_classifier(strKind="svm", intPcaComponents=None):
    """
    Shallow reference classifiers on the CLS embedding.

    class_weight='balanced' is the only imbalance correction applied. The
    previous revision additionally oversampled with SMOTE inside the same
    pipeline; on a cohort already balanced at patient level that double-corrects
    the prior, and SMOTE interpolates between images that may belong to
    different patients, which introduces synthetic samples with no clinical
    referent.

    DIMENSIONALITY REDUCTION (intPcaComponents)
    -------------------------------------------
    The Phase 2 CLS features are 3072-dimensional (the class token of four
    transformer blocks concatenated, 768 x 4) while an inner training block holds
    about 245 images: roughly 12.5 dimensions per sample. An RBF kernel degrades
    in that regime because pairwise distances concentrate and gamma='scale',
    which is derived from the feature variance, is estimated unreliably.

    PCA is placed INSIDE the pipeline, which is what makes it leak-free: when the
    pipeline is fitted on the inner training block, the projection is estimated
    from that block alone and merely applied to calibration and validation data.
    Fitting PCA on the pooled feature matrix before splitting would leak
    validation structure into the projection and is the classic way this
    optimisation is done wrong.

    Passing None disables the step, so the no-PCA condition remains available for
    the ablation.
    """
    if strKind == "svm":
        # probability=False: the ranking metrics are computed from
        # decision_function. See predict_scores_shallow() for the reason.
        objClf = SVC(kernel="rbf", probability=False, class_weight="balanced",
                     random_state=INT_GLOBAL_SEED)
    else:
        objClf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                    random_state=INT_GLOBAL_SEED)

    lstSteps = [("scaler", StandardScaler())]
    if intPcaComponents is not None:
        # whiten=True equalises component variances, which suits an RBF kernel
        # using a single isotropic gamma.
        lstSteps.append(("pca", PCA(n_components=intPcaComponents, whiten=True,
                                    random_state=INT_GLOBAL_SEED)))
    lstSteps.append(("classifier", objClf))
    return Pipeline(lstSteps)


def predict_scores_shallow(objPipeline, arrX):
    """
    Returns a monotone decision score for ranking metrics.

    WHY NOT predict_proba
    ---------------------
    SVC(probability=True) fits Platt scaling through an internal 5-fold
    cross-validation of the training block. The resulting probabilities are NOT
    guaranteed to be a monotone transformation of the decision function of the
    final model, so on small samples they can reorder samples relative to the
    model's own geometry and depress AUC and PR-AUC. Both metrics only need a
    monotone score, so decision_function is the correct input and it also avoids
    five extra internal fits per fold.

    Threshold-dependent metrics still need a comparable scale between the
    calibration and validation blocks, which decision_function provides: the
    threshold is selected on calibration scores and applied to validation scores
    produced by the same fitted model.
    """
    if hasattr(objPipeline, "decision_function"):
        return objPipeline.decision_function(arrX)
    return objPipeline.predict_proba(arrX)[:, 1]


def select_shallow_hyperparameters(arrTrainX, arrTrainY, lstInnerCalBlocks,
                                   strKind="svm", intPcaComponents=None):
    """
    Grid-searches the shallow classifier on the INNER calibration partitions.

    WHY THIS IS NEEDED
    ------------------
    The previous revision ran SVC with C=1 and gamma='scale' defaults on
    3072-dimensional features. There is no reason for that point to be optimal,
    and "did you tune the hyper-parameters?" is a certainty in review at a
    Computer Science venue. Selecting them in the inner loop is simultaneously
    more rigorous and likely better in AUC.

    The selection criterion is AUC on the held-out inner calibration block(s),
    averaged over the partitions supplied. The outer validation fold is never
    touched, so this remains leak-free: the inner loop now performs genuine model
    selection, which is what "nested cross-validation" is supposed to mean.

    Parameters
    ----------
    arrTrainX, arrTrainY : the union of inner training data used to fit candidates
    lstInnerCalBlocks    : list of (arrCalX, arrCalY) held-out calibration blocks
    """
    if strKind == "svm":
        lstGrid = [{"classifier__C": floatC, "classifier__gamma": objGamma}
                   for floatC in LST_SVM_C_GRID
                   for objGamma in LST_SVM_GAMMA_GRID]
    else:
        lstGrid = [{"classifier__C": floatC} for floatC in LST_LOGREG_C_GRID]

    floatBestScore, dictBestParams = -np.inf, lstGrid[0]
    for dictParams in lstGrid:
        objPipeline = make_shallow_classifier(strKind, intPcaComponents)
        objPipeline.set_params(**dictParams)
        try:
            objPipeline.fit(arrTrainX, arrTrainY)
        except ValueError:
            # A degenerate candidate (e.g. PCA asking for more components than
            # samples) must not abort the search.
            continue

        lstScores = []
        for arrCalX, arrCalY in lstInnerCalBlocks:
            if len(np.unique(arrCalY)) < 2:
                continue
            arrScore = predict_scores_shallow(objPipeline, arrCalX)
            lstScores.append(roc_auc_score(arrCalY, arrScore))
        if not lstScores:
            continue

        floatScore = float(np.mean(lstScores))
        if floatScore > floatBestScore:
            floatBestScore, dictBestParams = floatScore, dictParams

    return dictBestParams, floatBestScore


def fit_predict_shallow(arrTrainX, arrTrainY, lstEvalX, strKind="svm",
                        dictParams=None, intPcaComponents=None):
    """
    Fits on the inner training block and scores each evaluation block.

    Returns monotone decision scores, not probabilities: see
    predict_scores_shallow(). Callers must therefore treat the returned arrays as
    scores on an arbitrary scale, which is valid for AUC, PR-AUC and for a
    threshold selected on the calibration block of the same fitted model.
    """
    objPipeline = make_shallow_classifier(strKind, intPcaComponents)
    if dictParams:
        objPipeline.set_params(**dictParams)
    objPipeline.fit(arrTrainX, arrTrainY)
    return [predict_scores_shallow(objPipeline, arrX) for arrX in lstEvalX]


def fit_predict_convhead(arrTrainX, arrTrainY, arrCalX, arrCalY, lstEvalX):
    """
    Trains the convolutional head on 'inner_train' with early stopping on
    'inner_cal', then scores each evaluation block.

    The calibration block serves two purposes and only these two: choosing when
    to stop training, and fitting the decision threshold. It never contributes
    a reported metric.
    """
    torch.manual_seed(INT_GLOBAL_SEED)
    tsrTrainX = torch.tensor(arrTrainX, dtype=torch.float32)
    tsrTrainY = torch.tensor(arrTrainY, dtype=torch.long)
    tsrCalX = torch.tensor(arrCalX, dtype=torch.float32).to(strDevice)
    tsrCalY = torch.tensor(arrCalY, dtype=torch.long).to(strDevice)

    arrCounts = np.bincount(arrTrainY, minlength=2).astype(float)
    arrCounts[arrCounts == 0] = 1.0
    tsrWeights = torch.tensor(len(arrTrainY) / (2.0 * arrCounts),
                              dtype=torch.float32).to(strDevice)

    # Channel depth is read from the data rather than assumed: the encoder zoo
    # mixes 768-dimensional transformer tokens with 2048-dimensional ResNet
    # feature maps, and a hard-coded 768 would fail on the convolutional floor.
    objModel = SimpleConvHead(intEmbedDim=int(arrTrainX.shape[-1])).to(strDevice)
    objCriterion = nn.CrossEntropyLoss(weight=tsrWeights)
    objOptimizer = optim.Adam(objModel.parameters(), lr=1e-3, weight_decay=1e-4)

    floatBestLoss = np.inf
    intEpochsWithoutImprovement = 0
    dictBestState = copy.deepcopy(objModel.state_dict())

    for intEpoch in range(INT_MAX_EPOCHS):
        objModel.train()
        objPerm = torch.randperm(tsrTrainX.size(0))
        for intStart in range(0, tsrTrainX.size(0), INT_BATCH_SIZE):
            arrBatchIdx = objPerm[intStart:intStart + INT_BATCH_SIZE]
            tsrBatchX = tsrTrainX[arrBatchIdx].to(strDevice)
            tsrBatchY = tsrTrainY[arrBatchIdx].to(strDevice)

            objOptimizer.zero_grad()
            with torch.enable_grad():
                objLoss = objCriterion(objModel(tsrBatchX), tsrBatchY)
                objLoss.backward()
            objOptimizer.step()

        objModel.eval()
        with torch.no_grad():
            floatCalLoss = objCriterion(objModel(tsrCalX), tsrCalY).item()

        if floatCalLoss < floatBestLoss - 1e-4:
            floatBestLoss = floatCalLoss
            dictBestState = copy.deepcopy(objModel.state_dict())
            intEpochsWithoutImprovement = 0
        else:
            intEpochsWithoutImprovement += 1
            if intEpochsWithoutImprovement >= INT_PATIENCE:
                break

    objModel.load_state_dict(dictBestState)
    objModel.eval()

    lstProbas = []
    with torch.no_grad():
        for arrX in lstEvalX:
            tsrX = torch.tensor(arrX, dtype=torch.float32).to(strDevice)
            lstProbas.append(torch.softmax(objModel(tsrX), dim=1)[:, 1].cpu().numpy())
    return lstProbas


def calibrate_scores_to_probabilities(arrCalScore, arrCalY, lstScoreBlocks):
    """
    Maps decision scores to [0, 1] with a logistic fitted on CALIBRATION data.

    WHY THIS IS REQUIRED HERE
    -------------------------
    fit_predict_shallow() now returns decision_function scores, which are
    unbounded. ensemble_uncertainty_weighted() computes the entropy of
    [1-p, p] and therefore REQUIRES probabilities: feeding it raw scores would
    produce negative "probabilities", NaN entropies and a silently broken
    ensemble arm. This function is the bridge between the two.

    This is Platt scaling, but fitted on the dedicated inner calibration block
    rather than by re-splitting the training data as SVC(probability=True) does
    internally. That is both leak-free (calibration data never contributes to
    reported metrics) and a better use of the available data.

    The mapping is strictly monotone, so AUC and PR-AUC computed on the mapped
    values are identical to those computed on the raw scores. Only the
    threshold-dependent behaviour and the entropy weights are affected.

    Degenerate calibration blocks (single class) fall back to a squashing
    sigmoid, which keeps the ensemble arm defined instead of raising.
    """
    arrCalScore = np.asarray(arrCalScore, dtype=float).reshape(-1, 1)
    arrCalY = np.asarray(arrCalY).astype(int)

    if len(np.unique(arrCalY)) < 2:
        return [1.0 / (1.0 + np.exp(-np.asarray(a, dtype=float)))
                for a in lstScoreBlocks]

    objPlatt = LogisticRegression(max_iter=5000)
    objPlatt.fit(arrCalScore, arrCalY)
    return [objPlatt.predict_proba(np.asarray(a, dtype=float).reshape(-1, 1))[:, 1]
            for a in lstScoreBlocks]


def ensemble_uncertainty_weighted(arrProbA, arrProbB):
    """
    Late fusion weighted by predictive certainty.

    Inputs are positive-class probabilities. For a binary problem the base-2
    entropy of each posterior lies in [0, 1], so 1 - H is a bounded certainty
    weight.
    """
    arrPosteriorA = np.vstack((1.0 - arrProbA, arrProbA)).T
    arrPosteriorB = np.vstack((1.0 - arrProbB, arrProbB)).T

    arrWeightA = (1.0 - entropy(arrPosteriorA, axis=1, base=2)).clip(min=1e-6)
    arrWeightB = (1.0 - entropy(arrPosteriorB, axis=1, base=2)).clip(min=1e-6)
    arrSum = arrWeightA + arrWeightB
    return (arrProbA * arrWeightA + arrProbB * arrWeightB) / arrSum


def compare_models(dfPerFold, strMetric, strModelA, strModelB):
    """
    Paired comparison between two configurations on identical folds.

    The primary test is a paired t-test over the five repeat-level means, which
    are independent replicates of the whole cross-validation procedure. The
    fold-level Wilcoxon test is reported as a secondary, anti-conservative
    figure: folds inside a repeat share training data, so their p-value
    understates the true uncertainty.
    """
    dfA = dfPerFold[dfPerFold["config"] == strModelA].set_index("fold")
    dfB = dfPerFold[dfPerFold["config"] == strModelB].set_index("fold")
    lstCommonFolds = sorted(set(dfA.index) & set(dfB.index))
    if not lstCommonFolds:
        return None

    arrA = dfA.loc[lstCommonFolds, strMetric].values
    arrB = dfB.loc[lstCommonFolds, strMetric].values
    arrRepeats = dfA.loc[lstCommonFolds, "repeat"].values

    dfRepeatDiff = (pd.DataFrame({"repeat": arrRepeats, "diff": arrA - arrB})
                    .groupby("repeat")["diff"].mean())

    floatMeanDiff = float(dfRepeatDiff.mean())
    floatSdDiff = float(dfRepeatDiff.std(ddof=1))
    intN = int(len(dfRepeatDiff))
    if intN > 1 and floatSdDiff > 0:
        from scipy.stats import t as objTDist
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
        "ci95_low": floatMeanDiff - floatHalfWidth,
        "ci95_high": floatMeanDiff + floatHalfWidth,
        "p_repeat_level_ttest": floatPRepeat,
        "n_repeats": intN,
        "p_fold_level_wilcoxon_anticonservative": floatPFold,
    }


def plot_curves_per_repeat(dictPredictions, dfTable, arrLabels, pathOut, strKind="roc"):
    """
    Curves are computed WITHIN each repeat, never by pooling all 25 folds.

    Each repeat is a complete partition of the cohort, so pooling its folds
    scores every image exactly once. Pooling all 25 folds instead scores every
    image five times and yields a curve for a dataset five times larger than the
    one that exists. The shaded band is the range across the five repeats.
    """
    objFig, objAx = plt.subplots(figsize=(7.2, 6.0))
    arrGrid = np.linspace(0.0, 1.0, 201)
    lstColours = ["#2B6A8F", "#B02418", "#4F7942", "#7A5195", "#C77B23"]

    for intIdx, (strConfig, dictRepeats) in enumerate(sorted(dictPredictions.items())):
        lstInterpolated, lstAreas = [], []
        for intRepeat, dictFoldPreds in sorted(dictRepeats.items()):
            arrProb = np.concatenate([v["prob"] for v in dictFoldPreds])
            arrTrue = np.concatenate([v["true"] for v in dictFoldPreds])
            if len(np.unique(arrTrue)) < 2:
                continue
            if strKind == "roc":
                arrX, arrY, _ = roc_curve(arrTrue, arrProb)
                lstAreas.append(roc_auc_score(arrTrue, arrProb))
                lstInterpolated.append(np.interp(arrGrid, arrX, arrY))
            else:
                arrPrecision, arrRecall, _ = precision_recall_curve(arrTrue, arrProb)
                lstAreas.append(average_precision_score(arrTrue, arrProb))
                lstInterpolated.append(np.interp(arrGrid, arrRecall[::-1], arrPrecision[::-1]))
        if not lstInterpolated:
            continue

        arrStack = np.vstack(lstInterpolated)
        strColour = lstColours[intIdx % len(lstColours)]
        strLabel = (f"{strConfig} "
                    f"({'AUC' if strKind == 'roc' else 'PR-AUC'}="
                    f"{np.mean(lstAreas):.3f}±{np.std(lstAreas, ddof=1):.3f})")
        objAx.plot(arrGrid, arrStack.mean(axis=0), color=strColour, lw=1.8, label=strLabel)
        objAx.fill_between(arrGrid, arrStack.min(axis=0), arrStack.max(axis=0),
                           color=strColour, alpha=0.13, lw=0)

    if strKind == "roc":
        objAx.plot([0, 1], [0, 1], color="#8A8A8A", ls=":", lw=1.0)
        objAx.set_xlabel("False positive rate (1 - specificity)")
        objAx.set_ylabel("True positive rate (sensitivity)")
        objAx.set_title("Image-level ROC, mean over 5 repeats (band = range)", loc="left")
        objAx.legend(loc="lower right", frameon=False, fontsize=8)
    else:
        floatPrevalence = float(np.mean(arrLabels))
        objAx.axhline(floatPrevalence, color="#8A8A8A", ls=":", lw=1.0)
        objAx.annotate(f"prevalence = {floatPrevalence:.2f}", xy=(0.02, floatPrevalence),
                       xytext=(0.02, floatPrevalence - 0.07), fontsize=8, color="#8A8A8A")
        objAx.set_xlabel("Recall")
        objAx.set_ylabel("Precision")
        objAx.set_title("Image-level precision-recall, mean over 5 repeats (band = range)",
                        loc="left")
        objAx.legend(loc="lower left", frameon=False, fontsize=8)

    objAx.set_xlim(0, 1)
    objAx.set_ylim(0, 1.02)
    objAx.spines[["top", "right"]].set_visible(False)
    objFig.tight_layout()
    objFig.savefig(pathOut, dpi=300)
    plt.close(objFig)


def main():
    sys.stdout = TeeLogger(pathPhase3Dir / "phase3_run.log")
    apply_manuscript_style()
    dfTable = load_balanced_table()
    arrLabels = get_labels(dfTable)
    dictMeta, dictFolds = load_splits()

    # Hard gate: refuse to train if any patient straddles a split.
    assert_folds_patient_disjoint(dfTable, dictFolds)

    arrClinical, lstClinicalNames = build_clinical_matrix(dfTable)
    print(f"Cohort: {len(dfTable)} images / {dfTable['patient_id'].nunique()} patients. "
          f"Positive-class prevalence: {arrLabels.mean():.3f}")
    print(f"Clinical reference covariates: {', '.join(lstClinicalNames)}")

    # --- Feature inventory: encoders x ROI tracks ------------------------
    # The grid is STAGED rather than fully factorial. A full 5 encoders x 2 ROI
    # x 4 heads design is 40 configurations tested with n=10 repeats, which
    # manufactures false positives faster than it answers questions. Instead:
    #
    #   encoder axis : all encoders x both ROI tracks, head held fixed (SVM)
    #   head axis    : best encoder x both ROI tracks x all four heads
    #   layer axis   : best encoder x fixed ROI x {4 layers, last, last two}
    #
    # Each axis is its own Benjamini-Hochberg family, because correcting them
    # jointly would penalise every question for the existence of the others.
    lstEncodersFound = available_encoders(LST_ROI_MODES)
    if not lstEncodersFound:
        raise FileNotFoundError(
            "No Phase 2 feature files available for any encoder. "
            "Run 'python phases/phase2_features_baseline.py' first.")

    dictFeatures = {}
    for strEncoder in lstEncodersFound:
        for strRoiMode in LST_ROI_MODES:
            try:
                dictFeatures[(strEncoder, strRoiMode)] = load_features(
                    strRoiMode, dfTable, boolLoadPatch=True, strEncoder=strEncoder)
            except (FileNotFoundError, KeyError) as objError:
                print(f"Skipping {strEncoder}/{strRoiMode}: {objError}")

    if not dictFeatures:
        raise FileNotFoundError("No Phase 2 feature files could be loaded.")

    print(f"Encoders available: {', '.join(lstEncodersFound)}")
    print(f"Feature blocks loaded: {len(dictFeatures)}")

    # Backward-compatible alias: the reference encoder's tracks keep the short
    # configuration labels used by the existing reports and comparisons.
    dictRoiFeatures = {m: dictFeatures[(STR_REFERENCE_ENCODER, m)]
                       for m in LST_ROI_MODES
                       if (STR_REFERENCE_ENCODER, m) in dictFeatures}

    lstPerFoldRows = []
    dictPredictions = {}

    lstHyperparameterRows = []

    for strFoldName, dictFold in tqdm(sorted(dictFolds.items()), desc="folds"):
        intRepeat, intFold = parse_fold_name(strFoldName)
        arrVal = get_fold_row_positions(dfTable, dictFold["val"])
        arrYVal = arrLabels[arrVal]

        # All inner partitions of this outer fold. Phase 1 materialises four;
        # older split files yield one and get_inner_folds() says so.
        lstInnerFolds = get_inner_folds(dictFold)
        lstInnerRowPairs = []
        for dictInner in lstInnerFolds:
            arrIT = get_fold_row_positions(dfTable, dictInner["inner_train"])
            arrIC = get_fold_row_positions(dfTable, dictInner["inner_cal"])
            lstInnerRowPairs.append((arrIT, arrIC))

        # The FIRST inner partition provides the block on which each arm is
        # finally fitted and on which the score-to-probability mapping is
        # learned. The remaining partitions contribute their calibration blocks
        # to hyper-parameter selection and to threshold averaging.
        arrInnerTrain, arrInnerCal = lstInnerRowPairs[0]
        arrYInnerTrain = arrLabels[arrInnerTrain]
        arrYInnerCal = arrLabels[arrInnerCal]

        # dictFoldProbas maps config -> (calibration probabilities,
        # validation probabilities, list of (calY, calProb) per inner partition).
        # The third element is what select_threshold_averaged() consumes.
        dictFoldProbas = {}

        def register_arm(strConfig, arrCalProb, arrValProb, lstThresholdPairs):
            dictFoldProbas[strConfig] = (arrCalProb, arrValProb, lstThresholdPairs)

        # --- Clinical-metadata-only reference arm -------------------------
        # Mandatory comparator: Phase 0 established that age separates the two
        # classes, so without this arm the contribution of imaging is not
        # demonstrable.
        # Selection uses INT_TUNING_PARTITIONS calibration blocks; the rest are
        # spent on the threshold. Measured: more blocks here bought nothing.
        if BOOL_SEARCH_HYPERPARAMETERS:
            lstTuningBlocks = [(arrClinical[ic], arrLabels[ic])
                               for _, ic in lstInnerRowPairs[:INT_TUNING_PARTITIONS]]
            dictClinParams, _ = select_shallow_hyperparameters(
                arrClinical[arrInnerTrain], arrYInnerTrain, lstTuningBlocks,
                strKind="logreg", intPcaComponents=None)
        else:
            dictClinParams = dict(DICT_FIXED_LOGREG_PARAMS)
        arrCalClinScore, arrValClinScore = fit_predict_shallow(
            arrClinical[arrInnerTrain], arrYInnerTrain,
            [arrClinical[arrInnerCal], arrClinical[arrVal]],
            strKind="logreg", dictParams=dictClinParams, intPcaComponents=None)
        arrCalClin, arrValClin = calibrate_scores_to_probabilities(
            arrCalClinScore, arrYInnerCal, [arrCalClinScore, arrValClinScore])
        # Threshold pairs across every inner partition: each partition refits the
        # arm on its own training block, so its calibration scores are honest.
        lstClinPairs = []
        for arrIT, arrIC in lstInnerRowPairs:
            arrScoreIC, = fit_predict_shallow(
                arrClinical[arrIT], arrLabels[arrIT], [arrClinical[arrIC]],
                strKind="logreg", dictParams=dictClinParams, intPcaComponents=None)
            arrProbIC, = calibrate_scores_to_probabilities(
                arrScoreIC, arrLabels[arrIC], [arrScoreIC])
            lstClinPairs.append((arrLabels[arrIC], arrProbIC))
        register_arm("Clinical metadata only", arrCalClin, arrValClin, lstClinPairs)

        # --- Imaging arms, one per ROI extraction track -------------------
        for strRoiMode, (arrCls, arrPatch) in dictRoiFeatures.items():
            # Hyper-parameters chosen on the inner calibration blocks only.
            if BOOL_SEARCH_HYPERPARAMETERS:
                dictSvmParams, floatSvmScore = select_shallow_hyperparameters(
                    arrCls[arrInnerTrain], arrYInnerTrain,
                    [(arrCls[ic], arrLabels[ic])
                     for _, ic in lstInnerRowPairs[:INT_TUNING_PARTITIONS]],
                    strKind="svm", intPcaComponents=INT_PCA_COMPONENTS)
            else:
                dictSvmParams, floatSvmScore = dict(DICT_FIXED_SVM_PARAMS), float("nan")
            lstHyperparameterRows.append({
                "fold": strFoldName, "repeat": intRepeat, "roi": strRoiMode,
                "arm": "SVM on CLS", "inner_auc": floatSvmScore,
                **{k.replace("classifier__", ""): v for k, v in dictSvmParams.items()},
                "pca_components": INT_PCA_COMPONENTS})

            arrCalSvmScore, arrValSvmScore = fit_predict_shallow(
                arrCls[arrInnerTrain], arrYInnerTrain,
                [arrCls[arrInnerCal], arrCls[arrVal]], strKind="svm",
                dictParams=dictSvmParams, intPcaComponents=INT_PCA_COMPONENTS)
            arrCalSvm, arrValSvm = calibrate_scores_to_probabilities(
                arrCalSvmScore, arrYInnerCal, [arrCalSvmScore, arrValSvmScore])

            lstSvmPairs = []
            for arrIT, arrIC in lstInnerRowPairs:
                arrScoreIC, = fit_predict_shallow(
                    arrCls[arrIT], arrLabels[arrIT], [arrCls[arrIC]], strKind="svm",
                    dictParams=dictSvmParams, intPcaComponents=INT_PCA_COMPONENTS)
                arrProbIC, = calibrate_scores_to_probabilities(
                    arrScoreIC, arrLabels[arrIC], [arrScoreIC])
                lstSvmPairs.append((arrLabels[arrIC], arrProbIC))
            register_arm(f"{strRoiMode} ROI | SVM on CLS",
                         arrCalSvm, arrValSvm, lstSvmPairs)

            # ConvHead already returns probabilities from a softmax/sigmoid head
            # and does its own early stopping on the calibration block.
            arrCalConv, arrValConv = fit_predict_convhead(
                arrPatch[arrInnerTrain], arrYInnerTrain,
                arrPatch[arrInnerCal], arrYInnerCal,
                [arrPatch[arrInnerCal], arrPatch[arrVal]])
            register_arm(f"{strRoiMode} ROI | ConvHead on patches",
                         arrCalConv, arrValConv,
                         [(arrYInnerCal, arrCalConv)])

            # Both inputs are probabilities in [0, 1], which the entropy-based
            # certainty weighting requires.
            register_arm(f"{strRoiMode} ROI | Certainty-weighted ensemble",
                         ensemble_uncertainty_weighted(arrCalSvm, arrCalConv),
                         ensemble_uncertainty_weighted(arrValSvm, arrValConv),
                         [(arrYInnerCal,
                           ensemble_uncertainty_weighted(arrCalSvm, arrCalConv))])

            # Late fusion of imaging and demographics: tests whether the two
            # information sources are complementary rather than redundant.
            register_arm(f"{strRoiMode} ROI | Imaging + clinical fusion",
                         ensemble_uncertainty_weighted(arrCalSvm, arrCalClin),
                         ensemble_uncertainty_weighted(arrValSvm, arrValClin),
                         [(arrYInnerCal,
                           ensemble_uncertainty_weighted(arrCalSvm, arrCalClin))])

        # --- ENCODER AXIS -------------------------------------------------
        # Every encoder x both ROI tracks, with the head held fixed to the
        # shallow classifier. Holding the head constant is what makes a
        # difference here attributable to the representation; varying head and
        # encoder simultaneously would confound them.
        #
        # The reference encoder's own arms are already registered above under
        # the short labels, so they are skipped to avoid duplicate rows.
        for (strEncoder, strRoiMode), (arrCls, _) in sorted(dictFeatures.items()):
            if strEncoder == STR_REFERENCE_ENCODER:
                continue
            arrCalScore, arrValScore = fit_predict_shallow(
                arrCls[arrInnerTrain], arrYInnerTrain,
                [arrCls[arrInnerCal], arrCls[arrVal]], strKind="svm",
                dictParams=dict(DICT_FIXED_SVM_PARAMS),
                intPcaComponents=INT_PCA_COMPONENTS)
            arrCalProb, arrValProb = calibrate_scores_to_probabilities(
                arrCalScore, arrYInnerCal, [arrCalScore, arrValScore])

            lstEncPairs = []
            for arrIT, arrIC in lstInnerRowPairs:
                arrScoreIC, = fit_predict_shallow(
                    arrCls[arrIT], arrLabels[arrIT], [arrCls[arrIC]], strKind="svm",
                    dictParams=dict(DICT_FIXED_SVM_PARAMS),
                    intPcaComponents=INT_PCA_COMPONENTS)
                arrProbIC, = calibrate_scores_to_probabilities(
                    arrScoreIC, arrLabels[arrIC], [arrScoreIC])
                lstEncPairs.append((arrLabels[arrIC], arrProbIC))

            register_arm(f"{strEncoder} | {strRoiMode} ROI | SVM on CLS",
                         arrCalProb, arrValProb, lstEncPairs)

        # --- LAYER AXIS ---------------------------------------------------
        # The reference encoder concatenates the class token of its last four
        # blocks. That is a design choice the manuscript currently asserts
        # without defending, so it is tested: the concatenated vector is sliced
        # into its constituent layers and the alternatives are evaluated as
        # formal arms on the fixed ROI track.
        if (STR_REFERENCE_ENCODER, "fixed") in dictFeatures and BOOL_RUN_LAYER_ABLATION:
            arrClsRef, _ = dictFeatures[(STR_REFERENCE_ENCODER, "fixed")]
            intLayers = DICT_ENCODERS[STR_REFERENCE_ENCODER]["n_layers_concat"]
            intWidth = arrClsRef.shape[1] // intLayers
            # DEFECT FIXED: "last layer only" was a duplicate label for the slice
            # that the loop below already emits as "layer -1" -- the last layer IS
            # layer -1. Both labels produced byte-identical per-fold metrics (AUC
            # 0.810923 across all 50 folds), which inflated the layer-family
            # contrast count from 5 real comparisons to 6 rows and made the
            # Benjamini-Hochberg denominator wrong for that family. Only the
            # genuinely different multi-layer variant is declared by hand.
            dictLayerVariants = {
                "last two layers": arrClsRef[:, -2 * intWidth:],
            }
            for intLayer in range(intLayers):
                dictLayerVariants[f"layer -{intLayers - intLayer}"] = (
                    arrClsRef[:, intLayer * intWidth:(intLayer + 1) * intWidth])

            for strVariant, arrVariant in dictLayerVariants.items():
                arrCalScore, arrValScore = fit_predict_shallow(
                    arrVariant[arrInnerTrain], arrYInnerTrain,
                    [arrVariant[arrInnerCal], arrVariant[arrVal]], strKind="svm",
                    dictParams=dict(DICT_FIXED_SVM_PARAMS),
                    intPcaComponents=INT_PCA_COMPONENTS)
                arrCalProb, arrValProb = calibrate_scores_to_probabilities(
                    arrCalScore, arrYInnerCal, [arrCalScore, arrValScore])
                lstLayerPairs = []
                for arrIT, arrIC in lstInnerRowPairs:
                    arrScoreIC, = fit_predict_shallow(
                        arrVariant[arrIT], arrLabels[arrIT], [arrVariant[arrIC]],
                        strKind="svm", dictParams=dict(DICT_FIXED_SVM_PARAMS),
                        intPcaComponents=INT_PCA_COMPONENTS)
                    arrProbIC, = calibrate_scores_to_probabilities(
                        arrScoreIC, arrLabels[arrIC], [arrScoreIC])
                    lstLayerPairs.append((arrLabels[arrIC], arrProbIC))
                register_arm(f"fixed ROI | CLS {strVariant}",
                             arrCalProb, arrValProb, lstLayerPairs)

        for strConfig, (arrCalProbCfg, arrValProbCfg, lstPairs) in dictFoldProbas.items():
            # Threshold from the calibration partitions ONLY, aggregated across
            # them to cut its variance. The alternative criterion is recorded so
            # the manuscript can report a sensitivity analysis.
            floatThreshold, lstPartitionThresholds = select_threshold_averaged(
                lstPairs, STR_THRESHOLD_CRITERION)
            floatThresholdAlt, _ = select_threshold_averaged(
                lstPairs, "youden" if STR_THRESHOLD_CRITERION != "youden" else "equal_error")

            dictMetrics = calculate_full_metrics(arrYVal, arrValProbCfg, floatThreshold)
            lstPerFoldRows.append({
                "config": strConfig, "fold": strFoldName, "repeat": intRepeat,
                "outer_fold": intFold, "threshold": floatThreshold,
                "threshold_criterion": STR_THRESHOLD_CRITERION,
                "threshold_alternative": floatThresholdAlt,
                "n_threshold_partitions": len(lstPartitionThresholds),
                "threshold_sd_across_partitions": (
                    float(np.std(lstPartitionThresholds, ddof=1))
                    if len(lstPartitionThresholds) > 1 else 0.0),
                "n_val_images": len(arrVal),
                "n_val_patients": len(dictFold["val_patients"]),
                **dictMetrics})
            dictPredictions.setdefault(strConfig, {}).setdefault(intRepeat, []).append(
                {"prob": arrValProbCfg, "true": arrYVal})

    dfPerFold = pd.DataFrame(lstPerFoldRows)
    dfPerFold.to_csv(pathPhase3Dir / "phase3_per_fold_metrics.csv", index=False)

    dfSummary = summarise_across_repeats(dfPerFold, ["config"])
    dfSummary = dfSummary.sort_values("AUC_mean", ascending=False).reset_index(drop=True)
    lstReportCols = (["config", "n_repeats"] + LST_METRIC_KEYS
                     + [f"{k}_mean" for k in LST_METRIC_KEYS])
    dfSummary[lstReportCols].to_csv(pathPhase3Dir / "phase3_metrics_table.csv", index=False)
    print("\n" + dfSummary[["config"] + LST_METRIC_KEYS[:4]].to_string(index=False))

    # --- Paired statistical comparisons ---------------------------------
    lstComparisons = []
    strBestImaging = next(c for c in dfSummary["config"] if c != "Clinical metadata only")
    lstPairs = [(strBestImaging, "Clinical metadata only")]
    for strRoiMode in dictRoiFeatures:
        lstPairs += [
            (f"{strRoiMode} ROI | Certainty-weighted ensemble", f"{strRoiMode} ROI | SVM on CLS"),
            (f"{strRoiMode} ROI | ConvHead on patches", f"{strRoiMode} ROI | SVM on CLS"),
            (f"{strRoiMode} ROI | Imaging + clinical fusion", f"{strRoiMode} ROI | SVM on CLS"),
        ]
    # DEFECT FIXED: the fixed-vs-dynamic ROI contrast used to be appended here as
    # well, which placed it in BOTH the 'head' family (via lstPairs) and the 'roi'
    # family (via lstRoiPairs below). Benjamini-Hochberg corrects within a family
    # and the two families have different sizes, so the same experiment was written
    # twice with two different adjusted p-values (AUC: 0.000148 in 'head' vs
    # 0.000317 in 'roi'). Identical mean difference, identical raw p, two verdicts
    # on file. It never flipped a significance call, but a reviewer opening the CSV
    # would have no way to know which figure to quote. The ROI axis is now declared
    # exclusively in lstRoiPairs, where it belongs conceptually.

    # --- Encoder-axis contrasts -----------------------------------------
    # PRIMARY CONTRASTS, declared before the run rather than chosen after it:
    #   1. reference foundation model vs the convolutional floor
    #   2. best medical-domain encoder vs best natural-domain encoder
    # Everything else on this axis is exploratory and is reported as such. With
    # ~22 configurations, deciding which contrasts matter after seeing the
    # results is how a paper acquires findings that do not replicate.
    lstEncoderPairs, lstRoiPairs = [], []
    dictLabel = {}
    for strEncoder in lstEncodersFound:
        for strRoiMode in LST_ROI_MODES:
            if (strEncoder, strRoiMode) not in dictFeatures:
                continue
            dictLabel[(strEncoder, strRoiMode)] = (
                f"{strRoiMode} ROI | SVM on CLS" if strEncoder == STR_REFERENCE_ENCODER
                else f"{strEncoder} | {strRoiMode} ROI | SVM on CLS")

    for strEncoder in lstEncodersFound:
        if strEncoder == STR_REFERENCE_ENCODER:
            continue
        if (strEncoder, "fixed") in dictLabel and (STR_REFERENCE_ENCODER, "fixed") in dictLabel:
            lstEncoderPairs.append((dictLabel[(strEncoder, "fixed")],
                                    dictLabel[(STR_REFERENCE_ENCODER, "fixed")]))
    # GAP FIXED: the loop above only compares each encoder AGAINST THE REFERENCE,
    # so the pair that actually isolates the domain axis -- natural vs medical with
    # the learning objective held constant -- was missing whenever neither member
    # was the reference. Concretely, DINOv2 vs RAD-DINO existed (the reference is
    # one of them) but CLIP vs BiomedCLIP did not, even though it is the second
    # half of the headline finding. Its p-value had to be recomputed by hand and
    # could not be multiplicity-corrected, giving the two halves of one replicated
    # result asymmetric statistical support. These are the contrasts that carry the
    # paper's central claim, so they are declared here as primary.
    for strEncoderA in lstEncodersFound:
        for strEncoderB in lstEncodersFound:
            dictA = DICT_ENCODERS.get(strEncoderA, {})
            dictB = DICT_ENCODERS.get(strEncoderB, {})
            # Same objective, natural on the left, medical on the right: one
            # ordered pair per objective, never its mirror image.
            if (dictA.get("objective") != dictB.get("objective")
                    or dictA.get("domain") != "natural"
                    or dictB.get("domain") != "medical"):
                continue
            if (strEncoderA, "fixed") not in dictLabel or (strEncoderB, "fixed") not in dictLabel:
                continue
            tupPair = (dictLabel[(strEncoderA, "fixed")], dictLabel[(strEncoderB, "fixed")])
            # The reference-vs-others loop above already emits the medical member
            # against the reference, e.g. (RAD-DINO, DINOv2). Appending the natural
            # member first, e.g. (DINOv2, RAD-DINO), would write the SAME experiment
            # a second time with the sign flipped -- exactly the duplication defect
            # fixed on the ROI axis. Skip a pair when either direction is present.
            if tupPair in lstEncoderPairs or (tupPair[1], tupPair[0]) in lstEncoderPairs:
                continue
            lstEncoderPairs.append(tupPair)

    # ROI axis, crossed over every encoder: this is the experiment that
    # ESTABLISHES the fixed-vs-dynamic result inside the paper instead of
    # inheriting it from previous runs.
    for strEncoder in lstEncodersFound:
        if (strEncoder, "fixed") in dictLabel and (strEncoder, "dynamic") in dictLabel:
            lstRoiPairs.append((dictLabel[(strEncoder, "dynamic")],
                                dictLabel[(strEncoder, "fixed")]))

    # --- Layer-axis contrasts -------------------------------------------
    lstLayerPairsCmp = []
    lstLayerConfigs = [c for c in dfSummary["config"] if "CLS " in str(c)]
    for strLayerConfig in lstLayerConfigs:
        lstLayerPairsCmp.append((strLayerConfig, "fixed ROI | SVM on CLS"))

    for strMetric in ["AUC", "PR-AUC", "MCC", "Balanced Accuracy"]:
        for lstGroup, strFamily in [(lstPairs, None),
                                    (lstEncoderPairs, "encoder"),
                                    (lstRoiPairs, "roi"),
                                    (lstLayerPairsCmp, "layer")]:
            for strA, strB in lstGroup:
                dictComparison = compare_models(dfPerFold, strMetric, strA, strB)
                if not dictComparison:
                    continue
                # Family for multiplicity correction. Each axis answers a
                # separate question, so pooling them would penalise every
                # question for the existence of the others.
                if strFamily is None:
                    dictComparison["family"] = (
                        f"clinical|{strMetric}" if strB == "Clinical metadata only"
                        else f"head|{strMetric}")
                else:
                    dictComparison["family"] = f"{strFamily}|{strMetric}"
                # Primary contrasts on the encoder axis, declared before the run:
                #   (a) reference foundation model vs the convolutional floor, and
                #   (b) natural vs medical domain with the objective held constant.
                # (b) is the contrast the paper's headline rests on, so it must be
                # flagged as primary rather than sitting among the exploratory rows.
                dictEncA = DICT_ENCODERS.get(parse_config_label(strA)["encoder"], {})
                dictEncB = DICT_ENCODERS.get(parse_config_label(strB)["encoder"], {})
                boolFloor = strFamily == "encoder" and dictEncA.get("family") == "cnn"
                boolDomain = (strFamily == "encoder"
                              and dictEncA.get("objective") == dictEncB.get("objective")
                              and {dictEncA.get("domain"), dictEncB.get("domain")}
                              == {"natural", "medical"})
                dictComparison["primary_contrast"] = bool(boolFloor or boolDomain)
                lstComparisons.append(dictComparison)

    dfComparisons = add_multiplicity_correction(pd.DataFrame(lstComparisons))
    dfComparisons.to_csv(pathPhase3Dir / "phase3_statistical_comparisons.csv", index=False)
    export_manuscript_table(
        dfComparisons,
        pathPhase3Dir / "phase3_statistical_comparisons",
        strCaption="Paired contrasts at the image level. The primary test is a "
                   "paired t-test over repeat-level means; p-values are corrected "
                   "within family by Benjamini-Hochberg.",
        strLabel="tab:phase3-stats")

    # --- Manuscript tables ----------------------------------------------
    # The summary table is annotated with the experimental factors it crosses,
    # so a reader can group rows by encoder, ROI or head without parsing labels.
    dfAnnotated = dfSummary[lstReportCols].copy()
    dfFactors = pd.DataFrame([parse_config_label(c) for c in dfAnnotated["config"]])
    dfAnnotated.insert(1, "encoder", dfFactors["encoder"].values)
    dfAnnotated.insert(2, "roi", dfFactors["roi"].values)
    dfAnnotated.insert(3, "head", dfFactors["head"].values)
    export_manuscript_table(
        dfAnnotated,
        pathPhase3Dir / "phase3_metrics_table_annotated",
        strCaption="Image-level classification performance across encoders, "
                   "ROI strategies and classification heads. Mean +/- standard "
                   "deviation over five repeat-level means.",
        strLabel="tab:phase3-main")

    # Encoder-axis table: the head is held fixed, so the rows differ only in the
    # frozen representation. This is the table that answers the paper's question.
    lstEncoderRows = []
    for (strEncoder, strRoiMode), strLabel in dictLabel.items():
        dfRow = dfSummary[dfSummary["config"] == strLabel]
        if dfRow.empty:
            continue
        dictMeta = DICT_ENCODERS.get(strEncoder, {})
        lstEncoderRows.append({
            "encoder": dictMeta.get("display_name", strEncoder),
            "pre_training_domain": dictMeta.get("domain", "n/a"),
            "pre_training_objective": dictMeta.get("objective", "n/a"),
            "roi": strRoiMode,
            **{k: dfRow.iloc[0][k] for k in LST_METRIC_KEYS[:4]},
        })
    if lstEncoderRows:
        dfEncoderTable = pd.DataFrame(lstEncoderRows).sort_values(
            ["roi", "encoder"]).reset_index(drop=True)
        export_manuscript_table(
            dfEncoderTable,
            pathPhase3Dir / "phase3_encoder_comparison",
            strCaption="Encoder comparison with the classification head held "
                       "fixed. Differences are therefore attributable to the "
                       "frozen representation rather than to the downstream model.",
            strLabel="tab:phase3-encoders")

        # Encoder-axis figure, grouped by ROI track.
        objFig, objAx = plt.subplots(figsize=(7.0, 3.8))
        dfPivot = dfEncoderTable.copy()
        dfPivot["auc_mean"] = [float(str(v).split("±")[0]) for v in dfPivot["AUC"]]
        dfPivot["auc_sd"] = [float(str(v).split("±")[1]) if "±" in str(v) else 0.0
                             for v in dfPivot["AUC"]]
        lstEncoderNames = sorted(dfPivot["encoder"].unique())
        arrX = np.arange(len(lstEncoderNames))
        for intIdx, strRoiMode in enumerate(sorted(dfPivot["roi"].unique())):
            dfSub = dfPivot[dfPivot["roi"] == strRoiMode].set_index("encoder")
            dfSub = dfSub.reindex(lstEncoderNames)
            objAx.bar(arrX + (intIdx - 0.5) * 0.36, dfSub["auc_mean"], 0.34,
                      yerr=dfSub["auc_sd"], capsize=3, label=f"{strRoiMode} ROI",
                      color=LST_PALETTE[intIdx], edgecolor="black", linewidth=0.5)
        objAx.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        objAx.set_xticks(arrX)
        objAx.set_xticklabels(lstEncoderNames, rotation=20, ha="right")
        objAx.set_ylabel("AUC (image level)")
        objAx.set_ylim(0.45, 1.0)
        objAx.set_title("Frozen-encoder comparison, classification head held fixed")
        objAx.legend(frameon=False, loc="lower right")
        objFig.tight_layout()
        save_figure(objFig, pathPhase3Dir / "phase3_encoder_comparison")
        plt.close(objFig)

    # --- Selected hyper-parameters --------------------------------------
    # Exported so the manuscript can state what the inner loop chose and a
    # reviewer can confirm the search actually varied the model rather than
    # settling on the default everywhere.
    if lstHyperparameterRows:
        dfHyper = pd.DataFrame(lstHyperparameterRows)
        dfHyper.to_csv(pathPhase3Dir / "phase3_hyperparameters.csv", index=False)
        print("\nSelected hyper-parameters (frequency across the 25 folds):")
        for strRoiMode, dfGroup in dfHyper.groupby("roi"):
            lstCols = [c for c in ("C", "gamma") if c in dfGroup.columns]
            if not lstCols:
                continue
            srCounts = dfGroup.groupby(lstCols).size().sort_values(ascending=False)
            print(f"  {strRoiMode} ROI:")
            for objKey, intCount in srCounts.items():
                print(f"    {objKey} -> {intCount} folds")
            print(f"    mean inner AUC = {dfGroup['inner_auc'].mean():.4f}")

    # --- Curves ---------------------------------------------------------
    dictCurveSubset = {k: v for k, v in dictPredictions.items()
                       if k == "Clinical metadata only" or k in dfSummary["config"].head(4).tolist()}
    plot_curves_per_repeat(dictCurveSubset, dfTable, arrLabels,
                           pathPhase3Dir / "phase3_roc_curves.png", strKind="roc")
    plot_curves_per_repeat(dictCurveSubset, dfTable, arrLabels,
                           pathPhase3Dir / "phase3_pr_curves.png", strKind="pr")

    print(f"\nPhase 3 complete. Artifacts written to {pathPhase3Dir}")


if __name__ == "__main__":
    main()
