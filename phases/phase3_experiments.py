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

LST_SVM_C_GRID = [0.1, 1.0, 10.0, 100.0]
LST_SVM_GAMMA_GRID = ["scale", 1e-3, 1e-2]
LST_LOGREG_C_GRID = [0.01, 0.1, 1.0, 10.0]
INT_PCA_COMPONENTS = 100
LST_ROI_MODES = ["fixed", "dynamic"]
BOOL_RUN_LAYER_ABLATION = True
STR_THRESHOLD_CRITERION = "youden"
INT_TUNING_PARTITIONS = 1
BOOL_SEARCH_HYPERPARAMETERS = False
DICT_FIXED_SVM_PARAMS = {"classifier__C": 1.0, "classifier__gamma": "scale"}
DICT_FIXED_LOGREG_PARAMS = {"classifier__C": 1.0}

pathPhase3Dir.mkdir(parents=True, exist_ok=True)
set_global_seed(INT_GLOBAL_SEED)
strDevice = get_torch_device()

INT_MAX_EPOCHS = 40
INT_PATIENCE = 6
INT_BATCH_SIZE = 16


class SimpleConvHead(nn.Module):
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
    if strKind == "svm":
        objClf = SVC(kernel="rbf", probability=False, class_weight="balanced",
                     random_state=INT_GLOBAL_SEED)
    else:
        objClf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                    random_state=INT_GLOBAL_SEED)

    lstSteps = [("scaler", StandardScaler())]
    if intPcaComponents is not None:
        lstSteps.append(("pca", PCA(n_components=intPcaComponents, whiten=True,
                                    random_state=INT_GLOBAL_SEED)))
    lstSteps.append(("classifier", objClf))
    return Pipeline(lstSteps)


def predict_scores_shallow(objPipeline, arrX):
    if hasattr(objPipeline, "decision_function"):
        return objPipeline.decision_function(arrX)
    return objPipeline.predict_proba(arrX)[:, 1]


def select_shallow_hyperparameters(arrTrainX, arrTrainY, lstInnerCalBlocks,
                                   strKind="svm", intPcaComponents=None):
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
    objPipeline = make_shallow_classifier(strKind, intPcaComponents)
    if dictParams:
        objPipeline.set_params(**dictParams)
    objPipeline.fit(arrTrainX, arrTrainY)
    return [predict_scores_shallow(objPipeline, arrX) for arrX in lstEvalX]


def fit_predict_convhead(arrTrainX, arrTrainY, arrCalX, arrCalY, lstEvalX):
    torch.manual_seed(INT_GLOBAL_SEED)
    tsrTrainX = torch.tensor(arrTrainX, dtype=torch.float32)
    tsrTrainY = torch.tensor(arrTrainY, dtype=torch.long)
    tsrCalX = torch.tensor(arrCalX, dtype=torch.float32).to(strDevice)
    tsrCalY = torch.tensor(arrCalY, dtype=torch.long).to(strDevice)

    arrCounts = np.bincount(arrTrainY, minlength=2).astype(float)
    arrCounts[arrCounts == 0] = 1.0
    tsrWeights = torch.tensor(len(arrTrainY) / (2.0 * arrCounts),
                              dtype=torch.float32).to(strDevice)

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
    arrPosteriorA = np.vstack((1.0 - arrProbA, arrProbA)).T
    arrPosteriorB = np.vstack((1.0 - arrProbB, arrProbB)).T

    arrWeightA = (1.0 - entropy(arrPosteriorA, axis=1, base=2)).clip(min=1e-6)
    arrWeightB = (1.0 - entropy(arrPosteriorB, axis=1, base=2)).clip(min=1e-6)
    arrSum = arrWeightA + arrWeightB
    return (arrProbA * arrWeightA + arrProbB * arrWeightB) / arrSum


def compare_models(dfPerFold, strMetric, strModelA, strModelB):
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

    assert_folds_patient_disjoint(dfTable, dictFolds)

    arrClinical, lstClinicalNames = build_clinical_matrix(dfTable)
    print(f"Cohort: {len(dfTable)} images / {dfTable['patient_id'].nunique()} patients. "
          f"Positive-class prevalence: {arrLabels.mean():.3f}")
    print(f"Clinical reference covariates: {', '.join(lstClinicalNames)}")

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

        lstInnerFolds = get_inner_folds(dictFold)
        lstInnerRowPairs = []
        for dictInner in lstInnerFolds:
            arrIT = get_fold_row_positions(dfTable, dictInner["inner_train"])
            arrIC = get_fold_row_positions(dfTable, dictInner["inner_cal"])
            lstInnerRowPairs.append((arrIT, arrIC))

        arrInnerTrain, arrInnerCal = lstInnerRowPairs[0]
        arrYInnerTrain = arrLabels[arrInnerTrain]
        arrYInnerCal = arrLabels[arrInnerCal]

        dictFoldProbas = {}

        def register_arm(strConfig, arrCalProb, arrValProb, lstThresholdPairs):
            dictFoldProbas[strConfig] = (arrCalProb, arrValProb, lstThresholdPairs)

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
        
        lstClinPairs = []
        for arrIT, arrIC in lstInnerRowPairs:
            arrScoreIC, = fit_predict_shallow(
                arrClinical[arrIT], arrLabels[arrIT], [arrClinical[arrIC]],
                strKind="logreg", dictParams=dictClinParams, intPcaComponents=None)
            arrProbIC, = calibrate_scores_to_probabilities(
                arrScoreIC, arrLabels[arrIC], [arrScoreIC])
            lstClinPairs.append((arrLabels[arrIC], arrProbIC))
        register_arm("Clinical metadata only", arrCalClin, arrValClin, lstClinPairs)

        for strRoiMode, (arrCls, arrPatch) in dictRoiFeatures.items():
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

            arrCalConv, arrValConv = fit_predict_convhead(
                arrPatch[arrInnerTrain], arrYInnerTrain,
                arrPatch[arrInnerCal], arrYInnerCal,
                [arrPatch[arrInnerCal], arrPatch[arrVal]])
            register_arm(f"{strRoiMode} ROI | ConvHead on patches",
                         arrCalConv, arrValConv,
                         [(arrYInnerCal, arrCalConv)])

            register_arm(f"{strRoiMode} ROI | Certainty-weighted ensemble",
                         ensemble_uncertainty_weighted(arrCalSvm, arrCalConv),
                         ensemble_uncertainty_weighted(arrValSvm, arrValConv),
                         [(arrYInnerCal,
                           ensemble_uncertainty_weighted(arrCalSvm, arrCalConv))])

            register_arm(f"{strRoiMode} ROI | Imaging + clinical fusion",
                         ensemble_uncertainty_weighted(arrCalSvm, arrCalClin),
                         ensemble_uncertainty_weighted(arrValSvm, arrValClin),
                         [(arrYInnerCal,
                           ensemble_uncertainty_weighted(arrCalSvm, arrCalClin))])

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

        if (STR_REFERENCE_ENCODER, "fixed") in dictFeatures and BOOL_RUN_LAYER_ABLATION:
            arrClsRef, _ = dictFeatures[(STR_REFERENCE_ENCODER, "fixed")]
            intLayers = DICT_ENCODERS[STR_REFERENCE_ENCODER]["n_layers_concat"]
            intWidth = arrClsRef.shape[1] // intLayers
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

    lstComparisons = []
    strBestImaging = next(c for c in dfSummary["config"] if c != "Clinical metadata only")
    lstPairs = [(strBestImaging, "Clinical metadata only")]
    for strRoiMode in dictRoiFeatures:
        lstPairs += [
            (f"{strRoiMode} ROI | Certainty-weighted ensemble", f"{strRoiMode} ROI | SVM on CLS"),
            (f"{strRoiMode} ROI | ConvHead on patches", f"{strRoiMode} ROI | SVM on CLS"),
            (f"{strRoiMode} ROI | Imaging + clinical fusion", f"{strRoiMode} ROI | SVM on CLS"),
        ]

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
            if tupPair in lstEncoderPairs or (tupPair[1], tupPair[0]) in lstEncoderPairs:
                continue
            lstEncoderPairs.append(tupPair)

    for strEncoder in lstEncodersFound:
        if (strEncoder, "fixed") in dictLabel and (strEncoder, "dynamic") in dictLabel:
            lstRoiPairs.append((dictLabel[(strEncoder, "dynamic")],
                                dictLabel[(strEncoder, "fixed")]))

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

                if strFamily is None:
                    dictComparison["family"] = (
                        f"clinical|{strMetric}" if strB == "Clinical metadata only"
                        else f"head|{strMetric}")
                else:
                    dictComparison["family"] = f"{strFamily}|{strMetric}"

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

    dictCurveSubset = {k: v for k, v in dictPredictions.items()
                       if k == "Clinical metadata only" or k in dfSummary["config"].head(4).tolist()}
    plot_curves_per_repeat(dictCurveSubset, dfTable, arrLabels,
                           pathPhase3Dir / "phase3_roc_curves.png", strKind="roc")
    plot_curves_per_repeat(dictCurveSubset, dfTable, arrLabels,
                           pathPhase3Dir / "phase3_pr_curves.png", strKind="pr")

    print(f"\nPhase 3 complete. Artifacts written to {pathPhase3Dir}")


if __name__ == "__main__":
    main()
