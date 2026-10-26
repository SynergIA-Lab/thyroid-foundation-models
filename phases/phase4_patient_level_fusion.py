import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_multiplicity_correction, apply_manuscript_style,
                    assert_folds_patient_disjoint, available_encoders,
                    build_clinical_matrix, calculate_full_metrics,
                    compare_configs, DICT_ENCODERS, export_manuscript_table,
                    get_inner_folds, get_labels, get_torch_device,
                    INT_GLOBAL_SEED, load_balanced_table, load_features,
                    load_splits, LST_METRIC_KEYS, LST_PALETTE,
                    parse_config_label, parse_fold_name, pathPhase4Dir,
                    save_figure, select_threshold_averaged, set_global_seed,
                    STR_REFERENCE_ENCODER, summarise_across_repeats, TeeLogger)

pathPhase4Dir.mkdir(parents=True, exist_ok=True)
set_global_seed(INT_GLOBAL_SEED)
strDevice = get_torch_device()

INT_MAX_EPOCHS = 60
INT_PATIENCE = 10
INT_BATCH_SIZE = 8
STR_REFERENCE_VIEW = "ANTERIOR"
LST_ROI_MODES = ["fixed", "dynamic"]
INT_THRESHOLD_PARTITIONS = 4
STR_THRESHOLD_CRITERION = "youden"
class PatientDataset(Dataset):
    def __init__(self, lstPatientIds, dictPatientData):
        self.lstPatientIds = list(lstPatientIds)
        self.dictPatientData = dictPatientData

    def __len__(self):
        return len(self.lstPatientIds)

    def __getitem__(self, intIndex):
        strPid = self.lstPatientIds[intIndex]
        dictEntry = self.dictPatientData[strPid]
        tsrFeatures = torch.tensor(np.asarray(dictEntry["features"]), dtype=torch.float32)
        return tsrFeatures, int(dictEntry["label"])

def collate_patient_sets(batch):
    lstFeatures = [item[0] for item in batch]
    lstLabels = [item[1] for item in batch]
    tsrLabels = torch.tensor(lstLabels, dtype=torch.long)
    tsrPaddedFeatures = torch.nn.utils.rnn.pad_sequence(lstFeatures, batch_first=True, padding_value=0.0)
    
    lstLengths = [len(f) for f in lstFeatures]
    intMaxLen = max(lstLengths)
    tsrMask = torch.zeros((len(batch), intMaxLen), dtype=torch.bool)
    for i, l in enumerate(lstLengths):
        if l < intMaxLen:
            tsrMask[i, l:] = True
            
    return tsrPaddedFeatures, tsrLabels, tsrMask, lstLengths

class BaseFusion(nn.Module):
    def __init__(self, intEmbedDim=3072, intHiddenDim=64, intNumClasses=2, floatDropout=0.3):
        super().__init__()
        self.objClassifier = nn.Sequential(
            nn.Linear(intEmbedDim, intHiddenDim),
            nn.ReLU(inplace=True),
            nn.Dropout(floatDropout),
            nn.Linear(intHiddenDim, intNumClasses),
        )

    def pool(self, tsrFeatures, tsrMask):
        raise NotImplementedError

    def forward(self, tsrFeatures, tsrMask, boolReturnEmbeds=False):
        tsrPooled, tsrWeights = self.pool(tsrFeatures, tsrMask)
        tsrLogits = self.objClassifier(tsrPooled)
        if boolReturnEmbeds:
            return tsrLogits, tsrPooled, tsrWeights
        return tsrLogits


class MeanPoolFusion(BaseFusion):
    def pool(self, tsrFeatures, tsrMask):
        tsrValid = (~tsrMask).unsqueeze(-1).float()
        tsrPooled = (tsrFeatures * tsrValid).sum(dim=1) / tsrValid.sum(dim=1).clamp(min=1.0)
        return tsrPooled, tsrValid.squeeze(-1) / tsrValid.sum(dim=1).clamp(min=1.0)


class MaxPoolFusion(BaseFusion):
    def pool(self, tsrFeatures, tsrMask):
        tsrMasked = tsrFeatures.masked_fill(tsrMask.unsqueeze(-1), float("-inf"))
        tsrPooled = tsrMasked.max(dim=1).values
        tsrPooled = torch.nan_to_num(tsrPooled, neginf=0.0)
        return tsrPooled, torch.zeros(tsrFeatures.shape[:2], device=tsrFeatures.device)


class GatedAttentionFusion(BaseFusion):
    def __init__(self, intEmbedDim=3072, intAttnDim=64, **kwargs):
        super().__init__(intEmbedDim=intEmbedDim, **kwargs)
        self.objAttnV = nn.Sequential(nn.Linear(intEmbedDim, intAttnDim), nn.Tanh())
        self.objAttnU = nn.Sequential(nn.Linear(intEmbedDim, intAttnDim), nn.Sigmoid())
        self.objAttnW = nn.Linear(intAttnDim, 1)

    def pool(self, tsrFeatures, tsrMask):
        tsrScores = self.objAttnW(self.objAttnV(tsrFeatures) * self.objAttnU(tsrFeatures))
        tsrScores = tsrScores.squeeze(-1).masked_fill(tsrMask, float("-inf"))
        tsrWeights = torch.softmax(tsrScores, dim=1)
        tsrPooled = (tsrFeatures * tsrWeights.unsqueeze(-1)).sum(dim=1)
        return tsrPooled, tsrWeights


DICT_FUSION_MODELS = {
    "Mean pooling": MeanPoolFusion,
    "Max pooling": MaxPoolFusion,
    "Gated attention pooling": GatedAttentionFusion,
}

def build_patient_records(dfTable, arrCls, arrLabels):
    dictPatientData = {}
    for intRow, serRow in dfTable.reset_index(drop=True).iterrows():
        strPid = str(serRow["patient_id"])
        dictEntry = dictPatientData.setdefault(
            strPid, {"features": [], "views": [], "label": int(arrLabels[intRow])})
        if dictEntry["label"] != int(arrLabels[intRow]):
            raise ValueError(f"Patient {strPid} carries conflicting labels.")
        dictEntry["features"].append(arrCls[intRow])
        dictEntry["views"].append(str(serRow["view"]))
    return dictPatientData


def patients_of(dfTable, lstSampleIds):
    srSelected = dfTable.set_index("sample_id").loc[lstSampleIds, "patient_id"]
    return sorted(set(srSelected.astype(str)))


def train_fusion_model(objModelClass, lstTrainPids, lstCalPids, dictPatientData,
                       intEmbedDim):
    torch.manual_seed(INT_GLOBAL_SEED)
    objModel = objModelClass(intEmbedDim=intEmbedDim).to(strDevice)

    arrTrainLabels = np.array([dictPatientData[p]["label"] for p in lstTrainPids])
    arrCounts = np.bincount(arrTrainLabels, minlength=2).astype(float)
    arrCounts[arrCounts == 0] = 1.0
    tsrWeights = torch.tensor(len(arrTrainLabels) / (2.0 * arrCounts),
                              dtype=torch.float32).to(strDevice)

    objCriterion = nn.CrossEntropyLoss(weight=tsrWeights)
    objOptimizer = optim.Adam(objModel.parameters(), lr=1e-3, weight_decay=1e-4)

    objTrainLoader = DataLoader(PatientDataset(lstTrainPids, dictPatientData),
                                batch_size=INT_BATCH_SIZE, shuffle=True,
                                collate_fn=collate_patient_sets)
    objCalLoader = DataLoader(PatientDataset(lstCalPids, dictPatientData),
                              batch_size=INT_BATCH_SIZE, shuffle=False,
                              collate_fn=collate_patient_sets)

    floatBestLoss = np.inf
    intNoImprovement = 0
    dictBestState = copy.deepcopy(objModel.state_dict())

    for intEpoch in range(INT_MAX_EPOCHS):
        objModel.train()
        for tsrX, tsrY, tsrMask, _ in objTrainLoader:
            tsrX, tsrY, tsrMask = tsrX.to(strDevice), tsrY.to(strDevice), tsrMask.to(strDevice)
            objOptimizer.zero_grad()
            objLoss = objCriterion(objModel(tsrX, tsrMask), tsrY)
            objLoss.backward()
            objOptimizer.step()

        objModel.eval()
        floatCalLoss, intCalBatches = 0.0, 0
        with torch.no_grad():
            for tsrX, tsrY, tsrMask, _ in objCalLoader:
                tsrX, tsrY, tsrMask = tsrX.to(strDevice), tsrY.to(strDevice), tsrMask.to(strDevice)
                floatCalLoss += objCriterion(objModel(tsrX, tsrMask), tsrY).item()
                intCalBatches += 1
        floatCalLoss /= max(intCalBatches, 1)

        if floatCalLoss < floatBestLoss - 1e-4:
            floatBestLoss = floatCalLoss
            dictBestState = copy.deepcopy(objModel.state_dict())
            intNoImprovement = 0
        else:
            intNoImprovement += 1
            if intNoImprovement >= INT_PATIENCE:
                break

    objModel.load_state_dict(dictBestState)
    objModel.eval()
    return objModel


def score_patients(objModel, lstPids, dictPatientData):
    objLoader = DataLoader(PatientDataset(lstPids, dictPatientData),
                           batch_size=INT_BATCH_SIZE, shuffle=False,
                           collate_fn=collate_patient_sets)
    lstProb, lstTrue, lstEmbed, lstWeights = [], [], [], []
    with torch.no_grad():
        for tsrX, tsrY, tsrMask, lstLengths in objLoader:
            tsrX, tsrMask = tsrX.to(strDevice), tsrMask.to(strDevice)
            tsrLogits, tsrEmbeds, tsrWeights = objModel(tsrX, tsrMask, boolReturnEmbeds=True)
            lstProb.extend(torch.softmax(tsrLogits, dim=1)[:, 1].cpu().numpy())
            lstTrue.extend(tsrY.numpy())
            lstEmbed.extend(tsrEmbeds.cpu().numpy())
            for intRow, intLength in enumerate(lstLengths):
                lstWeights.append(tsrWeights[intRow, :intLength].cpu().numpy())
    return np.array(lstProb), np.array(lstTrue), np.array(lstEmbed), lstWeights


def plot_patient_tsne(arrEmbeds, arrLabels, lstPids, pathOut, strTitle):
    intPerplexity = max(5, min(30, (len(arrEmbeds) - 1) // 3))
    objTsne = TSNE(n_components=2, random_state=INT_GLOBAL_SEED,
                   perplexity=intPerplexity, init="pca", learning_rate="auto")
    arrTsne = objTsne.fit_transform(arrEmbeds)

    objFig, objAx = plt.subplots(figsize=(7.0, 5.8))
    for intLabel, strColour, strName in [(0, "#2B6A8F", "Diffuse goitre"),
                                         (1, "#B02418", "Multinodular goitre")]:
        arrSel = arrLabels == intLabel
        objAx.scatter(arrTsne[arrSel, 0], arrTsne[arrSel, 1], c=strColour,
                      s=34, alpha=0.85, edgecolors="black", linewidths=0.4,
                      label=f"{strName} (n={int(arrSel.sum())})")
    objAx.set_xlabel("t-SNE dimension 1")
    objAx.set_ylabel("t-SNE dimension 2")
    objAx.set_title(strTitle, loc="left", fontsize=10)
    objAx.legend(frameon=False, fontsize=8, loc="best")
    objAx.spines[["top", "right"]].set_visible(False)
    objFig.tight_layout()
    save_figure(objFig, Path(pathOut).with_suffix(""))
    plt.close(objFig)


def main():
    sys.stdout = TeeLogger(pathPhase4Dir / "phase4_run.log")
    apply_manuscript_style()
    dfTable = load_balanced_table()
    arrLabels = get_labels(dfTable)
    dictMeta, dictFolds = load_splits()
    assert_folds_patient_disjoint(dfTable, dictFolds)

    arrClinical, lstClinicalNames = build_clinical_matrix(dfTable)
    dfIndexed = dfTable.reset_index(drop=True)

    dfPatientClinical = (pd.DataFrame(arrClinical, columns=lstClinicalNames)
                         .assign(patient_id=dfIndexed["patient_id"].astype(str).values)
                         .groupby("patient_id").first())

    lstPerFoldRows, lstAttentionRows = [], []
    dictRepeatEmbeds = {}

    lstEncodersFound = available_encoders(LST_ROI_MODES)
    if not lstEncodersFound:
        raise FileNotFoundError(
            "No Phase 2 feature files available for any encoder. "
            "Run 'python phases/phase2_features_baseline.py' first.")
    print(f"Encoders available: {', '.join(lstEncodersFound)}")

    lstJobs = [(e, m) for e in lstEncodersFound for m in LST_ROI_MODES]
    for strEncoder, strRoiMode in lstJobs:
        boolIsReference = (strEncoder == STR_REFERENCE_ENCODER)
        try:
            arrCls, _ = load_features(strRoiMode, dfTable, boolLoadPatch=False,
                                      strEncoder=strEncoder)
        except (FileNotFoundError, KeyError) as objError:
            print(f"Skipping {strEncoder}/{strRoiMode}: {objError}")
            continue

        strPrefix = "" if boolIsReference else f"{strEncoder} | "

        dictFusionForThisJob = (DICT_FUSION_MODELS if boolIsReference else {})

        dictPatientData = build_patient_records(dfTable, arrCls, arrLabels)
        intEmbedDim = arrCls.shape[1]
        print(f"\n{strRoiMode.upper()} ROI: {len(dictPatientData)} patients, "
              f"{np.mean([len(v['features']) for v in dictPatientData.values()]):.2f} "
              f"projections/patient on average.")

        dictSingleView = {}
        for strPid, dictEntry in dictPatientData.items():
            lstIdx = [i for i, v in enumerate(dictEntry["views"]) if v == STR_REFERENCE_VIEW]
            if not lstIdx:
                lstIdx = [0]
            dictSingleView[strPid] = {
                "features": [dictEntry["features"][lstIdx[0]]],
                "views": [dictEntry["views"][lstIdx[0]]],
                "label": dictEntry["label"]}

        for strFoldName, dictFold in tqdm(sorted(dictFolds.items()),
                                          desc=f"{strEncoder}/{strRoiMode}"):
            intRepeat, intFold = parse_fold_name(strFoldName)
            lstInnerFolds = get_inner_folds(dictFold)
            lstTrainPids = patients_of(dfTable, lstInnerFolds[0]["inner_train"])
            lstCalPids = patients_of(dfTable, lstInnerFolds[0]["inner_cal"])
            lstValPids = patients_of(dfTable, dictFold["val"])

            dictFoldProbas = {}

            for strFusionName, objModelClass in dictFusionForThisJob.items():
                objModel = train_fusion_model(objModelClass, lstTrainPids, lstCalPids,
                                              dictPatientData, intEmbedDim)
                arrCalProb, arrCalTrue, _, _ = score_patients(objModel, lstCalPids, dictPatientData)
                arrValProb, arrValTrue, arrValEmbed, lstValWeights = score_patients(
                    objModel, lstValPids, dictPatientData)
                dictFoldProbas[f"{strPrefix}{strRoiMode} ROI | {strFusionName}"] = (
                    arrCalProb, arrCalTrue, arrValProb, arrValTrue)

                if strFusionName == "Gated attention pooling":
                    for strPid, arrWeights in zip(lstValPids, lstValWeights):
                        for strView, floatWeight in zip(dictPatientData[strPid]["views"],
                                                        arrWeights):
                            lstAttentionRows.append({
                                "encoder": strEncoder,
                                "roi": strRoiMode, "fold": strFoldName, "repeat": intRepeat,
                                "patient_id": strPid, "view": strView,
                                "attention_weight": float(floatWeight),
                                "label": dictPatientData[strPid]["label"]})
                    dictRepeatEmbeds.setdefault((strRoiMode, intRepeat), []).append(
                        (arrValEmbed, arrValTrue, lstValPids))

            objSingleModel = train_fusion_model(MeanPoolFusion, lstTrainPids, lstCalPids,
                                                dictSingleView, intEmbedDim)
            arrCalProb, arrCalTrue, _, _ = score_patients(objSingleModel, lstCalPids, dictSingleView)
            arrValProb, arrValTrue, _, _ = score_patients(objSingleModel, lstValPids, dictSingleView)
            dictFoldProbas[f"{strPrefix}{strRoiMode} ROI | Single {STR_REFERENCE_VIEW} projection"] = (
                arrCalProb, arrCalTrue, arrValProb, arrValTrue)

            dictThresholdPairs = {}
            for intPartition, dictInner in enumerate(lstInnerFolds):
                if intPartition >= INT_THRESHOLD_PARTITIONS:
                    break
                if intPartition == 0:
                    for strConfig, (arrCalProb, arrCalTrue, _, _) in dictFoldProbas.items():
                        dictThresholdPairs.setdefault(strConfig, []).append(
                            (arrCalTrue, arrCalProb))
                    continue

                lstPartTrainPids = patients_of(dfTable, dictInner["inner_train"])
                lstPartCalPids = patients_of(dfTable, dictInner["inner_cal"])

                for strFusionName, objModelClass in DICT_FUSION_MODELS.items():
                    objPartModel = train_fusion_model(
                        objModelClass, lstPartTrainPids, lstPartCalPids,
                        dictPatientData, intEmbedDim)
                    arrPartProb, arrPartTrue, _, _ = score_patients(
                        objPartModel, lstPartCalPids, dictPatientData)
                    dictThresholdPairs.setdefault(
                        f"{strRoiMode} ROI | {strFusionName}", []).append(
                            (arrPartTrue, arrPartProb))

                objPartSingle = train_fusion_model(
                    MeanPoolFusion, lstPartTrainPids, lstPartCalPids,
                    dictSingleView, intEmbedDim)
                arrPartProb, arrPartTrue, _, _ = score_patients(
                    objPartSingle, lstPartCalPids, dictSingleView)
                dictThresholdPairs.setdefault(
                    f"{strRoiMode} ROI | Single {STR_REFERENCE_VIEW} projection",
                    []).append((arrPartTrue, arrPartProb))

            for strConfig, (arrCalProb, arrCalTrue, arrValProb, arrValTrue) in dictFoldProbas.items():
                lstPairs = dictThresholdPairs.get(strConfig, [(arrCalTrue, arrCalProb)])
                floatThreshold, lstPartitionThresholds = select_threshold_averaged(
                    lstPairs, STR_THRESHOLD_CRITERION)
                floatThresholdAlt, _ = select_threshold_averaged(
                    lstPairs,
                    "youden" if STR_THRESHOLD_CRITERION != "youden" else "equal_error")

                dictMetrics = calculate_full_metrics(arrValTrue, arrValProb, floatThreshold)
                lstPerFoldRows.append({
                    "config": strConfig, "roi": strRoiMode, "fold": strFoldName,
                    "repeat": intRepeat, "outer_fold": intFold,
                    "threshold": floatThreshold,
                    "threshold_criterion": STR_THRESHOLD_CRITERION,
                    "threshold_alternative": floatThresholdAlt,
                    "n_threshold_partitions": len(lstPartitionThresholds),
                    "threshold_sd_across_partitions": (
                        float(np.std(lstPartitionThresholds, ddof=1))
                        if len(lstPartitionThresholds) > 1 else 0.0),
                    "n_val_patients": len(lstValPids),
                    **dictMetrics})

    if not lstPerFoldRows:
        raise FileNotFoundError("No Phase 2 feature files available.")

    dfPerFold = pd.DataFrame(lstPerFoldRows)
    dfPerFold.to_csv(pathPhase4Dir / "phase4_per_fold_metrics.csv", index=False)

    dfSummary = summarise_across_repeats(dfPerFold, ["config"])
    dfSummary = dfSummary.sort_values("AUC_mean", ascending=False).reset_index(drop=True)
    lstReportCols = ["config", "n_repeats"] + LST_METRIC_KEYS
    dfSummary[lstReportCols].to_csv(pathPhase4Dir / "phase4_metrics_table.csv", index=False)

    dfAnnotated = dfSummary[lstReportCols].copy()
    dfFactors = pd.DataFrame([parse_config_label(c) for c in dfAnnotated["config"]])
    dfAnnotated.insert(1, "encoder", dfFactors["encoder"].values)
    dfAnnotated.insert(2, "roi", dfFactors["roi"].values)
    dfAnnotated.insert(3, "aggregation", dfFactors["head"].values)
    export_manuscript_table(
        dfAnnotated,
        pathPhase4Dir / "phase4_metrics_table_annotated",
        strCaption="Patient-level performance across encoders, ROI strategies "
                   "and multi-projection aggregation operators. Mean +/- standard "
                   "deviation over repeat-level means.",
        strLabel="tab:phase4-main")
    print("\n" + dfSummary[["config"] + LST_METRIC_KEYS[:4]].to_string(index=False))

    strSingle = f"fixed ROI | Single {STR_REFERENCE_VIEW} projection"
    lstContrasts = [
        (strSingle, "fixed ROI | Gated attention pooling", "aggregation"),
        (strSingle, "fixed ROI | Mean pooling", "aggregation"),
        (strSingle, "fixed ROI | Max pooling", "aggregation"),
        ("fixed ROI | Gated attention pooling", "fixed ROI | Mean pooling", "aggregation"),
        ("fixed ROI | Gated attention pooling", "fixed ROI | Max pooling", "aggregation"),
        (strSingle, f"dynamic ROI | Single {STR_REFERENCE_VIEW} projection", "roi"),
        ("fixed ROI | Gated attention pooling",
         "dynamic ROI | Gated attention pooling", "roi"),
    ]
    for strEncoder in lstEncodersFound:
        if strEncoder == STR_REFERENCE_ENCODER:
            continue
        strEncSingleFixed = (f"{strEncoder} | fixed ROI | "
                             f"Single {STR_REFERENCE_VIEW} projection")
        strEncSingleDyn = (f"{strEncoder} | dynamic ROI | "
                           f"Single {STR_REFERENCE_VIEW} projection")
        lstContrasts.append((strSingle, strEncSingleFixed, "encoder"))
        lstContrasts.append((strEncSingleFixed, strEncSingleDyn, "roi"))

    lstComparisonRows = []
    for strMetric in ["AUC", "PR-AUC", "MCC", "Balanced Accuracy"]:
        for strModelA, strModelB, strFamily in lstContrasts:
            dictCmp = compare_configs(dfPerFold, strMetric, strModelA, strModelB)
            if dictCmp is not None:
                dictCmp["family"] = f"{strFamily}|{strMetric}"
                lstComparisonRows.append(dictCmp)

    if lstComparisonRows:
        dfComparisons = add_multiplicity_correction(pd.DataFrame(lstComparisonRows))
        dfComparisons.to_csv(pathPhase4Dir / "phase4_statistical_comparisons.csv",
                             index=False)
        export_manuscript_table(
            dfComparisons,
            pathPhase4Dir / "phase4_statistical_comparisons",
            strCaption="Paired contrasts at the patient level, corrected within "
                       "family by Benjamini-Hochberg.",
            strLabel="tab:phase4-stats")
        print(f"\nPaired comparisons: {len(dfComparisons)} contrasts, "
              "Benjamini-Hochberg corrected within family|metric.")
        for dictRow in dfComparisons[dfComparisons["metric"] == "AUC"].to_dict("records"):
            print("  [{}] {:<46s} vs {:<46s} diff={:+.4f} p={:.4g} p_adj={:.4g}".format(
                "SIG" if dictRow["significant_adjusted"] else "ns ",
                dictRow["model_a"][:46], dictRow["model_b"][:46],
                dictRow["mean_difference"], dictRow["p_repeat_level_ttest"],
                dictRow["p_adjusted_bh"]))

    dfAttention = pd.DataFrame(lstAttentionRows)
    if not dfAttention.empty:
        dfAttention.to_csv(pathPhase4Dir / "phase4_attention_weights.csv", index=False)
        dfViewWeights = (dfAttention.groupby(["roi", "view"])["attention_weight"]
                         .agg(["mean", "std", "count"]).reset_index())
        dfViewWeights.to_csv(pathPhase4Dir / "phase4_view_importance.csv", index=False)

        objFig, objAx = plt.subplots(figsize=(6.4, 4.2))
        for intIdx, strRoiMode in enumerate(sorted(dfAttention["roi"].unique())):
            dfSub = dfViewWeights[dfViewWeights["roi"] == strRoiMode]
            arrX = np.arange(len(dfSub)) + (intIdx - 0.5) * 0.34
            objAx.bar(arrX, dfSub["mean"], 0.32, yerr=dfSub["std"], capsize=3,
                      label=f"{strRoiMode} ROI", edgecolor="black", linewidth=0.5,
                      color=["#2B6A8F", "#B02418"][intIdx % 2])
        objAx.axhline(1.0 / 3.0, color="#8A8A8A", ls=":", lw=1.0)
        objAx.annotate("uniform weight (1/3)", xy=(0.02, 1.0 / 3.0), xycoords=("axes fraction", "data"),
                       xytext=(4, 5), textcoords="offset points", fontsize=8, color="#8A8A8A")
        objAx.set_xticks(np.arange(len(dfViewWeights["view"].unique())))
        objAx.set_xticklabels(sorted(dfViewWeights["view"].unique()))
        objAx.set_xlabel("Scintigraphic projection")
        objAx.set_ylabel("Mean learned attention weight")
        objAx.set_title("Which projection does the fusion model rely on?", loc="left", fontsize=10)
        objAx.legend(frameon=False, fontsize=8)
        objAx.spines[["top", "right"]].set_visible(False)
        objFig.tight_layout()

        save_figure(objFig, pathPhase4Dir / "phase4_view_importance")
        plt.close(objFig)

    if dictRepeatEmbeds:
        tupKey = sorted(dictRepeatEmbeds.keys())[0]
        lstParts = dictRepeatEmbeds[tupKey]
        arrEmbeds = np.concatenate([p[0] for p in lstParts])
        arrEmbedLabels = np.concatenate([p[1] for p in lstParts])
        lstPids = [pid for p in lstParts for pid in p[2]]
        plot_patient_tsne(
            arrEmbeds, arrEmbedLabels, lstPids,
            pathPhase4Dir / "phase4_tsne_patient_space.png",
            f"Out-of-fold patient embeddings, gated attention fusion\n"
            f"({tupKey[0]} ROI, repeat {tupKey[1]}: each patient plotted once)")

    print(f"\nPhase 4 complete. Artifacts written to {pathPhase4Dir}")


if __name__ == "__main__":
    main()
