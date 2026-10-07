"""
Phase 4: Patient-Level Fusion Benchmarking
==========================================

A patient contributes up to three scintigraphic projections (ANTERIOR, LAO, RAO).
This phase aggregates the per-projection embeddings into one representation per
patient and issues a single diagnosis, which is the clinically meaningful unit of
prediction.

Validation protocol
-------------------
Identical to Phase 3 and defined on the same folds: fit on 'inner_train',
calibrate the decision threshold on 'inner_cal', report on the outer 'val'
fold. Patient disjointness is asserted before training. Since the unit of
prediction is the patient, each fold's patient sets are derived from the fold's
sample lists and are disjoint by construction.

Fusion operators benchmarked
----------------------------
1. Mean pooling: the unweighted average of a patient's projection embeddings.
   This is the baseline every other operator must beat.
2. Gated attention pooling: learns a per-projection weight (Ilse et al., 2018,
   attention-based deep multiple-instance learning), so the model can
   down-weight an uninformative projection. The learned weights are exported
   per projection type, which is a clinically interpretable output.
3. Max pooling: element-wise maximum, a permutation-invariant alternative that
   rewards the single most discriminative projection.
4. Best single projection: the ANTERIOR view alone, which is the operationally
   relevant reference — if multi-view fusion does not beat one view, acquiring
   and processing three views is not justified.

A graph-convolution arm present in an earlier revision has been removed. With a
fully connected adjacency and uniform degree normalisation, every node receives
the identical aggregated vector, so its graph embedding reduces exactly to
relu(W · mean(x)): algebraically it is the mean-pooling baseline with one extra
linear layer, not an independent method, and benchmarking it against mean
pooling as if it were a distinct architecture is not defensible.

WHAT CHANGED IN THIS REVISION (and why it should improve the metrics)
--------------------------------------------------------------------
1. CROSS-FITTED DECISION THRESHOLD. This phase's threshold was selected on a
   single calibration block of 28-31 patients, which gave it sd ~0.16 and a range
   of 0.65 across the 25 folds; in 3 folds specificity saturated at 1.0 and in 3
   others sensitivity fell below 0.60. That is the direct cause of the ~0.10 gap
   between AUC (0.9137) and balanced accuracy (0.8118): the ranking is good and
   part of the advantage is then lost at the operating point. The threshold is
   now aggregated over up to INT_THRESHOLD_PARTITIONS inner partitions.

   Each extra partition REFITS the fusion operators on its own training patients.
   Reusing the partition-0 model to score another partition's calibration block
   would be scoring patients that model was trained on, i.e. exactly the leak
   this pipeline exists to prevent. The reported model is still the partition-0
   model; only the threshold is aggregated (standard cross-fitted calibration).

2. STATISTICAL CONTRASTS ARE NOW GENERATED HERE. This phase produces the paper's
   headline and its central claim is a comparison between aggregation
   strategies, yet it exported no test: the claim was unsupported anywhere in the
   repository and a reviewer running the code could not reproduce it. Contrasts
   are written to phase4_statistical_comparisons.csv with Benjamini-Hochberg
   adjusted p-values, computed within family (aggregation vs ROI) and per metric.

   All four metrics are reported deliberately, because they disagree: the
   single-projection advantage is significant in AUC and PR-AUC but not in MCC.

Neither change is guaranteed to raise a number; both remove a defect a reviewer
would otherwise find.

READ THIS BEFORE COMPARING WITH PHASE 3
---------------------------------------
Phase 3 reports AUC ~0.81 and this phase ~0.91. That difference is NOT a
methodological improvement: Phase 3's unit of analysis is the image (74-88 per
validation fold) and this phase's is the patient (28-30 per fold). Both consume
the identical feature files. Aggregating three projections of one thyroid into a
single decision averages acquisition noise and yields an intrinsically easier
task. Any manuscript sentence presenting the jump as the effect of a technique
would be wrong.

Execution Instructions:
-----------------------
Requires Phase 1 (folds) and Phase 2 (features). Produces the headline result
of the study.

   $ conda activate thyroid_env
   $ python phases/phase4_patient_level_fusion.py

Runtime: roughly 2.5 hours (every encoder on the single-projection arm, four
fusion operators on the reference encoder, both ROI tracks, 10 repeats, with the
decision threshold cross-fitted over four inner partitions).
"""

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

# Both ROI tracks are always evaluated. The fixed-vs-dynamic comparison is a
# result this paper must establish, not an assumption carried over from earlier
# runs: a reviewer can only reproduce it if the experiment is in the pipeline.
LST_ROI_MODES = ["fixed", "dynamic"]

# Number of inner partitions used to average the decision threshold. Each one
# beyond the first REFITS every fusion operator on its own training patients, so
# this constant multiplies the run time of this phase almost linearly. Set to 1
# to reproduce the previous single-block behaviour; 4 uses everything Phase 1
# now materialises.
INT_THRESHOLD_PARTITIONS = 4

# Threshold criterion, kept identical to Phase 3 so the two phases are
# methodologically consistent. Measured on the Phase 3 folds, 'equal_error' gives
# a more stable threshold but slightly worse threshold-dependent metrics (none of
# the differences significant), so Youden's J stays primary and the alternative
# is recorded per fold in 'threshold_alternative'.
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
    """
    Common classifier head. Subclasses implement pool() only.

    The head is intentionally small (3072 -> 64 -> 2): the patient-level cohort
    is even smaller than the image-level one (148 subjects total).
    """

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
    """Unweighted average over a patient's valid projections."""

    def pool(self, tsrFeatures, tsrMask):
        tsrValid = (~tsrMask).unsqueeze(-1).float()
        tsrPooled = (tsrFeatures * tsrValid).sum(dim=1) / tsrValid.sum(dim=1).clamp(min=1.0)
        return tsrPooled, tsrValid.squeeze(-1) / tsrValid.sum(dim=1).clamp(min=1.0)


class MaxPoolFusion(BaseFusion):
    """Element-wise maximum over a patient's valid projections."""

    def pool(self, tsrFeatures, tsrMask):
        tsrMasked = tsrFeatures.masked_fill(tsrMask.unsqueeze(-1), float("-inf"))
        tsrPooled = tsrMasked.max(dim=1).values
        tsrPooled = torch.nan_to_num(tsrPooled, neginf=0.0)
        return tsrPooled, torch.zeros(tsrFeatures.shape[:2], device=tsrFeatures.device)


class GatedAttentionFusion(BaseFusion):
    """
    Gated attention pooling over the projection set (Ilse et al., 2018).

    Each projection receives a scalar weight from a two-branch gated network;
    weights are normalised over the patient's valid projections only. The
    weights are the clinically interpretable output of this arm: they say which
    view the model relied on for a given patient.

    The previous revision used nn.MultiheadAttention followed by an unweighted
    mean over the attended tokens. That configuration lets projections exchange
    information but then averages them uniformly, so it produces no
    per-projection importance and its aggregation step is still a mean.
    """

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
    """
    Groups projection embeddings by patient.

    Row position in arrCls is safe here only because common.load_features() has
    already reordered the tensor to dfTable's row order using the stored
    'sample_ids' key. The view label of each projection is retained so that
    learned attention weights can be reported per projection type.
    """
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
    """Patient identifiers represented by a list of sample keys."""
    srSelected = dfTable.set_index("sample_id").loc[lstSampleIds, "patient_id"]
    return sorted(set(srSelected.astype(str)))


def train_fusion_model(objModelClass, lstTrainPids, lstCalPids, dictPatientData,
                       intEmbedDim):
    """
    Fits one fusion operator on the inner training patients with early stopping
    on the calibration patients. Returns the trained model.
    """
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
    """Returns positive-class probability, label, embedding and attention weight per patient."""
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
    """
    t-SNE of one repeat's out-of-fold patient embeddings.

    Each patient appears exactly once. The previous revision pooled embeddings
    from all 25 folds, plotting every patient five times, which inflates the
    apparent point count fivefold and makes visual separation look denser than
    the cohort supports.
    """
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
    # PDF as well as PNG: see the note on phase4_view_importance. A scatter plot
    # is line artwork, for which the journal's pixel floor is high and vector
    # output sidesteps it entirely.
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

    # Patient-level clinical covariates: one row per patient (constant within patient).
    dfPatientClinical = (pd.DataFrame(arrClinical, columns=lstClinicalNames)
                         .assign(patient_id=dfIndexed["patient_id"].astype(str).values)
                         .groupby("patient_id").first())

    lstPerFoldRows, lstAttentionRows = [], []
    dictRepeatEmbeds = {}

    # Staged grid, mirroring Phase 3: every encoder is evaluated with the
    # single-projection reference arm (which is the study's headline arm), while
    # the four fusion strategies are compared only on the reference encoder.
    # Running four fusion networks for five encoders would multiply a three-hour
    # phase by five to answer a question nobody asked.
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

        # Label convention: the reference encoder keeps the short two-part
        # labels used by the existing reports and comparison code; other
        # encoders are prefixed. common.parse_config_label() reads both.
        strPrefix = "" if boolIsReference else f"{strEncoder} | "

        # Fusion strategies are compared on the reference encoder only; every
        # other encoder contributes the single-projection arm, which is the one
        # the encoder axis is about.
        dictFusionForThisJob = (DICT_FUSION_MODELS if boolIsReference else {})

        dictPatientData = build_patient_records(dfTable, arrCls, arrLabels)
        intEmbedDim = arrCls.shape[1]
        print(f"\n{strRoiMode.upper()} ROI: {len(dictPatientData)} patients, "
              f"{np.mean([len(v['features']) for v in dictPatientData.values()]):.2f} "
              f"projections/patient on average.")

        # Single-projection reference set: the ANTERIOR view alone.
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
            # All inner partitions of this outer fold; the first one trains the
            # reported model, the rest contribute to threshold averaging.
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

            # Single-view reference, trained with the same operator and folds.
            objSingleModel = train_fusion_model(MeanPoolFusion, lstTrainPids, lstCalPids,
                                                dictSingleView, intEmbedDim)
            arrCalProb, arrCalTrue, _, _ = score_patients(objSingleModel, lstCalPids, dictSingleView)
            arrValProb, arrValTrue, _, _ = score_patients(objSingleModel, lstValPids, dictSingleView)
            dictFoldProbas[f"{strPrefix}{strRoiMode} ROI | Single {STR_REFERENCE_VIEW} projection"] = (
                arrCalProb, arrCalTrue, arrValProb, arrValTrue)

            # --- Cross-fitted decision threshold -------------------------
            # The threshold is the weakest link of this phase: it was selected on
            # a single calibration block of 28-31 patients, which gave it a
            # standard deviation of ~0.16 and a range of 0.65 across folds. In 3
            # of 25 folds specificity saturated at 1.0 and in 3 others
            # sensitivity fell below 0.60, which is what opens the ~0.10 gap
            # between AUC (0.9137) and balanced accuracy (0.8118).
            #
            # Each additional inner partition requires REFITTING the fusion
            # operator on its own training patients: reusing the partition-0
            # model to score another partition's calibration block would be
            # scoring patients the model was trained on, which is exactly the
            # leak this pipeline exists to avoid. The models are small, so the
            # cost is a multiplier on a fast operation, but it IS a multiplier of
            # about INT_THRESHOLD_PARTITIONS on this phase's run time.
            #
            # The reported model remains the partition-0 model; only the
            # threshold is aggregated. This is standard cross-fitted calibration.
            dictThresholdPairs = {}
            for intPartition, dictInner in enumerate(lstInnerFolds):
                if intPartition >= INT_THRESHOLD_PARTITIONS:
                    break
                if intPartition == 0:
                    # Already computed above; reuse instead of refitting.
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

    # Manuscript table annotated with the experimental factors, so rows can be
    # grouped by encoder / ROI / aggregation strategy without parsing labels.
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

    # --- Paired statistical comparisons ---------------------------------
    # This phase produces the headline of the paper (patient-level AUC) and its
    # central claim is a COMPARISON between aggregation strategies, yet it
    # shipped no test: the claim was unsupported anywhere in the repository and
    # a reviewer running the code could not reproduce it. Phase 3 already
    # exported its contrasts; this closes the gap.
    #
    # All four metrics are evaluated deliberately. Ranking metrics (AUC, PR-AUC)
    # and threshold-dependent ones (MCC, balanced accuracy) need not agree, and
    # here they do not: the single-projection advantage is significant in AUC and
    # PR-AUC but not in MCC. Reporting all four prevents claiming a global
    # superiority that only holds for discrimination.
    #
    # Families for multiplicity correction: 'aggregation' compares fusion
    # operators against each other, 'roi' compares the two ROI definitions. They
    # are separate questions and are corrected separately.
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

    # --- Encoder axis, at patient level ---------------------------------
    # Same staged logic as Phase 3: every encoder is contrasted against the
    # reference on the single-projection arm, and each encoder's own ROI
    # contrast is added so the fixed-vs-dynamic result is ESTABLISHED here too
    # rather than assumed from earlier runs.
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

    # --- Attention weights per projection type --------------------------
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
        # save_figure writes PNG and PDF. The PDF matters for submission: the
        # journal requires bitmapped line drawings to be at least 3543 pixels
        # wide at single-column size, which a 300 dpi raster of this figure does
        # not reach. Vector artwork has no resolution requirement.
        save_figure(objFig, pathPhase4Dir / "phase4_view_importance")
        plt.close(objFig)

    # --- t-SNE of one repeat's out-of-fold patient embeddings -----------
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
