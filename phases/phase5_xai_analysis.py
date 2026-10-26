import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (INT_GLOBAL_SEED, add_multiplicity_correction,
                    apply_manuscript_style, export_manuscript_table, get_labels,
                    get_torch_device, load_balanced_table, load_features,
                    load_splits, get_fold_row_positions, pathPhase5Dir,
                    save_figure, set_global_seed, TeeLogger)

pathDataDir = Path("data")
pathPhase5Dir.mkdir(parents=True, exist_ok=True)

set_global_seed(INT_GLOBAL_SEED)
strDevice = get_torch_device()

torch.set_grad_enabled(True)

INT_NUM_QUANTITATIVE = 100
FLOAT_TOP_FRACTION = 0.20
LST_PERTURBATION_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]
INT_BLUR_KERNEL = 51
FLOAT_BLUR_SIGMA = 15.0
STR_PRIMARY_SALIENCY = "rollout"
INT_PCA_COMPONENTS = 100
TUP_CLINICAL_ROI_WIDE = (0.20, 0.85, 0.15, 0.85)
TUP_CLINICAL_ROI_TIGHT = (0.28, 0.68, 0.28, 0.72)
TUP_CLINICAL_ROI = TUP_CLINICAL_ROI_TIGHT
STR_SALIENCY_METHOD = "class_gradient"
STR_EXPLAINED_ROI_MODE = "fixed"

def resize_with_padding(arrImage, tupleTargetSize=(224, 224)):
    intHeight, intWidth = arrImage.shape[:2]
    intTargetWidth, intTargetHeight = tupleTargetSize
    floatScale = min(intTargetWidth / intWidth, intTargetHeight / intHeight)
    
    intNewWidth = max(1, int(intWidth * floatScale))
    intNewHeight = max(1, int(intHeight * floatScale))
    arrResized = cv2.resize(arrImage, (intNewWidth, intNewHeight), interpolation=cv2.INTER_AREA)
    
    intPadWidth = intTargetWidth - intNewWidth
    intPadHeight = intTargetHeight - intNewHeight
    intTop = intPadHeight // 2
    intBottom = intPadHeight - intTop
    intLeft = intPadWidth // 2
    intRight = intPadWidth - intLeft
    
    return cv2.copyMakeBorder(arrResized, intTop, intBottom, intLeft, intRight, cv2.BORDER_CONSTANT, value=0)

def extract_attention_rollout(objModel, tsrImg):
    lstAttentions = []
    with torch.no_grad():
        x = objModel.prepare_tokens_with_masks(tsrImg)
        for blk in objModel.blocks:
            x_norm = blk.norm1(x)
            B, N, C = x_norm.shape
            num_heads = blk.attn.num_heads
            head_dim = C // num_heads
            qkv = blk.attn.qkv(x_norm).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
            attn = torch.softmax(attn, dim=-1)
            
            avg_attn = attn[0].mean(dim=0).cpu().numpy()
            lstAttentions.append(avg_attn)
            x = blk(x)
            
    arrRollout = np.eye(lstAttentions[0].shape[0])
    for arrAttn in lstAttentions:
        arrAttn = arrAttn + np.eye(arrAttn.shape[0])
        arrAttn = arrAttn / arrAttn.sum(axis=-1, keepdims=True)
        arrRollout = np.matmul(arrAttn, arrRollout)
        
    arrClsRollout = arrRollout[0, 1:]
    arrGrid = arrClsRollout.reshape(16, 16)
    arrGrid = (arrGrid - arrGrid.min()) / (arrGrid.max() - arrGrid.min() + 1e-8)
    arrHeatmap = cv2.resize(arrGrid, (224, 224), interpolation=cv2.INTER_CUBIC)
    arrHeatmap = np.clip(arrHeatmap, 0.0, 1.0)
    return arrHeatmap

def extract_class_gradient_saliency(objModel, tsrImg, objClassifier,
                                    intTargetClass=1):
    objModel.zero_grad(set_to_none=True)
    lstCaptured = []
    lstHandles = []
    lstBlocks = list(objModel.blocks)

    def _capture(objMod, objIn, objOut):
        tsrOut = objOut[0] if isinstance(objOut, (tuple, list)) else objOut
        tsrOut.retain_grad()
        lstCaptured.append(tsrOut)

    for objBlock in lstBlocks[-4:-1]:
        lstHandles.append(objBlock.register_forward_hook(_capture))

    try:
        with torch.enable_grad():
            lstInter = objModel.get_intermediate_layers(
                tsrImg, n=4, return_class_token=True)
            lstClsTokens = [objTuple[1] for objTuple in lstInter]
            tsrConcatCls = torch.cat(lstClsTokens, dim=1)      # [1, 3072]
    finally:
        for objHandle in lstHandles:
            objHandle.remove()

    lstPatchTokens = [tsrTokens[:, 1:] for tsrTokens in lstCaptured]
    arrCls = tsrConcatCls.detach().cpu().numpy()
    arrWeights = _classifier_linear_weights(objClassifier, arrCls, intTargetClass)
    tsrWeights = torch.from_numpy(arrWeights).float().to(tsrConcatCls.device)

    objScore = (tsrConcatCls * tsrWeights).sum()
    objScore.backward()

    lstMaps = []
    for intIdx, tsrTokens in enumerate(lstCaptured):
        if tsrTokens.grad is None:
            continue
        tsrGrad = tsrTokens.grad[0, 1:]                       # [N, 768]
        tsrAct = tsrTokens[0, 1:].detach()                    # [N, 768]
        if float(tsrGrad.abs().sum()) == 0.0:
            continue
        tsrChannelWeight = tsrGrad.mean(dim=0, keepdim=True)  # [1, 768]
        tsrMap = torch.relu((tsrAct * tsrChannelWeight).sum(dim=1))  # [N]
        lstMaps.append(tsrMap.cpu().numpy())

    objModel.zero_grad(set_to_none=True)

    if not lstMaps:
        raise RuntimeError(
            "Class-sensitive saliency produced no gradients on any captured "
            "block. This means the score is not connected to the patch tokens "
            "and the map would silently degrade to a class-agnostic one. "
            "Check that gradients are enabled and that the hooked blocks are "
            "not the terminal block.")

    arrTokens = np.mean(lstMaps, axis=0)
    intGrid = int(round(float(np.sqrt(len(arrTokens)))))
    if intGrid * intGrid != len(arrTokens):
        raise RuntimeError(
            f"{len(arrTokens)} patch tokens do not form a square grid; the "
            "saliency map cannot be reshaped to image space.")

    arrGrid = arrTokens.reshape(intGrid, intGrid)
    arrGrid = (arrGrid - arrGrid.min()) / (arrGrid.max() - arrGrid.min() + 1e-8)
    arrHeatmap = cv2.resize(arrGrid, (224, 224), interpolation=cv2.INTER_CUBIC)
    return np.clip(arrHeatmap, 0.0, 1.0)


def _classifier_linear_weights(objClassifier, arrCls, intTargetClass=1):
    arrCls = np.asarray(arrCls, dtype=float).reshape(1, -1)
    try:
        objFinal = objClassifier[-1] if hasattr(objClassifier, "__getitem__") else objClassifier
        if hasattr(objFinal, "coef_"):
            arrCoef = np.asarray(objFinal.coef_, dtype=float).ravel()
            if intTargetClass == 0:
                arrCoef = -arrCoef
            if hasattr(objClassifier, "named_steps"):
                for strName in list(objClassifier.named_steps)[:-1][::-1]:
                    objStep = objClassifier.named_steps[strName]
                    if hasattr(objStep, "components_"):          # PCA
                        arrComp = np.asarray(objStep.components_, dtype=float)
                        if getattr(objStep, "whiten", False):
                            arrCoef = arrCoef / np.sqrt(objStep.explained_variance_ + 1e-12)
                        arrCoef = arrCoef @ arrComp
                    elif hasattr(objStep, "scale_"):             # StandardScaler
                        arrCoef = arrCoef / np.asarray(objStep.scale_, dtype=float)
            if arrCoef.shape[0] == arrCls.shape[1]:
                return arrCoef
    except Exception:
        pass

    funScore = (objClassifier.decision_function
                if hasattr(objClassifier, "decision_function")
                else lambda a: objClassifier.predict_proba(a)[:, intTargetClass])
    floatEps = 1e-2
    arrGrad = np.zeros(arrCls.shape[1], dtype=float)
    for intDim in range(arrCls.shape[1]):
        arrPlus = arrCls.copy();  arrPlus[0, intDim] += floatEps
        arrMinus = arrCls.copy(); arrMinus[0, intDim] -= floatEps
        arrGrad[intDim] = (float(np.ravel(funScore(arrPlus))[0])
                           - float(np.ravel(funScore(arrMinus))[0])) / (2.0 * floatEps)
    if intTargetClass == 0:
        arrGrad = -arrGrad
    return arrGrad


TSR_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
TSR_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_normalised_tensor(arrRgb):
    tsrImg = torch.from_numpy(np.ascontiguousarray(arrRgb)).float().permute(2, 0, 1) / 255.0
    tsrImg = (tsrImg - TSR_IMAGENET_MEAN) / TSR_IMAGENET_STD
    return tsrImg.unsqueeze(0).to(strDevice)


def get_cls_token(objModel, tsrImg):
    with torch.no_grad():
        tsrImg = tsrImg.squeeze(0)
        tsr1 = tsrImg
        tsr2 = TF.hflip(tsrImg)
        tsr3 = TF.rotate(tsrImg, 5)
        tsr4 = TF.rotate(tsrImg, -5)
        tsr5 = TF.adjust_brightness(tsrImg, 1.1)
        tsrTTA = torch.stack([tsr1, tsr2, tsr3, tsr4, tsr5], dim=0)

        lstInter = objModel.get_intermediate_layers(tsrTTA, n=4, return_class_token=True)
        lst_layer_cls = [layer[1] for layer in lstInter]
        tsrConcatCls = torch.cat(lst_layer_cls, dim=1)
        tsrConcatCls = tsrConcatCls.mean(dim=0, keepdim=True)
        return tsrConcatCls.cpu().numpy()


def compute_anatomical_enrichment(arrHeatmap, tupRoiBox=None):
    tupRoiBox = TUP_CLINICAL_ROI if tupRoiBox is None else tupRoiBox
    intHeight, intWidth = arrHeatmap.shape
    floatY1, floatY2, floatX1, floatX2 = tupRoiBox
    intY1, intY2 = int(intHeight * floatY1), int(intHeight * floatY2)
    intX1, intX2 = int(intWidth * floatX1), int(intWidth * floatX2)

    floatInside = float(arrHeatmap[intY1:intY2, intX1:intX2].sum())
    floatTotal = float(arrHeatmap.sum()) + 1e-8
    floatFraction = floatInside / floatTotal
    floatAreaFraction = ((intY2 - intY1) * (intX2 - intX1)) / float(intHeight * intWidth)
    return floatFraction, floatAreaFraction, floatFraction / floatAreaFraction


def blurred_baseline(arrRgb):
    intKernel = INT_BLUR_KERNEL if INT_BLUR_KERNEL % 2 == 1 else INT_BLUR_KERNEL + 1
    return cv2.GaussianBlur(arrRgb, (intKernel, intKernel), FLOAT_BLUR_SIGMA)


def dispersed_control_mask(arrMaskTop, objRng):
    arrRandomMask = np.zeros(arrMaskTop.size, dtype=bool)
    arrRandomMask[objRng.choice(arrMaskTop.size, int(arrMaskTop.sum()), replace=False)] = True
    return arrRandomMask.reshape(arrMaskTop.shape)


def contiguous_control_mask(arrMaskTop, objRng, intAttempts=24):
    intH, intW = arrMaskTop.shape
    arrRows, arrCols = np.nonzero(arrMaskTop)
    if arrRows.size == 0:
        return arrMaskTop.copy(), 1.0

    intR0, intR1 = int(arrRows.min()), int(arrRows.max())
    intC0, intC1 = int(arrCols.min()), int(arrCols.max())
    lstDyRange = (-intR0, intH - 1 - intR1)
    lstDxRange = (-intC0, intW - 1 - intC1)

    arrBest, floatBestIou, boolBestWraps = None, np.inf, True
    for _ in range(intAttempts):
        boolCanFit = lstDyRange[1] > lstDyRange[0] and lstDxRange[1] > lstDxRange[0]
        if boolCanFit:
            intDy = int(objRng.integers(lstDyRange[0], lstDyRange[1] + 1))
            intDx = int(objRng.integers(lstDxRange[0], lstDxRange[1] + 1))
            boolWraps = False
        else:
            intDy = int(objRng.integers(intH // 4, 3 * intH // 4))
            intDx = int(objRng.integers(intW // 4, 3 * intW // 4))
            boolWraps = True

        arrShifted = np.roll(np.roll(arrMaskTop, intDy, axis=0), intDx, axis=1)
        intInter = int(np.logical_and(arrMaskTop, arrShifted).sum())
        intUnion = int(np.logical_or(arrMaskTop, arrShifted).sum())
        floatIou = intInter / intUnion if intUnion else 1.0

        boolBetter = ((not boolWraps and boolBestWraps)
                      or (boolWraps == boolBestWraps and floatIou < floatBestIou))
        if boolBetter:
            arrBest, floatBestIou, boolBestWraps = arrShifted, floatIou, boolWraps
    return arrBest, floatBestIou


def build_perturbations(arrRgb, arrHeatmap, objRng, floatFraction=None):
    floatFraction = FLOAT_TOP_FRACTION if floatFraction is None else floatFraction
    floatThreshold = np.percentile(arrHeatmap, 100.0 * (1.0 - floatFraction))
    arrMaskTop = arrHeatmap >= floatThreshold

    arrContiguous, floatControlIou = contiguous_control_mask(arrMaskTop, objRng)
    arrDispersed = dispersed_control_mask(arrMaskTop, objRng)
    arrBaseline = blurred_baseline(arrRgb)

    dictVariants = {}
    for strName, arrMask in [("attention", arrMaskTop),
                             ("contiguous", arrContiguous),
                             ("dispersed", arrDispersed)]:
        arrDeleted = arrRgb.copy()
        arrDeleted[arrMask] = arrBaseline[arrMask]
        arrInserted = arrBaseline.copy()
        arrInserted[arrMask] = arrRgb[arrMask]
        dictVariants[f"deletion_{strName}"] = arrDeleted
        dictVariants[f"insertion_{strName}"] = arrInserted
    return dictVariants, arrMaskTop, floatControlIou


def fit_reference_classifier(dfTable, arrLabels, arrCls, dictFolds):
    strFoldName = sorted(dictFolds.keys())[0]
    dictFold = dictFolds[strFoldName]
    arrTrainRows = get_fold_row_positions(dfTable, dictFold["train"])
    arrValRows = get_fold_row_positions(dfTable, dictFold["val"])

    objPipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=INT_PCA_COMPONENTS, whiten=True,
                    random_state=INT_GLOBAL_SEED)),
        ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced",
                           random_state=INT_GLOBAL_SEED))])
    objPipeline.fit(arrCls[arrTrainRows], arrLabels[arrTrainRows])

    objSurrogate = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced",
                                          random_state=INT_GLOBAL_SEED))])
    objSurrogate.fit(arrCls[arrTrainRows], arrLabels[arrTrainRows])

    return objPipeline, strFoldName, arrValRows, objSurrogate


def quantitative_xai_metrics(objModel, arrImage, objClassifier=None, objRng=None,
                             objSaliencyClassifier=None):
    objRng = objRng if objRng is not None else np.random.default_rng(INT_GLOBAL_SEED)
    arrPadded = resize_with_padding(arrImage, (224, 224))
    arrRgb = cv2.cvtColor(arrPadded, cv2.COLOR_GRAY2RGB)
    tsrImg = to_normalised_tensor(arrRgb)

    arrRollout = extract_attention_rollout(objModel, tsrImg)
    objForSaliency = (objSaliencyClassifier if objSaliencyClassifier is not None
                      else objClassifier)
    if STR_SALIENCY_METHOD == "class_gradient" and objForSaliency is not None:
        arrHeatmap = extract_class_gradient_saliency(objModel, tsrImg, objForSaliency)
    else:
        arrHeatmap = arrRollout

    floatFraction, floatAreaFraction, floatEnrichment = compute_anatomical_enrichment(arrHeatmap)
    floatFracWide, floatAreaWide, floatEnrichWide = compute_anatomical_enrichment(
        arrHeatmap, TUP_CLINICAL_ROI_WIDE)
    _, _, floatEnrichRollout = compute_anatomical_enrichment(arrRollout)

    floatMapDivergence = float(np.mean(np.abs(arrHeatmap - arrRollout)))

    dictMetrics = {
        "saliency_method": STR_SALIENCY_METHOD,
        "roi_attention_fraction": floatFraction,
        "roi_area_fraction": floatAreaFraction,
        "anatomical_enrichment": floatEnrichment,
        "roi_area_fraction_wide": floatAreaWide,
        "anatomical_enrichment_wide_box": floatEnrichWide,
        "anatomical_enrichment_rollout": floatEnrichRollout,
        "saliency_vs_rollout_divergence": floatMapDivergence,
    }

    if objClassifier is not None:
        floatBaseProb = float(objClassifier.predict_proba(get_cls_token(objModel, tsrImg))[0, 1])
        dictMetrics["base_probability"] = floatBaseProb

        for strMapName, arrMap in [("rollout", arrRollout), ("gradient", arrHeatmap)]:
            lstDelAttn, lstDelCont, lstDelDisp = [], [], []
            lstInsAttn, lstInsCont, lstInsDisp = [], [], []
            for floatFrac in LST_PERTURBATION_FRACTIONS:
                dictVariants, arrMaskTop, floatControlIou = build_perturbations(
                    arrRgb, arrMap, objRng, floatFrac)
                dictProbs = {}
                for strName, arrVariant in dictVariants.items():
                    arrClsVariant = get_cls_token(objModel, to_normalised_tensor(arrVariant))
                    dictProbs[strName] = float(
                        objClassifier.predict_proba(arrClsVariant)[0, 1])

                lstDelAttn.append(abs(floatBaseProb - dictProbs["deletion_attention"]))
                lstDelCont.append(abs(floatBaseProb - dictProbs["deletion_contiguous"]))
                lstDelDisp.append(abs(floatBaseProb - dictProbs["deletion_dispersed"]))
                lstInsAttn.append(1.0 - abs(floatBaseProb - dictProbs["insertion_attention"]))
                lstInsCont.append(1.0 - abs(floatBaseProb - dictProbs["insertion_contiguous"]))
                lstInsDisp.append(1.0 - abs(floatBaseProb - dictProbs["insertion_dispersed"]))

                if abs(floatFrac - FLOAT_TOP_FRACTION) < 1e-9:
                    dictMetrics.update({
                        f"deletion_shift_{strMapName}": lstDelAttn[-1],
                        f"deletion_shift_{strMapName}_contiguous": lstDelCont[-1],
                        f"deletion_shift_{strMapName}_dispersed": lstDelDisp[-1],
                        f"insertion_retention_{strMapName}": lstInsAttn[-1],
                        f"insertion_retention_{strMapName}_contiguous": lstInsCont[-1],
                        f"insertion_retention_{strMapName}_dispersed": lstInsDisp[-1],
                        f"control_mask_iou_{strMapName}": floatControlIou,
                        "masked_pixel_fraction": float(arrMaskTop.mean()),
                    })

            dictMetrics.update({
                f"deletion_auc_{strMapName}": float(np.mean(lstDelAttn)),
                f"deletion_auc_{strMapName}_contiguous": float(np.mean(lstDelCont)),
                f"deletion_auc_{strMapName}_dispersed": float(np.mean(lstDelDisp)),
                f"insertion_auc_{strMapName}": float(np.mean(lstInsAttn)),
                f"insertion_auc_{strMapName}_contiguous": float(np.mean(lstInsCont)),
                f"insertion_auc_{strMapName}_dispersed": float(np.mean(lstInsDisp)),
                # Advantages: positive means the saliency region beat the control.
                f"deletion_advantage_{strMapName}_contiguous":
                    float(np.mean(lstDelAttn) - np.mean(lstDelCont)),
                f"deletion_advantage_{strMapName}_dispersed":
                    float(np.mean(lstDelAttn) - np.mean(lstDelDisp)),
                f"insertion_advantage_{strMapName}_contiguous":
                    float(np.mean(lstInsAttn) - np.mean(lstInsCont)),
                f"insertion_advantage_{strMapName}_dispersed":
                    float(np.mean(lstInsAttn) - np.mean(lstInsDisp)),
            })

        dictMetrics.update({
            "deletion_shift_attention": dictMetrics[f"deletion_shift_{STR_PRIMARY_SALIENCY}"],
            "deletion_shift_random":
                dictMetrics[f"deletion_shift_{STR_PRIMARY_SALIENCY}_contiguous"],
            "deletion_advantage":
                dictMetrics[f"deletion_shift_{STR_PRIMARY_SALIENCY}"]
                - dictMetrics[f"deletion_shift_{STR_PRIMARY_SALIENCY}_contiguous"],
            "insertion_retention_attention":
                dictMetrics[f"insertion_retention_{STR_PRIMARY_SALIENCY}"],
            "insertion_retention_random":
                dictMetrics[f"insertion_retention_{STR_PRIMARY_SALIENCY}_contiguous"],
            "insertion_advantage":
                dictMetrics[f"insertion_retention_{STR_PRIMARY_SALIENCY}"]
                - dictMetrics[f"insertion_retention_{STR_PRIMARY_SALIENCY}_contiguous"],
        })

    return dictMetrics, arrPadded, arrHeatmap

def plot_xai_controls(dfMetrics, pathOut):
    objFig, arrAxes = plt.subplots(1, 2, figsize=(10.4, 4.4))

    objAx = arrAxes[0]
    floatBaseline = float(dfMetrics["roi_area_fraction"].iloc[0])
    objAx.hist(dfMetrics["anatomical_enrichment"], bins=20, color="#4F7942",
               edgecolor="black", linewidth=0.5)
    objAx.axvline(1.0, color="#B02418", lw=1.6)
    objAx.annotate(f"uniform attention = 1.0\n(ROI covers {floatBaseline:.0%} of frame)",
                   xy=(1.0, 0.94), xycoords=("data", "axes fraction"),
                   xytext=(6, 0), textcoords="offset points", fontsize=7.5,
                   color="#B02418", va="top")
    objAx.annotate(f"observed mean = {dfMetrics['anatomical_enrichment'].mean():.2f}",
                   xy=(0.02, 0.62), xycoords="axes fraction", fontsize=7.5, color="#2F4F2F")
    objAx.set_xlabel("Anatomical enrichment (ROI mass / ROI area fraction)")
    objAx.set_ylabel(f"Images (of {len(dfMetrics)})")
    objAx.set_title("Attention enrichment inside the clinical ROI", loc="left", fontsize=10)

    objAx = arrAxes[1]
    lstPairs = [("deletion_shift_attention", "deletion_shift_random", "Deletion\n(top 20 % removed)"),
                ("insertion_retention_attention", "insertion_retention_random",
                 "Insertion\n(only top 20 % kept)")]
    arrX = np.arange(len(lstPairs))
    floatWidth = 0.36
    for intIdx, (strColour, strLabel, intOffset) in enumerate(
            [("#2B6A8F", "Attention-guided region", -1), ("#8A8A8A", "Random region, equal area", 1)]):
        arrValues = [dfMetrics[p[intIdx]].mean() for p in lstPairs]
        arrErrors = [dfMetrics[p[intIdx]].std(ddof=1) for p in lstPairs]
        objAx.bar(arrX + intOffset * floatWidth / 2, arrValues, floatWidth, yerr=arrErrors,
                  capsize=3, color=strColour, edgecolor="black", linewidth=0.5,
                  error_kw=dict(lw=0.8), label=strLabel)
    objAx.set_xticks(arrX)
    objAx.set_xticklabels([p[2] for p in lstPairs], fontsize=8.5)
    objAx.set_ylabel("Effect on classifier probability")
    objAx.set_title("Faithfulness against area-matched control", loc="left", fontsize=10)
    objAx.legend(frameon=False, fontsize=8)

    for objAx in arrAxes:
        objAx.spines[["top", "right"]].set_visible(False)
    objFig.tight_layout()
    objFig.savefig(pathOut, dpi=300)
    plt.close(objFig)


def plot_faithfulness_matrix(dfTests, pathOutBase):
    if dfTests.empty:
        return
    dfCurve = dfTests[dfTests["statistic"] == "curve_area"]
    if dfCurve.empty:
        return

    objFig, arrAxes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for objAx, strTest, strTitle in zip(
            arrAxes, ["deletion", "insertion"],
            ["Necessity (deletion)", "Sufficiency (insertion)"]):
        dfSub = dfCurve[dfCurve["test"] == strTest]
        if dfSub.empty:
            continue
        lstLabels, lstAdv, lstColours = [], [], []
        for _, serRow in dfSub.iterrows():
            lstLabels.append(f"{serRow['saliency_map']}\nvs {serRow['control']}")
            lstAdv.append(serRow["advantage"])
            lstColours.append("#007A5A" if serRow["advantage"] > 0 else "#B02418")
        arrX = np.arange(len(lstAdv))
        objAx.bar(arrX, lstAdv, 0.6, color=lstColours, edgecolor="black", linewidth=0.7)
        objAx.axhline(0.0, color="black", linewidth=1.1)
        for intIdx, (floatAdv, serRow) in enumerate(zip(lstAdv, dfSub.itertuples())):
            strStar = "*" if bool(getattr(serRow, "significant_adjusted", False)) else ""
            objAx.text(intIdx, floatAdv, f"{floatAdv:+.4f}{strStar}",
                       ha="center", va="bottom" if floatAdv >= 0 else "top",
                       fontsize=9, fontweight="bold")
        objAx.set_xticks(arrX)
        objAx.set_xticklabels(lstLabels, fontsize=8.5)
        objAx.set_ylabel("Advantage over control (curve area)")
        objAx.set_title(strTitle, loc="left", fontsize=10.5)
        objAx.spines[["top", "right"]].set_visible(False)
        floatSpan = max(abs(min(lstAdv)), abs(max(lstAdv))) or 1.0
        objAx.set_ylim(min(0.0, min(lstAdv)) - 0.35 * floatSpan,
                       max(0.0, max(lstAdv)) + 0.35 * floatSpan)

    objFig.suptitle("Faithfulness: every saliency map against both area-matched controls "
                    "(* = significant after BH correction)", fontsize=11)
    objFig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(objFig, pathOutBase)
    plt.close(objFig)


def main():
    sys.stdout = TeeLogger(pathPhase5Dir / "phase5_run.log")
    apply_manuscript_style()
    dfTable = load_balanced_table()
    arrLabels = get_labels(dfTable)
    dictMeta, dictFolds = load_splits()
    arrCls, _ = load_features(STR_EXPLAINED_ROI_MODE, dfTable, boolLoadPatch=False)
    print(f"Explaining the '{STR_EXPLAINED_ROI_MODE}' ROI track "
          f"(the track reported as the headline result).")
    print(f"Saliency method: {STR_SALIENCY_METHOD} | "
          f"reference ROI box: {TUP_CLINICAL_ROI}")

    print("Loading pre-trained DINOv2...")
    objModel = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(strDevice)
    objModel.eval()

    objClassifier, strFoldName, arrValRows, objSurrogate = fit_reference_classifier(
        dfTable, arrLabels, arrCls, dictFolds)
    dfHeldOut = dfTable.reset_index(drop=True).iloc[arrValRows]
    print(f"Faithfulness classifier fitted on the training split of {strFoldName}; "
          f"XAI measured on its {len(dfHeldOut)} held-out images.")

    objRng = np.random.default_rng(INT_GLOBAL_SEED)
    dfSample = dfHeldOut.sample(min(INT_NUM_QUANTITATIVE, len(dfHeldOut)),
                                random_state=INT_GLOBAL_SEED)

    lstRecords = []
    for _, serRow in tqdm(dfSample.iterrows(), total=len(dfSample), desc="XAI"):
        pathImg = pathDataDir / serRow["class"] / serRow["view"] / serRow["filename"]
        arrRaw = cv2.imread(str(pathImg), cv2.IMREAD_GRAYSCALE)
        if arrRaw is None:
            continue
        dictMetrics, _, _ = quantitative_xai_metrics(
            objModel, arrRaw, objClassifier, objRng,
            objSaliencyClassifier=objSurrogate)
        dictMetrics.update({"sample_id": serRow["sample_id"], "class": serRow["class"],
                            "view": serRow["view"]})
        lstRecords.append(dictMetrics)

    dfMetrics = pd.DataFrame(lstRecords)

    if STR_SALIENCY_METHOD == "class_gradient" and not dfMetrics.empty:
        arrDivergence = dfMetrics["saliency_vs_rollout_divergence"].to_numpy()
        floatIdenticalFraction = float(np.mean(arrDivergence < 1e-9))
        if floatIdenticalFraction > 0.05:
            raise RuntimeError(
                f"Class-sensitive saliency is identical to attention rollout in "
                f"{floatIdenticalFraction:.1%} of images (mean divergence "
                f"{arrDivergence.mean():.3e}). The gradient path is not "
                f"connected: results would claim a method that was never run. "
                f"Fix extract_class_gradient_saliency() before reporting.")
        print(f"[gate] Class-sensitive saliency differs from rollout in "
              f"{100 * (1 - floatIdenticalFraction):.1f}% of images "
              f"(mean absolute divergence {arrDivergence.mean():.4f}).")

    dfMetrics.to_csv(pathPhase5Dir / "phase5_xai_quantitative.csv", index=False)

    lstRows = []
    for strMap in ["rollout", "gradient"]:
        for strTest, strStem, boolHigherIsBetter in [
                ("deletion", "deletion", True), ("insertion", "insertion", True)]:
            for strStat, strSuffix in [("curve_area", "auc"), ("point_20pct", "shift")]:
                if strStat == "point_20pct":
                    strBase = (f"deletion_shift_{strMap}" if strTest == "deletion"
                               else f"insertion_retention_{strMap}")
                else:
                    strBase = f"{strStem}_auc_{strMap}"
                for strControl in ["contiguous", "dispersed"]:
                    strCol, strCtrlCol = strBase, f"{strBase}_{strControl}"
                    if strCol not in dfMetrics.columns or strCtrlCol not in dfMetrics.columns:
                        continue
                    serA, serB = dfMetrics[strCol], dfMetrics[strCtrlCol]
                    try:
                        floatP = float(wilcoxon(serA, serB).pvalue)
                    except ValueError:
                        floatP = np.nan
                    lstRows.append({
                        "saliency_map": strMap,
                        "test": strTest,
                        "statistic": strStat,
                        "control": strControl,
                        "value": float(serA.mean()),
                        "control_value": float(serB.mean()),
                        "advantage": float((serA - serB).mean()),
                        "advantage_sd": float((serA - serB).std(ddof=1)),
                        "pct_images_favourable": float((serA > serB).mean()),
                        "p_wilcoxon_paired": floatP,
                        "is_primary": bool(strMap == STR_PRIMARY_SALIENCY
                                           and strControl == "contiguous"
                                           and strStat == "curve_area"),
                        "family": f"{strTest}|{strStat}",
                    })

    dfTests = pd.DataFrame(lstRows)
    if not dfTests.empty:
        dfTests = add_multiplicity_correction(
            dfTests.rename(columns={"p_wilcoxon_paired": "p_repeat_level_ttest"})
        ).rename(columns={"p_repeat_level_ttest": "p_wilcoxon_paired"})
    dfTests.to_csv(pathPhase5Dir / "phase5_xai_faithfulness_tests.csv", index=False)
    export_manuscript_table(dfTests, pathPhase5Dir / "phase5_xai_faithfulness_tests",
                            "Faithfulness of saliency maps against area-matched controls")

    lstEnrich = []
    for strLabel, strCol in [("rollout", "anatomical_enrichment_rollout"),
                             ("gradient", "anatomical_enrichment")]:
        try:
            floatP = float(wilcoxon(dfMetrics[strCol] - 1.0).pvalue)
        except ValueError:
            floatP = np.nan
        lstEnrich.append({"metric": f"anatomical_enrichment_{strLabel}",
                          "value": float(dfMetrics[strCol].mean()),
                          "sd": float(dfMetrics[strCol].std(ddof=1)),
                          "control": 1.0, "p_value": floatP})
    try:
        floatPMaps = float(wilcoxon(dfMetrics["anatomical_enrichment_rollout"],
                                    dfMetrics["anatomical_enrichment"]).pvalue)
    except ValueError:
        floatPMaps = np.nan
    lstEnrich.append({"metric": "enrichment_rollout_vs_gradient",
                      "value": float(dfMetrics["anatomical_enrichment_rollout"].mean()),
                      "sd": np.nan,
                      "control": float(dfMetrics["anatomical_enrichment"].mean()),
                      "p_value": floatPMaps})

    for strName, strA, strB in [("deletion", "deletion_shift_attention",
                                 "deletion_shift_random"),
                                ("insertion", "insertion_retention_attention",
                                 "insertion_retention_random")]:
        try:
            floatP = float(wilcoxon(dfMetrics[strA], dfMetrics[strB]).pvalue)
        except ValueError:
            floatP = np.nan
        lstEnrich.append({"metric": strName, "value": float(dfMetrics[strA].mean()),
                          "sd": np.nan, "control": float(dfMetrics[strB].mean()),
                          "p_value": floatP})

    pd.DataFrame(lstEnrich).to_csv(
        pathPhase5Dir / "phase5_xai_control_tests.csv", index=False)

    if not dfTests.empty:
        print("\n--- Faithfulness verdict (positive advantage = saliency beat its control) ---")
        for _, serRow in dfTests.sort_values(["saliency_map", "test", "control"]).iterrows():
            strFlag = "FAVOURABLE" if serRow["advantage"] > 0 else "unfavourable"
            strSig = ("sig" if bool(serRow.get("significant_adjusted", False))
                      else "not sig")
            print(f"  {serRow['saliency_map']:<9} {serRow['test']:<10} "
                  f"{serRow['statistic']:<11} vs {serRow['control']:<11} "
                  f"adv={serRow['advantage']:+.5f} ({serRow['pct_images_favourable']:.0%} "
                  f"of images) p_adj={serRow.get('p_adjusted_bh', np.nan):.4g} "
                  f"[{strFlag}, {strSig}]")

    plot_xai_controls(dfMetrics, pathPhase5Dir / "phase5_xai_controls.png")
    plot_faithfulness_matrix(dfTests, pathPhase5Dir / "phase5_xai_faithfulness")

    print("\nGenerating qualitative panel...")
    lstPanelRows = []
    for strClass in ["multinodular_goiter", "diffuse_goiter"]:
        dfClass = dfHeldOut[dfHeldOut["class"] == strClass]
        if len(dfClass):
            lstPanelRows.append(dfClass.sample(min(4, len(dfClass)),
                                               random_state=INT_GLOBAL_SEED))
    dfPanel = pd.concat(lstPanelRows) if lstPanelRows else dfSample.head(8)

    objFig, arrAxes = plt.subplots(len(dfPanel), 3, figsize=(8.4, 2.7 * len(dfPanel)))
    arrAxes = np.atleast_2d(arrAxes)
    for intIdx, (_, serRow) in enumerate(dfPanel.iterrows()):
        pathImg = pathDataDir / serRow["class"] / serRow["view"] / serRow["filename"]
        arrRaw = cv2.imread(str(pathImg), cv2.IMREAD_GRAYSCALE)
        dictMetrics, arrPadded, arrHeatmap = quantitative_xai_metrics(
            objModel, arrRaw,
            objSaliencyClassifier=(objSurrogate
                                   if STR_PRIMARY_SALIENCY == "gradient"
                                   else None))

        arrHeatColour = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * arrHeatmap), cv2.COLORMAP_INFERNO),
            cv2.COLOR_BGR2RGB)
        arrOverlay = cv2.addWeighted(cv2.cvtColor(arrPadded, cv2.COLOR_GRAY2RGB), 0.55,
                                     arrHeatColour, 0.45, 0)

        strLabel = ("Multinodular" if serRow["class"] == "multinodular_goiter" else "Diffuse")
        for intCol, (arrPanel, strCmap) in enumerate([(arrPadded, "gray"),
                                                      (arrHeatColour, None),
                                                      (arrOverlay, None)]):
            arrAxes[intIdx, intCol].imshow(arrPanel, cmap=strCmap)
            arrAxes[intIdx, intCol].axis("off")
        arrAxes[intIdx, 0].text(6, 18, f"{strLabel} / {serRow['view']}", color="white",
                                fontsize=8, backgroundcolor="black")

        strEnrichKey = ("anatomical_enrichment" if STR_PRIMARY_SALIENCY == "gradient"
                        else "anatomical_enrichment_rollout")
        arrAxes[intIdx, 2].text(6, 18, f"enrichment = {dictMetrics[strEnrichKey]:.2f}",
                                color="white", fontsize=8, backgroundcolor="black")
        if intIdx == 0:
            strMapTitle = ("Class-gradient saliency" if STR_PRIMARY_SALIENCY == "gradient"
                           else "Attention rollout")
            for intCol, strTitle in enumerate(["Padded scintigram", strMapTitle,
                                               "Overlay"]):
                arrAxes[intIdx, intCol].set_title(strTitle, fontsize=10)

    objFig.tight_layout()
    objFig.savefig(pathPhase5Dir / "phase5_xai_large_panel.png", dpi=300)
    plt.close(objFig)

    print(f"\nPhase 5 complete. Artifacts written to {pathPhase5Dir}")


if __name__ == "__main__":
    main()
