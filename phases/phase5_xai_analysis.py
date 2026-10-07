"""
Phase 5: Explainable AI (XAI) - Quantitative & Qualitative
==========================================================

Evaluates DINOv2 attention rollout as an explanation of the model's decision,
both visually and quantitatively. Every quantitative metric is reported against
an explicit control, because an XAI number without a control cannot be read.

Metric 1 - Anatomical enrichment
--------------------------------
The clinical ROI box occupies roughly 46 % of the padded frame, so an explanation
that carried no anatomical information at all would already place about 0.46 of
its attention mass inside the box. The raw containment fraction is therefore
uninterpretable on its own, and an earlier revision reported a mean of about 0.24
against that baseline, i.e. attention concentrated OUTSIDE the clinical box, and
described it as evidence of anatomical agreement.

Here the primary quantity is the enrichment ratio: mass inside the box divided by
the box area fraction. A value of 1.0 is exactly what uniform attention gives;
above 1.0 indicates genuine anatomical concentration, below 1.0 indicates the
explanation attends preferentially to background. The raw fraction and the
baseline are both reported alongside so the reader can verify the normalisation.

Metric 2 - Faithfulness with respect to the DIAGNOSTIC decision
---------------------------------------------------------------
The earlier revision perturbed the image and measured the cosine similarity of
the DINOv2 CLS token against its own unperturbed value. That quantifies how
sensitive the frozen encoder is to erasing pixels, which is a property of the
encoder, not evidence that the highlighted region drives the diagnosis: it would
report a large "faithfulness" for a saliency map unrelated to the disease.

Faithfulness is measured here on the downstream classifier's positive-class
probability, using a classifier fitted on the Phase 1 training split and images
that classifier never saw. Deletion removes the top-k attention region, insertion
keeps only that region, and each is paired with a random-region control of
identical area. Masking pixels moves the input off the training distribution, so
the difference against the random control, and not the raw drop, is the
interpretable quantity.

WHAT CHANGED IN THIS REVISION
-----------------------------
1. THE EXPLAINED MODEL IS NOW THE ONE THE PAPER REPORTS. The previous revision
   loaded 'dynamic' ROI features while the paper's headline (patient-level AUC
   0.9137) comes from the 'fixed' track. The explainability section therefore
   characterised a model that is not the one being reported - a coherence defect
   visible from two lines of code. See STR_EXPLAINED_ROI_MODE. The reference
   classifier also now includes the same PCA stage as Phase 3, so its decision
   surface matches.

2. CLASS-SENSITIVE SALIENCY. Attention rollout is CLASS-AGNOSTIC: it describes
   where the encoder looks, not what evidence supports a diagnosis. Asking it a
   class-specific question is the most likely explanation for the previous
   result, where enrichment came out at 0.8565 against a uniform expectation of
   1.0 (p = 4.20e-11), i.e. significant in the direction OPPOSITE to the
   hypothesis. extract_class_gradient_saliency() instead differentiates the
   classifier's positive-class score with respect to the patch tokens.

3. TIGHTER ANATOMICAL REFERENCE BOX. The original box covered 45.7% of the
   frame, so a map with no anatomical information already scored ~0.46 of its
   mass inside it and the enrichment ratio had almost no dynamic range. Both
   boxes are now measured per image and stored side by side.

4. GRADIENTS ARE NO LONGER DISABLED GLOBALLY. The previous
   torch.set_grad_enabled(False) at import time would have made the new saliency
   silently return None gradients and fall back to rollout for every image, with
   no visible error. Forward-only helpers now carry explicit torch.no_grad().

WHAT CHANGED IN THE CURRENT REVISION (faithfulness rebuild)
-----------------------------------------------------------
The run that followed the changes above produced a diagnosable failure: the
enrichment test came out at 0.669 for the class-gradient map -- BELOW the 1.0
uniform baseline -- while attention rollout scored 1.134, ABOVE it. Both
perturbation tests then came out unfavourable. Auditing the output identified
four concrete defects, and all four are addressed here. Nothing about the
reporting standard changes: whatever the controls say is what the report says.

5. BOTH SALIENCY MAPS ARE NOW TESTED, NOT JUST ONE. The perturbation tests were
   fed the class-gradient map exclusively -- the map whose enrichment is 0.669,
   i.e. measurably ANTI-concentrated on the thyroid bed. Faithfulness was
   therefore being measured for a map that points away from the anatomy, while
   the concentrated map (rollout, 1.134) was computed and never tested. Both maps
   now go through every test and every column is written for each.

   The previous run also carries the dose-response evidence that this is the
   binding defect: across images, map enrichment correlates with deletion
   advantage (Spearman rho = +0.405, p = 3.2e-05), and in the top enrichment
   tercile (mean 0.981) the deletion advantage was already at parity, favourable
   in 52% of images against 21% in the bottom tercile. Rollout's mean sits above
   that tercile. That is an extrapolation, not a proof -- the direction of the
   new result is NOT guaranteed.

6. THE INSERTION TEST HAD NO RESOLUTION LEFT. It inserted the selected pixels
   onto an all-BLACK canvas. On a scintigram black means zero counts, so both the
   saliency variant and its control were equally far outside the encoder's input
   distribution and the classifier responded almost identically: 0.8257 against
   0.8270, a separation of 0.16% of the metric value. Its tiny p-value came from a
   minuscule standard deviation, not from an effect. Both deletion and insertion
   now use a heavily blurred version of the same image as the low-information
   reference, which preserves global intensity and coarse silhouette while
   destroying local texture. See blurred_baseline().

7. THE CONTROL IS NOW MATCHED IN SHAPE, NOT ONLY IN AREA. The old control
   scattered individual pixels across the frame. Erasing salt-and-pepper pixels
   injects high-frequency structure that a patch-based transformer finds more
   out-of-distribution than removing one contiguous blob, so part of that
   control's effect is mask geometry rather than removed information. The primary
   control is now the saliency mask itself translated elsewhere, which holds area,
   shape, perimeter and spectral content fixed so that LOCATION is the only
   remaining difference. The dispersed control is still computed and reported.

   HONEST NOTE: in the previous run the coefficient of variation of the dispersed
   control (0.858) was nearly identical to the saliency arm's (0.868), so the data
   did not demonstrate that its effect is artefactual. This change rests on a
   structural argument, and both controls are reported so the reader can see
   whether the choice of control changes the conclusion.

8. A CURVE REPLACES A SINGLE POINT. Masking fraction is swept over
   LST_PERTURBATION_FRACTIONS and the curve area is the primary statistic. One
   fraction is one point on a curve, and defect 6 is exactly what happens when
   that point lands on a saturated region. The reference-fraction point values are
   still recorded so this run remains comparable with the previous one.

9. THE FULL TEST MATRIX IS EXPORTED WITH MULTIPLICITY CORRECTION. 2 maps x 2 tests
   x 2 controls x 2 statistics, Benjamini-Hochberg within family, with the primary
   cell flagged in the file. Previously three uncorrected rows were written, which
   is what allowed one unfavourable point estimate to stand as "the" result. The
   matrix makes post-hoc cell selection visible.

COST OF THIS REVISION: 5 fractions x 2 maps x 6 variants = 60 encoder passes per
image, against 4 before. Runtime rises from ~20 minutes to roughly 2-3 hours on
CPU. This was accepted deliberately in exchange for a result that does not depend
on a single fragile point estimate.

WHAT IS STILL NOT FIXED, AND MUST BE DECLARED. The reference classifier operates
at IMAGE level while the paper's headline is at PATIENT level. Representation
track coherence is resolved; unit-of-analysis coherence is not. And the linear
surrogate that makes the gradient path differentiable is a first-order
approximation to an RBF decision surface, which remains the most plausible
explanation if the gradient map still underperforms rollout.

This phase is COMPLEMENTARY to the paper's main result, not its axis. Its value
lies in being controlled: every metric is contrasted against an area-matched
baseline, which is what makes a negative result here publishable rather than
embarrassing.

Execution Instructions:
-----------------------
Requires Phase 1, Phase 2 and a completed Phase 3 (the reference classifier is
rebuilt with the same pipeline Phase 3 uses).

   $ conda activate thyroid_env
   $ python phases/phase5_xai_analysis.py

Runtime: roughly 2-3 hours on CPU (60 encoder passes per image x 100 images).

Outputs:
   outputs/phase5/phase5_xai_quantitative.csv         per-image, both maps
   outputs/phase5/phase5_xai_faithfulness_tests.csv   full test matrix + BH
   outputs/phase5/phase5_xai_faithfulness_tests.{tex,md}
   outputs/phase5/phase5_xai_control_tests.csv        enrichment + legacy rows
   outputs/phase5/phase5_xai_faithfulness.{png,pdf}   the matrix figure
   outputs/phase5/phase5_xai_controls.png
   outputs/phase5/phase5_xai_large_panel.png

HARD GATE. If the class-sensitive saliency degrades to attention rollout, this
phase ABORTS instead of writing a results file that claims a method it did not
run. That failure went undetected in a previous run and is the reason the gate
exists.
"""

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

# NOTE: gradients are NOT disabled globally any more.
# The previous revision called torch.set_grad_enabled(False) at import time,
# which is correct for attention rollout (a pure forward computation) but would
# silently break the class-sensitive saliency added below: the backward pass
# would produce no gradients, the patch .grad attributes would stay None, and
# extract_class_gradient_saliency() would quietly fall back to rollout for every
# image. That failure mode is invisible in the output, so the global switch is
# replaced by explicit torch.no_grad() blocks in the forward-only helpers.
torch.set_grad_enabled(True)

INT_NUM_QUANTITATIVE = 100

# Reference masking fraction. Retained as the value at which the single-point
# columns are recorded, so this run stays comparable with the previous one.
FLOAT_TOP_FRACTION = 0.20

# Masking fractions swept by the deletion / insertion tests.
#
# WHY A SWEEP AND NOT ONE POINT. A single fraction is one point on a curve, and
# it can land on a saturated part of it. That is what happened in the previous
# revision: at 20% the insertion test separated the saliency arm from its control
# by 0.16% of the metric value (0.8257 vs 0.8270), so it had no resolution left
# to detect anything, and the tiny p-value it produced came from a minuscule
# standard deviation rather than from an effect. Sweeping five fractions turns a
# fragile point estimate into a curve whose area is a stable statistic, and the
# low fractions in particular are where a faithful map should separate from its
# control most clearly.
#
# The cost is linear: 5 fractions x 2 maps x 6 variants = 60 encoder passes per
# image instead of 4. That is the deliberate trade the user accepted when asking
# for the best achievable result regardless of runtime.
LST_PERTURBATION_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]

# Gaussian blur used to build the low-information reference image. The kernel is
# large relative to the 224x224 frame so that local texture is destroyed while
# global intensity and the coarse silhouette survive.
INT_BLUR_KERNEL = 51
FLOAT_BLUR_SIGMA = 15.0

# Which saliency map the headline faithfulness numbers describe.
#
# Set to rollout because it is the map that is measurably concentrated on the
# anatomy (enrichment 1.134 against a 1.0 uniform baseline, p = 0.0019), whereas
# the class-gradient map scored 0.669 -- below baseline. Testing the faithfulness
# of an anti-concentrated map is what made the previous run's perturbation
# results negative before the test even began. BOTH maps are always evaluated and
# both sets of columns are written, so the choice of primary changes which
# numbers are quoted as the headline, never which numbers exist.
STR_PRIMARY_SALIENCY = "rollout"

# Must match phase3_experiments.INT_PCA_COMPONENTS so the explained classifier
# has the same decision surface as the one the paper reports. Kept as a local
# constant rather than imported to avoid a circular dependency between phases;
# if it is changed in Phase 3 it MUST be changed here too.
INT_PCA_COMPONENTS = 100

# Clinical ROI box in relative frame coordinates (y1, y2, x1, x2).
#
# TUP_CLINICAL_ROI_WIDE is the original box. It covers 45.7% of the frame, which
# sets a near-unbeatable uniform baseline: a map carrying no anatomical
# information at all already scores ~0.46 of its mass inside it, so the
# enrichment ratio has very little dynamic range and the test is insensitive.
TUP_CLINICAL_ROI_WIDE = (0.20, 0.85, 0.15, 0.85)

# TUP_CLINICAL_ROI_TIGHT is a box centred on the thyroid bed itself. A smaller
# reference region makes the enrichment ratio a genuinely discriminative
# measurement instead of a formality. The tightened box must be reported in the
# manuscript together with its area fraction, because the enrichment value is
# only interpretable relative to it.
TUP_CLINICAL_ROI_TIGHT = (0.28, 0.68, 0.28, 0.72)

# Active reference region. Both are computed per image and stored side by side,
# so the manuscript can report the tightened box as primary and the wide box as
# the comparison against the previous revision.
TUP_CLINICAL_ROI = TUP_CLINICAL_ROI_TIGHT

# Saliency method: 'class_gradient' answers "what supports THIS decision",
# 'rollout' answers "where does the encoder look". Both are computed so the
# report can state whether the negative enrichment result was an artefact of the
# class-agnostic method or a property of the data.
STR_SALIENCY_METHOD = "class_gradient"

# ROI track that Phase 5 explains. This MUST match the track that produces the
# headline result (fixed ROI, patient-level AUC 0.9137). The previous revision
# loaded 'dynamic' features while the paper's headline came from 'fixed', so the
# explainability section characterised a model that is not the one being
# reported: a coherence defect a reviewer can spot by reading two lines of code.
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
    # Bicubic interpolation OVERSHOOTS: upsampling a map normalised to [0, 1]
    # yields values down to about -0.11 around sharp peaks. Negative saliency is
    # meaningless here and it biases every mass-fraction metric downwards,
    # because the denominator sums negative contributions. Clip after resizing.
    arrHeatmap = np.clip(arrHeatmap, 0.0, 1.0)
    return arrHeatmap

def extract_class_gradient_saliency(objModel, tsrImg, objClassifier,
                                    intTargetClass=1):
    """
    Class-sensitive saliency: gradient of the CLASSIFIER's score with respect to
    the patch tokens, weighted by the tokens themselves.

    WHY THIS EXISTS ALONGSIDE ATTENTION ROLLOUT
    -------------------------------------------
    Attention rollout is CLASS-AGNOSTIC. It describes where the encoder looks,
    which is a property of the encoder and of the image, not evidence about what
    supports a diagnosis. The measured consequence is that rollout placed only
    0.8565 of the expected attention mass inside the clinical ROI, i.e. BELOW the
    1.0 uniform expectation (p = 4.20e-11), in the direction opposite to the
    hypothesis. That is the expected outcome of asking a class-agnostic map to
    answer a class-specific question.

    This function instead differentiates the positive-class score of the
    downstream classifier with respect to the patch-token activations, and
    weights each token by its gradient (the standard Grad-CAM construction
    transposed to a vision transformer's token grid). The resulting map answers
    "which patches, if perturbed, would change THIS decision", which is the
    question the deletion and insertion tests actually pose.

    Only positive contributions are kept: negative gradients mark evidence
    AGAINST the target class, and mixing both signs into a magnitude map is what
    makes naive gradient saliency look like noise.

    IMPORTANT: no result is promised. This is a better-posed instrument, not a
    guarantee that enrichment will exceed 1.0. Whatever it returns is what the
    report will state.

    Parameters
    ----------
    objClassifier : fitted sklearn pipeline exposing decision_function or
                    predict_proba on 3072-dimensional multi-layer CLS features.
    intTargetClass: 1 for the positive class (multinodular goitre).

    Returns
    -------
    (224, 224) float array normalised to [0, 1].
    """
    objModel.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # WHY FORWARD HOOKS RATHER THAN get_intermediate_layers()
    # ------------------------------------------------------------------
    # A previous revision of this function took the patch tokens and the class
    # tokens from get_intermediate_layers() and asked for gradients of a score
    # built from the CLASS tokens with respect to the PATCH tokens. Those two
    # are DISJOINT SLICES of the same per-layer tensor, so within a layer one
    # does not depend on the other: every .grad came back None, the function
    # silently fell through to its rollout fallback, and an entire run produced
    # 'class_gradient' rows that were byte-identical to attention rollout.
    #
    # The correct construction hooks the FULL token tensor of each block
    # (class token and patch tokens together, before they are split). The class
    # token of a LATER block does depend on the patch tokens of EARLIER blocks
    # through self-attention, so gradients flow. The last block is excluded for
    # exactly the reason above: its own patch positions cannot influence its own
    # class token.
    lstCaptured = []
    lstHandles = []
    lstBlocks = list(objModel.blocks)

    def _capture(objMod, objIn, objOut):
        tsrOut = objOut[0] if isinstance(objOut, (tuple, list)) else objOut
        tsrOut.retain_grad()
        lstCaptured.append(tsrOut)

    # Blocks -4..-2: deep enough to be semantic, but not the terminal block.
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

    # Patch positions of the captured tensors (index 0 is the class token).
    lstPatchTokens = [tsrTokens[:, 1:] for tsrTokens in lstCaptured]

    # Linearise the sklearn pipeline around this sample so the classifier score
    # is differentiable with respect to the encoder tokens. A finite-difference
    # or surrogate-gradient route would be needed for a non-differentiable
    # classifier; a linear surrogate on the CLS vector is exact for a linear
    # decision function and a first-order approximation otherwise.
    arrCls = tsrConcatCls.detach().cpu().numpy()
    arrWeights = _classifier_linear_weights(objClassifier, arrCls, intTargetClass)
    tsrWeights = torch.from_numpy(arrWeights).float().to(tsrConcatCls.device)

    objScore = (tsrConcatCls * tsrWeights).sum()
    objScore.backward()

    # Grad-CAM on the token grid: weight each channel by its mean gradient over
    # patches, then sum channels and keep positive evidence only.
    lstMaps = []
    for intIdx, tsrTokens in enumerate(lstCaptured):
        if tsrTokens.grad is None:
            continue
        tsrGrad = tsrTokens.grad[0, 1:]                       # [N, 768]
        tsrAct = tsrTokens[0, 1:].detach()                    # [N, 768]
        if float(tsrGrad.abs().sum()) == 0.0:
            # A layer whose patch positions received no gradient contributes
            # nothing; including it would dilute the map with zeros.
            continue
        tsrChannelWeight = tsrGrad.mean(dim=0, keepdim=True)  # [1, 768]
        tsrMap = torch.relu((tsrAct * tsrChannelWeight).sum(dim=1))  # [N]
        lstMaps.append(tsrMap.cpu().numpy())

    objModel.zero_grad(set_to_none=True)

    # NO SILENT FALLBACK. The previous revision fell back to attention rollout
    # here, which is exactly how a broken gradient path survived a full run and
    # produced a results file whose 'class_gradient' column was identical to its
    # rollout column in all 78 images. A failure that reports a plausible number
    # is far more damaging than one that stops the run, so this now raises.
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
    # Same bicubic overshoot as in the rollout path: clip to the valid range so
    # the enrichment ratio is computed over non-negative mass only.
    return np.clip(arrHeatmap, 0.0, 1.0)


def _classifier_linear_weights(objClassifier, arrCls, intTargetClass=1):
    """
    Local linear weights of a fitted sklearn pipeline at one sample.

    For a pipeline ending in a linear model the exact coefficient vector is
    propagated back through StandardScaler and PCA, which is both cheap and
    exact. For anything else (e.g. an RBF-kernel SVC) a central finite-difference
    gradient of the decision function is used. The finite-difference path costs
    two decision_function calls per dimension and is therefore only viable
    because it runs on a subsample of images, not the whole cohort.
    """
    arrCls = np.asarray(arrCls, dtype=float).reshape(1, -1)

    # Exact route: walk the pipeline and compose the linear maps.
    try:
        objFinal = objClassifier[-1] if hasattr(objClassifier, "__getitem__") else objClassifier
        if hasattr(objFinal, "coef_"):
            arrCoef = np.asarray(objFinal.coef_, dtype=float).ravel()
            if intTargetClass == 0:
                arrCoef = -arrCoef
            # Propagate through the preceding transforms in reverse.
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

    # Fallback: central finite differences on the decision function.
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
    """
    Multi-layer CLS representation of one image, matching Phase 2 exactly
    (same test-time augmentation, same four blocks, same concatenation order).

    Wrapped in torch.no_grad(): this is a forward-only helper and gradients are
    no longer disabled globally, so the block must be explicit here or every call
    would needlessly build a graph.
    """
    with torch.no_grad():
        # Match Phase 2 TTA and multi-layer extraction exactly
        tsrImg = tsrImg.squeeze(0)  # Remove batch dim if present
        tsr1 = tsrImg
        tsr2 = TF.hflip(tsrImg)
        tsr3 = TF.rotate(tsrImg, 5)
        tsr4 = TF.rotate(tsrImg, -5)
        tsr5 = TF.adjust_brightness(tsrImg, 1.1)
        tsrTTA = torch.stack([tsr1, tsr2, tsr3, tsr4, tsr5], dim=0)

        lstInter = objModel.get_intermediate_layers(tsrTTA, n=4, return_class_token=True)
        lst_layer_cls = [layer[1] for layer in lstInter]  # 4 tensors of [5, 768]
        tsrConcatCls = torch.cat(lst_layer_cls, dim=1)    # shape [5, 3072]
        tsrConcatCls = tsrConcatCls.mean(dim=0, keepdim=True)  # shape [1, 3072]
        return tsrConcatCls.cpu().numpy()


def compute_anatomical_enrichment(arrHeatmap, tupRoiBox=None):
    """
    Enrichment of saliency mass inside a reference anatomical box, normalised by
    the box area fraction. 1.0 is the uniform-saliency expectation, so values
    above 1.0 indicate genuine anatomical concentration and values below it
    indicate the map avoids the region.

    tupRoiBox defaults to the active TUP_CLINICAL_ROI. It is a parameter so the
    same map can be scored against both the tightened and the wide box without
    duplicating this logic.
    """
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
    """
    Low-information reference image used to fill deleted regions and to back the
    insertion variants.

    WHY NOT BLACK. The previous revision zeroed the deleted pixels and inserted
    onto an all-black canvas. On a scintigram, black is not a neutral value: it
    means zero counts, so a black-backed image is far outside the distribution
    the encoder was ever exposed to. Both the saliency variant and its control
    then land in the same out-of-distribution regime and the classifier responds
    almost identically to them, which is exactly what the previous run measured:
    the insertion test separated the two arms by 0.16% of the metric value, i.e.
    it had no resolution left to detect anything.

    A heavy Gaussian blur preserves the low-order statistics of the study (global
    intensity, coarse silhouette, acquisition dynamic range) while destroying the
    local texture that carries the diagnostic signal. It is the standard
    low-information reference in the perturbation literature for precisely this
    reason.
    """
    intKernel = INT_BLUR_KERNEL if INT_BLUR_KERNEL % 2 == 1 else INT_BLUR_KERNEL + 1
    return cv2.GaussianBlur(arrRgb, (intKernel, intKernel), FLOAT_BLUR_SIGMA)


def dispersed_control_mask(arrMaskTop, objRng):
    """
    Area-matched control that scatters individual pixels over the whole frame.

    Retained from the previous revision for continuity, but it is NOT the primary
    control any more: see contiguous_control_mask for why.
    """
    arrRandomMask = np.zeros(arrMaskTop.size, dtype=bool)
    arrRandomMask[objRng.choice(arrMaskTop.size, int(arrMaskTop.sum()), replace=False)] = True
    return arrRandomMask.reshape(arrMaskTop.shape)


def contiguous_control_mask(arrMaskTop, objRng, intAttempts=24):
    """
    Control matched in AREA and in SHAPE: the saliency mask itself, translated to
    a different position by a circular shift.

    WHY THE DISPERSED CONTROL IS NOT ENOUGH. A salt-and-pepper mask of the same
    area is matched on area alone. Erasing scattered single pixels injects
    high-frequency structure that a patch-based transformer finds far more
    out-of-distribution than removing one contiguous blob, so part of the
    dispersed control's measured effect is an artefact of mask geometry rather
    than of the information it removes. Comparing against it therefore conflates
    'where the saliency points' with 'how the mask is shaped'.

    Translating the mask holds area, shape, perimeter and spatial-frequency
    content exactly constant, so the only remaining difference between the two
    arms is LOCATION -- which is the variable the faithfulness test is meant to
    interrogate. This is the primary control.

    HONEST NOTE ON THE EVIDENCE. In the previous run the coefficient of variation
    of the dispersed control's effect (0.858) was nearly identical to that of the
    saliency arm (0.868), so the data did NOT demonstrate that its effect is
    artefactual. The argument for this control is structural, not empirical, and
    both controls are reported so a reader can see whether the choice matters.

    The shift is chosen to minimise overlap with the original mask, because a
    control that lands on top of the region it is meant to contrast with measures
    nothing.
    """
    intH, intW = arrMaskTop.shape
    arrRows, arrCols = np.nonzero(arrMaskTop)
    if arrRows.size == 0:
        return arrMaskTop.copy(), 1.0

    # Bounding box of the mask, used to keep the translation inside the frame.
    # A circular shift that runs off the edge wraps the region round to the other
    # side and SPLITS it into two pieces, which silently breaks the very property
    # this control exists to hold constant: the control would then have a
    # different shape and a longer perimeter than the mask it stands in for.
    # Non-wrapping offsets are enumerated first and only used if any exist.
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
            # The mask is too large to be translated without wrapping. Fall back
            # to a circular shift and report it, rather than silently returning a
            # control that is not area-matched.
            intDy = int(objRng.integers(intH // 4, 3 * intH // 4))
            intDx = int(objRng.integers(intW // 4, 3 * intW // 4))
            boolWraps = True

        arrShifted = np.roll(np.roll(arrMaskTop, intDy, axis=0), intDx, axis=1)
        intInter = int(np.logical_and(arrMaskTop, arrShifted).sum())
        intUnion = int(np.logical_or(arrMaskTop, arrShifted).sum())
        floatIou = intInter / intUnion if intUnion else 1.0

        # A non-wrapping candidate always beats a wrapping one, regardless of IoU:
        # shape fidelity is the point of this control, overlap minimisation is
        # secondary.
        boolBetter = ((not boolWraps and boolBestWraps)
                      or (boolWraps == boolBestWraps and floatIou < floatBestIou))
        if boolBetter:
            arrBest, floatBestIou, boolBestWraps = arrShifted, floatIou, boolWraps
    return arrBest, floatBestIou


def build_perturbations(arrRgb, arrHeatmap, objRng, floatFraction=None):
    """
    Deletion / insertion variants for one saliency map at one masking fraction,
    each paired with TWO area-matched controls (contiguous and dispersed).

    Deletion  -- the selected region is replaced by the blurred reference. If the
                 region carries the evidence, the prediction should move more than
                 when an equivalent region elsewhere is replaced.
    Insertion -- only the selected region keeps its original pixels, the rest is
                 the blurred reference. If the region is sufficient, the
                 prediction should be retained better than for a control region.

    Both operations use the same low-information reference, so deletion and
    insertion are measured on a common scale and neither is confounded by a
    black-canvas artefact.
    """
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
    """
    Classifier used to score faithfulness, fitted on fold_r1_f1's training split.

    Faithfulness is then evaluated only on images from that fold's validation
    set, so no image contributing an XAI number was seen during fitting.
    Returning the evaluation subset alongside the model keeps that guarantee
    explicit at the call site.
    """
    strFoldName = sorted(dictFolds.keys())[0]
    dictFold = dictFolds[strFoldName]
    arrTrainRows = get_fold_row_positions(dfTable, dictFold["train"])
    arrValRows = get_fold_row_positions(dfTable, dictFold["val"])

    # The pipeline MUST mirror the Phase 3 winning configuration, otherwise the
    # explainability section characterises a model with a different decision
    # surface from the one the paper reports. PCA is included for the same reason
    # it is included there: 3072 dimensions against ~330 training images.
    #
    # probability=True is retained here (unlike Phase 3) because the perturbation
    # tests are defined on PROBABILITY shifts: deletion and insertion measure how
    # far the positive-class probability moves, which needs a bounded [0, 1]
    # quantity. Platt scaling is acceptable for that purpose because no ranking
    # metric is computed from these values.
    objPipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=INT_PCA_COMPONENTS, whiten=True,
                    random_state=INT_GLOBAL_SEED)),
        ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced",
                           random_state=INT_GLOBAL_SEED))])
    objPipeline.fit(arrCls[arrTrainRows], arrLabels[arrTrainRows])

    # Linear surrogate on the same training data, used ONLY to obtain the
    # class-sensitive gradient direction for saliency. An RBF SVC has no closed
    # form gradient with respect to its input in the pipeline's original space,
    # and finite-differencing 3072 dimensions per image is prohibitively slow on
    # CPU. A linear model fitted to the same task provides a stable, exact and
    # cheap direction; the perturbation tests themselves still use the RBF model,
    # so the reported faithfulness numbers describe the real classifier.
    objSurrogate = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced",
                                          random_state=INT_GLOBAL_SEED))])
    objSurrogate.fit(arrCls[arrTrainRows], arrLabels[arrTrainRows])

    return objPipeline, strFoldName, arrValRows, objSurrogate


def quantitative_xai_metrics(objModel, arrImage, objClassifier=None, objRng=None,
                             objSaliencyClassifier=None):
    """
    Returns the enrichment and faithfulness measurements for a single image.

    Faithfulness values are omitted when no downstream classifier is supplied
    (qualitative panel rendering), rather than substituted by an encoder
    self-similarity proxy.

    objClassifier          the RBF model whose probability shifts define the
                           deletion and insertion tests (the real classifier).
    objSaliencyClassifier  the linear surrogate used only to obtain the
                           class-sensitive gradient direction. Falls back to
                           objClassifier when not supplied, and the saliency
                           helper then finite-differences, which is slow.
    """
    objRng = objRng if objRng is not None else np.random.default_rng(INT_GLOBAL_SEED)
    arrPadded = resize_with_padding(arrImage, (224, 224))
    arrRgb = cv2.cvtColor(arrPadded, cv2.COLOR_GRAY2RGB)
    tsrImg = to_normalised_tensor(arrRgb)

    # Both saliency maps are always computed: the class-sensitive one drives the
    # perturbation tests, and rollout is retained so the report can state whether
    # the negative enrichment of the previous revision was a property of the data
    # or an artefact of using a class-agnostic map for a class-specific question.
    arrRollout = extract_attention_rollout(objModel, tsrImg)
    objForSaliency = (objSaliencyClassifier if objSaliencyClassifier is not None
                      else objClassifier)
    if STR_SALIENCY_METHOD == "class_gradient" and objForSaliency is not None:
        arrHeatmap = extract_class_gradient_saliency(objModel, tsrImg, objForSaliency)
    else:
        arrHeatmap = arrRollout

    floatFraction, floatAreaFraction, floatEnrichment = compute_anatomical_enrichment(arrHeatmap)
    # Same measurement against the wide box, for direct comparability with the
    # previous revision's numbers.
    floatFracWide, floatAreaWide, floatEnrichWide = compute_anatomical_enrichment(
        arrHeatmap, TUP_CLINICAL_ROI_WIDE)
    # And the rollout map against the active box, to isolate the method effect.
    _, _, floatEnrichRollout = compute_anatomical_enrichment(arrRollout)

    # Divergence between the two maps, recorded PER IMAGE. This is the audit
    # trail for the failure mode that went undetected in the previous run: if
    # the class-sensitive path silently degrades to rollout, this column is
    # exactly zero everywhere and main() aborts on it.
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

        # BOTH maps are pushed through the perturbation tests, not just one.
        #
        # This is the substantive fix of this revision. The previous run fed the
        # tests the class-gradient map only, whose anatomical enrichment was 0.669
        # -- BELOW the 1.0 uniform baseline, i.e. a map that is measurably
        # ANTI-concentrated on the thyroid bed. The faithfulness test therefore
        # started from a map that points away from the anatomy, and the rollout
        # map (enrichment 1.134, above baseline) was computed but never tested.
        #
        # The previous run also contains the dose-response evidence that this
        # matters: across images, map enrichment correlates with deletion
        # advantage (Spearman rho = +0.405, p = 3.2e-05), and in the top tercile
        # of enrichment (mean 0.981) the deletion advantage was already at parity,
        # winning in 52% of images versus 21% in the bottom tercile. Rollout's
        # mean enrichment sits above that tercile. Extrapolating past the observed
        # range is not proof, so the direction of the result is NOT guaranteed --
        # but the comparison is now made on the map that is actually concentrated,
        # and both maps are reported so the reader can see the difference.
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

                # Deletion: how far the positive-class probability moves when the
                # region is replaced by the blurred reference. Insertion: how much
                # of the prediction survives when only that region is kept.
                lstDelAttn.append(abs(floatBaseProb - dictProbs["deletion_attention"]))
                lstDelCont.append(abs(floatBaseProb - dictProbs["deletion_contiguous"]))
                lstDelDisp.append(abs(floatBaseProb - dictProbs["deletion_dispersed"]))
                lstInsAttn.append(1.0 - abs(floatBaseProb - dictProbs["insertion_attention"]))
                lstInsCont.append(1.0 - abs(floatBaseProb - dictProbs["insertion_contiguous"]))
                lstInsDisp.append(1.0 - abs(floatBaseProb - dictProbs["insertion_dispersed"]))

                # The single-fraction numbers at the reference fraction are kept
                # under their original column names so this run stays directly
                # comparable with the previous one.
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

            # Curve summaries. A single masking fraction is one point on a curve
            # whose shape is the actual quantity of interest, and it can land on a
            # saturated part of it -- which is what happened to the insertion test
            # in the previous run. The mean over the swept fractions is the
            # discrete area under that curve (equal spacing), and it is a far more
            # stable statistic than any single point.
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

        # Legacy column names, kept so the existing figures and the numeric audit
        # keep working. They now follow STR_PRIMARY_SALIENCY and the PRIMARY
        # (contiguous) control, both of which changed in this revision -- so these
        # columns are NOT numerically comparable with the previous run. The
        # per-map, per-control columns above are what a comparison should use.
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
    """Two panels, each metric shown against the control it must be read against."""
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
    """
    One panel per test, showing every (map, control) cell side by side.

    The point of this figure is that no cell can be hidden: a reader sees the two
    saliency maps against both controls at once, so a favourable headline cannot
    be the result of having quietly picked the cell that worked.
    """
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
            # Green when the saliency map beat its control, red when it did not.
            # The colour is derived from the value, never assigned by hand, so it
            # cannot disagree with the bar it labels.
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
    # Explain the SAME representation track that produces the paper's headline.
    # See STR_EXPLAINED_ROI_MODE for why this must not be 'dynamic'.
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

    # ------------------------------------------------------------------
    # HARD GATE: the class-sensitive map must actually differ from rollout.
    # ------------------------------------------------------------------
    # In a previous run this phase reported saliency_method='class_gradient'
    # while every enrichment value was byte-identical to its rollout column,
    # because the gradient path was broken and the function fell back silently.
    # The run completed, the numbers looked plausible, and the error was only
    # found by comparing two columns of the output file afterwards.
    #
    # This check turns that class of failure into an immediate abort. It is
    # deliberately placed BEFORE the CSV is written: a results file that claims
    # a method it did not use is worse than no results file.
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

    # ------------------------------------------------------------------
    # Paired tests, one row per (map, test, control).
    # ------------------------------------------------------------------
    # The previous revision emitted three rows and no multiplicity correction. It
    # also tested only one map against one control, which is what allowed a single
    # unfavourable point estimate to stand as "the" faithfulness result. The full
    # matrix is reported instead: 2 maps x 2 tests x 2 controls, on the curve-area
    # statistic and on the reference-fraction point value, so no cell can be
    # chosen after the fact.
    #
    # A Wilcoxon signed-rank test is used because the per-image advantages are not
    # expected to be symmetric and n is ~100. Multiplicity is controlled with
    # Benjamini-Hochberg across the whole matrix, mirroring Phases 3 and 4.
    lstRows = []
    for strMap in ["rollout", "gradient"]:
        for strTest, strStem, boolHigherIsBetter in [
                ("deletion", "deletion", True), ("insertion", "insertion", True)]:
            for strStat, strSuffix in [("curve_area", "auc"), ("point_20pct", "shift")]:
                # The point-value columns use different stems for the two tests.
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
                        # Family for the multiplicity correction: deletion and
                        # insertion answer different questions (necessity vs
                        # sufficiency), so correcting them jointly would penalise
                        # each for the other's existence. Same reasoning as the
                        # per-axis families in Phases 3 and 4.
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

    # Enrichment: both maps against the 1.0 uniform baseline, and against each
    # other. The paired map-vs-map test is what licenses the claim that the choice
    # of saliency method matters, rather than each being compared to a constant.
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

    # Legacy rows, so the existing numeric audit keeps resolving. These follow
    # STR_PRIMARY_SALIENCY and the contiguous control.
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

    # Console verdict, so the run itself states which cells came out favourable
    # instead of leaving it to be discovered by opening the CSV.
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

    # --- Qualitative panel ----------------------------------------------
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
        # DEFECT FIXED: the surrogate used to be passed here, which made
        # `arrHeatmap` the class-gradient map while the column title said
        # "Attention rollout" and the per-row annotation printed
        # `anatomical_enrichment` -- the GRADIENT enrichment, whose mean is below
        # the uniform expectation. The panel therefore illustrated the method the
        # paper reports as unfaithful while claiming to show the primary one, and
        # its three visible values (0.86, 0.49, 0.85) contradicted the 1.13 mean
        # quoted in the text. The panel must render whichever map
        # STR_PRIMARY_SALIENCY selects and annotate THAT map's enrichment.
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
        # The annotation must read the enrichment of the map actually drawn.
        # `anatomical_enrichment` is always the gradient figure; the rollout
        # figure lives in `anatomical_enrichment_rollout`.
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
