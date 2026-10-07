"""
Frozen encoder zoo for the multi-encoder comparison
===================================================

This module is the single place where the five frozen encoders of the study are
built and where their heterogeneous APIs are hidden behind one uniform
interface. Phase 2 imports `build_encoder` and nothing else.

WHY A UNIFORM INTERFACE IS NECESSARY
------------------------------------
The five backbones expose completely different APIs: DINOv2 has
`get_intermediate_layers`, timm models have `forward_intermediates` or hooks,
open_clip returns a projected embedding by default rather than tokens, and
ResNet-50 has no tokens at all. Writing those differences inline in Phase 2
would make the extraction loop unreadable and, worse, would make it easy to
apply subtly different pooling to different encoders — which would confound the
comparison the study exists to make.

Every encoder returned by `build_encoder` therefore obeys the same contract:

    encoder.forward_features(tsrBatch) -> (tsrCls, tsrPatch)
        tsrCls   : [B, D]        multi-layer pooled representation
        tsrPatch : [B, P, d]     patch (or spatial) tokens of the last stage
    encoder.preprocess(arrRgbUint8) -> [3, 224, 224] normalised tensor
    encoder.name, encoder.key, encoder.patch_grid, encoder.embed_dim

WHAT IS HELD CONSTANT ACROSS ENCODERS (this is the experimental control)
------------------------------------------------------------------------
  * the same cropped images (crops are computed ONCE by the reference encoder)
  * the same test-time augmentation, applied identically
  * the same multi-layer concatenation depth where the architecture allows it
  * the same downstream heads, folds, thresholds and statistics

What differs is ONLY the pre-trained weights and their required input
normalisation. That is what makes a difference between encoders attributable to
the representation rather than to the pipeline.

AVAILABILITY IS CHECKED, NOT ASSUMED
------------------------------------
Some of these weights live behind `timm`, `open_clip` or the HuggingFace hub and
may be unavailable offline or license-gated. `build_encoder` raises
`EncoderUnavailable` with an actionable message; Phase 2 catches it, records the
encoder as skipped in its JSON report and continues. A partially completed zoo
degrades the study gracefully instead of aborting a multi-hour run.

Execution Instructions:
-----------------------
This module is not run directly. It is imported by:
   $ python phases/phase2_features_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DICT_ENCODERS, get_torch_device

strDevice = get_torch_device()

# ImageNet statistics, used by DINOv2, RAD-DINO and ResNet-50.
TSR_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
TSR_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# OpenAI CLIP statistics. Using ImageNet statistics for a CLIP backbone is a
# silent accuracy loss that is easy to miss, which is why each encoder carries
# its own normalisation instead of sharing a global constant.
TSR_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
TSR_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


class EncoderUnavailable(RuntimeError):
    """Raised when a backbone's weights or its library cannot be obtained."""


class FrozenEncoder:
    """
    Uniform wrapper around a frozen backbone.

    The wrapped module is always in eval mode with gradients disabled: this
    study never fine-tunes, because with 148 patients a full backbone would
    overfit long before it improved.
    """

    def __init__(self, strKey, objModule, funForward, tsrMean, tsrStd):
        dictMeta = DICT_ENCODERS[strKey]
        self.key = strKey
        self.name = dictMeta["display_name"]
        self.patch_grid = dictMeta["patch_grid"]
        self.embed_dim = dictMeta["embed_dim"]
        self.family = dictMeta["family"]
        self.domain = dictMeta["domain"]
        self.objective = dictMeta["objective"]
        self.n_layers_concat = dictMeta["n_layers_concat"]

        self.module = objModule.to(strDevice).eval()
        for objParam in self.module.parameters():
            objParam.requires_grad_(False)
        self._forward = funForward
        self._mean = tsrMean
        self._std = tsrStd

    def preprocess(self, arrRgbUint8):
        """uint8 HxWx3 -> normalised float tensor [3, 224, 224]."""
        tsrImg = torch.from_numpy(np.ascontiguousarray(arrRgbUint8)).float()
        tsrImg = tsrImg.permute(2, 0, 1) / 255.0
        return (tsrImg - self._mean) / self._std

    @torch.no_grad()
    def forward_features(self, tsrBatch):
        """[B, 3, 224, 224] -> (cls [B, D], patch [B, P, d])."""
        return self._forward(self.module, tsrBatch.to(strDevice))

    def count_parameters(self):
        return int(sum(p.numel() for p in self.module.parameters()))


# --------------------------------------------------------------------------
# Per-encoder forward functions
# --------------------------------------------------------------------------
def _forward_dinov2(objModule, tsrBatch):
    """
    DINOv2 exposes intermediate layers directly.

    The multi-layer concatenation is retained from the original pipeline: the
    class token of the last four blocks is concatenated (4 x 768 = 3072). The
    ablation in Phase 3 tests whether that concatenation is worth its
    dimensionality.
    """
    lstInter = objModule.get_intermediate_layers(tsrBatch, n=4, return_class_token=True)
    tsrCls = torch.cat([objTuple[1] for objTuple in lstInter], dim=1)
    tsrPatch = lstInter[-1][0]
    return tsrCls, tsrPatch


def _forward_timm_vit(objModule, tsrBatch):
    """
    Generic path for timm vision transformers (CLIP, BiomedCLIP, RAD-DINO).

    timm's `forward_intermediates` is used when present because it is the
    supported API; otherwise forward hooks on the last four blocks provide the
    same tensors. Both branches produce the SAME concatenation order (shallow to
    deep) as the DINOv2 path, which matters because the downstream ablation
    slices these vectors by layer.
    """
    lstFeatures = []

    if hasattr(objModule, "forward_intermediates"):
        try:
            objOut = objModule.forward_intermediates(
                tsrBatch, indices=4, return_prefix_tokens=True,
                norm=True, output_fmt="NLC", intermediates_only=True)
            lstFeatures = list(objOut)
        except (TypeError, ValueError):
            lstFeatures = []

    if not lstFeatures:
        lstCaptured = []
        lstHandles = []

        def _make_hook(lstSink):
            def _hook(objMod, objIn, objOut):
                lstSink.append(objOut)
            return _hook

        lstBlocks = getattr(objModule, "blocks", None)
        if lstBlocks is None:
            raise EncoderUnavailable(
                "Backbone exposes neither forward_intermediates nor .blocks; "
                "cannot extract multi-layer tokens.")
        for objBlock in list(lstBlocks)[-4:]:
            lstHandles.append(objBlock.register_forward_hook(_make_hook(lstCaptured)))
        try:
            objModule(tsrBatch)
        finally:
            for objHandle in lstHandles:
                objHandle.remove()
        lstFeatures = lstCaptured

    # Each captured tensor is [B, 1 + P, d] (class token first) or a
    # (patch, prefix) tuple depending on the timm version.
    lstCls, tsrPatch = [], None
    for objFeat in lstFeatures:
        if isinstance(objFeat, (tuple, list)):
            tsrTokens, tsrPrefix = objFeat[0], objFeat[1]
            tsrClsOne = tsrPrefix[:, 0] if tsrPrefix.ndim == 3 else tsrPrefix
            tsrPatch = tsrTokens
        else:
            tsrClsOne = objFeat[:, 0]
            tsrPatch = objFeat[:, 1:]
        lstCls.append(tsrClsOne)

    if not lstCls:
        raise EncoderUnavailable("No intermediate tokens captured from backbone.")
    return torch.cat(lstCls, dim=1), tsrPatch


def _forward_resnet50(objModule, tsrBatch):
    """
    Convolutional floor.

    A CNN has no class token, so the analogue of "multi-layer concatenation" is
    global-average-pooling the last two STAGES and concatenating them
    (1024 + 2048 = 3072). That deliberately matches the DINOv2 vector width, so
    the comparison is not confounded by feature dimensionality.

    The spatial map of the final stage (7x7x2048) plays the role of patch tokens
    for the convolutional head in Phase 3.
    """
    tsrX = objModule.conv1(tsrBatch)
    tsrX = objModule.bn1(tsrX)
    tsrX = objModule.relu(tsrX)
    tsrX = objModule.maxpool(tsrX)
    tsrX = objModule.layer1(tsrX)
    tsrX = objModule.layer2(tsrX)
    tsrStage3 = objModule.layer3(tsrX)
    tsrStage4 = objModule.layer4(tsrStage3)

    tsrCls = torch.cat([
        torch.flatten(nn.functional.adaptive_avg_pool2d(tsrStage3, 1), 1),
        torch.flatten(nn.functional.adaptive_avg_pool2d(tsrStage4, 1), 1),
    ], dim=1)

    intB, intC, intH, intW = tsrStage4.shape
    tsrPatch = tsrStage4.reshape(intB, intC, intH * intW).permute(0, 2, 1)
    return tsrCls, tsrPatch


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_dinov2():
    try:
        objModule = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    except Exception as objError:
        raise EncoderUnavailable(
            f"DINOv2 could not be loaded via torch.hub ({objError}). "
            "It requires network access on first run; afterwards it is cached "
            "under ~/.cache/torch/hub.") from objError
    return FrozenEncoder("dinov2", objModule, _forward_dinov2,
                         TSR_IMAGENET_MEAN, TSR_IMAGENET_STD)


def load_clip():
    try:
        import timm
    except ImportError as objError:
        raise EncoderUnavailable(
            "timm is not installed. Install it with 'pip install timm' to "
            "include CLIP in the encoder comparison.") from objError
    try:
        objModule = timm.create_model(
            "vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0)
    except Exception as objError:
        raise EncoderUnavailable(
            f"CLIP ViT-B/16 weights unavailable ({objError}).") from objError
    return FrozenEncoder("clip", objModule, _forward_timm_vit,
                         TSR_CLIP_MEAN, TSR_CLIP_STD)


def load_biomedclip():
    """
    BiomedCLIP (Microsoft), pre-trained on PMC-15M biomedical figure-caption
    pairs. It is the medical-domain counterpart of CLIP, which is what turns the
    domain effect into a replicated observation rather than a single contrast.
    """
    try:
        import open_clip
    except ImportError as objError:
        raise EncoderUnavailable(
            "open_clip is not installed. Install it with 'pip install open_clip_torch'.") from objError

    try:
        model = open_clip.create_model('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        objModule = model.visual.trunk
        return FrozenEncoder("biomedclip", objModule, _forward_timm_vit,
                             TSR_CLIP_MEAN, TSR_CLIP_STD)
    except Exception as objError:
        raise EncoderUnavailable(
            f"BiomedCLIP weights unavailable ({objError}). It is distributed "
            "through the HuggingFace hub; 'pip install open_clip_torch' plus network "
            "access on first run is the usual fix. Verify the licence before use.")


def load_raddino():
    """
    RAD-DINO (Microsoft), DINOv2-style self-supervised pre-training on chest
    radiographs. Same objective as DINOv2, medical domain: the cleanest
    available isolation of the domain axis.
    """
    try:
        from transformers import AutoModel
    except ImportError as objError:
        raise EncoderUnavailable(
            "transformers is not installed. Install it with "
            "'pip install transformers' to include RAD-DINO.") from objError
    try:
        objHfModel = AutoModel.from_pretrained("microsoft/rad-dino")
    except Exception as objError:
        raise EncoderUnavailable(
            f"RAD-DINO weights unavailable ({objError}). Verify network access "
            "and the model licence on the HuggingFace hub.") from objError

    def _forward_raddino(objModule, tsrBatch):
        objOut = objModule(pixel_values=tsrBatch, output_hidden_states=True)
        lstHidden = objOut.hidden_states[-4:]
        tsrCls = torch.cat([h[:, 0] for h in lstHidden], dim=1)
        return tsrCls, lstHidden[-1][:, 1:]

    return FrozenEncoder("raddino", objHfModel, _forward_raddino,
                         TSR_IMAGENET_MEAN, TSR_IMAGENET_STD)


def load_resnet50():
    """Conventional ImageNet transfer-learning floor."""
    try:
        from torchvision.models import ResNet50_Weights, resnet50
        objModule = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception as objError:
        raise EncoderUnavailable(
            f"ResNet-50 ImageNet weights unavailable ({objError}).") from objError
    return FrozenEncoder("resnet50", objModule, _forward_resnet50,
                         TSR_IMAGENET_MEAN, TSR_IMAGENET_STD)


DICT_LOADERS = {
    "dinov2": load_dinov2,
    "clip": load_clip,
    "biomedclip": load_biomedclip,
    "raddino": load_raddino,
    "resnet50": load_resnet50,
}


def build_encoder(strKey):
    """Builds one encoder by registry key. Raises EncoderUnavailable."""
    if strKey not in DICT_LOADERS:
        raise KeyError(f"Unknown encoder '{strKey}'. "
                       f"Available: {sorted(DICT_LOADERS)}")
    return DICT_LOADERS[strKey]()


def apply_tta(tsrImage):
    """
    Test-time augmentation, IDENTICAL for every encoder.

    Five views: identity, horizontal flip, +/-5 degree rotations and a mild
    brightness change. Applying different augmentation to different encoders
    would confound the comparison, so this lives here rather than in each
    extraction path.
    """
    return torch.stack([
        tsrImage,
        TF.hflip(tsrImage),
        TF.rotate(tsrImage, 5),
        TF.rotate(tsrImage, -5),
        TF.adjust_brightness(tsrImage, 1.1),
    ], dim=0)