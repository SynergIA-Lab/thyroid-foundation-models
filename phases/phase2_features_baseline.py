import gc
import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (apply_manuscript_style, encoder_metadata_frame,
                    export_manuscript_table, feature_file, get_torch_device,
                    INT_GLOBAL_SEED, load_balanced_table, LST_ENCODER_ORDER,
                    LST_PALETTE, pathBalancedCsv, save_figure, set_global_seed,
                    STR_REFERENCE_ENCODER, TeeLogger)
from encoders import apply_tta, build_encoder, EncoderUnavailable

LST_ROI_MODES = ["fixed", "dynamic"]
LST_ENCODERS_TO_RUN = [s.strip() for s in
                       os.environ.get("THYROID_ENCODERS", ",".join(LST_ENCODER_ORDER)).split(",")
                       if s.strip()]

pathDataDir = Path("data")
pathPhase1Dir = Path("outputs/phase1")
pathPhase2Dir = Path("outputs/phase2")
pathPhase2Dir.mkdir(parents=True, exist_ok=True)

set_global_seed(INT_GLOBAL_SEED)
strDevice = get_torch_device()
torch.set_grad_enabled(False)
print(f"DINOv2 Inference Hardware Device: {strDevice}")

def get_dinov2_model():
    print("Loading pre-trained DINOv2 (vitb14) model via PyTorch Hub...")
    objModel = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    objModel = objModel.to(strDevice)
    objModel.eval()
    return objModel

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
    
    intColor = [0, 0, 0] if len(arrImage.shape) == 3 else 0
    return cv2.copyMakeBorder(arrResized, intTop, intBottom, intLeft, intRight, cv2.BORDER_CONSTANT, value=intColor)

def crop_thyroid_roi_fixed(arrImage, tupleTargetSize=(224, 224)):
    intHeight, intWidth = arrImage.shape[:2]
    intStartY = int(intHeight * 0.20)
    intEndY = int(intHeight * 0.85)
    intStartX = int(intWidth * 0.15)
    intEndX = int(intWidth * 0.85)
    arrCropped = arrImage[intStartY:intEndY, intStartX:intEndX]
    if arrCropped.size == 0:
        return resize_with_padding(arrImage, tupleTargetSize)
    return resize_with_padding(arrCropped, tupleTargetSize)

def get_dynamic_roi_bbox(arrImage, objModel):
    arrPadded = resize_with_padding(arrImage, (224, 224))
    arrRgb = cv2.cvtColor(arrPadded, cv2.COLOR_GRAY2RGB)
    tsrImg = torch.from_numpy(arrRgb).float().permute(2, 0, 1) / 255.0
    tsrMean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tsrStd = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tsrImg = (tsrImg - tsrMean) / tsrStd
    tsrImg = tsrImg.unsqueeze(0).to(strDevice)
    
    with torch.no_grad():
        x = objModel.prepare_tokens_with_masks(tsrImg)
        for i, blk in enumerate(objModel.blocks):
            if i < len(objModel.blocks) - 1:
                x = blk(x)
                
        blk = objModel.blocks[-1]
        x_norm = blk.norm1(x)
        
        B, N, C = x_norm.shape
        num_heads = blk.attn.num_heads
        head_dim = C // num_heads
        
        qkv = blk.attn.qkv(x_norm).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        tsrAttn = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
        tsrAttn = torch.softmax(tsrAttn, dim=-1)
        
    tsrClsAttn = tsrAttn[0, :, 0, 1:]
    arrAvgAttn = tsrClsAttn.mean(dim=0).cpu().numpy().reshape(16, 16)
    
    arrAvgAttn = (arrAvgAttn - arrAvgAttn.min()) / (arrAvgAttn.max() - arrAvgAttn.min() + 1e-8)
    
    floatThresh = np.percentile(arrAvgAttn, 70)
    arrMask = (arrAvgAttn > floatThresh).astype(np.uint8)
    
    intHeight, intWidth = arrImage.shape[:2]
    arrMaskResized = cv2.resize(arrMask, (intWidth, intHeight), interpolation=cv2.INTER_NEAREST)
    
    lstCoords = cv2.findNonZero(arrMaskResized)
    if lstCoords is not None:
        x, y, w, h = cv2.boundingRect(lstCoords)
        pad_x = int(w * 0.10)
        pad_y = int(h * 0.10)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(intWidth, x + w + pad_x)
        y2 = min(intHeight, y + h + pad_y)
        arrCropped = arrImage[y1:y2, x1:x2]
        return resize_with_padding(arrCropped, (224, 224)), arrAvgAttn
    else:
        return resize_with_padding(arrImage, (224, 224)), arrAvgAttn

class ThyroidDataset(Dataset):
    def __init__(self, dfData, pathBaseDir, objModel, strMode="fixed"):
        self.dfData = dfData
        self.pathBaseDir = pathBaseDir
        self.objModel = objModel
        self.strMode = strMode

    def __len__(self):
        return len(self.dfData)

    def __getitem__(self, intIndex):
        serRow = self.dfData.iloc[intIndex]
        pathImg = self.pathBaseDir / serRow['class'] / serRow['view'] / serRow['filename']
        
        arrImgGrayscale = cv2.imread(str(pathImg), cv2.IMREAD_GRAYSCALE)
        
        if self.strMode == "fixed":
            arrRoiImg = crop_thyroid_roi_fixed(arrImgGrayscale)
        else:
            arrRoiImg, _ = get_dynamic_roi_bbox(arrImgGrayscale, self.objModel)
            
        arrRoiRgb = cv2.cvtColor(arrRoiImg, cv2.COLOR_GRAY2RGB)
        
        tsrImage = torch.from_numpy(arrRoiRgb).float().permute(2, 0, 1) / 255.0
        tsrMean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tsrStd = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tsrImageNorm = (tsrImage - tsrMean) / tsrStd
        
        tsr1 = tsrImageNorm
        tsr2 = TF.hflip(tsrImageNorm)
        tsr3 = TF.rotate(tsrImageNorm, 5)
        tsr4 = TF.rotate(tsrImageNorm, -5)
        tsr5 = TF.adjust_brightness(tsrImageNorm, 1.1)
        
        tsrTTA = torch.stack([tsr1, tsr2, tsr3, tsr4, tsr5], dim=0)
        
        return tsrTTA, serRow['class']

def precompute_roi_crops(dfMetadata, objReferenceModel):
    pathCache = pathPhase2Dir / "phase2_roi_crops.npz"
    arrSampleIds = np.asarray(dfMetadata["sample_id"].astype(str).tolist(), dtype=np.str_)

    if pathCache.exists():
        dictCache = np.load(pathCache, allow_pickle=False)
        if (len(dictCache["sample_ids"]) == len(arrSampleIds)
                and bool(np.all(np.asarray(dictCache["sample_ids"], dtype=str) == arrSampleIds))):
            print(f"[crops] Reusing cached ROI crops: {pathCache.name}")
            return dictCache["fixed"], dictCache["dynamic"]
        print("[crops] Cache does not match the current cohort; recomputing.")

    print("\n--- Pre-computing ROI crops (shared by every encoder) ---")
    lstFixed, lstDynamic = [], []
    for _, serRow in tqdm(dfMetadata.iterrows(), total=len(dfMetadata), desc="crops"):
        pathImg = pathDataDir / serRow["class"] / serRow["view"] / serRow["filename"]
        arrGray = cv2.imread(str(pathImg), cv2.IMREAD_GRAYSCALE)
        if arrGray is None:
            raise FileNotFoundError(f"Unreadable image: {pathImg}")
        arrFixed = crop_thyroid_roi_fixed(arrGray)
        arrDynamic, _ = get_dynamic_roi_bbox(arrGray, objReferenceModel)
        lstFixed.append(cv2.cvtColor(arrFixed, cv2.COLOR_GRAY2RGB))
        lstDynamic.append(cv2.cvtColor(arrDynamic, cv2.COLOR_GRAY2RGB))

    arrFixedAll = np.stack(lstFixed).astype(np.uint8)
    arrDynamicAll = np.stack(lstDynamic).astype(np.uint8)
    np.savez_compressed(pathCache, fixed=arrFixedAll, dynamic=arrDynamicAll,
                        sample_ids=arrSampleIds)
    print(f"[crops] Cached to {pathCache.name} "
          f"(fixed={arrFixedAll.shape}, dynamic={arrDynamicAll.shape})")
    return arrFixedAll, arrDynamicAll


class CropDataset(Dataset):
    def __init__(self, arrCrops, objEncoder):
        self.arrCrops = arrCrops
        self.objEncoder = objEncoder

    def __len__(self):
        return len(self.arrCrops)

    def __getitem__(self, intIndex):
        tsrImage = self.objEncoder.preprocess(self.arrCrops[intIndex])
        return apply_tta(tsrImage), intIndex


def extract_features_for_encoder(objEncoder, strMode, arrCrops, dfMetadata):
    print(f"\n--- {objEncoder.name} | {strMode.upper()} ROI ---")
    objDataLoader = DataLoader(CropDataset(arrCrops, objEncoder),
                               batch_size=8, shuffle=False, num_workers=0)

    lstCls, lstPatch = [], []
    floatStart = time.time()
    with torch.no_grad():
        for tsrBatch, _ in tqdm(objDataLoader, desc=f"{objEncoder.key}/{strMode}"):
            intBatch = tsrBatch.shape[0]
            tsrFlat = tsrBatch.view(-1, 3, 224, 224)
            tsrCls, tsrPatch = objEncoder.forward_features(tsrFlat)
            # Mean-pool over the five test-time augmented views.
            lstCls.append(tsrCls.view(intBatch, 5, -1).mean(dim=1).cpu().numpy())
            tsrPatch = tsrPatch.view(intBatch, 5, tsrPatch.shape[1],
                                     tsrPatch.shape[2]).mean(dim=1)
            lstPatch.append(tsrPatch.cpu().numpy().astype(np.float32))
    floatElapsed = time.time() - floatStart

    arrCls = np.concatenate(lstCls, axis=0)
    arrPatch = np.concatenate(lstPatch, axis=0)
    if len(arrCls) != len(dfMetadata):
        raise RuntimeError(
            f"{objEncoder.key}/{strMode}: extracted {len(arrCls)} embeddings for "
            f"{len(dfMetadata)} metadata rows.")

    arrL2 = np.linalg.norm(arrCls, axis=1)
    dictMetrics = {
        "encoder": objEncoder.key,
        "encoder_display_name": objEncoder.name,
        "roi_mode": strMode,
        "pre_training_domain": objEncoder.domain,
        "pre_training_objective": objEncoder.objective,
        "architecture_family": objEncoder.family,
        "parameters_millions": round(objEncoder.count_parameters() / 1e6, 2),
        "embedding_dim": int(arrCls.shape[1]),
        "patch_tokens": int(arrPatch.shape[1]),
        "extraction_time_seconds": float(floatElapsed),
        "latency_per_image_ms": float(floatElapsed / len(dfMetadata) * 1000.0),
        "cls_l2_norm_mean": float(np.mean(arrL2)),
        "cls_l2_norm_std": float(np.std(arrL2)),
        "tensor_memory_mb": float((arrCls.nbytes + arrPatch.nbytes) / (1024 * 1024)),
    }

    arrSampleIds = np.asarray(dfMetadata["sample_id"].astype(str).tolist(), dtype=np.str_)
    pathOut = feature_file(strMode, objEncoder.key)
    np.savez_compressed(pathOut, cls=arrCls, patch=arrPatch,
                        sample_ids=arrSampleIds,
                        encoder=np.asarray(objEncoder.key, dtype=np.str_),
                        cohort_table=np.asarray(pathBalancedCsv.as_posix(), dtype=np.str_))
    print(f"[{objEncoder.key}/{strMode}] saved {pathOut.name} "
          f"(cls={arrCls.shape}, patch={arrPatch.shape}, "
          f"{dictMetrics['latency_per_image_ms']:.0f} ms/image)")
    return arrCls, arrPatch, dictMetrics


def extract_features(strMode, dfMetadata, objModel):
    print(f"\n--- Extracting Features: {strMode.upper()} ROI ---")
    objDataset = ThyroidDataset(dfMetadata, pathDataDir, objModel, strMode=strMode)
    objDataLoader = DataLoader(objDataset, batch_size=8, shuffle=False, num_workers=0)
    
    lstClsTokens = []
    lstPatchTokens = []
    
    floatStartTime = time.time()
    with torch.no_grad():
        for tsrBatchImgs, _ in tqdm(objDataLoader):
            intBatchSize = tsrBatchImgs.shape[0]
            tsrBatchImgs = tsrBatchImgs.view(-1, 3, 224, 224).to(strDevice)
            
            lstInter = objModel.get_intermediate_layers(tsrBatchImgs, n=4, return_class_token=True)
            
            lst_layer_cls = [layer[1] for layer in lstInter] 
            tsrConcatCls = torch.cat(lst_layer_cls, dim=1)   
            
            tsrConcatCls = tsrConcatCls.view(intBatchSize, 5, -1).mean(dim=1) 
            lstClsTokens.append(tsrConcatCls.cpu().numpy())
            
            tsrPatch = lstInter[-1][0] 
            tsrPatch = tsrPatch.view(intBatchSize, 5, tsrPatch.shape[1], tsrPatch.shape[2]).mean(dim=1) 
            lstPatchTokens.append(tsrPatch.cpu().numpy())
    floatEndTime = time.time()
            
    arrCls = np.concatenate(lstClsTokens, axis=0)
    arrPatch = np.concatenate(lstPatchTokens, axis=0)
    
    floatTotalTime = floatEndTime - floatStartTime
    floatTimePerImgMs = (floatTotalTime / len(dfMetadata)) * 1000.0
    arrL2Norms = np.linalg.norm(arrCls, axis=1)
    dictMetrics = {
        "extraction_time_seconds": float(floatTotalTime),
        "latency_per_image_ms": float(floatTimePerImgMs),
        "cls_l2_norm_mean": float(np.mean(arrL2Norms)),
        "cls_l2_norm_std": float(np.std(arrL2Norms)),
        "tensor_memory_mb": float((arrCls.nbytes + arrPatch.nbytes) / (1024 * 1024))
    }
    
    if len(arrCls) != len(dfMetadata):
        raise RuntimeError(
            f"Extracted {len(arrCls)} embeddings for {len(dfMetadata)} metadata rows."
        )
    
    arrSampleIds = np.asarray(dfMetadata['sample_id'].astype(str).tolist(), dtype=np.str_)
    pathOutputNpz = pathPhase2Dir / f"dinov2_features_{strMode}.npz"
    np.savez_compressed(pathOutputNpz, cls=arrCls, patch=arrPatch,
                        sample_ids=arrSampleIds,
                        cohort_table=np.asarray(pathBalancedCsv.as_posix(), dtype=np.str_))
    print(f"[{strMode.upper()}] Features saved to: {pathOutputNpz} "
          f"(cls={arrCls.shape}, patch={arrPatch.shape})")
    return arrCls, arrPatch, dictMetrics

def extract_roi_metrics(dfMetadata, objModel):
    print("\n--- Extracting Quantitative ROI Metrics ---")
    lstMetrics = []
    
    for intIdx, serRow in tqdm(dfMetadata.iterrows(), total=len(dfMetadata)):
        pathImg = pathDataDir / serRow['class'] / serRow['view'] / serRow['filename']
        arrImage = cv2.imread(str(pathImg), cv2.IMREAD_GRAYSCALE)
        
        intHeight, intWidth = arrImage.shape[:2]
        intStartY = int(intHeight * 0.20)
        intEndY = int(intHeight * 0.85)
        intStartX = int(intWidth * 0.15)
        intEndX = int(intWidth * 0.85)
        
        arrMaskFixed = np.zeros((intHeight, intWidth), dtype=np.uint8)
        arrMaskFixed[intStartY:intEndY, intStartX:intEndX] = 1

        arrPadded = resize_with_padding(arrImage, (224, 224))
        arrRgb = cv2.cvtColor(arrPadded, cv2.COLOR_GRAY2RGB)
        tsrImg = torch.from_numpy(arrRgb).float().permute(2, 0, 1) / 255.0
        tsrMean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tsrStd = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tsrImg = (tsrImg - tsrMean) / tsrStd
        tsrImg = tsrImg.unsqueeze(0).to(strDevice)
        
        with torch.no_grad():
            x = objModel.prepare_tokens_with_masks(tsrImg)
            for i, blk in enumerate(objModel.blocks):
                if i < len(objModel.blocks) - 1: x = blk(x)
            blk = objModel.blocks[-1]
            x_norm = blk.norm1(x)
            B, N, C = x_norm.shape
            num_heads = blk.attn.num_heads
            head_dim = C // num_heads
            qkv = blk.attn.qkv(x_norm).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            tsrAttn = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
            tsrAttn = torch.softmax(tsrAttn, dim=-1)
            
        tsrClsAttn = tsrAttn[0, :, 0, 1:]
        arrAvgAttn = tsrClsAttn.mean(dim=0).cpu().numpy().reshape(16, 16)
        arrAvgAttn = (arrAvgAttn - arrAvgAttn.min()) / (arrAvgAttn.max() - arrAvgAttn.min() + 1e-8)
        
        floatThresh = np.percentile(arrAvgAttn, 70)
        arrMaskDyn16 = (arrAvgAttn > floatThresh).astype(np.uint8)
        arrMaskDyn = cv2.resize(arrMaskDyn16, (intWidth, intHeight), interpolation=cv2.INTER_NEAREST)
        
        arrIntersection = (arrMaskFixed & arrMaskDyn)
        arrUnion = (arrMaskFixed | arrMaskDyn)
        floatIoU = arrIntersection.sum() / (arrUnion.sum() + 1e-8)
        
        floatAreaFixed = arrMaskFixed.sum() / (intHeight * intWidth)
        floatAreaDyn = arrMaskDyn.sum() / (intHeight * intWidth)
        
        floatSignalFixed = arrImage[arrMaskFixed == 1].mean() if arrMaskFixed.sum() > 0 else 0
        floatBgFixed = arrImage[arrMaskFixed == 0].mean() if (arrMaskFixed == 0).sum() > 0 else 0
        floatSBRFixed = floatSignalFixed / (floatBgFixed + 1e-8)
        
        floatSignalDyn = arrImage[arrMaskDyn == 1].mean() if arrMaskDyn.sum() > 0 else 0
        floatBgDyn = arrImage[arrMaskDyn == 0].mean() if (arrMaskDyn == 0).sum() > 0 else 0
        floatSBRDyn = floatSignalDyn / (floatBgDyn + 1e-8)
        
        lstCoords = cv2.findNonZero(arrMaskDyn)
        if lstCoords is not None:
            x, y, w, h = cv2.boundingRect(lstCoords)
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.10)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(intWidth, x + w + pad_x)
            y2 = min(intHeight, y + h + pad_y)
            arrCroppedDyn = arrImage[y1:y2, x1:x2]
        else:
            arrCroppedDyn = arrImage
            
        arrCroppedFixed = arrImage[intStartY:intEndY, intStartX:intEndX]
        
        def compute_entropy(arr):
            if arr.size == 0: return 0
            arrHist = cv2.calcHist([arr], [0], None, [256], [0, 256])
            arrHist = arrHist.ravel() / arrHist.sum()
            arrHist = arrHist[arrHist > 0]
            return -np.sum(arrHist * np.log2(arrHist))
            
        floatEntropyFixed = float(compute_entropy(arrCroppedFixed))
        floatEntropyDyn = float(compute_entropy(arrCroppedDyn))
        
        intNumComponents, _ = cv2.connectedComponents(arrMaskDyn)
        intFragmentation = max(1, intNumComponents - 1)
        
        dictMetrics = {
            "sample_id": serRow["sample_id"],
            "patient_id": serRow["patient_id"],
            "class": serRow["class"],
            "view": serRow["view"],
            "iou_fixed_dynamic": float(floatIoU),
            "relative_area_fixed": float(floatAreaFixed),
            "relative_area_dynamic": float(floatAreaDyn),
            "sbr_fixed": float(floatSBRFixed),
            "sbr_dynamic": float(floatSBRDyn),
            "entropy_fixed": floatEntropyFixed,
            "entropy_dynamic": floatEntropyDyn,
            "fragmentation_dynamic": intFragmentation
        }
        lstMetrics.append(dictMetrics)
        
    dfMetrics = pd.DataFrame(lstMetrics)
    pathMetricsCsv = pathPhase2Dir / "phase2_roi_quantitative_metrics.csv"
    dfMetrics.to_csv(pathMetricsCsv, index=False)
    print(f"ROI Quantitative Metrics saved to: {pathMetricsCsv}")
    
    dfSummary = dfMetrics.groupby("class").agg({
        "iou_fixed_dynamic": ["mean", "std"],
        "relative_area_fixed": ["mean", "std"],
        "relative_area_dynamic": ["mean", "std"],
        "sbr_fixed": ["mean", "std"],
        "sbr_dynamic": ["mean", "std"],
        "entropy_fixed": ["mean", "std"],
        "entropy_dynamic": ["mean", "std"],
        "fragmentation_dynamic": ["mean", "std"]
    }).reset_index()
    
    dfSummary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in dfSummary.columns.values]
    pathSummaryCsv = pathPhase2Dir / "phase2_roi_metrics_summary_by_class.csv"
    dfSummary.to_csv(pathSummaryCsv, index=False)
    print(f"ROI Summary Metrics saved to: {pathSummaryCsv}")

def main():
    sys.stdout = TeeLogger(pathPhase2Dir / "phase2_run.log")
    apply_manuscript_style()
    
    dfMetadata = load_balanced_table()
    print(f"Analysis cohort: {len(dfMetadata)} images from "
          f"{dfMetadata['patient_id'].nunique()} patients.")
    
    objReference = build_encoder(STR_REFERENCE_ENCODER)
    print(f"Reference encoder: {objReference.name} "
          f"({objReference.count_parameters()/1e6:.1f}M parameters)")

    arrCropsFixed, arrCropsDynamic = precompute_roi_crops(dfMetadata, objReference.module)
    dictCropsByMode = {"fixed": arrCropsFixed, "dynamic": arrCropsDynamic}

    extract_roi_metrics(dfMetadata, objReference.module)

    lstEncoderMetrics = []
    dictSkipped = {}
    for strEncoderKey in LST_ENCODERS_TO_RUN:
        try:
            objEncoder = (objReference if strEncoderKey == STR_REFERENCE_ENCODER
                          else build_encoder(strEncoderKey))
        except EncoderUnavailable as objError:
            dictSkipped[strEncoderKey] = str(objError)
            print(f"\n[SKIP] {strEncoderKey}: {objError}")
            continue

        for strMode in LST_ROI_MODES:
            _, _, dictMetrics = extract_features_for_encoder(
                objEncoder, strMode, dictCropsByMode[strMode], dfMetadata)
            lstEncoderMetrics.append(dictMetrics)

        if strEncoderKey != STR_REFERENCE_ENCODER:
            del objEncoder
            gc.collect()

    dfEncoderMetrics = pd.DataFrame(lstEncoderMetrics)
    export_manuscript_table(
        dfEncoderMetrics,
        pathPhase2Dir / "phase2_encoder_extraction_metrics",
        strCaption="Frozen encoders evaluated in this study: pre-training "
                   "characteristics, embedding geometry and CPU extraction cost.",
        strLabel="tab:encoders")

    dictReference = {m["roi_mode"]: m for m in lstEncoderMetrics
                     if m["encoder"] == STR_REFERENCE_ENCODER}
    dictHardwareMetrics = {
        "hardware_device": strDevice,
        "reference_encoder": STR_REFERENCE_ENCODER,
        "model_architecture": "dinov2_vitb14_multilayer",
        "roi_crops_shared_across_encoders": True,
        "roi_crop_provenance": (
            "Both ROI crops are computed once by the reference encoder and reused "
            "by every other encoder, so that the ROI axis and the encoder axis "
            "remain separable and the convolutional floor (which has no attention "
            "map) can participate."),
        "total_patients": int(dfMetadata['patient_id'].nunique()),
        "total_images": len(dfMetadata),
        "encoders_evaluated": sorted({m["encoder"] for m in lstEncoderMetrics}),
        "encoders_skipped": dictSkipped,
        "per_encoder_metrics": lstEncoderMetrics,
    }
    if "fixed" in dictReference and "dynamic" in dictReference:
        dictHardwareMetrics["fixed_roi_metrics"] = dictReference["fixed"]
        dictHardwareMetrics["dynamic_roi_metrics"] = dictReference["dynamic"]
        dictHardwareMetrics["overhead_dynamic_ms"] = float(
            dictReference["dynamic"]["latency_per_image_ms"]
            - dictReference["fixed"]["latency_per_image_ms"])

    pathMetricsJson = pathPhase2Dir / "phase2_computational_metrics.json"
    with open(pathMetricsJson, "w", encoding="utf-8") as f:
        json.dump(dictHardwareMetrics, f, indent=4)
    print(f"\nComputational metrics saved to: {pathMetricsJson}")

    export_manuscript_table(
        encoder_metadata_frame(),
        pathPhase2Dir / "phase2_encoder_registry",
        strCaption="Pre-training domain and objective of each frozen encoder. "
                   "The two axes are crossed so that domain and objective "
                   "effects can be separated.",
        strLabel="tab:encoder-registry")

    if not dfEncoderMetrics.empty:
        objFig, objAx = plt.subplots(figsize=(6.6, 3.6))
        dfPivot = dfEncoderMetrics.pivot_table(
            index="encoder_display_name", columns="roi_mode",
            values="latency_per_image_ms")
        arrX = np.arange(len(dfPivot))
        for intIdx, strMode in enumerate(dfPivot.columns):
            objAx.bar(arrX + (intIdx - 0.5) * 0.36, dfPivot[strMode], 0.34,
                      label=f"{strMode} ROI", color=LST_PALETTE[intIdx],
                      edgecolor="black", linewidth=0.5)
        objAx.set_xticks(arrX)
        objAx.set_xticklabels(dfPivot.index, rotation=20, ha="right")
        objAx.set_ylabel("Latency per image (ms, CPU)")
        objAx.set_title("Feature-extraction cost per encoder")
        objAx.legend(frameon=False)
        objFig.tight_layout()
        save_figure(objFig, pathPhase2Dir / "phase2_encoder_cost")
        plt.close(objFig)
    
    serDemoRow = dfMetadata.iloc[0]
    pathDemoImg = pathDataDir / serDemoRow['class'] / serDemoRow['view'] / serDemoRow['filename']
    
    arrOriginal = cv2.imread(str(pathDemoImg), cv2.IMREAD_GRAYSCALE)
    arrCroppedFixed = crop_thyroid_roi_fixed(arrOriginal)
    arrCroppedDynamic, arrAttnMap = get_dynamic_roi_bbox(arrOriginal, objReference.module)
    
    objFig, arrAxes = plt.subplots(1, 4, figsize=(15, 4.4))
    for objAx, (arrPanel, strTitle, strCmap) in zip(arrAxes, [
            (arrOriginal, 'Original scintigram', 'gray'),
            (arrCroppedFixed, 'Track A: fixed geometric ROI', 'gray'),
            (arrAttnMap, 'DINOv2 CLS self-attention', 'inferno'),
            (arrCroppedDynamic, 'Track B: attention-derived ROI', 'gray')]):
        objAx.imshow(arrPanel, cmap=strCmap)
        objAx.set_title(strTitle, fontsize=10)
        objAx.axis('off')
    
    objFig.suptitle(f"Region-of-interest extraction tracks (case {serDemoRow['patient_id']}, "
                    f"{serDemoRow['view']} projection)", fontsize=11)
    objFig.tight_layout()
    save_figure(objFig, pathPhase2Dir / "phase2_roi_comparison")
    plt.close(objFig)
    
if __name__ == "__main__":
    main()
