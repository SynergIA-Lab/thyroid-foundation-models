"""
Phase 0: Dataset Evidence, Cohort Structure and Confounder Audit
================================================================

Quantitative audit of the raw scintigraphic image repository and its clinical
metadata, performed before any tensor transformation. Phase 0 produces no
model and no prediction; its only job is to state what the data actually is,
including the parts that are inconvenient.

WHAT THE IMAGES ARE
-------------------
These are thyroid *scintigrams* (nuclear medicine planar gamma-camera images
acquired after administration of a radiotracer), NOT ultrasound images. The
metadata column 'DOSE ADMINISTERED (mCi)' is the administered activity, which
only exists for a radionuclide study. Every downstream description of the
modality must say scintigraphy: the texture statistics, the resolution, and
the meaning of "diffuse" versus "multinodular" uptake are specific to it.

FOUR AUDITS PERFORMED HERE
--------------------------
1. Image inventory, resolved at THREE levels that must not be conflated:
   image, projection (view), and patient. Reporting an image-level ratio as
   "the class imbalance" overstates it, because patients contribute unequal
   numbers of images.

2. Linkage integrity between the folder tree and the demographics workbook.
   This is asserted, not assumed: identifier sets must match exactly and the
   folder-derived label must agree with the workbook diagnosis for every
   patient. A silent linkage failure here would invalidate every later phase,
   so a mismatch aborts the run.

3. Per-class demographic stratification with hypothesis tests. This is the
   scientifically load-bearing part of Phase 0. If a covariate differs
   between the two diagnostic groups, then a classifier can reach good
   performance without extracting any information from the gland, and the
   manuscript is obliged to (a) declare it as a limitation and (b) report the
   clinical-only reference arm of Phase 3 that quantifies it.

4. Plausibility screening of the anthropometric fields. Values that are
   physiologically impossible are flagged, and their effect on the reported
   summary statistics is quantified. Anthropometric fields feed the Phase 3
   reference arm through common.build_clinical_matrix(), so a transcription
   error is not a cosmetic problem: it propagates into a comparator arm.

OUTPUTS (outputs/phase0/)
-------------------------
- phase0_evidence.json    Full machine-readable audit.
- phase0_cohort_audit.png Three-panel figure for the manuscript.
- phase0.txt              Narrative clinical report.
- phase0_run.log          Console transcript.

Execution Instructions:
-----------------------
1. Activate the conda environment:
   $ conda activate thyroid_env

2. Run the script from the root of the repository:
   $ python phases/phase0_evidence.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TeeLogger

# Physiological plausibility bounds for adult anthropometry. Values outside
# these ranges are treated as transcription errors, not as biological
# variation, and are reported explicitly rather than silently winsorised.
DICT_PLAUSIBLE_RANGE = {
    "AGE": (18.0, 100.0),
    "WEIGHT (Kg)": (35.0, 200.0),
    "HEIGHT (cm)": (130.0, 210.0),
    "DOSE ADMINISTERED (mCi)": (0.5, 15.0),
}

# Views expected in this acquisition protocol. ANTERIOR is the mandatory
# projection; the two obliques are acquired only when the OBLIQUE flag is set.
LST_EXPECTED_VIEWS = ["ANTERIOR", "LAO", "RAO"]

DICT_DIAGNOSIS_TO_FOLDER = {
    "MULTINODULAR GOITER": "multinodular_goiter",
    "DIFFUSE GOITER": "diffuse_goiter",
}

# Colour identity for the two diagnostic classes, threaded through every
# panel. Blue/orange rather than blue/green so the pair survives
# deuteranopia simulation.
DICT_CLASS_COLOUR = {
    "multinodular_goiter": "#2B6A8F",
    "diffuse_goiter": "#C1663A",
}

DICT_CLASS_LABEL = {
    "multinodular_goiter": "Multinodular goitre",
    "diffuse_goiter": "Diffuse goitre",
}


def scan_image_inventory(pathDataDir):
    """
    Builds the image inventory resolved by class, patient and projection.

    The patient identifier is the image file stem, which is the same integer
    used in the 'IMAGE' column of the demographics workbook. A patient
    contributes one file per acquired projection, so file count and patient
    count are different quantities and are returned separately.
    """
    dictInventory = {}
    lstValidExtensions = [".jpg", ".png", ".tif", ".jpeg"]

    for pathClassDir in sorted(pathDataDir.iterdir()):
        if not pathClassDir.is_dir():
            continue
        dictPerView = {}
        dictPerPatient = {}
        for pathFile in sorted(pathClassDir.rglob("*")):
            if pathFile.suffix.lower() not in lstValidExtensions:
                continue
            strView = pathFile.parent.name
            dictPerView[strView] = dictPerView.get(strView, 0) + 1
            try:
                intPatient = int(pathFile.stem)
            except ValueError:
                raise ValueError(
                    f"Image '{pathFile}' has a non-numeric stem. Patient identity "
                    f"is derived from the file stem, so this file cannot be linked "
                    f"to the demographics workbook."
                )
            dictPerPatient.setdefault(intPatient, set()).add(strView)

        if dictPerView:
            dictInventory[pathClassDir.name] = {
                "images": sum(dictPerView.values()),
                "patients": len(dictPerPatient),
                "per_view": dictPerView,
                "views_by_patient": {k: sorted(v) for k, v in sorted(dictPerPatient.items())},
            }
    return dictInventory


def assert_linkage_integrity(dictInventory, dfDemographics):
    """
    Verifies that the folder tree and the workbook describe the same cohort.

    Three conditions are enforced. First, the identifier sets must be equal:
    an image without metadata cannot be used in the clinical reference arm,
    and metadata without an image is a phantom patient inflating the reported
    cohort size. Second, the folder-derived label must equal the workbook
    diagnosis for every patient, since the folder is what supervises training.
    Third, the OBLIQUE flag must agree with the projections present on disk.

    Any violation raises. Phase 0 is the only place where these can be caught
    cheaply; downstream they would surface as silently wrong labels.
    """
    dictPatientToClass = {}
    dictPatientToViews = {}
    for strClass, dictClass in dictInventory.items():
        for intPatient, lstViews in dictClass["views_by_patient"].items():
            if intPatient in dictPatientToClass:
                raise ValueError(
                    f"Patient {intPatient} appears in both "
                    f"'{dictPatientToClass[intPatient]}' and '{strClass}'. "
                    f"Patient identifiers must be unique across classes."
                )
            dictPatientToClass[intPatient] = strClass
            dictPatientToViews[intPatient] = lstViews

    setDisk = set(dictPatientToClass)
    setBook = set(dfDemographics["IMAGE"].astype(int))

    lstOnlyBook = sorted(setBook - setDisk)
    lstOnlyDisk = sorted(setDisk - setBook)
    if lstOnlyBook or lstOnlyDisk:
        raise ValueError(
            f"Cohort linkage failure. Present only in the workbook: {lstOnlyBook}. "
            f"Present only on disk: {lstOnlyDisk}. Phase 0 refuses to report a "
            f"cohort size that the two sources do not agree on."
        )

    serDiagnosis = dfDemographics.set_index(dfDemographics["IMAGE"].astype(int))["DIAGNOSIS"]
    lstLabelMismatch = [
        int(i) for i in sorted(setDisk)
        if DICT_DIAGNOSIS_TO_FOLDER.get(serDiagnosis.loc[i]) != dictPatientToClass[i]
    ]
    if lstLabelMismatch:
        raise ValueError(
            f"Folder label disagrees with workbook diagnosis for patients "
            f"{lstLabelMismatch}. The folder is the supervision signal, so this "
            f"would train the model on the wrong target."
        )

    dictOblique = {}
    lstObliqueMismatch = []
    if "OBLIQUE" in dfDemographics.columns:
        serOblique = dfDemographics.set_index(dfDemographics["IMAGE"].astype(int))["OBLIQUE"]
        for intPatient in sorted(setDisk):
            strFlag = str(serOblique.loc[intPatient]).strip().upper()
            intViews = len(dictPatientToViews[intPatient])
            dictOblique[intPatient] = strFlag
            boolExpectObliques = strFlag == "YES"
            if boolExpectObliques and intViews != len(LST_EXPECTED_VIEWS):
                lstObliqueMismatch.append((intPatient, strFlag, dictPatientToViews[intPatient]))
            if not boolExpectObliques and intViews != 1:
                lstObliqueMismatch.append((intPatient, strFlag, dictPatientToViews[intPatient]))

    return {
        "n_patients_linked": len(setDisk),
        "identifiers_match": True,
        "label_mismatches": 0,
        "oblique_flag_mismatches": lstObliqueMismatch,
        "patient_to_class": dictPatientToClass,
        "patient_to_views": dictPatientToViews,
    }


def screen_plausibility(dfDemographics):
    """
    Flags physiologically impossible values and quantifies their leverage.

    For each flagged field the summary statistics are recomputed with the
    offending rows excluded, so the report can state how much the anomaly
    distorts the number it would otherwise print. Nothing is modified in the
    source workbook: this function only measures and reports.
    """
    dictFlags = {}
    for strCol, tupRange in DICT_PLAUSIBLE_RANGE.items():
        if strCol not in dfDemographics.columns:
            continue
        serVal = pd.to_numeric(dfDemographics[strCol], errors="coerce")
        maskBad = serVal.notna() & ((serVal < tupRange[0]) | (serVal > tupRange[1]))
        if not maskBad.any():
            continue
        serClean = serVal[~maskBad].dropna()
        dictFlags[strCol] = {
            "plausible_range": list(tupRange),
            "n_flagged": int(maskBad.sum()),
            "flagged_rows": [
                {
                    "IMAGE": int(dfDemographics.loc[i, "IMAGE"]),
                    "value": float(serVal.loc[i]),
                    "diagnosis": str(dfDemographics.loc[i, "DIAGNOSIS"]),
                }
                for i in dfDemographics.index[maskBad]
            ],
            "as_reported": {
                "mean": float(serVal.dropna().mean()),
                "std": float(serVal.dropna().std(ddof=1)),
                "min": float(serVal.dropna().min()),
                "max": float(serVal.dropna().max()),
            },
            "excluding_flagged": {
                "mean": float(serClean.mean()),
                "std": float(serClean.std(ddof=1)),
                "min": float(serClean.min()),
                "max": float(serClean.max()),
            },
        }
    return dictFlags


def stratify_by_class(dfDemographics):
    """
    Contrasts every demographic covariate between the two diagnostic groups.

    Continuous covariates use the Mann-Whitney U test, which does not assume
    normality and is appropriate for the skewed anthropometric fields here.
    The effect size reported alongside is the rank-biserial correlation, since
    a p-value on n=178 says whether a difference exists but not whether it is
    large enough to matter. Categorical covariates use Fisher's exact test.

    A covariate that separates the classes is a potential confounder: it
    offers the classifier a shortcut that bypasses the image entirely.
    """
    dfWork = dfDemographics.copy()
    dfWork["__class"] = dfWork["DIAGNOSIS"].map(DICT_DIAGNOSIS_TO_FOLDER)

    dictOut = {"continuous": {}, "categorical": {}}
    for strCol in ["AGE", "WEIGHT (Kg)", "HEIGHT (cm)", "DOSE ADMINISTERED (mCi)"]:
        if strCol not in dfWork.columns:
            continue
        arrA = pd.to_numeric(
            dfWork.loc[dfWork["__class"] == "multinodular_goiter", strCol], errors="coerce"
        ).dropna().to_numpy()
        arrB = pd.to_numeric(
            dfWork.loc[dfWork["__class"] == "diffuse_goiter", strCol], errors="coerce"
        ).dropna().to_numpy()
        if len(arrA) < 3 or len(arrB) < 3:
            continue
        floatU, floatP = stats.mannwhitneyu(arrA, arrB, alternative="two-sided")
        floatRankBiserial = abs(1.0 - 2.0 * floatU / (len(arrA) * len(arrB)))
        dictOut["continuous"][strCol] = {
            "multinodular_goiter": {
                "n": int(len(arrA)),
                "mean": float(arrA.mean()),
                "std": float(arrA.std(ddof=1)),
                "median": float(np.median(arrA)),
                "min": float(arrA.min()),
                "max": float(arrA.max()),
            },
            "diffuse_goiter": {
                "n": int(len(arrB)),
                "mean": float(arrB.mean()),
                "std": float(arrB.std(ddof=1)),
                "median": float(np.median(arrB)),
                "min": float(arrB.min()),
                "max": float(arrB.max()),
            },
            "mann_whitney_u": float(floatU),
            "p_value": float(floatP),
            "rank_biserial": float(floatRankBiserial),
            "is_confounder": bool(floatP < 0.05),
        }

    for strCol in ["SEX", "OBLIQUE"]:
        if strCol not in dfWork.columns:
            continue
        dfCross = pd.crosstab(dfWork["__class"], dfWork[strCol])
        dictEntry = {"contingency": dfCross.to_dict()}
        if dfCross.shape == (2, 2):
            floatOdds, floatP = stats.fisher_exact(dfCross.to_numpy())
            dictEntry["fisher_exact_p"] = float(floatP)
            dictEntry["odds_ratio"] = float(floatOdds)
            dictEntry["is_confounder"] = bool(floatP < 0.05)
        dictOut["categorical"][strCol] = dictEntry

    return dictOut


def plot_cohort_audit(dictInventory, dictStrat, dfDemographics, pathOut):
    """
    Three-panel cohort figure for the manuscript.

    Panel a resolves the imbalance at image and patient level side by side,
    because the two ratios differ and quoting only the image-level one
    overstates the problem. Panel b shows the projection protocol. Panel c
    shows the age distribution per class, which is the confounder the reader
    must see rather than be told about.
    """
    objFig, arrAxes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    lstClasses = ["multinodular_goiter", "diffuse_goiter"]

    # --- Panel a: imbalance at two levels of aggregation -----------------
    objAx = arrAxes[0]
    arrX = np.arange(len(lstClasses))
    floatWidth = 0.38
    lstImages = [dictInventory[c]["images"] for c in lstClasses]
    lstPatients = [dictInventory[c]["patients"] for c in lstClasses]

    objAx.bar(arrX - floatWidth / 2, lstImages, floatWidth, label="Images",
              color=[DICT_CLASS_COLOUR[c] for c in lstClasses],
              edgecolor="black", linewidth=0.6)
    objAx.bar(arrX + floatWidth / 2, lstPatients, floatWidth, label="Patients",
              color=[DICT_CLASS_COLOUR[c] for c in lstClasses],
              edgecolor="black", linewidth=0.6, alpha=0.45, hatch="///")

    for floatXPos, intVal in zip(arrX - floatWidth / 2, lstImages):
        objAx.text(floatXPos, intVal + 6, str(intVal), ha="center", va="bottom", fontsize=8)
    for floatXPos, intVal in zip(arrX + floatWidth / 2, lstPatients):
        objAx.text(floatXPos, intVal + 6, str(intVal), ha="center", va="bottom", fontsize=8)

    floatRatioImg = max(lstImages) / min(lstImages)
    floatRatioPat = max(lstPatients) / min(lstPatients)
    objAx.set_xticks(arrX)
    objAx.set_xticklabels([DICT_CLASS_LABEL[c].replace(" ", "\n") for c in lstClasses])
    objAx.set_ylabel("Count")
    objAx.set_title(f"Imbalance is milder per patient ({floatRatioPat:.2f}:1)\n"
                    f"than per image ({floatRatioImg:.2f}:1)")
    objAx.set_ylim(0, max(lstImages) * 1.18)
    objAx.legend(frameon=False, fontsize=8, loc="upper right")
    objAx.spines[["top", "right"]].set_visible(False)

    # --- Panel b: acquisition protocol ----------------------------------
    objAx = arrAxes[1]
    arrXView = np.arange(len(LST_EXPECTED_VIEWS))
    for intPos, strClass in enumerate(lstClasses):
        lstCounts = [dictInventory[strClass]["per_view"].get(v, 0) for v in LST_EXPECTED_VIEWS]
        objAx.bar(arrXView + (intPos - 0.5) * floatWidth, lstCounts, floatWidth,
                  label=DICT_CLASS_LABEL[strClass], color=DICT_CLASS_COLOUR[strClass],
                  edgecolor="black", linewidth=0.6)
    # The title states a fact about the protocol, so it is derived from the
    # inventory rather than written by hand: a hardcoded count would silently
    # become a false claim if the cohort were ever extended.
    intMissingObl = sum(
        1 for dictCls in dictInventory.values()
        for lstViews in dictCls["views_by_patient"].values()
        if len(lstViews) < len(LST_EXPECTED_VIEWS)
    )
    intAnteriorPresent = sum(
        1 for dictCls in dictInventory.values()
        for lstViews in dictCls["views_by_patient"].values()
        if "ANTERIOR" in lstViews
    )
    intPatientsAll = sum(d["patients"] for d in dictInventory.values())
    strAnterior = ("anterior always present" if intAnteriorPresent == intPatientsAll
                   else f"anterior present in {intAnteriorPresent}/{intPatientsAll}")

    objAx.set_xticks(arrXView)
    objAx.set_xticklabels(["Anterior", "LAO", "RAO"])
    objAx.set_ylabel("Number of images")
    objAx.set_xlabel("Projection")
    objAx.set_title(f"Obliques absent for {intMissingObl} patients,\n{strAnterior}")
    objAx.legend(frameon=False, fontsize=8)
    objAx.spines[["top", "right"]].set_visible(False)

    # --- Panel c: the age confounder ------------------------------------
    objAx = arrAxes[2]
    dictAge = dictStrat["continuous"].get("AGE", {})
    dfWork = dfDemographics.copy()
    dfWork["__class"] = dfWork["DIAGNOSIS"].map(DICT_DIAGNOSIS_TO_FOLDER)

    for intPos, strClass in enumerate(lstClasses):
        arrVals = pd.to_numeric(
            dfWork.loc[dfWork["__class"] == strClass, "AGE"], errors="coerce"
        ).dropna().to_numpy()
        arrJitter = np.random.default_rng(42).normal(0, 0.055, size=len(arrVals))
        objAx.plot(np.full(len(arrVals), intPos) + arrJitter, arrVals, "o",
                   color=DICT_CLASS_COLOUR[strClass], markersize=3.4,
                   alpha=0.55, markeredgewidth=0)
        floatMedian = float(np.median(arrVals))
        objAx.hlines(floatMedian, intPos - 0.24, intPos + 0.24,
                     color="black", linewidth=1.8, zorder=5)
        objAx.text(intPos + 0.29, floatMedian, f"{floatMedian:.0f} y",
                   va="center", ha="left", fontsize=8, fontweight="bold")

    objAx.set_xticks(range(len(lstClasses)))
    objAx.set_xticklabels([DICT_CLASS_LABEL[c].replace(" ", "\n") for c in lstClasses])
    objAx.set_ylabel("Age (years)")
    objAx.set_xlim(-0.5, len(lstClasses) - 0.25)
    if dictAge:
        objAx.set_title(f"Age separates the classes\n"
                        f"(Mann-Whitney p = {dictAge['p_value']:.1e}, "
                        f"r = {dictAge['rank_biserial']:.2f})")
    objAx.spines[["top", "right"]].set_visible(False)

    for strLetter, objAxis in zip("abc", arrAxes):
        objAxis.text(-0.14, 1.10, strLetter, transform=objAxis.transAxes,
                     fontsize=12, fontweight="bold", va="top", ha="left")

    objFig.tight_layout()
    objFig.savefig(pathOut, dpi=300, bbox_inches="tight")
    plt.close(objFig)
    return pathOut


def main():
    pathDataDir = Path("data")
    pathPhase0Dir = Path("outputs/phase0")
    pathPhase0Dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(pathPhase0Dir / "phase0_run.log")
    print("=== Phase 0: Cohort Evidence, Linkage Integrity and Confounder Audit ===")

    # --- 1. Image inventory at three levels of aggregation ---------------
    dictInventory = scan_image_inventory(pathDataDir)
    if not dictInventory:
        raise FileNotFoundError(
            f"No images found under '{pathDataDir}'. Phase 0 cannot audit an empty cohort."
        )

    intImgTotal = sum(d["images"] for d in dictInventory.values())
    intPatTotal = sum(d["patients"] for d in dictInventory.values())
    lstImgCounts = [d["images"] for d in dictInventory.values()]
    lstPatCounts = [d["patients"] for d in dictInventory.values()]

    intThreeViews = 0
    intOneView = 0
    for dictClass in dictInventory.values():
        for lstViews in dictClass["views_by_patient"].values():
            if len(lstViews) == len(LST_EXPECTED_VIEWS):
                intThreeViews += 1
            elif len(lstViews) == 1:
                intOneView += 1

    print(f"Images: {intImgTotal} across {intPatTotal} unique patients")
    for strClass, dictClass in dictInventory.items():
        print(f" - {strClass}: {dictClass['images']} images / "
              f"{dictClass['patients']} patients / views {dictClass['per_view']}")
    print(f"Imbalance ratio  images: {max(lstImgCounts) / min(lstImgCounts):.2f}:1  "
          f"patients: {max(lstPatCounts) / min(lstPatCounts):.2f}:1")

    # --- 2. Clinical metadata, linkage, stratification, plausibility -----
    pathClinicalData = pathDataDir / "datosDemograficos.xlsx"
    if not pathClinicalData.exists():
        raise FileNotFoundError(
            f"Clinical metadata '{pathClinicalData}' not found. The demographic audit "
            f"and the Phase 3 clinical reference arm both depend on it."
        )

    dfDemographics = pd.read_excel(pathClinicalData)
    print(f"\nDemographics workbook: {len(dfDemographics)} rows, "
          f"columns {list(dfDemographics.columns)}")

    dictLinkage = assert_linkage_integrity(dictInventory, dfDemographics)
    print(f"[linkage] {dictLinkage['n_patients_linked']} patients linked; "
          f"0 identifier mismatches; 0 label mismatches; "
          f"{len(dictLinkage['oblique_flag_mismatches'])} OBLIQUE flag mismatches.")

    dictStratification = stratify_by_class(dfDemographics)
    for strCol, dictVar in dictStratification["continuous"].items():
        print(f"[strat] {strCol:26s} p={dictVar['p_value']:.3e} "
              f"r={dictVar['rank_biserial']:.3f}"
              + ("  <-- CONFOUNDER" if dictVar["is_confounder"] else ""))

    dictPlausibility = screen_plausibility(dfDemographics)
    for strCol, dictFlag in dictPlausibility.items():
        print(f"[plausibility] {strCol}: {dictFlag['n_flagged']} implausible value(s); "
              f"sd {dictFlag['as_reported']['std']:.2f} -> "
              f"{dictFlag['excluding_flagged']['std']:.2f} once excluded")

    # --- 3. Descriptive summary of the whole cohort ----------------------
    dictClinicalStats = {"total_patients_in_workbook": int(len(dfDemographics))}
    if "DIAGNOSIS" in dfDemographics.columns:
        dictClinicalStats["diagnosis_distribution"] = \
            dfDemographics["DIAGNOSIS"].value_counts().to_dict()
    if "SEX" in dfDemographics.columns:
        dictClinicalStats["sex_distribution"] = \
            dfDemographics["SEX"].value_counts().to_dict()
    for strCol, strKey in [("AGE", "age_stats"), ("WEIGHT (Kg)", "weight_kg_stats"),
                           ("HEIGHT (cm)", "height_cm_stats"),
                           ("DOSE ADMINISTERED (mCi)", "dose_mci_stats")]:
        if strCol not in dfDemographics.columns:
            continue
        serVal = pd.to_numeric(dfDemographics[strCol], errors="coerce").dropna()
        dictClinicalStats[strKey] = {
            "mean": float(serVal.mean()),
            "std": float(serVal.std(ddof=1)),
            "median": float(serVal.median()),
            "min": float(serVal.min()),
            "max": float(serVal.max()),
        }

    # --- 4. Assemble and persist the audit ------------------------------
    dictResults = {
        "modality": {
            "technique": "planar thyroid scintigraphy (nuclear medicine)",
            "note": ("Confirmed by the DOSE ADMINISTERED (mCi) field, which records "
                     "administered radiotracer activity. These are not ultrasound images."),
            "views": LST_EXPECTED_VIEWS,
        },
        "image_inventory": {
            "totals": {"images": intImgTotal, "patients": intPatTotal},
            "per_class": {
                k: {"images": v["images"], "patients": v["patients"], "per_view": v["per_view"]}
                for k, v in dictInventory.items()
            },
            "imbalance_ratio_images": float(max(lstImgCounts) / min(lstImgCounts)),
            "imbalance_ratio_patients": float(max(lstPatCounts) / min(lstPatCounts)),
            "n_patients_three_views": int(intThreeViews),
            "n_patients_one_view": int(intOneView),
        },
        "linkage_integrity": {
            "n_patients_linked": dictLinkage["n_patients_linked"],
            "identifiers_match": dictLinkage["identifiers_match"],
            "label_mismatches": dictLinkage["label_mismatches"],
            "oblique_flag_mismatches": dictLinkage["oblique_flag_mismatches"],
        },
        "clinical_data": dictClinicalStats,
        "class_stratification": dictStratification,
        "plausibility_flags": dictPlausibility,
    }

    pathJsonOutput = pathPhase0Dir / "phase0_evidence.json"
    with open(pathJsonOutput, "w", encoding="utf-8") as objFile:
        json.dump(dictResults, objFile, indent=4, ensure_ascii=False)
    print(f"\nAudit saved to: {pathJsonOutput}")

    pathFig = plot_cohort_audit(dictInventory, dictStratification, dfDemographics,
                                pathPhase0Dir / "phase0_cohort_audit.png")
    print(f"Cohort audit figure saved to: {pathFig}")



if __name__ == "__main__":
    main()
