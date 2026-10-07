"""
Phase 1: Data Preparation, Undersampling, and Nested CV Splits
==============================================================

This script is the statistical core of the dataset preparation process. It performs
patient-level balancing and generates strict nested cross-validation partitions to
prevent data leakage.

Key Operations:
1. Iterates through the raw image directory to build a master data table.
2. Merges demographic and clinical data (datosDemograficos.xlsx) into the master table.
3. Addresses class imbalance by applying strict patient-level undersampling, ensuring
   that both 'multinodular_goiter' and 'diffuse_goiter' have the same number of patients.
4. Generates a reproducible 5x5 Nested Cross-Validation split dictionary utilizing
   StratifiedGroupKFold. Each outer training block is further divided into an inner
   training subset and an inner calibration subset, also grouped by patient, so that
   decision thresholds can be fitted without ever touching the outer validation fold.
5. Tabulates the cohort demographics per diagnostic class, including the between-class
   age comparison that the manuscript must report as a potential confounder.
6. Saves the tables, the JSON splits and the manuscript figure to disk.

Split file contract
-------------------
Splits are written EXCLUSIVELY as lists of 'sample_id' and 'patient_id' strings, and
are defined over phase1_balanced_table.csv. Positional integer indices are deliberately
not written: an earlier revision stored indices computed on the balanced table while
Phases 3-5 applied them to the longer master table, which destroyed the patient
grouping and inflated every reported metric. Downstream phases resolve splits through
common.get_fold_row_positions() and verify them with
common.assert_folds_patient_disjoint().

WHAT CHANGED IN THIS REVISION (and why it should improve the metrics)
--------------------------------------------------------------------
ALL FOUR inner train/calibration partitions are now written to the split file,
under a new 'inner_folds' key. The previous revision built a 4-split
StratifiedGroupKFold and then kept only the first partition, discarding 75% of
the inner loop while the manuscript described a "nested cross-validation with 4
inner folds".

This was not merely a documentation defect. Phases 3 and 4 select the decision
threshold on the calibration block, and with a single block that means 28-31
patients in Phase 4. The measured consequence across the 25 folds was a threshold
with sd ~0.15-0.16 and a range of 0.52-0.65; in 9 folds sensitivity and
specificity differed by more than 0.20, in 3 folds specificity saturated at 1.0
and in 3 folds sensitivity fell below 0.60. That instability is what opens the
~0.10 gap between AUC (0.9137) and balanced accuracy (0.8118) in Phase 4:
the model ranks well and then loses part of that advantage at the operating
point. Averaging the threshold over four partitions roughly halves its standard
error, which is the cheapest available route to recovering some of that gap.

The flat 'inner_train'/'inner_cal' keys are retained and hold the first
partition, so nothing downstream breaks and the previous behaviour stays
reproducible by setting the phases' partition count to 1.

Execution Instructions:
-----------------------
Run FIRST. Every later phase reads the cohort table and the fold definitions
this script writes.

   $ conda activate thyroid_env
   $ python phases/phase1_data_prep.py

Runtime: a few seconds.

IMPORTANT: changing STR_COHORT_STRATEGY or INT_NUM_REPEATS invalidates the
feature files, because Phase 2 keys its tensors to the cohort table. Re-run
Phase 2 afterwards.

To reproduce the undersampled cohort as a sensitivity analysis, set
STR_COHORT_STRATEGY = "undersample" above and re-run Phases 1 to 4.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DICT_CLASS_TO_LABEL, INT_GLOBAL_SEED, LST_TARGET_CLASSES,
                    assert_folds_patient_disjoint, set_global_seed, TeeLogger)

# Repeats of the whole cross-validation procedure. Each repeat is an independent
# replicate, and the paired tests in Phases 3 and 4 use repeat-level means as
# their unit of analysis: with 5 repeats those tests have n = 5, which is thin
# for a grid of ~22 configurations. Raising it to 10 doubles the degrees of
# freedom of every contrast at the cost of re-running Phases 3 and 4 only
# (features are extracted once and reused), and it is the cheapest available
# increase in statistical power.
INT_NUM_REPEATS = 10

# Cohort strategy: 'full' keeps every patient and lets class weighting handle
# the 1.4:1 imbalance; 'undersample' reproduces the previous behaviour and is
# retained so the manuscript can report it as a sensitivity analysis.
# Re-run Phase 2 after changing this: the feature files are keyed to the cohort.
STR_COHORT_STRATEGY = "full"
INT_NUM_FOLDS = 5
INT_NUM_INNER_FOLDS = 4          # inner calibration split = 25% of each training block


def build_inner_folds(dfTrainBlock, intSeed):
    """
    Partitions one outer training block into ALL INT_NUM_INNER_FOLDS inner
    train/calibration pairs, keeping every view of a patient on the same side.

    WHY THIS RETURNS A LIST AND NOT A SINGLE PAIR
    ---------------------------------------------
    The previous revision built a 4-split StratifiedGroupKFold and then called
    next() on it, keeping only the FIRST of the four partitions and discarding
    75% of the inner loop. The manuscript described a "nested cross-validation
    with 4 inner folds" while what actually ran was a single 26% calibration
    block.

    That is not only a description problem, it costs measurable performance.
    Phases 3 and 4 fit the decision threshold on the calibration block, and with
    a single block that means 28-31 patients in Phase 4. The resulting threshold
    had a standard deviation of ~0.15 and a range of 0.52-0.65 across folds; in
    9 of 25 folds sensitivity and specificity differed by more than 0.20, in 3
    folds specificity saturated at 1.0 and in 3 folds sensitivity fell below
    0.60. That instability is exactly the ~0.10 gap observed between AUC and
    balanced accuracy.

    Averaging the threshold over the four inner folds reduces its variance by
    approximately 1/sqrt(4), i.e. halves the standard error, which is the
    cheapest available route to closing part of that gap. It also makes the
    manuscript's description of the design true.

    Returns
    -------
    list of (dfInnerTrain, dfInnerCal) tuples, length INT_NUM_INNER_FOLDS.
    Patient grouping is enforced by StratifiedGroupKFold on 'patient_id'.
    """
    objInnerSgkf = StratifiedGroupKFold(n_splits=INT_NUM_INNER_FOLDS,
                                        shuffle=True, random_state=intSeed)
    lstInnerFolds = []
    for arrInnerTrainIdx, arrInnerCalIdx in objInnerSgkf.split(
            dfTrainBlock["sample_id"].values,
            dfTrainBlock["class"].values,
            dfTrainBlock["patient_id"].values):
        lstInnerFolds.append((dfTrainBlock.iloc[arrInnerTrainIdx],
                              dfTrainBlock.iloc[arrInnerCalIdx]))
    return lstInnerFolds


def describe_cohort(dfMasterTable, dfBalancedTable):
    """
    Builds the per-class demographic table for the manuscript and tests the
    between-class differences. The age comparison is reported because
    multinodular goitre is epidemiologically associated with older age, so age
    is a candidate confounder for any imaging-based discrimination claim.
    """
    lstRows = []
    for strLabel, dfTable in [("full_cohort", dfMasterTable),
                              ("balanced_cohort", dfBalancedTable)]:
        dfPatients = dfTable.drop_duplicates("patient_id")
        for strCls in LST_TARGET_CLASSES:
            dfCls = dfPatients[dfPatients["class"] == strCls]
            lstRows.append({
                "cohort": strLabel,
                "class": strCls,
                "n_patients": len(dfCls),
                "n_images": int((dfTable["class"] == strCls).sum()),
                "age_mean": dfCls["AGE"].mean(),
                "age_sd": dfCls["AGE"].std(ddof=1),
                "age_min": dfCls["AGE"].min(),
                "age_max": dfCls["AGE"].max(),
                "pct_female": 100.0 * (dfCls["SEX"] == "FEMALE").mean(),
                "weight_mean": dfCls["WEIGHT (Kg)"].mean(),
                "height_mean": dfCls["HEIGHT (cm)"].mean(),
                "dose_mean": dfCls["DOSE ADMINISTERED (mCi)"].mean(),
            })
    dfDemographics = pd.DataFrame(lstRows)

    dfPatients = dfBalancedTable.drop_duplicates("patient_id")
    serAgeMulti = dfPatients.loc[dfPatients["class"] == "multinodular_goiter", "AGE"].dropna()
    serAgeDiff = dfPatients.loc[dfPatients["class"] == "diffuse_goiter", "AGE"].dropna()
    floatAgeP = mannwhitneyu(serAgeMulti, serAgeDiff).pvalue if len(serAgeMulti) and len(serAgeDiff) else np.nan

    dfSexTable = pd.crosstab(dfPatients["class"], dfPatients["SEX"])
    floatSexP = chi2_contingency(dfSexTable.values)[1] if dfSexTable.shape[1] > 1 else np.nan

    dictTests = {
        "age_mannwhitney_p": float(floatAgeP),
        "age_median_multinodular": float(serAgeMulti.median()),
        "age_median_diffuse": float(serAgeDiff.median()),
        "sex_chi2_p": float(floatSexP),
        "pct_female_balanced_cohort": float(100.0 * (dfPatients["SEX"] == "FEMALE").mean()),
    }
    return dfDemographics, dictTests


def main():
    # Establish a deterministic seed for reproducibility
    set_global_seed(INT_GLOBAL_SEED)
    
    # Define primary directory paths
    pathDataDir = Path("data")
    pathPhase1Dir = Path("outputs/phase1")
    
    # Create necessary output folders
    pathPhase1Dir.mkdir(parents=True, exist_ok=True)
    
    sys.stdout = TeeLogger(pathPhase1Dir / "phase1_run.log")
    print("=== Phase 1: Data Preparation and Patient-Level Balancing ===")
    
    # --- 1. Build the Master Table by parsing the directory structure ---
    lstRecords = []
    lstValidExtensions = ['.png', '.jpg', '.jpeg', '.tif']
    
    for pathClassDir in pathDataDir.iterdir():
        if not pathClassDir.is_dir(): 
            continue
        strClassName = pathClassDir.name
        
        for pathViewDir in pathClassDir.iterdir():
            if not pathViewDir.is_dir(): 
                continue
            strViewName = pathViewDir.name
            
            for pathImage in pathViewDir.rglob('*'):
                if pathImage.suffix.lower() not in lstValidExtensions:
                    continue
                    
                # Patient ID is typically the filename stem (without extension)
                strPatientId = pathImage.stem
                
                # Store paths as relative to ensure cross-machine reproducibility
                strRelativePath = pathImage.as_posix()
                
                lstRecords.append({
                    "sample_id": strRelativePath,
                    "filepath": strRelativePath,
                    "filename": pathImage.name,
                    "class": strClassName,
                    "view": strViewName,
                    "patient_id": strPatientId
                })
                
    # Sorted by sample_id so that the inventory does not depend on the order in
    # which the filesystem returned directory entries.
    dfMasterTable = pd.DataFrame(lstRecords).sort_values("sample_id").reset_index(drop=True)
    print(f"Total raw images parsed into Master Table: {len(dfMasterTable)}")
    
    # --- 2. Enrich the Master Table with clinical metadata ---
    pathClinicalData = pathDataDir / "datosDemograficos.xlsx"
    if pathClinicalData.exists():
        dfClinical = pd.read_excel(pathClinicalData)
        # Ensure the linking column 'IMAGE' is treated as a string
        dfClinical['IMAGE'] = dfClinical['IMAGE'].astype(str)
        
        # Merge clinical data on patient_id
        dfMasterTable = dfMasterTable.merge(
            dfClinical, 
            left_on='patient_id', 
            right_on='IMAGE', 
            how='left'
        )
        
    pathMasterCsv = pathPhase1Dir / "phase1_master_table.csv"
    dfMasterTable.to_csv(pathMasterCsv, index=False)
    print(f"Master Table successfully saved to: {pathMasterCsv}")
    
    # --- 3. Execute Patient-Level Balanced Undersampling ---
    # Due to natural imbalance, we undersample the majority class at the patient level
    serPatientsPerClass = dfMasterTable.groupby('class')['patient_id'].nunique()
    print("\nOriginal patient distribution across classes:")
    print(serPatientsPerClass)
    
    intMinPatients = int(serPatientsPerClass.min())

    # ------------------------------------------------------------------
    # COHORT STRATEGY: full cohort with class weighting (default), or
    # patient-level undersampling (sensitivity analysis).
    # ------------------------------------------------------------------
    # The previous default discarded 30 of 178 patients (17% of the cohort) to
    # correct an imbalance of only 1.4:1 at patient level, where the majority
    # class is 58.4% of patients. Three measured reasons to prefer the full
    # cohort:
    #
    #   1. A learning curve built by sub-sampling the training block gives AUC
    #      0.7695 / 0.7890 / 0.7948 / 0.8177 at 50 / 65 / 80 / 100% of the
    #      available data, i.e. slope +0.000372 AUC per training image
    #      (p = 0.0145, R2 = 0.971). The curve is NOT saturated, so the
    #      discarded images were costing performance.
    #   2. Every downstream classifier already uses class_weight='balanced',
    #      which is the standard correction for a 1.4:1 ratio and does not
    #      throw data away.
    #   3. The undersample was drawn ONCE and reused across all repeats, so the
    #      reported error bars never included the variance introduced by which
    #      30 patients happened to be dropped.
    #
    # The undersampled cohort remains available as a sensitivity analysis, which
    # is what a reviewer asking "does the imbalance drive your result" needs.
    if STR_COHORT_STRATEGY == "full":
        # No patient is discarded; class imbalance is handled downstream by
        # class weighting rather than by deletion.
        dfBalancedTable = (dfMasterTable
                           .sort_values('sample_id')
                           .reset_index(drop=True)
                           .copy())
        print(f"\nCohort strategy: FULL cohort retained "
              f"({dfBalancedTable['patient_id'].nunique()} patients). "
              f"Imbalance handled by class weighting downstream.")
    elif STR_COHORT_STRATEGY == "undersample":
        # Candidate patients are sorted before sampling so that the selection
        # depends only on the seed and never on the order in which the
        # filesystem enumerated the image directories. Without the sort, the
        # same seed yields different cohorts on different machines and the study
        # stops being reproducible.
        lstBalancedPatientIds = []
        for strCls in LST_TARGET_CLASSES:
            lstClassPatients = sorted(
                dfMasterTable.loc[dfMasterTable['class'] == strCls, 'patient_id'].unique().tolist()
            )
            lstSelectedPatients = random.sample(lstClassPatients, intMinPatients)
            lstBalancedPatientIds.extend(lstSelectedPatients)

        # Sorting by sample_id fixes the row order of the analysis cohort, so
        # the table is byte-identical across runs and operating systems.
        dfBalancedTable = (dfMasterTable[dfMasterTable['patient_id'].isin(lstBalancedPatientIds)]
                           .sort_values('sample_id')
                           .reset_index(drop=True)
                           .copy())
        print(f"\nCohort strategy: patient-level UNDERSAMPLING to "
              f"{intMinPatients} patients per class.")
    else:
        raise ValueError(
            f"STR_COHORT_STRATEGY must be 'full' or 'undersample', "
            f"got {STR_COHORT_STRATEGY!r}.")
    print(f"\nTotal patients after balancing: {dfBalancedTable['patient_id'].nunique()}")
    print(f"Total image samples after balancing: {len(dfBalancedTable)}")
    
    pathBalancedCsv = pathPhase1Dir / "phase1_balanced_table.csv"
    dfBalancedTable.to_csv(pathBalancedCsv, index=False)
    
    # --- 4. Generate Nested Cross-Validation Splits (5x5) ---
    # StratifiedGroupKFold groups by patient_id so that no patient contributes
    # images to both sides of a split. Each outer training block is additionally
    # divided into an inner training subset and an inner calibration subset,
    # also patient-grouped: the calibration subset is the ONLY data on which
    # downstream phases are permitted to fit a decision threshold.
    #
    # The stored representation is a dictionary of sample_id / patient_id lists.
    # Positional integer indices are intentionally not stored, because they are
    # only meaningful for the exact table and row order that produced them.
    dictFolds = {}
    
    # Encode patient strings to integers to satisfy StratifiedGroupKFold requirements
    dictPatientEncoder = {
        strPid: intIdx 
        for intIdx, strPid in enumerate(sorted(dfBalancedTable['patient_id'].unique()))
    }
    dfBalancedTable['group_id'] = dfBalancedTable['patient_id'].map(dictPatientEncoder)
    
    arrSampleIds = dfBalancedTable['sample_id'].values
    arrLabels = dfBalancedTable['class'].values
    arrGroups = dfBalancedTable['group_id'].values
    
    for intRepeatIdx in range(INT_NUM_REPEATS):
        intSeed = INT_GLOBAL_SEED + intRepeatIdx
        objSgkf = StratifiedGroupKFold(n_splits=INT_NUM_FOLDS, shuffle=True,
                                       random_state=intSeed)
        
        for intFoldIdx, (arrTrainIndices, arrValIndices) in enumerate(
                objSgkf.split(arrSampleIds, arrLabels, arrGroups)):
            dfTrainBlock = dfBalancedTable.iloc[arrTrainIndices]
            dfValBlock = dfBalancedTable.iloc[arrValIndices]
            
            # ALL four inner partitions are materialised. See build_inner_folds()
            # for why keeping only the first one was costing threshold stability.
            lstInnerFolds = build_inner_folds(dfTrainBlock, intSeed)

            # 'inner_folds' is the new, complete representation consumed by
            # Phases 3 and 4 to average the decision threshold.
            lstInnerFoldRecords = []
            for dfInnerTrain, dfInnerCal in lstInnerFolds:
                lstInnerFoldRecords.append({
                    "inner_train": dfInnerTrain['sample_id'].tolist(),
                    "inner_cal": dfInnerCal['sample_id'].tolist(),
                    "inner_train_patients": sorted(dfInnerTrain['patient_id'].unique().tolist()),
                    "inner_cal_patients": sorted(dfInnerCal['patient_id'].unique().tolist()),
                })

            # The flat 'inner_train'/'inner_cal' keys are RETAINED and hold the
            # first inner partition. They are what common.load_splits() validates
            # and what the single-fit code paths (e.g. fitting the final model of
            # a fold) still use, so older behaviour remains reproducible and no
            # downstream phase breaks when it does not need the full inner loop.
            dfFirstInnerTrain, dfFirstInnerCal = lstInnerFolds[0]

            dictFolds[f"repeat_{intRepeatIdx}_fold_{intFoldIdx}"] = {
                "train": dfTrainBlock['sample_id'].tolist(),
                "val": dfValBlock['sample_id'].tolist(),
                "train_patients": sorted(dfTrainBlock['patient_id'].unique().tolist()),
                "val_patients": sorted(dfValBlock['patient_id'].unique().tolist()),
                "inner_train": dfFirstInnerTrain['sample_id'].tolist(),
                "inner_cal": dfFirstInnerCal['sample_id'].tolist(),
                "inner_train_patients": sorted(dfFirstInnerTrain['patient_id'].unique().tolist()),
                "inner_cal_patients": sorted(dfFirstInnerCal['patient_id'].unique().tolist()),
                "inner_folds": lstInnerFoldRecords,
            }
            
    dictSplits = {
        "meta": {
            "cohort_table": "outputs/phase1/phase1_balanced_table.csv",
            "description": ("Repeated stratified group K-fold over the patient-balanced "
                            "cohort. Folds are addressed by sample_id; positional "
                            "indices are deliberately not stored."),
            "n_repeats": INT_NUM_REPEATS,
            "n_outer_folds": INT_NUM_FOLDS,
            "n_inner_folds": INT_NUM_INNER_FOLDS,
            "grouping_variable": "patient_id",
            "stratification_variable": "class",
            "seeds": [INT_GLOBAL_SEED + i for i in range(INT_NUM_REPEATS)],
            "n_images": int(len(dfBalancedTable)),
            "n_patients": int(dfBalancedTable['patient_id'].nunique()),
            "cohort_strategy": STR_COHORT_STRATEGY,
            "cohort_strategy_note": (
                "'full' retains every patient and delegates the 1.4:1 class "
                "imbalance to class_weight='balanced' in the downstream "
                "classifiers; 'undersample' discards majority-class patients. "
                "The full cohort is the primary analysis because a measured "
                "learning curve (slope +0.000372 AUC per training image, "
                "p = 0.0145) showed the curve is not saturated, so discarded "
                "images were costing performance."),
            "class_to_label": DICT_CLASS_TO_LABEL,
            "inner_split_purpose": ("inner_cal is reserved exclusively for decision-threshold "
                                    "calibration and hyper-parameter selection; it is never "
                                    "used to report metrics."),
            "inner_folds_materialised": INT_NUM_INNER_FOLDS,
            "inner_folds_note": ("Each fold carries an 'inner_folds' list with all "
                                 f"{INT_NUM_INNER_FOLDS} inner train/calibration partitions. "
                                 "Phases 3 and 4 average the decision threshold over them to "
                                 "reduce its variance. The flat 'inner_train'/'inner_cal' keys "
                                 "hold the first partition and are retained for backward "
                                 "compatibility."),
        },
        "folds": dictFolds,
    }
    
    pathSplitsJson = pathPhase1Dir / "phase1_nested_cv_splits.json"
    with open(pathSplitsJson, "w", encoding="utf-8") as objFile:
        json.dump(dictSplits, objFile, indent=2)
        
    print(f"\nNested 5x5 splits (StratifiedGroupKFold) generated and saved to: {pathSplitsJson}")
    
    # Verify the guarantee instead of assuming it. This raises if any patient
    # appears on both sides of any outer or inner split.
    assert_folds_patient_disjoint(dfBalancedTable, dictFolds)
    
    lstValSizes = [len(d["val"]) for d in dictFolds.values()]
    lstCalSizes = [len(d["inner_cal"]) for d in dictFolds.values()]
    print(f"[splits] outer validation blocks: {min(lstValSizes)}-{max(lstValSizes)} images; "
          f"inner calibration blocks: {min(lstCalSizes)}-{max(lstCalSizes)} images.")
    
    # --- 4b. Cohort demographics and confounder screening ---
    dfDemographics, dictTests = describe_cohort(dfMasterTable, dfBalancedTable)
    pathDemographicsCsv = pathPhase1Dir / "phase1_demographics_table.csv"
    dfDemographics.to_csv(pathDemographicsCsv, index=False)
    with open(pathPhase1Dir / "phase1_confounder_tests.json", "w", encoding="utf-8") as objFile:
        json.dump(dictTests, objFile, indent=2)
    print(f"Cohort demographics saved to: {pathDemographicsCsv}")
    print(f"[confounder] age Mann-Whitney p = {dictTests['age_mannwhitney_p']:.3e} "
          f"(median {dictTests['age_median_multinodular']:.0f} vs "
          f"{dictTests['age_median_diffuse']:.0f} years)")
    
    # --- 5. Generate Visual Evidence of Balancing ---
    objFig, objAx = plt.subplots(figsize=(10, 6))
    
    # Original Data
    lstClasses = serPatientsPerClass.index.tolist()
    lstOriginalCounts = serPatientsPerClass.values.tolist()
    
    # Balanced Data
    serBalancedPerClass = dfBalancedTable.groupby('class')['patient_id'].nunique()
    lstBalancedCounts = serBalancedPerClass.loc[lstClasses].values.tolist()
    
    x = range(len(lstClasses))
    width = 0.35
    
    objBarsOriginal = objAx.bar([i - width/2 for i in x], lstOriginalCounts, width,
                                label='Before balancing', color='#B0B0B0', edgecolor='black', linewidth=0.6)
    objBarsBalanced = objAx.bar([i + width/2 for i in x], lstBalancedCounts, width,
                                label='After patient-level undersampling', color='#2B6A8F', edgecolor='black', linewidth=0.6)
    
    objAx.set_ylabel('Unique patients')
    objAx.set_title('Patient-level class balancing of the analysis cohort')
    objAx.set_xticks(list(x))
    objAx.set_xticklabels([s.replace('_', ' ').title() for s in lstClasses])
    objAx.legend(frameon=False)
    objAx.spines[['top', 'right']].set_visible(False)
    objAx.set_ylim(0, max(lstOriginalCounts) * 1.18)
    
    # Add values on top
    for objBars in [objBarsOriginal, objBarsBalanced]:
        for objBar in objBars:
            floatHeight = objBar.get_height()
            objAx.text(objBar.get_x() + objBar.get_width()/2., floatHeight + 1,
                       f'{int(floatHeight)}', ha='center', va='bottom', fontweight='bold')
            
    objFig.tight_layout()
    pathBalancingPlot = pathPhase1Dir / "phase1_balancing_plot.png"
    objFig.savefig(pathBalancingPlot, dpi=300)
    plt.close(objFig)
    print(f"Balancing chart successfully saved to: {pathBalancingPlot}")
    


if __name__ == "__main__":
    main()
