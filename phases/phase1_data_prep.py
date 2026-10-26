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

INT_NUM_REPEATS = 10
STR_COHORT_STRATEGY = "full"
INT_NUM_FOLDS = 5
INT_NUM_INNER_FOLDS = 4          


def build_inner_folds(dfTrainBlock, intSeed):
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
    set_global_seed(INT_GLOBAL_SEED)
    
    pathDataDir = Path("data")
    pathPhase1Dir = Path("outputs/phase1")
    
    pathPhase1Dir.mkdir(parents=True, exist_ok=True)
    
    sys.stdout = TeeLogger(pathPhase1Dir / "phase1_run.log")
    print("=== Phase 1: Data Preparation and Patient-Level Balancing ===")
    
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
                    
                strPatientId = pathImage.stem
                strRelativePath = pathImage.as_posix()
                
                lstRecords.append({
                    "sample_id": strRelativePath,
                    "filepath": strRelativePath,
                    "filename": pathImage.name,
                    "class": strClassName,
                    "view": strViewName,
                    "patient_id": strPatientId
                })
                
    dfMasterTable = pd.DataFrame(lstRecords).sort_values("sample_id").reset_index(drop=True)
    print(f"Total raw images parsed into Master Table: {len(dfMasterTable)}")
    
    pathClinicalData = pathDataDir / "datosDemograficos.xlsx"
    if pathClinicalData.exists():
        dfClinical = pd.read_excel(pathClinicalData)
        dfClinical['IMAGE'] = dfClinical['IMAGE'].astype(str)
        
        dfMasterTable = dfMasterTable.merge(
            dfClinical, 
            left_on='patient_id', 
            right_on='IMAGE', 
            how='left'
        )
        
    pathMasterCsv = pathPhase1Dir / "phase1_master_table.csv"
    dfMasterTable.to_csv(pathMasterCsv, index=False)
    print(f"Master Table successfully saved to: {pathMasterCsv}")
    
    serPatientsPerClass = dfMasterTable.groupby('class')['patient_id'].nunique()
    print("\nOriginal patient distribution across classes:")
    print(serPatientsPerClass)
    
    intMinPatients = int(serPatientsPerClass.min())

    if STR_COHORT_STRATEGY == "full":
        dfBalancedTable = (dfMasterTable
                           .sort_values('sample_id')
                           .reset_index(drop=True)
                           .copy())
        print(f"\nCohort strategy: FULL cohort retained "
              f"({dfBalancedTable['patient_id'].nunique()} patients). "
              f"Imbalance handled by class weighting downstream.")
    elif STR_COHORT_STRATEGY == "undersample":
        lstBalancedPatientIds = []
        for strCls in LST_TARGET_CLASSES:
            lstClassPatients = sorted(
                dfMasterTable.loc[dfMasterTable['class'] == strCls, 'patient_id'].unique().tolist()
            )
            lstSelectedPatients = random.sample(lstClassPatients, intMinPatients)
            lstBalancedPatientIds.extend(lstSelectedPatients)

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
    
    dictFolds = {}
    
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
            
            lstInnerFolds = build_inner_folds(dfTrainBlock, intSeed)

            lstInnerFoldRecords = []
            for dfInnerTrain, dfInnerCal in lstInnerFolds:
                lstInnerFoldRecords.append({
                    "inner_train": dfInnerTrain['sample_id'].tolist(),
                    "inner_cal": dfInnerCal['sample_id'].tolist(),
                    "inner_train_patients": sorted(dfInnerTrain['patient_id'].unique().tolist()),
                    "inner_cal_patients": sorted(dfInnerCal['patient_id'].unique().tolist()),
                })

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
    assert_folds_patient_disjoint(dfBalancedTable, dictFolds)
    
    lstValSizes = [len(d["val"]) for d in dictFolds.values()]
    lstCalSizes = [len(d["inner_cal"]) for d in dictFolds.values()]
    print(f"[splits] outer validation blocks: {min(lstValSizes)}-{max(lstValSizes)} images; "
          f"inner calibration blocks: {min(lstCalSizes)}-{max(lstCalSizes)} images.")
    
    dfDemographics, dictTests = describe_cohort(dfMasterTable, dfBalancedTable)
    pathDemographicsCsv = pathPhase1Dir / "phase1_demographics_table.csv"
    dfDemographics.to_csv(pathDemographicsCsv, index=False)
    with open(pathPhase1Dir / "phase1_confounder_tests.json", "w", encoding="utf-8") as objFile:
        json.dump(dictTests, objFile, indent=2)
    print(f"Cohort demographics saved to: {pathDemographicsCsv}")
    print(f"[confounder] age Mann-Whitney p = {dictTests['age_mannwhitney_p']:.3e} "
          f"(median {dictTests['age_median_multinodular']:.0f} vs "
          f"{dictTests['age_median_diffuse']:.0f} years)")
    
    objFig, objAx = plt.subplots(figsize=(10, 6))
    lstClasses = serPatientsPerClass.index.tolist()
    lstOriginalCounts = serPatientsPerClass.values.tolist()
    
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
