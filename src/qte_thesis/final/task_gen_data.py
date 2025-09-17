from __future__ import annotations
from pathlib import Path
import pandas as pd
from qte_thesis.config import BLD_data
from itertools import product, combinations
import numpy as np
from qte_thesis.analysis.quantile_regression import PenalizedQuantileRegression

def task_generate_dataset(
    depends_1: Path = BLD_data/"cps_2020_2024.csv",
    depends_2: Path =BLD_data/"minimal_wage.csv",
    produces: Path = BLD_data/"real_data.csv",
) -> None:

    all_data = pd.read_csv(depends_1)

    # Keep only employed Employment Status == 10
    employed = all_data[all_data["EMPSTAT"]==10]
    # Drop Unpaid Family Worker
    employed = employed[employed["CLASSWKR"]!=29]
    # Keep only traceable working hours
    employed = employed[(employed["UHRSWORKT"]!=997)&(employed["UHRSWORKT"]!=999)]
    # Generate income reporting var
    employed["INC_YEAR"] = employed["YEAR"]-1
    # Generate hourly wage
    employed["HOURLY_WAGE"] = employed["INCWAGE"]/(employed["UHRSWORKLY"]*employed["WKSWORK1"])
    # Drop outliers
    employed = employed[(employed["HOURLY_WAGE"]>3)|(employed["HOURLY_WAGE"]<200)]

    minimum_wage = pd.read_csv(depends_2)
    # drop entries with no changing minimum wage
    data_col = ["2019", "2020", "2021", "2022", "2023"]
    minimum_wage = minimum_wage[
        minimum_wage[data_col].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    ].reset_index(drop=True)

    minimum_wage = minimum_wage[minimum_wage["notes"].isna()]
    print(len(minimum_wage))

    # Years present in the minimum_wage table
    year_cols = [c for c in minimum_wage.columns if c.isdigit()]

    # Long format: fips, YEAR, min_wage
    mw_long = minimum_wage.melt(
        id_vars=["fips"],
        value_vars=year_cols,
        var_name="YEAR",
        value_name="min_wage"
    )

    # Align dtypes for a clean join
    mw_long["YEAR"] = mw_long["YEAR"].astype(int)
    mw_long["fips"] = pd.to_numeric(mw_long["fips"], errors="coerce").astype("Int64")
    employed["STATEFIP"] = pd.to_numeric(employed["STATEFIP"], errors="coerce").astype("Int64")
    employed["INC_YEAR"] = pd.to_numeric(employed["INC_YEAR"], errors="coerce").astype(int)

    # Merge: add one column 'min_wage' to employed
    employed = (
        employed.merge(mw_long, left_on=["STATEFIP", "INC_YEAR"], right_on=["fips", "YEAR"], how="inner")
                .drop(columns=["fips"])
    )
    columns_to_keep = [
        'REGION','STATEFIP','COUNTY','METFIPS','METRO','CBSASZ','INDIVIDCC','PLACEFIPS',
        'OWNERSHP','PUBHOUS','RENTSUB','HEATSUB','HEATVAL','FOODSTMP','STAMPNO',
        'STAMPMO','ATELUNCH','LUNCHSUB','FRELUNCH','STAMPVAL','UNITSSTR','PHONE','RELATE',
        'AGE','SEX','RACE','MARST','ASIAN','VETSTAT','BPL','YRIMMIG','CITIZEN',
        'MBPL','FBPL','NATIVITY','HISPAN','OCC','IND','CLASSWKR','WKSTAT','EDUC','SCHLCOLL',
        'DIFFHEAR','DIFFEYE','DIFFREM','DIFFPHYS','DIFFMOB','DIFFCARE','DIFFANY','OCCLY',
        'INDLY','MIGSTA1','WHYMOVE','MIGRATE1','DISABWRK','HEALTH','HOURLY_WAGE','min_wage'
    ]

    existing = [c for c in columns_to_keep if c in employed.columns]
    missing = sorted(set(columns_to_keep) - set(existing))
    if missing:
        print("Missing columns dropped:", missing)

    employed = employed.loc[:, existing].copy()

    # columns to dummy-encode (drop continuous)
    categorical_columns = [
        'REGION','STATEFIP','COUNTY','METFIPS','METRO','CBSASZ','OWNERSHP','PUBHOUS','RENTSUB',
        'HEATSUB','FOODSTMP','STAMPMO','LUNCHSUB','UNITSSTR','PHONE','SEX','RACE','MARST','ASIAN',
        'VETSTAT','BPL','YRIMMIG','CITIZEN','MBPL','FBPL','NATIVITY','HISPAN','OCC','IND',
        'CLASSWKR','WKSTAT','EDUC','SCHLCOLL','DIFFHEAR','DIFFEYE','DIFFREM','DIFFPHYS','DIFFMOB',
        'DIFFCARE','DIFFANY','OCCLY','INDLY','MIGSTA1','WHYMOVE','MIGRATE1','DISABWRK','HEALTH'
    ]

    # keep only those that actually exist
    categorical_columns = [c for c in categorical_columns if c in employed.columns]

    employed = pd.get_dummies(
        employed,
        columns=categorical_columns,
        drop_first=True,      
        dtype=int,            
        dummy_na=False        
    )

    # def add_interactions(df, left_prefix, right_prefix, sep="_"):
    #     left = [c for c in df.columns if c.startswith(f"{left_prefix}{sep}")]
    #     right = [c for c in df.columns if c.startswith(f"{right_prefix}{sep}")]
    #     for a, b in product(left, right):
    #         df[f"{a}*{b}"] = df[a].values * df[b].values
    #     return df

    # control_interactions = ['SEX', 'RACE', "EDUC","MARST","CITIZEN"]

    # for left, right in combinations(control_interactions, 2):
    #     employed = add_interactions(employed, left, right)

    # def add_age_interactions(df, dummy_prefix="STATEFIP", sep="_"):
    #     dummies = [
    #         c for c in df.columns
    #         if c.startswith(f"{dummy_prefix}{sep}") and ("*" not in c)
    #     ]
    #     age = pd.to_numeric(df["AGE"], errors="coerce")
    #     age_c = (age - age.mean()).to_numpy(dtype=np.float64)
    #     for d in dummies:
    #         z = pd.to_numeric(df[d], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    #         df[f"AGE*{d}"] = age_c * z
    #     return df

    # for pref in control_interactions:
    #     employed = add_age_interactions(employed, pref)

    employed.to_csv(produces, index=False)