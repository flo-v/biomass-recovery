from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from typing import Tuple
import json

# from biomassrecovery.utils import wild_bootstrap as wb


def _filter_pct_agreement(pct_agreement, recovery_sample):
    # Filter for points with at least x% agreement on recovery age.
    nbins = np.arange(
        0, np.max(recovery_sample[~np.isnan(recovery_sample)]) + 2
    )
    hist = (
        np.apply_along_axis(
            lambda a: np.histogram(a, bins=nbins)[0],
            axis=1,
            arr=recovery_sample,
        )
        / recovery_sample.shape[1]
    )
    return np.max(hist, axis=1) >= (pct_agreement / 100)


def _filter_pct_nonnan(pct_agreement, recovery_sample):
    # Filter for points with at least x% non-nan values (x% recovering forest)
    # print("Filtering happens@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    non_na_ratio = np.sum(~np.isnan(recovery_sample), axis=1) / recovery_sample.shape[1]
    res = non_na_ratio >= pct_agreement / 100
    # print(non_na_ratio)
    # print(res)
    return res


def _mode(arr, axis):
    if arr.shape[0] > 0:
        nbins = np.arange(0, np.max(arr[~np.isnan(arr)]) + 2)
        hist = np.apply_along_axis(
            lambda a: np.histogram(a, bins=nbins)[0],
            axis=axis,
            arr=arr,
        )
        mode_count = np.max(hist, axis=axis)
        mode_val = np.argmax(hist, axis=axis)
        nan_count = np.sum(np.isnan(arr), axis=axis)

        mode_is_nan = np.where(nan_count > mode_count)

        mode_count[mode_is_nan] = nan_count[mode_is_nan]
        mode_val = np.argmax(hist, axis=axis)
        mode_val[mode_is_nan] = -1
        return mode_count, mode_val
    return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

def convert_to_json_compatible(value):
    return {str(k): int(v) for k, v in value.items()}

def filter_shots(opts, finterface, chunk_id: Tuple[int, str]):
    year, token = chunk_id
    master_df = finterface.load_data(token=token, year=year, data_type="master")
    recovery_sample = master_df.iloc[:,1:1001].to_numpy()

    # due to preliminary filtering (quickfilter_shots(): at least 5 out of 9 of the mean-location-estimate surrounding pixels need to be recovering) 
    # differences in low numbers of pctnonan (eg. 10 and 20) have less filtering impact because shots are already selected due to surrounding pixels (which have probabilistically most weight in lcoation distribution)

    # if clause unnecessary as script doesn't run without --filter_pctnonan flag
    # if 'pctnonan' in opts.filter_regime.keys():
    filter_idx = _filter_pct_nonnan(opts.filter_regime['pctnonan'], recovery_sample)
    filtered_recovery = recovery_sample[filter_idx]
    del recovery_sample # Free up some memory
    
    if 'maxstd' in opts.filter_regime.keys():
        # the way this was originally coded (using "np.std()") filters out every shot containing one sampled NA value making the pctnonan flag useless
        # this does filtering according to std allowing NA values in the distribution:
        filter_idx2 = np.nanstd(filtered_recovery, axis=1) <= opts.filter_regime['maxstd']

    filtered = master_df[filter_idx]
    del master_df # Free up some memory

    if 'maxstd' in opts.filter_regime.keys():
        filtered = filtered[filter_idx2]
    # Note: Cannot assign df["shot_number"] = filtered["shot_number"]
    # This implicitly converts to float64 (for unknown reasons)
    # which is not big enough to hold the shot numbers, and silently makes them NaN.

    # summaries recovery_period for both recovery_land_types
    recovery_other_land = []
    recovery_forest = []
    for i in range(filtered.shape[0]):
        # other_land (non_forest)
        tmp_idx = (filtered.iloc[i, 1001:2001] == 0).to_numpy()
        temp_recovery_period = filtered.iloc[i, 1:1001].loc[tmp_idx]
        unique_elements, counts = np.unique(temp_recovery_period, return_counts=True)
        recovery_other_land.append(dict(zip(unique_elements, counts)))
        # forest
        tmp_idx = (filtered.iloc[i, 1001:2001] == 1).to_numpy()
        temp_recovery_period = filtered.iloc[i, 1:1001].loc[tmp_idx]
        unique_elements, counts = np.unique(temp_recovery_period, return_counts=True)
        recovery_forest.append(dict(zip(unique_elements, counts)))

    filtered = pd.DataFrame({
        'shot_number': filtered.iloc[:, 0], 
        'recovery_other_land': recovery_other_land, 
        'recovery_forest': recovery_forest
        })
    
    # Convert dictionary columns to JSON strings
    filtered['recovery_other_land'] = filtered['recovery_other_land'].apply(lambda x: json.dumps(convert_to_json_compatible(x)))
    filtered['recovery_forest'] = filtered['recovery_forest'].apply(lambda x: json.dumps(convert_to_json_compatible(x)))

    finterface.save_data(
        token=token, year=year, data_type="filtered", data=filtered
    )

    return filtered

"""
def run_median_regression_model(experiment_id, dataframe):
    formula = "agcd_{experiment_id} ~ r_{experiment_id}".format(
        experiment_id=experiment_id
    )
    result = wb.wild_bootstrap(
        data=dataframe, formula=formula, tau=0.5, num_samples=10
    )
    result_df = pd.DataFrame(result, columns=["b0", "b1"])
    return result_df


def run_ols_medians_model(experiment_id, dataframe):
    recovery_col = "r_{}".format(experiment_id)
    agcd_col = "agcd_{}".format(experiment_id)
    recovery_periods = dataframe[recovery_col].unique()
    median_agcds = np.array(
        [
            dataframe.loc[dataframe[recovery_col] == x, agcd_col].median()
            for x in recovery_periods
        ]
    )
    df = pd.DataFrame(
        {
            "recovery": recovery_periods,
            "agcd": median_agcds,
        }
    )
    ols = smf.ols("agcd ~ recovery", df).fit()
    ols_ci = ols.conf_int().loc["recovery"].tolist()
    ols_intercept_ci = ols.conf_int().loc["Intercept"].tolist()
    result = [
        [
            ols.params["Intercept"],
            ols_intercept_ci[0],
            ols_intercept_ci[1],
            ols.params["recovery"],
            ols_ci[0],
            ols_ci[1],
            ols.rsquared,
        ]
    ]
    result_df = pd.DataFrame(
        result, columns=["a", "la", "ua", "b", "lb", "ub", "rs"]
    )
    return result_df
"""