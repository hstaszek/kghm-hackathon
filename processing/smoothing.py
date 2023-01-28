from typing import List
import pandas as pd
import numpy as np


def wma_smoothing(df: pd.DataFrame, window_size: int = 5, outlier_n_stdev: int = 2, ignore_columns: List[str] = []) -> pd.DataFrame:
    '''
    Apply smoothing with weighted-moving-average.  
    '''

    raw_df: pd.DataFrame = df.copy().drop(columns=ignore_columns)

    #smoothed = list()
    smoothed = np.zeros(shape=raw_df.shape)

    stdevs = raw_df.std(axis=0)
    means = raw_df.mean(axis=0)

    print(stdevs)
    print(means)

    wmas = np.ones(shape=(window_size, raw_df.shape[1]))  * np.array([means for i in range(window_size)])


    for i, row in enumerate(df.values):
        wmas[range(window_size-1)] = wmas[range(1, window_size)] 
        wmas[window_size-1] = row

        outlier_mask = np.abs((row - means) / stdevs) > outlier_n_stdev

        wmas[window_size-1] = row
        wmas[window_size-1][outlier_mask] = np.mean(wmas[0:window_size-1, :], axis=0)[outlier_mask]

        smoothed[i] = wmas[window_size-1]


    smoothed_df = pd.DataFrame(data=smoothed, columns=raw_df.columns)

    if len(ignore_columns) > 0:
        pd.concat([smoothed_df, df[ignore_columns]], ignore_index=True)

    return smoothed_df


r = pd.read_csv('tmp.csv')
wma_smoothing(r)

