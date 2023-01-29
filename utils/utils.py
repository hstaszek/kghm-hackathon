import json
from typing import List

import pandas as pd
import numpy as np

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, ARIMA

#from grupy import grupy_sekcji_1


def remove_blacklisted(df: pd.DataFrame, file: str = 'blacklist15.txt') -> pd.DataFrame:
    '''
    Remove dataframe columns based on names in blacklist text file.
    '''
    blacklist = list()

    # Drop blacklisted (correlated) columns
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            blacklist.append(line[:-1])

    blacklist = np.array(blacklist, dtype='str')

    to_drop = list()
    for col in df.columns.to_list():
        if col in blacklist:
            to_drop.append(col)    

    _df = df.drop(columns=to_drop)
    return _df


def select_whitelisted(df: pd.DataFrame, file: str, y_col: str) -> pd.DataFrame:
    with open(file, 'r') as f:
        whitelist = json.load(f)
    return df[whitelist + [y_col]]


def remove_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove columns like 'Czas' and 'idx' if they appear in DF and remove zero-std features.
    '''
    if 'Czas' in df.columns:
        df = df.drop(columns='Czas')

    if 'idx' in df.columns:
        df = df.drop(columns='idx')

    if 'idx.0' in df.columns:
        df = df.drop(columns='idx.0')

    if 'idx.1' in df.columns:
        df = df.drop(columns='idx.1')

    
    columns_without_zeros = df.columns.to_numpy()[~np.isclose(df.std(), 0.0, rtol=1e-4)]


    return df[columns_without_zeros]


def filter_device_groups(df: pd.DataFrame, file: str, groups: List[str], y_name: str,  anti_groups: List[str] = []):
    with open(file, 'r') as f:
        json_load = json.load(f)
    unloaded_lists = [col for group in groups for col in json_load.get(group)]
    unloaded_anty_lists = [col for group in anti_groups for col in json_load.get(group)]
    
    selected_list = list()
    for device in unloaded_lists:
        for col in df.columns:
            if device in col:
                allow = True
                for anty_device in unloaded_anty_lists:
                    if anty_device in col:
                        allow = False
                if allow:
                    selected_list.append(col)
    selected_list += [y_name]
    return df[selected_list]

def get_highly_correlated_columns_to_remove(correlated_dict: pd.DataFrame) -> list:
    '''
    Get colums that can be safely removed without loss of information. 

    `correlated_dict`: should be like {col: [corr_columns]}
    '''
    to_keep = set()
    to_remove = set()

    for feature, _list_of_correlated in correlated_dict.items():
        for correlated in _list_of_correlated:
            to_remove.add(correlated)
        
        if feature not in to_remove:
            to_keep.add(feature)

    print('To remove: ', to_remove)
    print('To keep:', to_keep)

    return list(to_remove)


def smooth_seven_minute_intervals(df: pd.DataFrame, columns_to_smooth: List[str], minutes: int = 7) -> pd.DataFrame:

    smoothed_df = df.copy()

    for col in columns_to_smooth:

        data = df[col].to_numpy()
        print(data.shape)
        kernel_size = minutes
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved = np.convolve(data, kernel, mode='same')

        smoothed_df[col] = data_convolved

    return smoothed_df


def simple_smooth(df: pd.DataFrame, columns_to_smooth: List[str], window_size: int = 5) -> pd.DataFrame:

    smoothed = df.copy()
    for col in columns_to_smooth:
        # f = SimpleExpSmoothing(df[col].to_numpy()).fit()
        # print(f)
        # df[col] = f.predict(df[col].to_numpy())
        smoothed[col] = df[col].rolling(window=window_size, min_periods=1, closed='both').mean()

    return smoothed

def holt(df: pd.DataFrame, cols: List[str], alpha: int = 0.5):
    for col in cols:
        tmp = SimpleExpSmoothing(df[col]).fit(smoothing_level=alpha, optimized=False, use_brute=True).fittedvalues
        df[col] = tmp.to_numpy().reshape((-1, 1))

    return df


if __name__ == '__main__':

    # TEST SMOOTHING
    hc = 'HC212B'
    prefix = f'tr_20230127213409_{hc}_'
    y_name = f'LMAM_{hc}_PLKL90---_TPG'
    groups = ["1", "2", "3"]

    df = pd.read_csv(f'dumbdata/{prefix}{y_name}.csv')

    df = remove_blacklisted(df, file='feature_lists/blacklist15.txt')
    df = remove_redundant_columns(df)
    df = filter_device_groups(df, file='feature_lists/15groups.json', groups=groups, y_name=y_name)

    print(df[y_name].iloc[:10])

    #ret = smooth_seven_minute_intervals(df, [y_name], minutes=20)
    ret = holt(df, [y_name])
    print(ret[y_name].iloc[:10])


    import matplotlib.pyplot as plt


    to_plot1 = df[y_name].iloc[200:500]
    to_plot2 = ret[y_name].iloc[200:500]


    fig = plt.figure(figsize=(15,15))
    #plt.plot(range(len(to_plot1)), to_plot1, label='original')
    plt.plot(range(len(to_plot2)), to_plot2, label='smoothed')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # # TEST device
    # hc = 'HC212B'
    # #prefix = f'tr_20230127213409_{hc}_'
    # y_name = f'LMAM_HC212B_PL-------_TPS'
    # groups = ["1", "4", "5"]


    # # hc = 'HC201A'
    # # y_name = f'LMAM_{hc}_PLKL90---_TPG'
    # # groups = ["1", "2", "3"]

    # #df = pd.read_csv(f'dumbdata/{prefix}{y_name}.csv')
    # df = pd.read_csv(f'maybe.csv')

    # df = remove_blacklisted(df, file='feature_lists/blacklist15.txt')
    # print('LMAM_HC212B_NDXRPI1--_PV' in df.columns)
    # df = remove_redundant_columns(df)
    # print('LMAM_HC212B_NDXRPI1--_PV' in df.columns)
    # df = filter_device_groups(df, file='feature_lists/15groups.json', groups=groups, y_name=y_name, anti_groups=['y2'])
    # print(df.columns.value_counts())