import json
from typing import List

import pandas as pd
import numpy as np

from grupy import grupy_sekcji_1


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


def filter_device_groups(df: pd.DataFrame, file: str, groups: List[str]):
    with open(file, 'r') as f:
        json_load = json.load(f)

    unloaded_lists = [col for group in groups for col in json_load.get(group)]

    return df[unloaded_lists]
