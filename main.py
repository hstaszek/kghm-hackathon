# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import logging.config
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from processing.raw.extract import extract_fn, parse_schema
from processing.raw.transform import transform_fn

logging.config.fileConfig("logging.conf")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press ⌘F8 to toggle the breakpoint.
    pandas_numpy_test = pd.Series([1, 3, 5, np.nan, 6, 8])
    print(pandas_numpy_test)
    xgb_test = xgb.DMatrix(pandas_numpy_test, label=np.random.randint(2, size=6))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # print_hi("PyCharm")

    BASEDIR = "data"
    DATASET = "ZWRL_1M2C_S1_20200908_15"
    SCHEMA = "His_2c_zmienne_s1.xlsx"
    TARGET_POSTFIX = datetime.now().strftime("%Y%m%d%H%M%S")

    schema_path = os.path.join(BASEDIR, DATASET, SCHEMA)
    src_dir = os.path.join(BASEDIR, DATASET)
    target_dir = os.path.join(BASEDIR, "target", DATASET)

    target_paths = {
        "extract": os.path.join(target_dir, "raw", f"raw_{TARGET_POSTFIX}.csv"),
        "transform": os.path.join(target_dir, "train", f"tr_{TARGET_POSTFIX}.csv"),
    }

    schema_path = parse_schema(schema_path=schema_path)
    extract_fn(src_dir=src_dir, target_path=target_paths.get("extract"))
    transform_fn(
        src_path=target_paths.get("raw"),
        src_schema_path=schema_path,
        target_path=target_paths.get("transform")
    )
