import json
import logging
import os

import pandas as pd

from processing.raw.configs import csv_in_parameters, csv_out_parameters

log = logging.getLogger('base')


def transform_fn(src_path: str, src_schema_path: str, target_path: str):
    y_col = "LMAM_HC201A_PLKL90---_TPG"

    with open(src_schema_path, "r") as f:
        schema = json.load(f)

    input_df = pd.read_csv(src_path, dtype=schema, **csv_in_parameters)

    input_df["Czas"] = pd.to_datetime(input_df["Czas"], format="%d.%m.%Y %H:%M:%S")
    input_df.sort_values(by="Czas", axis=0, inplace=True)
    input_df[y_col] = input_df[y_col].shift(1)

    output_df = input_df[input_df[y_col].notnull() & input_df[y_col] != 0.0]

    log.info(f"saving target file: {target_path}")
    output_df.to_csv(target_path, **csv_out_parameters)
    log.info("extract_fn ends")
