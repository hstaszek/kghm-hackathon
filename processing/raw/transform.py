import json
import logging
import os

import pandas as pd

from processing.raw.configs import csv_in_parameters, csv_out_parameters

log = logging.getLogger('base')


def transform_fn(src_path: str, src_schema_path: str, target_path: str):
    y_cols = {
        "HC201A": [
            "LMAM_HC201A_PL-------_TPS",
            "LMAM_HC201A_PLDT01---_TDI",
            "LMAM_HC201A_PLKL90---_TPG"
        ],
        "HC212B": [
            "LMAM_HC212B_PL-------_TPS",
            "LMAM_HC212B_PLDT01---_TDI",
            "LMAM_HC212B_PLKL90---_TPG"
        ]
    }

    with open(src_schema_path, "r") as f:
        schema = json.load(f)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    log.info(f"created target dir: {os.path.dirname(target_path)}")

    input_df = pd.read_csv(src_path, dtype=schema, **csv_in_parameters)
    input_df["Czas"] = pd.to_datetime(input_df["Czas"], format="%d.%m.%Y %H:%M:%S")
    input_df.sort_values(by="Czas", axis=0, inplace=True)

    in_rows_number = input_df.shape[0]
    log.info(f"rows number before filtering: {in_rows_number}")
    for device, cols in y_cols.items():
        log.info(f"creating files for device: {device}")
        for col in cols:
            output_df = input_df.copy(deep=True)
            output_df[col] = output_df[col].shift(1)
            output_df.drop([c for c in cols if c != col], axis=1, inplace=True)
            output_df = output_df[output_df[col].notnull() & output_df[col] != 0.0]
            output_df = output_df[[c for c in output_df.columns if c != col] + [col]]

            sub_target_path = target_path.format(f"{device}_{col}")
            log.info(f"saving target file: {sub_target_path}")
            output_df.to_csv(sub_target_path, **csv_out_parameters)

            out_rows_number = output_df.shape[0]
            filtered_percent = round(out_rows_number*100/in_rows_number, 2)
            log.info(f"rows number after filtering: {out_rows_number} ({filtered_percent}%)")

    log.info("transform_fn ends")
