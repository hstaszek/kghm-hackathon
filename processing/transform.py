import json
import logging
import os
from typing import Dict

import pandas as pd

from processing.configs import csv_in_parameters, csv_out_parameters

log = logging.getLogger('base')


def transform_fn(conf: Dict, schema_path: str, postfix: str, section: str, src_path: str = None):
    if not src_path:
        src_path = conf.get("source_path")
        src_path = os.path.join(src_path, f"raw_{postfix}.csv")

    target_path = conf.get("target_path")
    if target_path.endswith("*"):
        target_dir = os.path.dirname(target_path)
        target_path = os.path.join(target_dir, f"tr_{postfix}_{{}}.csv")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    y_cols = {
        "ZWRL_1M2C_S1_20200908_15": {
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
        },
        "ZWRL_1M2C_S3_20210805_12": {
            "HC203B": [
                "LMAM_HC203B_PL-------_TPS",
                "LMAM_HC203B_PLDT01---_TDI",
                "LMAM_HC203B_PLKL90---_TPG"
            ],
            "HC232B": [
                "LMAM_HC232B_PL-------_TPS",
                "LMAM_HC232B_PLDT01---_TDI",
                "LMAM_HC232B_PLKL90---_TPG"
            ]
        }
    }

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    log.info(f"created target dir: {os.path.dirname(target_path)}")

    input_df = pd.read_csv(src_path, dtype=schema, **csv_in_parameters)
    input_df["Czas"] = pd.to_datetime(input_df["Czas"], format="%d.%m.%Y %H:%M:%S")
    input_df.sort_values(by="Czas", axis=0, inplace=True)

    in_rows_number = input_df.shape[0]
    log.info(f"initial - shape: {input_df.shape}")
    log.info(f"initial - rows number: {in_rows_number}")
    if section.upper() not in y_cols:
        raise ValueError(f"section {section} not exists")

    target_cols = y_cols.get(section.upper())
    for device, cols in target_cols.items():
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
            log.info(f"output - share: {output_df.shape}")
            log.info(f"output - rows number: {out_rows_number} ({filtered_percent}%)")

    log.info("transform_fn ends")
