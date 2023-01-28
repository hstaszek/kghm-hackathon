import logging
import os
import json

import pandas as pd

from processing.configs import csv_in_parameters, csv_out_parameters
from processing.schema import RAW_SCHEMA

log = logging.getLogger('base')


def parse_schema(schema_path: str):
    target_schema = pd.read_excel(schema_path, dtype={"tagname": "object"}, usecols=[0])
    target_schema = target_schema["tagname"].tolist()
    target_schema = {"Czas": "object", **{col: "float64" for col in target_schema}}

    target_schema_path = os.path.join(
        os.path.dirname(schema_path),
        os.path.basename(schema_path).replace("xlsx", "json")
    )
    with open(target_schema_path, "w") as f:
        json.dump(target_schema, f)
    log.info(f"target schema: {target_schema_path}")
    return target_schema_path


def collect_fn(src_path: str, out_path: str):
    loading_paths = [src_path]
    if src_path.endswith("*"):
        src_dir = os.path.dirname(src_path)
        loading_paths = []
        for path in os.listdir(src_dir):
            if path.endswith(".csv"):
                loading_paths.append(os.path.join(src_dir, path))
    loading_paths_number = len(loading_paths)
    log.info(f"number of files: {loading_paths_number}")

    target_path = os.path.join(out_path, "001_extract.csv")

    target_df = None
    is_initial = True
    for i, path in enumerate(loading_paths):
        log.info(f"{i + 1}/{loading_paths_number} processing: {path}")
        var_df = pd.read_csv(path, dtype=RAW_SCHEMA, usecols=["Zmienna", "Wartosc", "Czas"], **csv_in_parameters)

        unique_vars = var_df["Zmienna"].unique()
        if len(unique_vars) > 1:
            log.info(f"{i + 1}/{loading_paths_number} ! more than 2 vars {unique_vars}")

        for var in unique_vars:
            side_var_df = var_df[var_df["Zmienna"] == var]
            side_var_df = side_var_df[["Czas", "Wartosc"]].rename(columns={'Wartosc': var})
            if is_initial:
                target_df = side_var_df
                is_initial = False
                continue
            target_df = target_df.merge(side_var_df, how="outer", on="Czas")

    log.info(f"saving target file: {target_path}")
    target_df.to_csv(target_path, **csv_out_parameters)
    log.info("extract_fn ends")
    return target_path


def transform_fn(src_path: str, schema_path: str, section: str, out_path: str):
    target_path = os.path.join(out_path, f"002_{{}}.csv")

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log.info(f"created target dir: {os.path.dirname(out_path)}")

    input_df = pd.read_csv(src_path, dtype=schema, **csv_in_parameters)
    input_df["Czas"] = pd.to_datetime(input_df["Czas"], format="%d.%m.%Y %H:%M:%S")
    input_df.sort_values(by="Czas", axis=0, inplace=True)

    # TODO: apply mapping

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
