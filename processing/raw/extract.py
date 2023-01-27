import logging
import os

import pandas as pd

from processing.raw.schema import RAW_SCHEMA

log = logging.getLogger('base')

csv_in_parameters = {
    "sep": ",",
    "header": 0,
    "skip_blank_lines": True,
    "encoding": "utf-8"
}
csv_out_parameters = {
    "mode": "w",
    "index": True,
    "index_label": "idx",
    "sep": ",",
    "encoding": "utf-8",
}


def extract_fn(src_dir: str, target_schema_path: str, target_path: str):
    target_schema = pd.read_excel(target_schema_path, dtype={"tagname": "object"}, usecols=[0])
    target_schema = target_schema["tagname"].tolist()
    target_schema = {"Czas": "object", **{col: "float64" for col in target_schema}}
    log.info(f"target schema: {target_schema}")

    loading_paths = []
    for path in os.listdir(src_dir):
        if path.endswith(".csv"):
            loading_paths.append(os.path.join(src_dir, path))
    loading_paths_number = len(loading_paths)
    log.info(f"number of files: {loading_paths_number}")

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    log.info(f"created target dir: {os.path.dirname(target_path)}")

    target_df = None
    is_initial = True
    for i, path in enumerate(loading_paths):
        log.info(f"{i+1}/{loading_paths_number} processing: {path}")
        var_df = pd.read_csv(path, dtype=RAW_SCHEMA, usecols=["Zmienna", "Wartosc", "Czas"], **csv_in_parameters)

        unique_vars = var_df["Zmienna"].unique()
        if len(unique_vars) > 1:
            log.info(f"{i+1}/{loading_paths_number} ! more than 2 vars {unique_vars}")

        for var in unique_vars:
            side_var_df = var_df[var_df["Zmienna"] == var]
            side_var_df = side_var_df[["Czas", "Wartosc"]].rename(columns={'Wartosc': var})
            if is_initial:
                target_df = side_var_df
                is_initial = False
                continue
            target_df = target_df.merge(side_var_df, how="outer", on="Czas")

    log.info(f"saving target file: {target_path}")
    target_df.to_csv(
        target_path,
        **csv_out_parameters
    )
    log.info("extract_fn ends")
