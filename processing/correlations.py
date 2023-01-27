import pandas as pd
import numpy as np
import os


def find_corelations(src_csv: str, schema_path: str, target_path: str):
    raw_df = pd.read_csv(src_csv)
    print(raw_df.head(10))

    desc_excel = pd.read_excel(schema_path, dtype={"tagname": "object"}, usecols=[0, 1])

    target_path = target_path + "/correlations.csv"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    for c1 in raw_df.columns:
        for c2 in raw_df.columns:
            if c1 != c2 and c1 != 'Czas' and c2 != 'Czas':
                corr2 = raw_df[c1].corr(raw_df[c2])
                if np.isclose(corr2, 1.0) == 1:
                    with open(target_path, "a+") as file:
                        descr_c1 = desc_excel[desc_excel['tagname'] == c1]['description'].values[0]
                        descr_c2 = desc_excel[desc_excel['tagname'] == c2]['description'].values[0]
                        final_val = f"{c1},{descr_c1},{c2},{descr_c2},{corr2},\n"
                        print(final_val)
                        file.write(final_val)
