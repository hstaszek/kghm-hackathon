import json
import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn import metrics
from xgboost import XGBRegressor

from processing.configs import csv_in_parameters
from utils.utils import simple_exp_smooth
from utils.utils import remove_redundant_columns, filter_device_groups

log = logging.getLogger('base')

SHIFT = -4
SHIFT_HCS = 0
SMOOTH = 15

UP_TARGETS = ["LMAM_HC20??_PL-------_TPS", "LMAM_HC20??_PLDT01---_TDI", "LMAM_HC20??_PLKL90---_TPG"]
DOWN_TARGETS = ["LMAM_HC2?2B_PL-------_TPS", "LMAM_HC2?2B_PLDT01---_TDI", "LMAM_HC2?2B_PLKL90---_TPG"]

RANGES = {
    1: {"UP": [200, 9000], "DOWN": [200, 6200]},
    3: {"UP": [200, 9000], "DOWN": [200, 6200]}
}

def train_fn(in_path: str, section: str, out_path: str):
    log.info("train fn process starts")

    section_map = {
        "ZWRL_1M2C_S1_20200908_15": 1,
        "ZWRL_1M2C_S3_20210805_12": 3
    }
    summary = {"section": section_map.get(section), "section_name": section}
    section = section_map.get(section)

    log.info(f"reading file from: {in_path}")
    initial_df = pd.read_csv(in_path, **csv_in_parameters)

    log.info(f"processing section: {section}")
    for branch in ["UP", "DOWN"]:
        log.info(f"processing branch: {branch}")
        if branch not in summary:
            summary[branch] = {}

        if branch == 'UP':
            y_names = UP_TARGETS
            groups = ['1', '2', '3']
            anti_groups = ["y1"]
            with open('configurations/15common_up.json', 'r') as f:
                to_take = json.load(f)
        else:
            y_names = DOWN_TARGETS
            groups = ['1', '4', '5']
            anti_groups = ["y2"]
            with open('configurations/15common_down.json', 'r') as f:
                to_take = json.load(f)

        for i, y_name in enumerate(y_names):
            log_post = f"[{i + 1}/{len(y_names)}][s{section}][{branch}][{y_name}]"
            log.info(f"{log_post} run training")
            b_summary = summary[branch]
            if y_name not in b_summary:
                b_summary[y_name] = {}
            y_summary = b_summary[y_name]

            df = initial_df.copy(deep=True)
            df = remove_redundant_columns(df)

            log.info(f'{log_post} predicted feature: {y_name}')

            df = filter_device_groups(
                df,
                file='data/15groups_unified.json',
                groups=groups,
                y_name=y_name,
                anti_groups=anti_groups
            )

            legal = list()
            for col in list(set(to_take) | set([y_name])):
                if col in df.columns:
                    legal.append(col)
                else:
                    log.info(f'{log_post} KEY NOT IN INDEX!: {col}')
            df = df[legal]
            log.info(f'{log_post} columns to use: {df.columns.to_list()}')
            y_summary["columns"] = df.columns.to_list()

            df = simple_exp_smooth(df, [y_name], alpha=0.05)
            df = simple_exp_smooth(df, df.drop(columns=y_name).columns.to_list(), alpha=0.9)

            df[y_name] = df[y_name].shift(SHIFT)
            df = df[df[y_name].notnull() & df[y_name] != 0]

            ranges = RANGES.get(section).get(branch)
            x = df.drop(columns=y_name).iloc[ranges[0]:ranges[1]]
            y = df[y_name].iloc[ranges[0]:ranges[1]]
            x_train, y_train = x, y

            # KFOLD validation
            k_splits = 10
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 5,
                'n_estimators': 60,
                'colsample_bylevel': 0.4,
                'colsample_bytree': 0.7,
                'learning_rate': 0.2,
                'subsample': 0.5,
                'alpha': 0.9,
                'lambda': 0.9,
                'gamma': 0.9,
                'random_state': 20
            }
            y_summary["model"] = {"type": "XGBRegressor", "parameters": params}
            fold_metrics = defaultdict(lambda: [])
            y_pred_history = list()
            for k in tqdm.tqdm(range(k_splits), desc='Training k-fold'):
                n = len(x_train)

                _from = k * n // k_splits
                _to = (k + 1) * n // k_splits

                fold_x_train = pd.concat([x_train.iloc[0:_from + 1], x_train.iloc[_to:]])
                fold_y_train = pd.concat([y_train.iloc[0:_from + 1], y_train.iloc[_to:]])

                fold_x_test = x_train.iloc[_from:_to]
                fold_y_test = y_train.iloc[_from:_to]

                model = XGBRegressor(**params)
                # model = Lasso(positive=False, fit_intercept=True)
                model.fit(fold_x_train, fold_y_train)

                y_pred = model.predict(fold_x_test)

                fold_mae = metrics.mean_absolute_error(fold_y_test, y_pred)
                fold_mse = metrics.mean_squared_error(fold_y_test, y_pred)
                fold_mape = metrics.mean_absolute_percentage_error(fold_y_test, y_pred)
                fold_r2 = metrics.r2_score(fold_y_test, y_pred)

                fold_metrics['mae'].append(round(fold_mae, 3))
                fold_metrics['mape'].append(round(fold_mape, 3))
                fold_metrics['mse'].append(round(fold_mse, 3))
                fold_metrics['r2'].append(round(fold_r2, 3))

                y_pred_history.append(y_pred)

            model = XGBRegressor(**params)
            model.fit(x_train, y_train)
            model_out_path = os.path.join(out_path, f"004_model_{section}_{branch}_{y_name}.json")
            log.info(f"save model at {model_out_path}")
            model.save_model(model_out_path)

            y_pred_history = np.concatenate(y_pred_history)
            log.info(f'{log_post} K-FOLD scores;')
            for key, value in fold_metrics.items():
                log.info(f'{log_post} {key}: {value};')

            mean_mae = np.mean(fold_metrics['mae'])
            mean_mape = np.mean(fold_metrics['mape'])
            mean_mse = np.mean(fold_metrics['mse'])
            mean_r2 = np.mean(fold_metrics['r2'])
            y_summary["evaluation"] = {
                "k_fold_splits": k_splits,
                "k_fold_scores:": {k: v for k, v in fold_metrics.items()},
                "mean_mae": mean_mae,
                "mean_mape": mean_mape,
                "mean_mse": mean_mse,
                "mean_r2": mean_r2
            }

            stats = [
                f"Mean mae {mean_mae:.2f};",
                f"Mean mape {mean_mape:.2f}",
                f"Mean mse {mean_mse:.2f};",
                f"Mean r2 {mean_r2:.2f};"
            ]
            log.info(f"{log_post} Mean mae {mean_mae:.2f};")
            log.info(f"{log_post} Mean mape {mean_mape:.2f}")
            log.info(f"{log_post} Mean mse {mean_mse:.2f};")
            log.info(f"{log_post} Mean r2 {mean_r2:.2f};")

            image_out_path = os.path.join(out_path, "images")
            os.makedirs(image_out_path, exist_ok=True)
            log.info(f"{log_post} created target dir: {image_out_path}")

            x_range = range(len(y_pred_history))
            _ = plt.figure(figsize=(14, 8))
            plt.plot(x_range, y_train, label='original')
            plt.plot(x_range, y_pred_history, label='predicted')
            plt.title(f'{y_name}\n{stats}\nModel:{type(model)}')
            plt.savefig(os.path.join(image_out_path, f"s{section}_{branch}_{y_name.replace('?', '-')}.png"))
            plt.legend()

    summ_out_path = os.path.join(out_path, "003_summary.json")
    log.info(f"saving summary at: {summ_out_path}")
    with open(summ_out_path, "w") as f:
        json.dump(summary, f)

    log.info("train fn process ends")
