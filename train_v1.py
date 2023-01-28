from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from pandas import DataFrame
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from utils.utils import remove_blacklisted, remove_redundant_columns, filter_device_groups, select_whitelisted, \
    smooth_seven_minute_intervals
import joblib


def fit_branch(
        in_df: DataFrame, in_model: XGBRegressor, y_col: str, in_groups: List[str],
        whitelist: Optional[str] = None,
        smooth_columns: Optional[List[str]] = None
    ):
    if whitelist:
        in_df = select_whitelisted(in_df, file=whitelist, y_col=y_col)
    else:
        in_df = remove_blacklisted(in_df, file='feature_lists/blacklist15.txt')
        in_df = remove_redundant_columns(in_df)
        in_df = filter_device_groups(in_df, file='feature_lists/15groups.json', groups=in_groups, y_name=y_col)

    if smooth_columns:
        in_df = smooth_seven_minute_intervals(in_df, columns_to_smooth=smooth_columns)

    print(len(in_df.columns))

    print(in_df.isna().sum())

    X = in_df.drop(columns=y_col)
    y = in_df[y_col]

    test_size = 1000
    split_index = 3000

    X_train = pd.concat([X.iloc[0:split_index + 1], X.iloc[split_index + test_size:]])
    X_test = X.iloc[split_index:split_index + test_size + 1]

    y_train = pd.concat([y.iloc[0:split_index + 1], y.iloc[split_index + test_size:]])
    y_test = y.iloc[split_index:split_index + test_size + 1]

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=44)

    scores = cross_val_score(
        in_model,
        X_train,
        y_train,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=4
    )

    in_model.fit(X_train, y_train)
    test_pred = in_model.predict(X_test)
    preds = in_model.predict(X)

    return in_model, (X, y, preds), (X_train, y_train, []), (X_test, y_test, test_pred), scores


def evaluate(model, hc, data, test, c_scores, with_plots=False):
    _, y, _ = data
    _, y_test, test_pred = test

    # joblib.dump(model, f'xgb15{hc}.joblib')

    # force scores to be positive
    abs_scores = np.absolute(c_scores)
    mse = metrics.mean_squared_error(y_test.to_numpy(), test_pred, multioutput="raw_values")[0]
    mae = metrics.mean_absolute_error(y_test.to_numpy(), test_pred, multioutput="raw_values")[0]
    mape = metrics.mean_absolute_percentage_error(y_test.to_numpy(), test_pred, multioutput="raw_values")[0]
    r_2 = metrics.r2_score(y_test.to_numpy(), test_pred, multioutput="raw_values")[0]
    print(f"{y_col} {mae} {mse} {mape} {r_2}")

    print('-' * 20)
    print('Scores')
    print(f'Mean MSE: {abs_scores.mean():.2f} +- {abs_scores.std():.2f}')
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"R2: {r_2}")
    print('-' * 20)
    print('Stats for Y')
    print(f'Min {y.min():.2f}')
    print(f'Max {y.max():.2f}')
    print(f'Mean {y.mean():.2f}')
    print(f'Std {y.std():.2f}')
    print('-' * 20)

    if with_plots:
        # # plot feature importance
        # fig, ax = plt.subplots(ncols=3, figsize=(16, 8))
        # plot_importance(model, max_num_features=20, ax=ax[0], title='weight', importance_type='weight')
        # plot_importance(model, max_num_features=20, ax=ax[1], title='gain', importance_type='gain')
        # plot_importance(model, max_num_features=20, ax=ax[2], title='cover', importance_type='cover')
        # plt.suptitle('XGBOOST internal feature importance')
        # plt.tight_layout()
        # plt.show()

        fig = plt.figure()

        error = y_test - test_pred
        plt.hist(error, bins=20)
        plt.title('Errors')
        plt.show()

        fig = plt.figure()

        test_n = len(test_pred)
        plt.plot(range(test_n), y_test, label='True')
        plt.plot(range(test_n), test_pred, label='Predicted')
        plt.legend()
        plt.show()

        fig = plt.figure()
        # Plot
        plt.hist(y, bins=20)
        plt.title(f'Y value for {y_col}')
        plt.xlabel('jednostka?')
        plt.tight_layout()
        plt.show()

    return [y_col, mae, mse, mape, r_2]


all_scores = []
stage_df = pd.DataFrame({})

hc = "HC201A"
src_path = "output/target/ZWRL_1M2C_S1_20200908_15/20230127223039/train/tr_{hc}_{y_col}.csv"
y_name = ['LMAM_{}_PLKL90---_TPG', 'LMAM_{}_PLDT01---_TDI', 'LMAM_{}_PL-------_TPS']
groups = ["1", "2", "3"]
model_params = {}
# whitelist = "feature_lists/15cleaned_features.json"
whitelist = None
smooth_columns = None

for y_col in y_name:
    y_col = y_col.format(hc)
    print(f"processing {y_col}")

    src_file = src_path.format(hc=hc, y_col=y_col)
    print("reading from:", src_file)
    df = pd.read_csv(src_file)
    model = XGBRegressor(**model_params)
    _, data, train, test, scores = fit_branch(df, model, y_col, groups, whitelist, smooth_columns)

    _, _, preds = data
    _, _, test_pred = test
    # stage_df[y_col] = preds

    results = evaluate(model, hc, data, test, scores)
    print(*results)
    all_scores.append(results)


print(f"\n{'='*20}")

hc = 'HC212B'
src_path = "output/target/ZWRL_1M2C_S1_20200908_15/20230127223039/train/tr_{hc}_{y_col}.csv"
y_name = ['LMAM_{}_PLKL90---_TPG', 'LMAM_{}_PLDT01---_TDI', 'LMAM_{}_PL-------_TPS']
groups = ["1", "4", "5"]
model_params = {}
whitelist = None

for y_col in y_name:
    y_col = y_col.format(hc)
    print(f"processing {y_col}")

    src_file = src_path.format(hc=hc, y_col=y_col)
    print("reading from:", src_file)
    df = pd.read_csv(src_file)

    # for col in stage_df.columns:
    #     print("swap column:", col)
    #     df[col].merge(on="Czas")

    df = df[:7300]
    # model = XGBRegressor(**model_params)
    model = ensemble.RandomForestRegressor(n_jobs=4, **model_params)
    _, data, train, test, scores = fit_branch(df, model, y_col, groups, whitelist)

    results = evaluate(model, hc, data, test, scores, with_plots=True)
    print(*results)
    all_scores.append(results)

print("\nfinal results:")
for metrics in all_scores:
    print(*metrics)
