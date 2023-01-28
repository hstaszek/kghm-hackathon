import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
from xgboost import XGBRegressor

from training.evaluation import calculate_basic_info, calculate_basic_metrics, plot_importances, plot_erros, \
    plot_predicted, plot_histogram
from utils.utils import remove_redundant_columns

all_scores = []
stage_df = pd.DataFrame({})

TEST_SIZE = 1000
SPLIT_INDEX = 3000

hc = "HC201A"

src_path = "../output/target/ZWRL_1M2C_S1_20200908_15/20230127223039/train/tr_{hc}_{y_col}.csv"
y_name = ['LMAM_{}_PLKL90---_TPG', 'LMAM_{}_PLDT01---_TDI', 'LMAM_{}_PL-------_TPS']
model_params = {}

for y_col in y_name:
    y_col = y_col.format(hc)
    print(f"processing {y_col}")

    src_file = src_path.format(hc=hc, y_col=y_col)
    print("reading from:", src_file)
    df = pd.read_csv(src_file)
    model = XGBRegressor(**model_params)

    df = remove_redundant_columns(df)

    print("number of columns:", len(df.columns))

    X = df.drop(columns=y_col)
    y = df[y_col]

    X_train = pd.concat([X.iloc[0:SPLIT_INDEX + 1], X.iloc[SPLIT_INDEX + TEST_SIZE:]])
    X_test = X.iloc[SPLIT_INDEX:SPLIT_INDEX + TEST_SIZE + 1]

    y_train = pd.concat([y.iloc[0:SPLIT_INDEX + 1], y.iloc[SPLIT_INDEX + TEST_SIZE:]])
    y_test = y.iloc[SPLIT_INDEX:SPLIT_INDEX + TEST_SIZE + 1]

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=44)
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    abs_scores = np.absolute(scores)
    print('-' * 20)
    print('Scores')
    print(f'Mean MSE: {abs_scores.mean():.2f} +- {abs_scores.std():.2f}')

    calculate_basic_info(y)
    results = calculate_basic_metrics(y_test, y_pred)
    plot_importances(model, y_col)
    plot_erros(y_test, y_pred)
    plot_predicted(y_test, y_pred)
    plot_histogram(y)

    all_scores.append([y_col, *results])


hc = 'HC212B'

src_path = "../output/target/ZWRL_1M2C_S1_20200908_15/20230127223039/train/tr_{hc}_{y_col}.csv"
y_name = ['LMAM_{}_PLKL90---_TPG', 'LMAM_{}_PLDT01---_TDI', 'LMAM_{}_PL-------_TPS']
model_params = {}

for y_col in y_name:
    y_col = y_col.format(hc)
    print(f"processing {y_col}")

    src_file = src_path.format(hc=hc, y_col=y_col)
    print("reading from:", src_file)
    df = pd.read_csv(src_file)
    model = XGBRegressor(**model_params)

    df = remove_redundant_columns(df)

    print("number of columns:", len(df.columns))

    X = df.drop(columns=y_col)
    y = df[y_col]

    X_train = pd.concat([X.iloc[0:SPLIT_INDEX + 1], X.iloc[SPLIT_INDEX + TEST_SIZE:]])
    X_test = X.iloc[SPLIT_INDEX:SPLIT_INDEX + TEST_SIZE + 1]

    y_train = pd.concat([y.iloc[0:SPLIT_INDEX + 1], y.iloc[SPLIT_INDEX + TEST_SIZE:]])
    y_test = y.iloc[SPLIT_INDEX:SPLIT_INDEX + TEST_SIZE + 1]

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=44)
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    abs_scores = np.absolute(scores)
    print('-' * 20)
    print('Scores')
    print(f'Mean MSE: {abs_scores.mean():.2f} +- {abs_scores.std():.2f}')

    calculate_basic_info(y)
    results = calculate_basic_metrics(y_test, y_pred)
    plot_importances(model, y_col)
    plot_erros(y_test, y_pred)
    plot_predicted(y_test, y_pred)
    plot_histogram(y)

    all_scores.append([y_col, *results])

