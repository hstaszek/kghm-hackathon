import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from utils.utils import remove_blacklisted, remove_redundant_columns, filter_device_groups
from collections import defaultdict
import json
import tqdm
from utils.utils import smooth_seven_minute_intervals, simple_smooth, holt

SHIFT = -4
SHIFT_HCS = 0
SMOOTH = 15

BRANCH = 'DOWN' # / 'DOWN'
DATA_DIR = 'data/'
SECTION = 3
GOAL_INDEX = 2

UP = ["LMAM_HC20??_PL-------_TPS", "LMAM_HC20??_PLDT01---_TDI", "LMAM_HC20??_PLKL90---_TPG"]
DOWN = ["LMAM_HC2?2B_PL-------_TPS", "LMAM_HC2?2B_PLDT01---_TDI", "LMAM_HC2?2B_PLKL90---_TPG"]


df = pd.read_csv(f'{DATA_DIR}/s{SECTION}_002_transform.csv')


if BRANCH == 'UP':
    y_name = UP[GOAL_INDEX]
    groups = ['1', '2', '3']
    with open('data/15common_up.json', 'r') as f:
        to_take = json.load(f)
else:
    y_name = DOWN[GOAL_INDEX]
    groups = ['1', '4', '5']
    with open('data/15common_down.json', 'r') as f:
        to_take = json.load(f)

    
#df = remove_blacklisted(df, file='feature_lists/blacklist15.txt')
df = remove_redundant_columns(df)


print(f'Predicted feature: {y_name}')
print(f'We use: {df.drop(columns=y_name).columns.to_list()}')

df = filter_device_groups(df, file='data/15groups_unified.json', groups=groups, y_name=y_name, anti_groups=[f'y1' if BRANCH == 'UP' else 'y2'])

print(f'Predicted feature: {y_name}')
print(f'We use: {df.drop(columns=y_name).columns.to_list()}')

legal = list()
for col in list(set(to_take) | set([y_name])):
    if col in df.columns:
        legal.append(col)
    else:
        print(f'KEY NOT IN INDEX!: {col}')
df = df[legal]


df = holt(df, [y_name], alpha=0.05)
df = holt(df, df.drop(columns=y_name).columns.to_list(), alpha=0.6)
    
df[y_name] = df[y_name].shift(SHIFT)
df = df[df[y_name].notnull() & df[y_name] != 0]


X = df.drop(columns=y_name).iloc[200:9000]
y = df[y_name].iloc[200:9000]

X_train = X
y_train = y

# KFOLD validation
k_splits = 10

fold_metrics = defaultdict(lambda: [])
y_pred_history = list()

for k in tqdm.tqdm(range(k_splits), desc='Training k-fold'):
    n = len(X_train)

    _from = k * n // k_splits
    _to = (k + 1) * n // k_splits

    fold_X_train = pd.concat([X_train.iloc[0:_from+1], X_train.iloc[_to:]])
    fold_y_train = pd.concat([y_train.iloc[0:_from+1], y_train.iloc[_to:]])

    fold_X_test = X_train.iloc[_from:_to]
    fold_y_test = y_train.iloc[_from:_to]

    params={ 
        'objective':'reg:squarederror',
        'max_depth': 5, 
        'n_estimators': 60,
        'colsample_bylevel': 0.4,
        'colsample_bytree': 0.7,
        'learning_rate': 0.2,
        'subsample': 0.5,
        'alpha': 0.9,
        'lambda': 0.9,
        'gamma': 0.9,
        'random_state':20
         }
    
    model = XGBRegressor(**params)
    #model = Lasso(positive=False, fit_intercept=True)
    model.fit(fold_X_train, fold_y_train)


    y_pred = model.predict(fold_X_test)

    fold_mae = metrics.mean_absolute_error(fold_y_test, y_pred)
    fold_mse = metrics.mean_squared_error(fold_y_test, y_pred)
    fold_mape = metrics.mean_absolute_percentage_error(fold_y_test, y_pred)
    fold_r2 = metrics.r2_score(fold_y_test, y_pred)

    fold_metrics['mae'].append(round(fold_mae, 3))
    fold_metrics['mape'].append(round(fold_mape, 3))
    fold_metrics['mse'].append(round(fold_mse, 3))
    fold_metrics['r2'].append(round(fold_r2, 3))

    y_pred_history.append(y_pred)

y_pred_history = np.concatenate(y_pred_history)

print('K-FOLD scores;')
for key, value in fold_metrics.items():
    print(f'{key}: {value};')

print(f"Mean mae {np.mean(fold_metrics['mae']):.2f};")
print(f"Mean mape {np.mean(fold_metrics['mape']):.2f}")
print(f"Mean mse {np.mean(fold_metrics['mse']):.2f};")
print(f"Mean r2 {np.mean(fold_metrics['r2']):.2f};")

stats = [
f"Mean mae {np.mean(fold_metrics['mae']):.2f};",
f"Mean mape {np.mean(fold_metrics['mape']):.2f}",
f"Mean mse {np.mean(fold_metrics['mse']):.2f};",
f"Mean r2 {np.mean(fold_metrics['r2']):.2f};"
]

x_range = range(len(y_pred_history))
fig = plt.figure(figsize=(14,8))
plt.plot(x_range, y_train, label='original')
plt.plot(x_range, y_pred_history, label='predicted')
plt.title(f'{y_name}\n{stats}\nModel:{type(model)}')
plt.savefig(f"images/unifu/s{SECTION}_{BRANCH}_{y_name.replace('?', '-')}.png")
plt.legend()
plt.show()