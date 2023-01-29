import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

s1_raw = pd.read_csv('data/s1_002_transform.csv')
s2_raw = pd.read_csv('data/s3_002_transform.csv')

with open('models/s1/003_summary.json') as f:
    s1_summary = json.load(f)

with open('models/s3/003_summary.json') as f:
    s3_summary = json.load(f)


features = list()
mae = list()
mse = list()
mape = list()


for target, data in s1_summary['UP'].items():
    mae.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mae']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mae']):.2f}")
    mse.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mse']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mse']):.2f}")
    mape.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mape']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mape']):.2f}")
    features.append('s1 ' + target)

for target, data in s1_summary['DOWN'].items():
    mae.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mae']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mae']):.2f}")
    mse.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mse']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mse']):.2f}")
    mape.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mape']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mape']):.2f}")
    features.append('s1 ' + target)

for target, data in s3_summary['UP'].items():
    mae.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mae']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mae']):.2f}")
    mse.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mse']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mse']):.2f}")
    mape.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mape']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mape']):.2f}")
    features.append('s3 ' + target)

for target, data in s3_summary['DOWN'].items():
    mae.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mae']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mae']):.2f}")
    mse.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mse']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mse']):.2f}")
    mape.append(f"{np.mean(data['evaluation']['k_fold_scores:']['mape']):.2f} +- {np.std(data['evaluation']['k_fold_scores:']['mape']):.2f}")
    features.append('s3 ' + target)

tab = pd.DataFrame(index=features)
tab['MAE'] = mae
tab['MAPE'] = mape
tab['MSE'] = mse

from collections import defaultdict

all_gain = defaultdict(lambda: 0)
all_gain_counts = defaultdict(lambda: 0)


for i in range(6):
    with open(f'cover{i}.json', 'r') as f:
        dic = json.load(f)

        for key, value in dic.items():
            all_gain[key] += value
            all_gain_counts[key] += 1


to_print = list()
for feature, value in all_gain.items():
    to_print.append((value/all_gain_counts[feature], feature))

print(sorted(to_print))