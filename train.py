import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from utils.utils import remove_blacklisted, remove_redundant_columns
import joblib

hc = 'HC201A'
prefix = f'tr_20230127213409_{hc}_'
y_name = f'LMAM_{hc}_PLKL90---_TPG'

df = pd.read_csv(f'dumbdata/{prefix}{y_name}.csv')

df = remove_blacklisted(df, file='feature_lists/blacklist15.txt')
df = remove_redundant_columns(df)


X = df.drop(columns=y_name).to_numpy()
y = df[y_name].to_numpy()


test_size = 500
split_index = 5000
X_train, X_test = np.concatenate([X[0:split_index + 1], X[split_index + test_size:]]), X[split_index:split_index + test_size + 1]
y_train, y_test = np.concatenate([y[0:split_index + 1], y[split_index + test_size:]]), y[split_index:split_index + test_size + 1]

model = XGBRegressor(
    # n_estimators=1000, 
    # max_depth=7, 
    # eta=0.1, 
    # subsample=0.7, 
    # colsample_bytree=0.8
    )


cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=44)

scores = cross_val_score(model, 
                         X_train, 
                         y_train, 
                         scoring='neg_mean_squared_error', 
                         cv=cv, 
                         n_jobs=4
                    )


model.fit(X_train, y_train)
test_pred = model.predict(X_test)


# plot feature importance
plot_importance(model)
plt.show()

joblib.dump(model, 'xgb15.joblib')



# force scores to be positive
scores = np.absolute(scores)
print('-'*20)
print('Scores')
print(f'Mean MSE: {scores.mean():.2f} +- {scores.std():.2f}')
print('-'*20)
print('Stats for Y')
print(f'Min {y.min():.2f}')
print(f'Max {y.max():.2f}')
print(f'Mean {y.mean():.2f}')
print(f'Std {y.std():.2f}')


fig = plt.figure()

test_n = len(test_pred)
plt.plot(range(test_n), y_test, label='True')
plt.plot(range(test_n), test_pred, label='Predicted')
plt.legend()
plt.show()

fig = plt.figure()
# Plot
plt.hist(y, bins=20)
plt.title(f'Y value for {y_name}')
plt.xlabel('jednostka?')
plt.tight_layout()
plt.show()