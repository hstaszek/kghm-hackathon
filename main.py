# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import xgboost as xgb

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    pandas_numpy_test = pd.Series([1, 3, 5, np.nan, 6, 8])
    print(pandas_numpy_test)
    xgb_test = xgb.DMatrix(pandas_numpy_test, label=np.random.randint(2, size=6))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
