from matplotlib import pyplot as plt
from sklearn import metrics
from xgboost import plot_importance


def calculate_basic_metrics(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true.to_numpy(), y_pred, multioutput="raw_values")[0]
    mae = metrics.mean_absolute_error(y_true.to_numpy(), y_pred, multioutput="raw_values")[0]
    mape = metrics.mean_absolute_percentage_error(y_true.to_numpy(), y_pred, multioutput="raw_values")[0]
    r_2 = metrics.r2_score(y_true.to_numpy(), y_pred, multioutput="raw_values")[0]

    print('-' * 20)
    print('Scores')
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"R2: {r_2}")
    print('-' * 20)

    return [mae, mse, mape, r_2]


def calculate_basic_info(y):
    print('-' * 20)
    print('Stats for Y')
    print(f'Min {y.min():.2f}')
    print(f'Max {y.max():.2f}')
    print(f'Mean {y.mean():.2f}')
    print(f'Std {y.std():.2f}')


def plot_importances(model, title="XGBOOST internal feature importance"):
    # plot feature importance
    fig, ax = plt.subplots(ncols=3, figsize=(16, 8))
    plot_importance(model, max_num_features=20, ax=ax[0], title='weight', importance_type='weight')
    plot_importance(model, max_num_features=20, ax=ax[1], title='gain', importance_type='gain')
    plot_importance(model, max_num_features=20, ax=ax[2], title='cover', importance_type='cover')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_erros(y_test, test_pred, title="Errors"):
    error = y_test - test_pred
    plt.hist(error, bins=20)
    plt.title(title)
    plt.show()


def plot_predicted(y_test, test_pred):
    test_n = len(test_pred)
    plt.plot(range(test_n), y_test, label='True')
    plt.plot(range(test_n), test_pred, label='Predicted')
    plt.legend()
    plt.show()


def plot_histogram(y, title="Y value"):
    plt.hist(y, bins=20)
    plt.title(title)
    plt.xlabel('jednostka?')
    plt.tight_layout()
    plt.show()
