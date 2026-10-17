import numpy as np

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    compare_classical_models,
    sigmoid,
    train_linear_regression,
    train_logistic_classifier,
)


def make_regression_data(scale_first_column=1.0):
    """Данные, где ответ известен точно: y = 2*x1 - 1*x2 + 0.5*x3 + 3."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = x @ np.array([2.0, -1.0, 0.5]) + 3.0
    x[:, 0] = x[:, 0] * scale_first_column      # можно "испортить" масштаб
    return x, y


def make_classification_data():
    """Два облака точек: класс 0 около (-2, -2), класс 1 около (2, 2)."""
    rng = np.random.default_rng(1)
    x0 = rng.normal(loc=-2.0, size=(100, 2))
    x1 = rng.normal(loc=2.0, size=(100, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 100 + [1] * 100)
    order = rng.permutation(len(y))             # перемешиваем
    return x[order], y[order]


# Линейная регрессия 
def test_linear_regression_finds_true_weights():
    x, y = make_regression_data()
    data = RegressionInput(x[:150], y[:150], x[150:], y[150:])
    result = train_linear_regression(data)
    assert np.allclose(result.weights, [2.0, -1.0, 0.5], atol=0.01)
    assert abs(result.bias - 3.0) < 0.01


def test_linear_regression_loss_history():
    x, y = make_regression_data()
    data = RegressionInput(x[:150], y[:150], x[150:], y[150:], epochs=300)
    result = train_linear_regression(data)
    assert len(result.loss_history) == 300           # по одному числу на эпоху
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.loss_history[-1] < 1e-3


def test_linear_regression_predictions_shape_and_quality():
    x, y = make_regression_data()
    data = RegressionInput(x[:150], y[:150], x[150:], y[150:])
    result = train_linear_regression(data)
    assert len(result.predictions) == 50
    assert np.mean((result.predictions - y[150:]) ** 2) < 1e-3


def test_linear_regression_works_with_huge_feature_scale():
    # первый признак умножен на 1000: без масштабирования обучение бы "взорвалось"
    x, y = make_regression_data(scale_first_column=1000.0)
    data = RegressionInput(x[:150], y[:150], x[150:], y[150:])
    result = train_linear_regression(data)
    assert np.all(np.isfinite(result.loss_history))
    assert np.mean((result.predictions - y[150:]) ** 2) < 1.0


def test_linear_regression_constant_column_does_not_crash():
    x, y = make_regression_data()
    x[:, 1] = 5.0                                    # постоянный признак (разброс = 0)
    data = RegressionInput(x[:150], y[:150], x[150:], y[150:])
    result = train_linear_regression(data)
    assert np.all(np.isfinite(result.predictions))


# Логистическая регрессия 
def test_sigmoid_values():
    assert sigmoid(0) == 0.5
    assert sigmoid(100) > 0.999
    assert sigmoid(-100) < 0.001
    assert np.isfinite(sigmoid(-10000))              # без переполнения


def test_logistic_separates_two_blobs():
    x, y = make_classification_data()
    data = ClassificationInput(x[:150], y[:150], x[150:], y[150:])
    result = train_logistic_classifier(data)
    accuracy = np.mean(result.predictions == y[150:])
    assert accuracy > 0.95


def test_logistic_outputs_format():
    x, y = make_classification_data()
    data = ClassificationInput(x[:150], y[:150], x[150:], y[150:], epochs=200)
    result = train_logistic_classifier(data)
    assert len(result.loss_history) == 200
    assert result.loss_history[-1] < result.loss_history[0]
    assert np.all((result.probabilities >= 0) & (result.probabilities <= 1))
    assert set(np.unique(result.predictions)).issubset({0, 1})
    assert len(result.probabilities) == 50
    assert len(result.weights) == 2


def test_logistic_first_loss_is_ln2():
    # при нулевых весах модель всегда даёт 0.5, а кросс-энтропия равна ln(2) = 0.6931
    x, y = make_classification_data()
    data = ClassificationInput(x[:150], y[:150], x[150:], y[150:], epochs=5)
    result = train_logistic_classifier(data)
    assert abs(result.loss_history[0] - np.log(2)) < 1e-6


# Сравнение моделей 
def test_compare_models_returns_valid_metrics():
    x, y = make_classification_data()
    data = ClassificationInput(x[:150], y[:150], x[150:], y[150:])
    result = compare_classical_models(data)
    for metrics in (result.logistic_regression, result.decision_tree, result.knn):
        for value in (metrics.roc_auc, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1):
            assert 0.0 <= value <= 1.0
    assert result.logistic_regression.accuracy > 0.9