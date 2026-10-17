"""Лабораторная №3: линейные модели, написанные вручную."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


# Вспомогательные функции
def scale_features(train_x, test_x):
    """Стандартизация: из каждого признака вычитаем среднее и делим на разброс.

    Среднее и разброс считаем ТОЛЬКО по обучающей части (по тесту нельзя,
    иначе мы "подсмотрим" в тестовые данные).
    """
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0              # если признак постоянный, делить на 0 нельзя
    train_scaled = (train_x - mean) / std
    test_scaled = (test_x - mean) / std
    return train_scaled, test_scaled, mean, std


def sigmoid(z):
    """Сигмоида: превращает любое число в число от 0 до 1."""
    z = np.clip(z, -500, 500)        # защита от переполнения в exp
    return 1.0 / (1.0 + np.exp(-z))


def calculate_metrics(y_true, probabilities, predictions) -> ClassificationMetrics:
    """Считает пять метрик классификации."""
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )


# Задача 1. Линейная регрессия
def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    # Шаг 1. Превращаем входные данные в числовые массивы NumPy
    train_x = np.asarray(data.train_features, dtype=float)
    train_y = np.asarray(data.train_target, dtype=float)
    test_x = np.asarray(data.test_features, dtype=float)

    # Шаг 2. Масштабируем признаки, чтобы градиентный спуск сходился
    train_scaled, test_scaled, mean, std = scale_features(train_x, test_x)

    n_samples, n_features = train_scaled.shape
    weights = np.zeros(n_features)   # начальные веса - нули
    bias = 0.0                       # начальное смещение
    loss_history = []

    # Шаг 3. Градиентный спуск: много раз немного поправляем веса
    for epoch in range(data.epochs):
        predictions = train_scaled @ weights + bias     # прогноз модели
        error = predictions - train_y                   # ошибка по каждому объекту

        loss = np.mean(error ** 2)                      # MSE
        loss_history.append(float(loss))

        # Градиенты MSE: в какую сторону менять веса, чтобы ошибка уменьшилась
        grad_weights = 2 * (train_scaled.T @ error) / n_samples
        grad_bias = 2 * np.mean(error)

        weights = weights - data.learning_rate * grad_weights
        bias = bias - data.learning_rate * grad_bias

    # Шаг 4. Переводим веса обратно в исходный масштаб признаков,
    # чтобы они подходили к данным, которые нам передали
    original_weights = weights / std
    original_bias = bias - np.sum(original_weights * mean)

    # Шаг 5. Прогноз для тестовой части
    test_predictions = test_x @ original_weights + original_bias

    return RegressionResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        predictions=test_predictions,
    )


# Задача 2. Логистическая регрессия
def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    train_x = np.asarray(data.train_features, dtype=float)
    train_y = np.asarray(data.train_target, dtype=float)
    test_x = np.asarray(data.test_features, dtype=float)

    train_scaled, test_scaled, mean, std = scale_features(train_x, test_x)

    n_samples, n_features = train_scaled.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    tiny = 1e-12                     # чтобы не брать log(0)

    for epoch in range(data.epochs):
        probabilities = sigmoid(train_scaled @ weights + bias)   # вероятность класса 1

        # Бинарная кросс-энтропия (чем меньше, тем лучше)
        loss = -np.mean(
            train_y * np.log(probabilities + tiny)
            + (1 - train_y) * np.log(1 - probabilities + tiny)
        )
        loss_history.append(float(loss))

        # Градиенты кросс-энтропии (формула получается очень простой)
        error = probabilities - train_y
        grad_weights = (train_scaled.T @ error) / n_samples
        grad_bias = np.mean(error)

        weights = weights - data.learning_rate * grad_weights
        bias = bias - data.learning_rate * grad_bias

    original_weights = weights / std
    original_bias = bias - np.sum(original_weights * mean)

    test_probabilities = sigmoid(test_x @ original_weights + original_bias)
    test_predictions = (test_probabilities >= 0.5).astype(int)   # порог 0.5

    return ClassificationResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions,
    )


# Задача 3. Сравнение моделей
def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    train_x = np.asarray(data.train_features, dtype=float)
    train_y = np.asarray(data.train_target, dtype=int)
    test_x = np.asarray(data.test_features, dtype=float)
    test_y = np.asarray(data.test_target, dtype=int)

    # Масштабируем так же, как в собственной модели (важно для KNN и логистики)
    train_scaled, test_scaled, mean, std = scale_features(train_x, test_x)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(train_scaled, train_y)
        probabilities = model.predict_proba(test_scaled)[:, 1]   # вероятность класса 1
        predictions = model.predict(test_scaled)
        metrics[name] = calculate_metrics(test_y, probabilities, predictions)

    return BaselineComparisonResult(
        logistic_regression=metrics["logistic_regression"],
        decision_tree=metrics["decision_tree"],
        knn=metrics["knn"],
    )