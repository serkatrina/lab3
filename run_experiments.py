"""Запуск экспериментов лабораторной №3 (запуск: python run_experiments.py).

Скрипт: готовит данные, обучает модели, считает метрики, рисует графики
и записывает результаты в experiment_results.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    calculate_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

FOLDER = Path(__file__).parent
SEED = 42            # одно и то же число даёт одно и то же разбиение данных


def metrics_to_dict(metrics) -> dict:
    return {
        "roc_auc": metrics.roc_auc,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }


def unscaled_gradient_descent(x_train, y_train, learning_rate, epochs=800):
    """Градиентный спуск БЕЗ масштабирования признаков (только для эксперимента)."""
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    weights = np.zeros(x.shape[1])
    bias = 0.0
    history = []
    with np.errstate(all="ignore"):          # не засыпать вывод предупреждениями о переполнении
        for epoch in range(epochs):
            error = x @ weights + bias - y
            history.append(float(np.mean(error ** 2)))
            weights = weights - learning_rate * 2 * (x.T @ error) / len(y)
            bias = bias - learning_rate * 2 * np.mean(error)
    return history


# Часть 1. Линейная регрессия на Student_Performance.csv
def run_regression() -> dict:
    df = pd.read_csv(FOLDER / "Student_Performance.csv")

    # Текст "Yes"/"No" переводим в числа 1/0
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

    y = df["Performance Index"]
    x = df.drop(columns=["Performance Index"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

    data = RegressionInput(x_train, y_train, x_test, y_test)    # lr=0.05, epochs=800
    result = train_linear_regression(data)

    test_mse = float(np.mean((result.predictions - np.asarray(y_test)) ** 2))

    # Для сравнения: готовая модель sklearn и "глупый" прогноз средним значением
    sk_model = LinearRegression().fit(x_train, y_train)
    sk_mse = float(np.mean((sk_model.predict(x_test) - np.asarray(y_test)) ** 2))
    mean_mse = float(np.mean((np.asarray(y_train).mean() - np.asarray(y_test)) ** 2))

    print("=== Линейная регрессия ===")
    print("Признаки:", list(x.columns))
    print("Размеры: train", x_train.shape, "test", x_test.shape)
    print("Веса:", np.round(result.weights, 4), "смещение:", round(result.bias, 4))
    print("MSE на 1-й эпохе:", round(result.loss_history[0], 4))
    print("MSE на 100-й эпохе:", round(result.loss_history[99], 4))
    print("MSE на последней эпохе (train):", round(result.loss_history[-1], 4))
    print("MSE на test (своя модель):", round(test_mse, 4))
    print("MSE на test (sklearn LinearRegression):", round(sk_mse, 4))
    print("MSE на test (прогноз средним):", round(mean_mse, 4))

    # Эксперимент: как скорость обучения влияет на результат
    print("\nЭксперимент: learning_rate (epochs=800)")
    for lr in [0.001, 0.01, 0.05, 0.1, 0.5]:
        r = train_linear_regression(RegressionInput(x_train, y_train, x_test, y_test, lr, 800))
        mse = float(np.mean((r.predictions - np.asarray(y_test)) ** 2))
        print("  lr =", lr, " train MSE =", round(r.loss_history[-1], 4), " test MSE =", round(mse, 4))

    # Эксперимент: сколько эпох нужно
    print("\nЭксперимент: epochs (lr=0.05)")
    for epochs in [10, 50, 200, 800, 3000]:
        r = train_linear_regression(RegressionInput(x_train, y_train, x_test, y_test, 0.05, epochs))
        mse = float(np.mean((r.predictions - np.asarray(y_test)) ** 2))
        print("  epochs =", epochs, " test MSE =", round(mse, 4))

    # Эксперимент: что будет без масштабирования признаков
    print("\nЭксперимент: БЕЗ масштабирования (epochs=800)")
    for lr in [0.05, 0.001, 0.0001]:
        history = unscaled_gradient_descent(x_train, y_train, lr)
        print("  lr =", lr, " MSE на 11-й эпохе =", history[10], " на последней =", history[-1])

    # График хода обучения
    plt.figure(figsize=(7, 4))
    plt.plot(result.loss_history)
    plt.xlabel("Эпоха")
    plt.ylabel("MSE (train)")
    plt.title("Линейная регрессия: ход обучения")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FOLDER / "loss_regression.png")
    plt.close()

    return {"test_mse": test_mse}


# Части 2 и 3. Классификация на german.csv
def run_classification() -> dict:
    # В этом файле разделитель - точка с запятой!
    df = pd.read_csv(FOLDER / "german.csv", sep=";")

    y = df["Creditability"]                 # целевая переменная - первый столбец
    x = df.drop(columns=["Creditability"])

    # stratify=y сохраняет долю классов одинаковой в train и test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )

    data = ClassificationInput(x_train, y_train, x_test, y_test)    # lr=0.1, epochs=1000
    own = train_logistic_classifier(data)
    own_metrics = calculate_metrics(np.asarray(y_test), own.probabilities, own.predictions)
    baselines = compare_classical_models(data)

    print("\n=== Классификация (german.csv) ===")
    print("Размеры: train", x_train.shape, "test", x_test.shape)
    print("Доля класса 1 в train:", round(float(np.mean(y_train)), 4), "в test:", round(float(np.mean(y_test)), 4))
    print("Кросс-энтропия: 1-я эпоха", round(own.loss_history[0], 4),
          "последняя", round(own.loss_history[-1], 4))
    rows = [("own_logistic", own_metrics),
            ("logistic_regression", baselines.logistic_regression),
            ("decision_tree", baselines.decision_tree),
            ("knn", baselines.knn)]
    print("{:22s} {:>8s} {:>8s} {:>9s} {:>8s} {:>8s}".format("модель", "roc_auc", "accuracy", "precision", "recall", "f1"))
    for name, m in rows:
        print("{:22s} {:8.4f} {:8.4f} {:9.4f} {:8.4f} {:8.4f}".format(
            name, m.roc_auc, m.accuracy, m.precision, m.recall, m.f1))

    # Эксперимент: скорость обучения
    print("\nЭксперимент: learning_rate (epochs=1000), своя логистическая регрессия")
    for lr in [0.001, 0.01, 0.1, 1.0]:
        r = train_logistic_classifier(ClassificationInput(x_train, y_train, x_test, y_test, lr, 1000))
        m = calculate_metrics(np.asarray(y_test), r.probabilities, r.predictions)
        print("  lr =", lr, " loss =", round(r.loss_history[-1], 4),
              " roc_auc =", round(m.roc_auc, 4), " accuracy =", round(m.accuracy, 4))

    # Эксперимент: как зависят метрики от порога классификации
    print("\nЭксперимент: порог классификации (своя модель)")
    for threshold in [0.3, 0.5, 0.7]:
        preds = (own.probabilities >= threshold).astype(int)
        m = calculate_metrics(np.asarray(y_test), own.probabilities, preds)
        print("  порог =", threshold, " accuracy =", round(m.accuracy, 4),
              " precision =", round(m.precision, 4), " recall =", round(m.recall, 4),
              " f1 =", round(m.f1, 4))

    plt.figure(figsize=(7, 4))
    plt.plot(own.loss_history)
    plt.xlabel("Эпоха")
    plt.ylabel("Бинарная кросс-энтропия (train)")
    plt.title("Логистическая регрессия: ход обучения")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FOLDER / "loss_logistic.png")
    plt.close()

    return {
        "own_logistic": metrics_to_dict(own_metrics),
        "logistic_regression": metrics_to_dict(baselines.logistic_regression),
        "decision_tree": metrics_to_dict(baselines.decision_tree),
        "knn": metrics_to_dict(baselines.knn),
    }


def main() -> None:
    results = {
        "linear_regression": run_regression(),
        "classification": run_classification(),
    }
    with open(FOLDER / "experiment_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print("\nРезультаты записаны в experiment_results.json")

    # Версии библиотек: эту строку нужно вставить в раздел "Воспроизводимость" отчёта
    import sys
    import matplotlib
    import sklearn
    print("\nСтрока для REPORT.md (раздел 5, Воспроизводимость):")
    print("Python {}, NumPy {}, pandas {}, scikit-learn {}, matplotlib {}".format(
        sys.version.split()[0], np.__version__, pd.__version__,
        sklearn.__version__, matplotlib.__version__))


if __name__ == "__main__":
    main()