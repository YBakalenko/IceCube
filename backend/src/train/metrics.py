"""
Программа: Получение метрик
Версия: 1.0
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from ..data.get_data import get_config


def angular_dist_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Расчет среднего углового расстояния между векторами в декартовой системе координат.
    Возвращает результат в радианах
    :param y_true: тестовые данные с декартовыми координатами в формате [x, y, z]
    :param y_pred: предсказанные данные с декартовыми координатами в формате [x, y, z]
    """
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        raise ValueError("All arguments must be finite")

    y_pred_norm = np.array([np.linalg.norm(y_pred, axis=1)]).T
    y_pred = np.divide(y_pred, y_pred_norm)
    x_t = y_true.iloc[:, 0]
    y_t = y_true.iloc[:, 1]
    z_t = y_true.iloc[:, 2]
    x_p = y_pred[:, 0]
    y_p = y_pred[:, 1]
    z_p = y_pred[:, 2]
    scalar_prod = x_t * x_p + y_t * y_p + z_t * z_p
    score = np.average(np.arccos(scalar_prod))
    return score


def r2_adjusted(y_true: pd.Series, y_pred: np.ndarray,
                x_test: pd.DataFrame) -> float:
    """
    Коэффициент детерминации (множественная регрессия)
    :param y_true: тестовые данные целевой переменной
    :param y_pred: предсказанные данные целевой переменной
    :param x_test: тестовые данные с признаками
    """
    n_objects = len(y_true)
    n_features = x_test.shape[1]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)


def get_metrics(y_test: pd.Series,
                y_pred: np.ndarray,
                x_test: pd.DataFrame,
                dpt: int = 3) -> dict:
    """
    Генерация словаря с метриками для задачи регрессии
    :param y_test: тестовые данные целевой переменной
    :param y_pred: предсказанные данные целевой переменной
    :param x_test: тестовые данные с признаками
    :param dpt: число знаков после запятой для округления
    :return: словарь с метриками модели
    """
    dict_metrics = {
        'mae': round(mean_absolute_error(y_test, y_pred), dpt),
        'mse': round(mean_squared_error(y_test, y_pred), dpt),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), dpt),
        'r2_adjusted': round(r2_adjusted(y_test, y_pred, x_test), dpt),
        'mape': round(mean_absolute_percentage_error(y_test, y_pred), dpt),
        'angular_distance': round(angular_dist_score(y_test, y_pred), dpt),
    }

    return dict_metrics


def save_metrics(score: dict) -> None:
    """
    Получение и сохранение метрик
    :param score: словарь метрик обученной модели
    """
    train_cfg = get_config()['train']

    metrics_path = train_cfg['metrics_path']
    with open(metrics_path, 'w') as f:
        json.dump(score, f)


def load_metrics() -> dict:
    """
    Получение метрик из файла
    :return: словарь с матриками модели из файла
    """
    train_cfg = get_config()['train']

    # get params
    with open(train_cfg["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
