"""
Программа: Разделение данных на train/test
Версия: 1.0
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from .get_data import get_config


def get_train_test_data(batch_data: pd.DataFrame,
                        test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Получение train/test данных разбитых по отдельности на объект-признаки и целевую переменную
    :param batch_data: батч датасет для обучения
    :param test_size: пропорция test данных
    :return: набор данных train/test
    """
    config = get_config()
    preproc_cfg = config['preprocessing']
    train_cfg = config['train']

    x = batch_data.drop(columns=preproc_cfg['target_columns'] + preproc_cfg['target_cart_columns'])
    y = batch_data[preproc_cfg['target_cart_columns']]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=train_cfg['random_state'])

    return x_train, x_test, y_train, y_test
