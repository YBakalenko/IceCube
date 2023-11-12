"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Union, BinaryIO
from ..data.get_data import get_config, get_batch
from ..transform.transform import pipeline_preprocess_evaluate, dtypes_convert


def cartesian_to_spherical(cart: np.ndarray) -> np.ndarray:
    """
    Обратное преобразование векторов декартовых координат в сферическую систему
    :param cart: массив векторов в декартовых координатах в формате [x, y, z]
    :return: массив векторов в сферических координатах в формате [azimuth, zenith]
    """
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    rxy_sq = x**2 + y**2
    zenith = np.arctan2(np.sqrt(rxy_sq), z)
    zenith = np.where(zenith < 0, zenith + 2 * np.pi, zenith)
    azimuth = np.arctan2(y, x)
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return np.array([azimuth, zenith], dtype='float32').T


def pipeline_evaluate(batch_df: pd.DataFrame = None, batch_path: Union[str, BinaryIO] = None) -> pd.DataFrame:
    """
    Предобработка входных данных и получение предсказаний
    :param batch_df: датасет батч-файла
    :param batch_path: путь до батч-файла с данными
    :return: датафрейм предсказаний
    """
    # get params
    config = get_config()
    preprocess_cfg = config['preprocessing']
    train_cfg = config['train']

    # preprocessing
    if batch_path:
        batch_df = dtypes_convert(get_batch(batch_path=batch_path))
    batch_df = pipeline_preprocess_evaluate(data=batch_df)

    model = joblib.load(os.path.join(train_cfg['model_path']))
    prediction = model.predict(batch_df)
    prediction = cartesian_to_spherical(prediction)
    cols = preprocess_cfg['target_columns']
    result_df = pd.DataFrame(prediction, columns=cols, index=batch_df.index, dtype=float)

    return result_df
