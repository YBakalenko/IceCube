"""
Программа: Задачи довычисления для вывода на экран
Версия: 1.0
"""

import pandas as pd
import numpy as np
from scipy import constants
from ..data.get_data import get_config


def flatten_multiindex(df: pd.DataFrame, is_join: bool = True) -> list:
    """
    Приведение иерархического мультииндекса столбцов в плоский вид
    :param df: датафрейм
    :param is_join: флаг объединения имен через '_', иначе - отбрасываем второй уровень
    """
    if is_join:
        result = [
            '_'.join(col).strip()
            if len(col[1]) > 0 else ' '.join(col).strip()
            for col in df.columns.values
        ]
    else:
        result = [col[0] for col in df.columns.values]
    return result


def get_valid_time():
    """
    Получение максимального теоретического времени прохождения нейтрино
    аппарата IceCube
    :return: время в нс
    """
    preproc_cfg = get_config()['preprocessing']
    sg_cfg = preproc_cfg['file_dirs']['sensor_geometry']

    sg_path = sg_cfg['local_dir'] + sg_cfg['filename']
    sensor_geometry = pd.read_csv(sg_path)
    min_coordinate = np.array([
        min(sensor_geometry[preproc_cfg['coordinate_columns'][0]]),
        min(sensor_geometry[preproc_cfg['coordinate_columns'][1]]),
        min(sensor_geometry[preproc_cfg['coordinate_columns'][2]])
    ])
    max_coordinate = np.array([
        max(sensor_geometry[preproc_cfg['coordinate_columns'][0]]),
        max(sensor_geometry[preproc_cfg['coordinate_columns'][1]]),
        max(sensor_geometry[preproc_cfg['coordinate_columns'][2]])
    ])
    dist = np.linalg.norm(max_coordinate - min_coordinate)
    valid_time = dist / constants.speed_of_light * 1e9
    return np.round(valid_time)


def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """
    Преобразование вектора сферических координат в декартову систему координат
    :param spherical: массив сферических координат в формате [azimuth, zenith]
    """
    azimuth = spherical[:, 0]
    zenith = spherical[:, 1]
    dx = np.sin(zenith) * np.cos(azimuth)
    dy = np.sin(zenith) * np.sin(azimuth)
    dz = np.cos(zenith)

    return np.array([dx, dy, dz]).T


def charge_normalize(df_event: pd.DataFrame) -> np.ndarray:
    """
    Вычисление средневзвешенных по значениям импульсов координат срабатывания фотодатчиков
    в рамках одного события. Используется для вычисления опорных координат луча нейтрино
    :param df_event: датафрейм с импульсами одного события с координатами датчиков импульсов
    :return: массив средневзвешенных по импульсам координат
    """
    preproc_cfg = get_config()['preprocessing']
    coord_cols = preproc_cfg['coordinate_columns']
    charge_col = preproc_cfg['charge_column']

    df_event_sub = df_event[coord_cols + [charge_col]]
    # Вычисляем весовой коэффициент каждого импульса
    df_event_sub.loc[:, 'coef'] = df_event_sub.loc[:, charge_col] / df_event_sub.loc[:, charge_col].sum()
    for ax in coord_cols:
        df_event_sub.loc[:, ax] *= df_event_sub.loc[:, 'coef']
    cn = df_event_sub[coord_cols].sum()
    return np.array(cn)
