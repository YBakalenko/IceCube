"""
Программа: Получение сырых данных для обучения модели
Версия: 1.0
"""
import yaml
import pandas as pd
from typing import Text


def get_config():
    """
    Загрузка конфигурационного файла проекта
    :return: словарь настроек
    """
    config_path = '../config/params.yml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    return config


def get_batch(batch_path: Text) -> pd.DataFrame:
    """
    Получение данных из батч-файла по заданному пути
    :param batch_path: путь к батч-файлу
    :return: датасет
    """
    return pd.read_parquet(batch_path)
