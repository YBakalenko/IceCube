"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

from io import BytesIO
import io
import os
import yaml
from typing import Dict, Tuple, Text
import streamlit as st
import pandas as pd


@st.cache_data
def get_config():
    """
    Загрузка конфигурационного файла проекта
    :return: словарь настроек
    """
    config_path = '../config/params.yml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    return config


@st.cache_data
def get_dataset(dataset_path: Text) -> pd.DataFrame:
    """
    Получение данных из файла датасета по заданному пути и преобразование размерностей для экономии ресурсов
    :param dataset_path: путь к файлу
    :return: датасет
    """
    src_check_file = os.path.isfile(dataset_path)
    assert src_check_file, f'Dataframe file {dataset_path} is missing.'
    _, file_extension = os.path.splitext(dataset_path)
    if file_extension == '.parquet':
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    return df


@st.cache_data
def load_data(data_path: str, data_type: str) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data_path: путь к batch-файлу
    :param data_type: тип датасета (train/test)
    :return: датасет в формате BytesIO
    """
    dataset = pd.read_parquet(data_path).reset_index()
    st.write('Batch data')
    st.write(dataset.head())

    # Преобразовать dataframe в объект BytesIO (для последующего анализа в виде файла в FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # запись в BytesIO буфер
    dataset.to_parquet(dataset_bytes_obj, index=False)
    # Сбросить указатель, чтобы избежать ошибки с пустыми данными
    dataset_bytes_obj.seek(0)

    files = {
        'file': (f'{data_type}_batch.parquet', dataset_bytes_obj, 'multipart/form-data')
    }
    return dataset, files
