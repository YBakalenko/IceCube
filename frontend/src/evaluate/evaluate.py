"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import pandas as pd
import requests
import streamlit as st
from ..data.get_data import get_config


def evaluate_input(endpoint: str) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param endpoint: endpoint
    """
    config = get_config()
    preprocess_cfg = config['preprocessing']
    evaluate_cfg = config['evaluate']
    path = evaluate_cfg['template_batch']['local_dir'] + evaluate_cfg['template_batch']['filename']
    batch_template = pd.read_parquet(path)

    # для ввода данных используем шаблон батч-датафрейма
    edited_df = st.data_editor(batch_template, num_rows='dynamic')

    # evaluate and return prediction (text)
    button_ok = st.button('Predict', key='input')
    if button_ok:
        edited_df[preprocess_cfg['aux_column']].fillna(False, inplace=True)
        # отправляем файл в backend
        with st.spinner('Получаем результаты от модели...'):
            output = requests.post(endpoint, json=edited_df.reset_index().to_json())
        if output.status_code == 200:
            st.success('Success!')
            predictions = pd.read_json(output.json()['prediction'], orient='columns')
            st.write(predictions.head())
        else:
            st.error('Введены некорректные данные')


def evaluate_from_file(endpoint: str, files: dict):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param endpoint: endpoint
    :param files:
    """
    button_pred = st.button('Predict', key='file')
    if button_pred:
        with st.spinner('Получаем результаты от модели...'):
            output = requests.post(endpoint, files=files, timeout=8000)
        if output.status_code == 200:
            predictions = pd.read_json(output.json()['prediction'], orient='columns')
            st.success('Success!')
            st.write('Predicted angles:')
            st.write(predictions.head())
        else:
            st.error('Batch-файл содержит некорректные данные')
