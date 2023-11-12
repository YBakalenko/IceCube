"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
import numpy as np
from optuna.visualization import plot_param_importances, plot_optimization_history
from ..data.get_data import get_config


def start_train(endpoint: str) -> None:
    """
    Тренировка модели с выводом результатов
    :param endpoint: endpoint
    """
    # get config
    train_cfg = get_config()['train']
    # Last metrics
    if os.path.exists(train_cfg['metrics_path']):
        with open(train_cfg['metrics_path']) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {
            'mae': 0,
            'mse': 0,
            'rmse': 0,
            'r2_adjusted': 0,
            'mape': 0,
            'angular_distance': np.pi
        }
    # Train
    with st.spinner('Модель подбирает параметры...'):
        output = requests.post(endpoint, timeout=8000)
    st.success('Success!')
    new_metrics = output.json()['metrics']

    # diff metrics
    mae, mse, rmse, r2_adjusted, mape, angular_distance = st.columns(6)
    mae.metric(
        'MAE',
        new_metrics['mae'],
        f"{new_metrics['mae'] - old_metrics['mae']:.2f}",
        delta_color='inverse'
    )
    mse.metric(
        'MSE',
        new_metrics['mse'],
        f"{new_metrics['mse'] - old_metrics['mse']:.2f}",
        delta_color='inverse'
    )
    rmse.metric(
        'RMSE',
        new_metrics['rmse'],
        f"{new_metrics['rmse'] - old_metrics['rmse']:.2f}",
        delta_color='inverse'
    )
    r2_adjusted.metric(
        'R2 adjusted', new_metrics['r2_adjusted'],
        f"{new_metrics['r2_adjusted']-old_metrics['r2_adjusted']:.2f}"
    )
    mape.metric(
        'MAPE',
        new_metrics['mape'],
        f"{new_metrics['mape']-old_metrics['mape']:.2f}",
        delta_color='inverse'
    )
    angular_distance.metric(
        'Angular distance',
        new_metrics['angular_distance'],
        f"{new_metrics['angular_distance'] - old_metrics['angular_distance']:.2f}",
        delta_color='inverse'
    )

    # plot study
    study = joblib.load(os.path.join(train_cfg['study_path']))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
