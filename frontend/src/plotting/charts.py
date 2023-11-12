"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from ..data.get_data import get_config
from ..data.support import flatten_multiindex, get_valid_time, charge_normalize, spherical_to_cartesian


@st.cache_resource(show_spinner=False)
def sensors_3d(sensor_geometry: pd.DataFrame) -> px.scatter_3d:
    """
    Отрисовка геометрии оптических нитей IceCube
    :param sensor_geometry: датасет с данными сенсоров IceCube
    :return: 3D-график сенсоров
    """
    preproc_cfg = get_config()['preprocessing']
    fig = px.scatter_3d(sensor_geometry,
                        x=preproc_cfg['coordinate_columns'][0],
                        y=preproc_cfg['coordinate_columns'][1],
                        z=preproc_cfg['coordinate_columns'][2],
                        color=sensor_geometry.index,
                        opacity=0.7)
    fig.update_traces(marker_size=2)

    return fig


@st.cache_resource(show_spinner=False)
def plot_meta(train_meta: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Отрисовка статистики направлений лучей нейтрино
    :param train_meta: датасет с метаданными прошлых наблюдений
    :return: тепловая карта событий
    """
    preproc_cfg = get_config()['preprocessing']
    azimuth_col = preproc_cfg['target_columns'][0]
    zenith_col = preproc_cfg['target_columns'][1]
    fig = plt.figure(figsize=(10, 5))
    plt.hist2d(data=train_meta,
               x=azimuth_col,
               y=zenith_col,
               bins=(50, 50),
               cmap='jet')
    plt.xlabel(azimuth_col)
    plt.ylabel(zenith_col)
    plt.colorbar()

    return fig


@st.cache_resource(show_spinner=False)
def plot_charge_hist(batch_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Отрисовка статистики величин импульсов нейтрино
    :param batch_df: датасет из batch-файла
    :return: гистограмма со статистикой величин импульсов нейтрино
    """
    preproc_cfg = get_config()['preprocessing']
    fig = plt.figure(figsize=(10, 5))
    plt.hist(batch_df[preproc_cfg['charge_column']], log=True)
    plt.ylabel('Measurements count')
    plt.xlabel('Charge value')

    return fig


@st.cache_resource(show_spinner=False)
def barplot_aux(batch_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Отрисовка графика barplot признака aux
    :param batch_df: датасет batch-файла
    :return: поле рисунка
    """
    preproc_cfg = get_config()['preprocessing']
    data_group = (
        batch_df[preproc_cfg['aux_column']]
        .value_counts(normalize=True)
        .rename('percentage')
        .mul(100)
        .reset_index()
    )

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(10, 5))

    ax = sns.barplot(x=data_group.index, y='percentage', data=data_group)

    for patch in ax.patches:
        percentage = '{:.1f}%'.format(patch.get_height())
        ax.annotate(percentage,
                    (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=10)

    plt.title('Auxiliary percentage', fontsize=12)
    plt.xlabel('Auxiliary', fontsize=10)
    plt.ylabel('Percent', fontsize=10)
    plt.ylim((0, 100))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    return fig


@st.cache_resource(show_spinner=False)
def histplot_time(batch_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot признаков auxiliary и величин импульсов
    :param batch_df: датасет batch-файла
    :return: поле рисунка
    """
    preproc_cfg = get_config()['preprocessing']
    batch_aux = []
    auxiliaries = [True, False]
    for i, aux in enumerate(auxiliaries):
        batch_aux.append(batch_df[batch_df[preproc_cfg['aux_column']] == aux])
        # Находим времена импульсов относительно начала событий
        batch_aux[i] = batch_aux[i].groupby(preproc_cfg['event_column'])\
            .agg({preproc_cfg['time_column']: ['min', 'max']})\
            .reset_index()
        batch_aux[i].columns = flatten_multiindex(batch_aux[i])
        col_delta = '_'.join(['delta', str(aux)])
        batch_aux[i][col_delta] = batch_aux[i].iloc[:, -1] - batch_aux[i].iloc[:, -2]
    batch_aux = pd.concat(batch_aux, axis=1)
    valid_time = get_valid_time()

    fig = plt.figure(figsize=(10, 5))
    sns.histplot(batch_aux[['delta_False', 'delta_True']], kde=True)
    plt.axvline(x=valid_time, ymin=0, ymax=1, linestyle='--', color='darkgreen')
    ylim = plt.gca().get_ylim()
    plt.text(x=(valid_time - 700), y=0.5 * ylim[0] + 0.5 * ylim[1], s='Valid time', rotation=90)
    plt.title('Event time distribution', fontsize=12)
    plt.ylabel('Events count', fontsize=10)
    plt.xlabel('Time window', fontsize=10)

    return fig


@st.cache_resource(show_spinner=False)
def draw_subplot(fig: go.Figure,
                 ax_num: int,
                 event_pulses_: pd.DataFrame,
                 sensor_geometry: pd.DataFrame,
                 mc: np.ndarray,
                 c_ray: np.ndarray,
                 scale: float = 400):
    """
    Построение 3D-графика импульсов и расчетного вектора события, отфильтрованных по признаку
    auxiliary и наложенных на геометрическую модель фотодатчиков, в объекте go.Figure
    :param fig: объект go.Figure
    :param ax_num: идентификатор объекта 3D-графика: 0 для aux == False и 1 для aux == True
    :param event_pulses_: датафрейм с импульсами одного события, отфильтрованными по признаку aux
    :param sensor_geometry: датафрейм с координатами фотодатчиков IceCube
    :param mc: массив декартовых координат X, Y, Z по значениям импульсов срабатывания фотодатчиков
    :param c_ray: массив значений декартовых координат X, Y, Z расчетного вектора события
    :param scale: коэффициент масштабирования длины вектора на графике
    :return: 3D-график с импульсами и лучом нейтрино для различных aux
    """
    preproc_cfg = get_config()['preprocessing']
    coord_cols = preproc_cfg['coordinate_columns']
    charge_col = preproc_cfg['charge_column']
    time_col = preproc_cfg['time_column']
    # отрисовываем все фотодатчики IceCube в виде фона
    fig.add_trace(go.Scatter3d(x=sensor_geometry[coord_cols[0]],
                               y=sensor_geometry[coord_cols[1]],
                               z=sensor_geometry[coord_cols[2]],
                               name=None,
                               showlegend=False,
                               mode='markers',
                               marker=dict(size=1, color='black'),
                               opacity=0.2),
                  col=(ax_num + 1),
                  row=1)
    # отрисовываем фотодатчики, зафиксировавшие импульсы
    fig.add_trace(go.Scatter3d(x=event_pulses_[coord_cols[0]],
                               y=event_pulses_[coord_cols[1]],
                               z=event_pulses_[coord_cols[2]],
                               name=None,
                               showlegend=False,
                               opacity=0.8,
                               mode='markers',
                               marker=dict(size=event_pulses_[charge_col] * 15,
                                           color=event_pulses_[time_col],
                                           colorscale='turbo',
                                           showscale=True)),
                  col=(ax_num + 1),
                  row=1)
    # рисуем расчетный вектор события
    fig.add_trace(go.Scatter3d(
        x=[mc[0] - c_ray[0] * scale, mc[0] + c_ray[0] * scale],
        y=[mc[1] - c_ray[1] * scale, mc[1] + c_ray[1] * scale],
        z=[mc[2] - c_ray[2] * scale, mc[2] + c_ray[2] * scale],
        name='target',
        showlegend=not bool(ax_num),
        opacity=0.8,
        mode='lines+markers',
        marker=dict(size=5, symbol=['cross', 'circle-open']),
        line=dict(color='red', width=3)),
                  col=(ax_num + 1),
                  row=1)


@st.cache_resource(show_spinner=False)
def event_plot(event_id: int,
               batch_data: pd.DataFrame,
               sensor_geometry: pd.DataFrame,
               train_meta: pd.DataFrame) -> go.Figure:
    """
    Создание 3D-графика импульсов и расчетного вектора события для сравнения импульсов
    с признаком auxiliary == True и False
    :param event_id: идентификатор события
    :param batch_data: датасет событий измерения импульсов нейтрино с детализацией импульсов
    :param sensor_geometry: датасет с координатами фотодатчиков
    :param train_meta: датафрейм с метаданными событий с расчетными векторами направления нейтрино
    в сферических координатах
    """
    preproc_cfg = get_config()['preprocessing']
    target_cols = preproc_cfg['target_columns']
    charge_col = preproc_cfg['charge_column']
    sensor_col = preproc_cfg['sensor_column']
    aux_col = preproc_cfg['aux_column']
    event_col = preproc_cfg['event_column']

    # Выбираем нужное событие из метаданных
    event_metadata = dict(train_meta[train_meta[event_col] == event_id].iloc[0])
    azimuth = event_metadata[target_cols[0]]
    zenith = event_metadata[target_cols[1]]
    # Выбираем импульсы, соответствующие запрошенному событию
    event_pulses = batch_data[batch_data.index == event_id]
    # Накладываем на датасет с импульсами координаты фотодатчиков
    event_pulses = event_pulses.reset_index().merge(
        sensor_geometry, on=sensor_col, how='left')
    event_pulses.loc[:, charge_col] /= event_pulses[charge_col].max()
    # Единичные координаты вектора луча нейтрино
    c_ray = spherical_to_cartesian(np.array([[azimuth, zenith]]))

    auxiliaries = [False, True]
    fig = make_subplots(
        cols=2,
        specs=[[{
            'type': 'scene'
        }, {
            'type': 'scene'
        }]],
        subplot_titles=[f'{aux_col}={aux}' for aux in auxiliaries],
        horizontal_spacing=0.1,
    )
    mc = np.zeros(3, dtype=int)
    for i, aux in enumerate(auxiliaries):
        event_pulses_ = event_pulses[event_pulses[aux_col] == aux]
        mc = charge_normalize(
            df_event=event_pulses[event_pulses[aux_col] == aux])
        draw_subplot(fig, i, event_pulses_, sensor_geometry, mc, c_ray[0])

    fig.update_layout(
        height=500,
        width=1000,
        showlegend=True,
        legend=dict(y=-0.1, x=0.5),
        title_text=f"Event #{event_id}"
        f" / azimuth={azimuth:0.3}; zenith={zenith:0.3}\n"
        f"-> x={mc[0]:0.2}; y={mc[1]:0.2}; z={mc[2]:0.2}",
    )
    return fig
