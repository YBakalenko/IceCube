"""
Программа: Предобработка данных
Версия: 1.0
"""

import os
import warnings
import pandas as pd
import numpy as np
from ..data.get_data import get_config, get_batch

warnings.filterwarnings("ignore")


def sg_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Получение данных об их принадлежности к нитям
    :param df: датафрейм sensor_geometry
    """
    preproc_cfg = get_config()['preprocessing']

    df_out = df.copy()
    # Номер нити, к которой относится датчик
    df_out[preproc_cfg['line_column']] = df_out[preproc_cfg['sensor_column']] // 60 + 1
    # Флаг принадлежности датчика к центральным нитям

    return df_out


def dtypes_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразование числовых полей датафрейма к меньшей размерности для экономии вычислительных
     ресурсов
    :param df: датафрейм
    """
    f_cols = df.select_dtypes('float').columns
    i_cols = df.select_dtypes('integer').columns

    df[f_cols] = df[f_cols].apply(pd.to_numeric, downcast='float')
    df[i_cols] = df[i_cols].apply(pd.to_numeric, downcast='unsigned')

    return df


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


def batch_prepare(batch_df: pd.DataFrame,
                  drop_aux: bool = True,
                  doms_agg: bool = True) -> pd.DataFrame:
    """
    Подготовка батча данных: агрегация импульсов, нормализация времен (отсчет от 0)
    :param batch_df: датафрейм с батчем данных импульсов
    :param drop_aux: флаг для отброса ненадежных импульсов с auxiliary==True
    :param doms_agg: флаг для агрегирования импульсов и их времени по каждому датчику,
                     иначе - на выход подаются все необработанные импульсы, в том числе,
                     если в одном событии присутствуют импульсы с одного модуля
    """
    preproc_cfg = get_config()['preprocessing']

    if drop_aux:
        batch_df = batch_df[~batch_df[preproc_cfg['aux_column']]]
    if doms_agg:
        batch_df = batch_df.groupby([
            preproc_cfg['event_column'], preproc_cfg['sensor_column']
        ]).agg(preproc_cfg['prepare_aggregators']).reset_index()
        batch_df.columns = flatten_multiindex(batch_df, is_join=False)

    # Находим времена импульсов относительно начала событий
    times = batch_df.groupby(preproc_cfg['event_column']).agg(
        preproc_cfg['time_aggregator']).reset_index()
    times.columns = flatten_multiindex(times)
    min_time_column = list(times.columns)[-1]
    batch_df = batch_df.merge(times, on=preproc_cfg['event_column'], how='left')
    batch_df[preproc_cfg['time_column']] = (batch_df[preproc_cfg['time_column']] -
                                            batch_df[min_time_column])
    batch_df.drop(columns=[min_time_column], inplace=True)

    return batch_df


def cut_pulses(batch_df: pd.DataFrame,
               max_pulses: int = 128,
               drop_aux: bool = True) -> pd.DataFrame:
    """
    Выкидываем последние и ненадёжные импульсы в событии если их больше max_pulses
    :param batch_df: датафрейм с импульсами
    :param max_pulses: количество импульсов на одно событие (отсечка)
    :param drop_aux: флаг для отброса ненадежных импульсов с auxiliary==True
    """
    preproc_cfg = get_config()['preprocessing']

    if drop_aux:
        batch_df = batch_df.sort_values(
            [preproc_cfg['event_column'], preproc_cfg['time_column']])
    else:
        batch_df = batch_df.sort_values([
            preproc_cfg['event_column'], preproc_cfg['aux_column'],
            preproc_cfg['time_column']
        ])
    batch_df = batch_df.reset_index(drop=True)
    batch_df = batch_df.groupby(preproc_cfg['event_column']).head(max_pulses)
    batch_df = batch_df.reset_index(drop=True)
    if not drop_aux:
        batch_df = batch_df.sort_values(
            [preproc_cfg['event_column'], preproc_cfg['time_column']])
        batch_df = batch_df.reset_index(drop=True)

    return batch_df


def get_event_features(batch_df: pd.DataFrame,
                       apply_aux: bool = True) -> pd.DataFrame:
    """
    Набираем агрегированные фичи, характеризующие событие в целом
    :param batch_df: датафрейм с импульсами и координатами датчиков
    :param apply_aux: флаг обнуления фичей при отсутствии надежных импульсов
    """
    preproc_cfg = get_config()['preprocessing']

    for key, mult in preproc_cfg['multiplication'].items():
        batch_df[key] = batch_df[mult[0]] * batch_df[mult[1]]

    if apply_aux:
        for col in batch_df.columns:
            if col not in [
                preproc_cfg['event_column'], preproc_cfg['sensor_column'],
                preproc_cfg['line_column']
            ]:
                batch_df[col] = batch_df[col] * (
                        1 - batch_df[preproc_cfg['aux_column']])

    batch_df = batch_df.groupby(preproc_cfg['event_column']).agg(
        preproc_cfg['aggregators'])
    batch_df.columns = flatten_multiindex(batch_df, is_join=True)
    batch_df = batch_df.reset_index()

    for key, division in preproc_cfg['divisions'].items():
        batch_df[key] = np.log10(batch_df[division[0]] / batch_df[division[1]])

    for col in preproc_cfg['log_scale_transform']:
        batch_df[col] = np.log(1 + batch_df[col])

    for key, col in preproc_cfg['log_features'].items():
        batch_df[key] = np.log10(batch_df[col]) / 10

    batch_df.drop(columns=preproc_cfg['drop_columns'], inplace=True)
    # На случай ошибок агрегирования std значений
    batch_df.fillna(value=0, inplace=True)

    return batch_df


def batch_transform(batch_df: pd.DataFrame,
                    sensor_geometry: pd.DataFrame,
                    meta: pd.DataFrame,
                    max_pulses: int = 128,
                    drop_aux: bool = True,
                    doms_agg: bool = True,
                    is_evaluate: bool = False):
    """
    Трансформация батча импульсов в таблицу событий с фичами, характеризующими каждое из них
    :param batch_df: датафрейм батча с импульсами
    :param sensor_geometry: датафрейм с геометрией датчиков аппарата IceCube
    :param meta: метаданные событий с векторами направления нейтрино в сферических координатах
    :param max_pulses: количество импульсов на одно событие (отсечка)
    :param drop_aux: флаг для отброса ненадежных импульсов с auxiliary==True
    :param doms_agg: флаг для агрегирования импульсов и их времени по каждому датчику,
                     иначе - на выход подаются все необработанные импульсы, в том числе,
                     если в одном событии присутствуют импульсы с одного модуля
    :param is_evaluate: флаг того, что батч не предназначен для обучения (не имеет таргета)
    """
    preproc_cfg = get_config()['preprocessing']
    # Подготовка данных импульсов перед преобразованием пакета
    batch_df = batch_prepare(batch_df=batch_df,
                             drop_aux=drop_aux,
                             doms_agg=doms_agg)

    # Выкидываем последние и ненадёжные импульсы в событии если их больше max_pulses
    batch_df = cut_pulses(batch_df=batch_df,
                          max_pulses=max_pulses,
                          drop_aux=drop_aux)
    # Объединяем с геометрией датчиков для получения координат
    batch_df = batch_df.merge(sensor_geometry,
                              on=preproc_cfg['sensor_column'],
                              how='left')

    # Набираем агрегированные фичи, характеризующие событие в целом
    batch_df = get_event_features(batch_df, apply_aux=True)

    # Добавляем целевые переменные в сферических и декартовых координатах
    if not is_evaluate:
        batch_df[preproc_cfg['target_columns']] = batch_df.merge(
            meta, on=preproc_cfg['event_column'],
            how='left')[preproc_cfg['target_columns']]
        # Для снижения нелинейности задачи преобразуем к декартовым координатам
        batch_df[preproc_cfg['target_cart_columns']] = spherical_to_cartesian(
            batch_df[preproc_cfg['target_columns']].to_numpy())
    # Удаляем строки, если не нашлись таргеты соответствующих эвентов
    batch_df.dropna(axis=0, inplace=True)
    # Итоговый датафрейм для обучения
    batch_df.set_index(preproc_cfg['event_column'], drop=True, inplace=True)
    batch_df = dtypes_convert(batch_df)

    return batch_df


def pipeline_preprocess_train(overwrite: bool = False) -> None:
    """
    Преобразование всего датачета батчей с импульсами в формат датафреймов для обучения
    и сохраняет обработанные файлы в отдельной папке
    :param overwrite: флаг перезаписи ранее сохраненных обработанных файлов
    """
    config = get_config()
    preproc_cfg = config['preprocessing']
    train_cfg = config['train']

    sg_file_path = preproc_cfg['file_dirs']['sensor_geometry']['local_dir'] + '/' + preproc_cfg[
        'file_dirs']['sensor_geometry']['filename']
    sensor_geometry = dtypes_convert(pd.read_csv(sg_file_path))
    sensor_geometry = sg_transform(sensor_geometry)
    meta_file_path = preproc_cfg['file_dirs']['train_meta_sample']['local_dir'] + '/' + preproc_cfg[
        'file_dirs']['train_meta_sample']['filename']
    meta = pd.read_parquet(meta_file_path)

    src_filepath = preproc_cfg['file_dirs']['train_batches_sample']['local_dir']
    src_filename = preproc_cfg['file_dirs']['train_batches_sample']['filename']
    src_path = src_filepath + src_filename
    dst_filepath = train_cfg['file_dirs']['processed_batches_sample']['local_dir']
    dst_filename = train_cfg['file_dirs']['processed_batches_sample']['filename']
    dst_path = dst_filepath + dst_filename
    dst_check_file = os.path.isfile(dst_path)
    if overwrite or not dst_check_file:
        # Обрабатываем batch-файл
        batch_data = dtypes_convert(get_batch(src_path))
        batch_data = batch_transform(batch_df=batch_data,
                                     sensor_geometry=sensor_geometry,
                                     meta=meta,
                                     max_pulses=10000,
                                     drop_aux=True,
                                     doms_agg=True,
                                     is_evaluate=False)
        # сохраняем batch-файл
        batch_data.to_parquet(dst_path, engine='fastparquet')


def pipeline_preprocess_evaluate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразование всего датачета батчей с импульсами в формат датафреймов для инференса
    и сохранение обработанных файлов в отдельной папке
    :param data: датафрейм тестового батч-файла
    :return batch_data: датафрейм, готовый на вход обученной модели, без таргетов
    """
    # get config
    config = get_config()
    preproc_cfg = config['preprocessing']
    evaluate_cfg = config['evaluate']
    # dataframes transform
    sg_file_path = preproc_cfg['file_dirs']['sensor_geometry'][
                       'local_dir'] + '/' + preproc_cfg['file_dirs']['sensor_geometry']['filename']
    sensor_geometry = dtypes_convert(pd.read_csv(sg_file_path))
    sensor_geometry = sg_transform(sensor_geometry)
    meta_file_path = evaluate_cfg['file_dirs']['test_meta'][
                         'local_dir'] + '/' + evaluate_cfg['file_dirs']['test_meta']['filename']
    test_meta = dtypes_convert(pd.read_parquet(meta_file_path))
    # Обрабатываем batch-файл
    batch_data = batch_transform(batch_df=data,
                                 sensor_geometry=sensor_geometry,
                                 meta=test_meta,
                                 max_pulses=10000,
                                 drop_aux=True,
                                 doms_agg=True,
                                 is_evaluate=True)

    return batch_data
