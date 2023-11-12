"""
Программа: Тренировка данных
Версия: 1.0
"""

import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, sum_models
from sklearn.model_selection import KFold
import pandas as pd
import optuna
import joblib
from optuna import Study
from typing import Tuple
from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics, get_metrics
from ..data.get_data import get_config, get_batch


def train_batch(x_train: pd.DataFrame,
                y_train: pd.DataFrame,
                x_val: pd.DataFrame,
                y_val: pd.DataFrame,
                model_parameters) -> type(CatBoostRegressor):
    """
    Обучение модели CatBoostRegressor для одного батча
    :param x_train: датафрейм с признаками для обучения
    :param y_train: датафрейм с таргетами для обучения
    :param x_val: датафрейм с признаками для проверки
    :param y_val: датафрейм с таргетами для проверки
    :param model_parameters: гиперпараметры модели CatBoostRegressor
    """
    train_cfg = get_config()['train']
    clf = CatBoostRegressor(allow_writing_files=False,
                            loss_function='MultiRMSE',
                            eval_metric='MultiRMSE',
                            random_state=train_cfg['random_state'],
                            **model_parameters)
    eval_set = [(x_val, y_val)]
    clf.fit(x_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=100)

    return clf


def train_cv(n_folds: int = 5, *model_parameters) -> Tuple[type(CatBoostRegressor), dict, dict]:
    """
    Обучение модели CatBoostRegressor для всех батчей с кросс-валидацией
    :param n_folds: количество фолдов для кросс-валидации
    :param model_parameters: гиперпараметры модели CatBoostRegressor
    :return: обученная модель класса CatBoostRegressor, словари метрик на train и test
    """
    config = get_config()
    preproc_cfg = config['preprocessing']
    train_cfg = config['train']
    cv = KFold(n_splits=n_folds,
               shuffle=True,
               random_state=train_cfg['random_state'])

    filepath = train_cfg['file_dirs']['batches_sample']['local_dir']
    filename = train_cfg['file_dirs']['batches_sample']['filename']
    path = filepath + filename

    # Обрабатываем batch-файл с кросс-валидацией
    batch_data = get_batch(path)
    x_train, x_test, y_train, y_test = get_train_test_data(batch_data, preproc_cfg['test_size'])
    cv_models = []
    for idx, (train_idx, test_idx) in enumerate(cv.split(x_train, y_train)):
        x_train_, x_val = x_train.iloc[train_idx], x_train.iloc[test_idx]
        y_train_, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]
        model = train_batch(x_train_, y_train_, x_val, y_val, model_parameters)
        cv_models.append(model)

    # находим среднее моделей с кросс-валидацией
    models_cv_avg = sum_models(cv_models, weights=[1.0 / len(cv_models)] * len(cv_models))
    score_test = get_metrics(y_test,
                             models_cv_avg.predict(x_test),
                             x_test)
    score_train = get_metrics(y_train,
                              models_cv_avg.predict(x_train),
                              x_train)
    score_train = pd.DataFrame(score_train).T.mean().to_dict()
    score_test = pd.DataFrame(score_test).T.mean().to_dict()

    return models_cv_avg, score_train, score_test


def objective(trial,
              n_folds: int = 5,
              n_estimators: int = 1000,
              learning_rate: float = 0.01) -> float:
    """
    Целевая функция оптимизации всех основных гиперпараметров, кроме шага (learning_rate)
    и числа базовых алгоритмов (n_estimators)
    :param trial: номер итерации
    :param n_folds: количество выборок при выполнении кросс-валидации
    :param n_estimators: количество базовых алгоритмов
    :param learning_rate: шаг обучения
    """
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [n_estimators]),
        'learning_rate': trial.suggest_categorical('learning_rate', [learning_rate]),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'l2_leaf_reg': trial.suggest_uniform('l2_leaf_reg', 1e-5, 1e2),
        'random_strength': trial.suggest_uniform('random_strength', 10, 50),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type',
                                                    ['Bayesian', 'Bernoulli', 'MVS', 'No']),
        'border_count': trial.suggest_categorical('border_count', [128, 254]),
        'grow_policy': trial.suggest_categorical('grow_policy',
                                                 ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'od_wait': trial.suggest_int('od_wait', 500, 2000),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
        'use_best_model': trial.suggest_categorical('use_best_model', [True])
    }

    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 100)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.1, 1, log=True)

    model, score_train, score_test = train_cv(n_folds=n_folds, *params)

    return score_test['angular_distance']


def trial_func(trial) -> float:
    """
    Вызов функции поиска целевых значений оптимизации
    :param trial: итератор модели optuna
    """
    train_cfg = get_config()['train']
    return objective(trial=trial,
                     n_folds=train_cfg['n_folds'],
                     n_estimators=train_cfg['n_estimators'],
                     learning_rate=train_cfg['Learning_rate'])


def params_optimizer() -> Study:
    """
    Пайплайн для оптимизации гиперпараметров модели
    :return: [CatBoostRegressor tuning study result]
    """
    train_cfg = get_config()['train']
    if os.path.isfile(train_cfg['study_path']):
        study = joblib.load(train_cfg['study_path'])
    else:
        study = optuna.create_study(direction='minimize', study_name='CatBoost_main')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(func=trial_func,
                       n_trials=train_cfg['n_trials'],
                       show_progress_bar=True,
                       n_jobs=-1)

    return study


def train(model_parameters) -> Tuple[type(CatBoostRegressor), dict, dict]:
    """
    Обучение модели CatBoostRegressor для всех батчей
    :param model_parameters: гиперпараметры модели CatBoostRegressor
    :return: обученная модель класса CatBoostRegressor, словари метрик на train и test
    """
    config = get_config()
    train_cfg = config['train']
    preproc_cfg = config['preprocessing']

    filepath = train_cfg['file_dirs']['processed_batches_sample']['local_dir']
    filename = train_cfg['file_dirs']['processed_batches_sample']['filename']
    path = filepath + filename
    # Проверяем наличие batch-файла
    check_file = os.path.isfile(path)
    assert check_file, f'Batch file {path} is missing.'
    # Обрабатываем batch-файл
    batch_data = get_batch(path)
    x_train, x_test, y_train, y_test = get_train_test_data(batch_data, preproc_cfg['test_size'])
    x_train_, x_val, y_train_, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.16,
            shuffle=True,
            random_state=train_cfg['random_state'])
    model = train_batch(x_train_, y_train_, x_val, y_val, model_parameters)
    score_test = get_metrics(y_test, model.predict(x_test), x_test)
    score_train = get_metrics(y_train, model.predict(x_train), x_train)

    # сохраняем метрики в файл
    save_metrics(score_test)

    return model, score_train, score_test
