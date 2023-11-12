"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

import os
import joblib
from ..transform.transform import pipeline_preprocess_train
from ..data.get_data import get_config
from ..train.train import params_optimizer, train


def pipeline_train() -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели
    """
    # get params
    config = get_config()
    train_cfg = config['train']

    # preprocessing
    pipeline_preprocess_train()

    # find optimal params
    study = params_optimizer()

    # train with optimal params
    clf, score_train, score_test = train(study.best_params)

    # save result (study, model)
    joblib.dump(clf, os.path.join(train_cfg['model_path']))
    joblib.dump(study, os.path.join(train_cfg['study_path']))
