"""
Программа: Модель для предсказания направления возникновения лучей нейтрино
аппарата IceCube на основе метаданных прошлых исследований
Версия: 1.0
"""
import warnings
import optuna
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request

from src.pipelines.pipeline import pipeline_train
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics
from src.data.get_data import get_config

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()


@app.post('/train')
def training() -> dict:
    """
    Обучение модели, логирование метрик
    :return: словарь с метриками модели
    """
    pipeline_train()
    metrics = load_metrics()

    return {'metrics': metrics}


@app.post('/predict')
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по батч-данным из файла
    """
    result = pipeline_evaluate(batch_path=file.file)
    assert isinstance(result, pd.DataFrame), 'Результат не соответствует типу pandas.DataFrame'
    return {'prediction': result.head().to_json()}


@app.post('/predict_input')
async def prediction(request: Request):
    """
    :param request: тело запроса
    Предсказание модели по батч-данным, введенным вручную
    """
    # get config
    preproc_cfg = get_config()['preprocessing']
    data = await request.json()
    batch_df = pd.read_json(data, orient='columns')
    batch_df.set_index(preproc_cfg['event_column'], drop=True, inplace=True)
    result_df = pipeline_evaluate(batch_df=batch_df)
    assert isinstance(result_df, pd.DataFrame), 'Результат не соответствует типу pandas.DataFrame'
    return {'prediction': result_df.head().to_json()}


if __name__ == '__main__':
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host='127.0.0.1', port=18507)
