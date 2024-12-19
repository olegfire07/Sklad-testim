# ml_models.py

import numpy as np
from sklearn.linear_model import LinearRegression
import logging
import joblib

def train_ml_model(df, target_column):
    """
    Обучает модель линейной регрессии на основе исторических данных.
    
    :param df: DataFrame с данными.
    :param target_column: Столбец целевой переменной.
    :return: Обученная модель.
    """
    try:
        if 'Месяц' not in df.columns or target_column not in df.columns:
            raise ValueError("Необходимы столбцы 'Месяц' и целевой столбец для ML-модели.")
        X = df[['Месяц']].values
        y = df[target_column].values
        model = LinearRegression()
        model.fit(X, y)
        logging.info("ML-модель успешно обучена.")
        return model
    except Exception as e:
        logging.error(f"Ошибка при обучении ML-модели: {e}")
        raise

def predict_with_model(model, future_months):
    """
    Прогнозирует значения с помощью обученной модели.
    
    :param model: Обученная ML-модель.
    :param future_months: Список будущих месяцев для прогноза.
    :return: Прогнозируемые значения.
    """
    try:
        X_future = np.array(future_months).reshape(-1, 1)
        predictions = model.predict(X_future)
        logging.info("Прогнозирование с использованием ML-модели выполнено успешно.")
        return predictions
    except Exception as e:
        logging.error(f"Ошибка при прогнозировании с ML-моделью: {e}")
        raise

def save_ml_model(model, filepath="ml_model.pkl"):
    """
    Сохраняет обученную ML-модель на диск.
    
    :param model: Обученная ML-модель.
    :param filepath: Путь к файлу для сохранения модели.
    """
    try:
        joblib.dump(model, filepath)
        logging.info(f"ML-модель успешно сохранена в {filepath}.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении ML-модели: {e}")

def load_ml_model(filepath="ml_model.pkl"):
    """
    Загружает ML-модель с диска.
    
    :param filepath: Путь к файлу с моделью.
    :return: Загруженная ML-модель.
    """
    try:
        model = joblib.load(filepath)
        logging.info(f"ML-модель успешно загружена из {filepath}.")
        return model
    except Exception as e:
        logging.error(f"Ошибка при загрузке ML-модели: {e}")
        return None
