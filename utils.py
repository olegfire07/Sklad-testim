# utils.py

import numpy as np
import streamlit as st
import pandas as pd
import base64
import io
from calculations import calculate_financials

def calculate_irr_utils(cash_flows):
    """
    Рассчитывает IRR (внутренняя норма доходности).
    
    :param cash_flows: Список денежных потоков, где первый элемент - начальные вложения (отрицательное значение).
    :return: IRR (в %) или None, если расчёт невозможен.
    """
    if not cash_flows or len(cash_flows) < 2:
        st.write("Недостаточно данных для расчёта IRR.")
        return None

    if cash_flows[0] >= 0:
        st.write("Первый денежный поток должен быть отрицательным (начальные вложения).")
        return None

    try:
        irr = np.irr(cash_flows)
        if irr is None or np.isnan(irr):
            st.write("Невозможно рассчитать IRR: значение не определено.")
            return None
        return irr * 100  # Преобразуем в процентное значение
    except Exception as e:
        st.write(f"Ошибка расчёта IRR: {e}")
        return None

def calculate_roi_utils(total_income, total_expenses):
    """
    Рассчитывает ROI (коэффициент возврата инвестиций).
    
    :param total_income: Общий доход.
    :param total_expenses: Общие расходы.
    :return: ROI (в %).
    """
    try:
        return ((total_income - total_expenses) / total_expenses) * 100
    except ZeroDivisionError:
        return float('inf')

def monte_carlo_simulation_utils(income, expenses, horizon, simulations, deviation, seed=None):
    """
    Выполняет моделирование Монте-Карло.
    
    :param income: Базовый доход.
    :param expenses: Базовые расходы.
    :param horizon: Горизонт планирования (в месяцах).
    :param simulations: Количество симуляций.
    :param deviation: Отклонение (стандартное).
    :param seed: Сид для повторяемости.
    :return: DataFrame с результатами симуляции.
    """
    np.random.seed(seed)
    incomes = np.random.normal(loc=income, scale=deviation * income, size=(simulations, horizon))
    expenses = np.random.normal(loc=expenses, scale=deviation * expenses, size=(simulations, horizon))
    profits = incomes - expenses

    results = {
        "Месяц": np.arange(1, horizon + 1),
        "Средний Доход": np.mean(incomes, axis=0),
        "Средний Расход": np.mean(expenses, axis=0),
        "Средняя Прибыль": np.mean(profits, axis=0)
    }

    return pd.DataFrame(results)

def calculate_total_bep_utils(financials):
    """
    Рассчитывает общую точку безубыточности.
    
    :param financials: Финансовые показатели.
    :return: Точка безубыточности (в % от доходов).
    """
    try:
        total_income = financials["total_income"]
        total_expenses = financials["total_expenses"]
        if total_income == 0:
            return float('inf')
        return (total_expenses / total_income) * 100
    except KeyError as e:
        st.write(f"Ключ отсутствует: {e}")
        return None

def generate_download_link(df: pd.DataFrame, filename="results.csv", link_text="Скачать CSV"):
    """
    Генерирует ссылку для скачивания DataFrame в формате CSV.

    :param df: DataFrame для скачивания.
    :param filename: Имя файла.
    :param link_text: Текст ссылки.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def generate_excel_download(df: pd.DataFrame, filename="results.xlsx", link_text="Скачать Excel"):
    """
    Генерирует ссылку для скачивания DataFrame в формате Excel.

    :param df: DataFrame для скачивания.
    :param filename: Имя файла.
    :param link_text: Текст ссылки.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def normalize_shares(changed_share_key, new_value):
    """
    Нормализует доли видов хранения при изменении одной из долей.

    :param changed_share_key: Ключ изменённой доли.
    :param new_value: Новое значение доли.
    """
    st.session_state.shares[changed_share_key] = new_value
    other_keys = [k for k in st.session_state.shares.keys() if k != changed_share_key]
    total_other = sum(st.session_state.shares[k] for k in other_keys)
    if total_other > 0:
        for k in other_keys:
            st.session_state.shares[k] = max(0, st.session_state.shares[k] - new_value / len(other_keys))
    # Дополнительно можно нормализовать, чтобы сумма долей была равна 1
    total_sum = sum(st.session_state.shares.values())
    if total_sum > 0:
        for k in st.session_state.shares.keys():
            st.session_state.shares[k] /= total_sum

def perform_sensitivity_analysis(params, param_key, param_values, disable_extended):
    """
    Выполняет анализ чувствительности по заданному параметру.

    :param params: Объект WarehouseParams с параметрами склада.
    :param param_key: Ключ параметра для анализа.
    :param param_values: Список значений параметра для анализа.
    :param disable_extended: Флаг отключения расширенных параметров.
    :return: DataFrame с результатами анализа.
    """
    base_val = getattr(params, param_key)
    results = []
    for v in param_values:
        setattr(params, param_key, v)
        fin = calculate_financials(params, disable_extended)
        results.append({"Параметр": v, "Прибыль (руб.)": fin["profit"]})
    setattr(params, param_key, base_val)
    return pd.DataFrame(results)

def safe_display_irr(irr_value):
    """
    Безопасно отображает IRR в интерфейсе.

    :param irr_value: Значение IRR (или None).
    """
    if irr_value is None:
        st.metric("IRR (%)", "Невозможно рассчитать")
    else:
        st.metric("IRR (%)", f"{irr_value:.2f}%")

def prepare_cash_flows(base_financials, params):
    """
    Формирует корректный массив денежных потоков для расчёта IRR.

    :param base_financials: Финансовые данные.
    :param params: Параметры склада.
    :return: Массив cash_flows.
    """
    initial_investment = -(
        params.one_time_setup_cost +
        params.one_time_equipment_cost +
        params.one_time_other_costs
    )
    recurring_profits = [base_financials["profit"]] * params.time_horizon
    return [initial_investment] + recurring_profits

def integrate_irr_in_main(base_financials, params):
    """
    Интегрирует расчёт IRR в основной код, формируя и используя cash_flows.

    :param base_financials: Финансовые данные.
    :param params: Параметры склада.
    :return: Значение IRR.
    """
    cash_flows = prepare_cash_flows(base_financials, params)
    irr_value = calculate_irr(cash_flows)
    return irr_value
