# main.py

import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib

from data_model import WarehouseParams, validate_inputs
from calculations import (
    calculate_areas,
    calculate_items,
    calculate_financials,
    calculate_additional_metrics,
    calculate_roi,
    calculate_irr,  # Обновлённая функция
    monte_carlo_simulation,
    calculate_total_bep,
    min_loan_amount_for_bep
)
from utils import normalize_shares
from ui import (
    display_tab1,
    display_tab2,
    display_tab3,
    display_tab4
)
from ml_models import train_ml_model, predict_with_model

st.set_page_config(page_title="Экономическая модель склада 📦", layout="wide")

st.markdown("# Экономическая модель склада 📦")
st.markdown("Все расчёты выполняются автоматически при изменении параметров в боковой панели. Просто меняйте параметры — результаты обновятся.")

# Вспомогательные функции для ввода должны быть определены до их использования
def input_storage_share(share_key, current_share):
    """
    Ввод доли площади для заданного вида хранения.
    
    :param share_key: Ключ доли (например, 'storage_share')
    :param current_share: Текущее значение доли (от 0 до 1)
    :return: Новое значение доли
    """
    share_labels = {
        "storage_share": "Простое",
        "loan_share": "Займы",
        "vip_share": "VIP",
        "short_term_share": "Краткосрочное"
    }
    new_share = st.slider(
        f"{share_labels.get(share_key, share_key).upper()} (%)",
        0.0, 
        100.0, 
        current_share * 100, 
        step=1.0, 
        help="Доля площади для данного вида хранения."
    ) / 100.0
    normalize_shares(share_key, new_share)
    return st.session_state.shares[share_key]

if 'shares' not in st.session_state:
    st.session_state.shares = {
        'storage_share': 0.25,
        'loan_share': 0.25,
        'vip_share': 0.25,
        'short_term_share': 0.25
    }

with st.sidebar:
    st.markdown("## Ввод параметров")
    
    with st.sidebar.expander("### Основные параметры склада"):
        total_area = st.number_input(
            "📏 Общая площадь (м²)", 
            value=250, 
            step=10, 
            min_value=1, 
            help="Общая арендуемая площадь склада."
        )
        rental_cost_per_m2 = st.number_input(
            "💰 Аренда (руб./м²/мес.)", 
            value=1000, 
            step=50, 
            min_value=1, 
            help="Ежемесячная аренда за 1 м²."
        )
        useful_area_ratio = st.slider(
            "📐 Доля полезной площади (%)", 
            40, 
            80, 
            50, 
            5, 
            help="Процент полезной площади от общей."
        ) / 100.0

    with st.sidebar.expander("### Управление распределением площади"):
        mode = st.radio(
            "Режим распределения площади", 
            ["Автоматический", "Ручной"], 
            index=0, 
            help="Автоматический: доли суммируются до 100%. Ручной: ввод площадей вручную."
        )

        if mode == "Автоматический":
            st.markdown("#### Доли видов хранения (%)")
            no_storage_for_storage = st.checkbox(
                "🚫 Отключить простое хранение", 
                value=False, 
                help="Если включено, простое хранение = 0%."
            )
            no_storage_for_loan = st.checkbox(
                "🚫 Отключить займы", 
                value=False, 
                help="Если включено, займы = 0%."
            )
            no_storage_for_vip = st.checkbox(
                "🚫 Отключить VIP", 
                value=False, 
                help="Если включено, VIP = 0%."
            )
            no_storage_for_short_term = st.checkbox(
                "🚫 Отключить краткосрочное", 
                value=False, 
                help="Если включено, краткосрочное = 0%."
            )

            if no_storage_for_storage:
                st.session_state.shares['storage_share'] = 0.0
            if no_storage_for_loan:
                st.session_state.shares['loan_share'] = 0.0
            if no_storage_for_vip:
                st.session_state.shares['vip_share'] = 0.0
            if no_storage_for_short_term:
                st.session_state.shares['short_term_share'] = 0.0

            storage_options = []
            if not no_storage_for_storage: storage_options.append("storage_share")
            if not no_storage_for_loan: storage_options.append("loan_share")
            if not no_storage_for_vip: storage_options.append("vip_share")
            if not no_storage_for_short_term: storage_options.append("short_term_share")

            for share_key in storage_options:
                current_share = st.session_state.shares[share_key]
                st.session_state.shares[share_key] = input_storage_share(share_key, current_share)

            storage_share = st.session_state.shares['storage_share']
            loan_share = st.session_state.shares['loan_share']
            vip_share = st.session_state.shares['vip_share']
            short_term_share = st.session_state.shares['short_term_share']

            storage_area_manual = 0.0
            loan_area_manual = 0.0
            vip_area_manual = 0.0
            short_term_area_manual = 0.0
        else:
            st.markdown("#### Ручной ввод площадей (м²)")
            temp_usable = total_area * useful_area_ratio
            storage_area_manual = st.number_input(
                "Простое (м²)", 
                value=50.0, 
                step=10.0, 
                min_value=0.0, 
                help="Площадь под простое хранение."
            )
            loan_area_manual = st.number_input(
                "Займы (м²)", 
                value=50.0, 
                step=10.0, 
                min_value=0.0, 
                help="Площадь под займы."
            )
            vip_area_manual = st.number_input(
                "VIP (м²)", 
                value=50.0, 
                step=10.0, 
                min_value=0.0, 
                help="Площадь под VIP хранение."
            )
            short_term_area_manual = st.number_input(
                "Краткосрочное (м²)", 
                value=50.0, 
                step=10.0, 
                min_value=0.0, 
                help="Площадь под краткосрочное хранение."
            )

            total_manual_set = storage_area_manual + loan_area_manual + vip_area_manual + short_term_area_manual
            leftover = temp_usable - total_manual_set
            st.write(f"Не распределено: {leftover:.2f} м² из {temp_usable:.2f} м² полезной площади.")

            storage_share = st.session_state.shares['storage_share']
            loan_share = st.session_state.shares['loan_share']
            vip_share = st.session_state.shares['vip_share']
            short_term_share = st.session_state.shares['short_term_share']

    with st.sidebar.expander("### Тарифы и плотности"):
        storage_fee = st.number_input(
            "💳 Тариф простого (руб./м²/мес.)", 
            value=1500, 
            step=100, 
            min_value=0,
            help="Тариф за простой склад (руб/м²/мес)."
        )
        shelves_per_m2 = st.number_input(  # Изменено
            "📚 Полок на 1 м²", 
            value=3, 
            step=1, 
            min_value=1, 
            max_value=100,
            help="Количество полок на 1 м². Влияет на количество вещей и требует дополнительной площади."
        )
        short_term_daily_rate = st.number_input(
            "🕒 Тариф краткосрочного (руб./день/м²)", 
            value=60.0, 
            step=10.0, 
            min_value=0.0,
            help="Тариф для краткосрочного хранения (руб/день/м²)."
        )
        vip_extra_fee = st.number_input(
            "👑 Наценка VIP (руб./м²/мес.)", 
            value=100.0, 
            step=50.0, 
            min_value=0.0,
            help="Дополнительная наценка за VIP хранение."
        )

    with st.sidebar.expander("### Оценка и займы"):
        item_evaluation = st.slider(
            "🔍 Оценка вещи (%)", 
            0, 
            100, 
            80, 
            5, 
            help="Процент оценки стоимости вещи при займе."
        ) / 100.0
        item_realization_markup = st.number_input(
            "📈 Наценка реализации (%)", 
            value=20.0, 
            step=5.0, 
            min_value=0.0, 
            max_value=100.0,
            help="Наценка при реализации вещей."
        )
        average_item_value = st.number_input(
            "💲 Средняя оценка одной вещи (руб.)", 
            value=15000, 
            step=500, 
            min_value=0,
            help="Средняя стоимость одной вещи."
        )
        loan_interest_rate = st.number_input(
            "💳 Ставка займов (%/день)", 
            value=0.317, 
            step=0.01, 
            min_value=0.0,
            help="Процентная ставка по займам в день."
        )

    with st.sidebar.expander("### Реализация (%)"):
        realization_share_storage = st.slider(
            "Простое", 
            0, 
            100, 
            50, 
            5, 
            help="Процент реализуемых вещей из простого хранения."
        ) / 100.0
        realization_share_loan = st.slider(
            "Займы", 
            0, 
            100, 
            50, 
            5, 
            help="Процент реализуемых вещей из залоговых."
        ) / 100.0
        realization_share_vip = st.slider(
            "VIP", 
            0, 
            100, 
            50, 
            5, 
            help="Процент реализуемых вещей из VIP."
        ) / 100.0
        realization_share_short_term = st.slider(
            "Краткосрочное", 
            0, 
            100, 
            50, 
            5, 
            help="Процент реализуемых вещей из краткосрочного."
        ) / 100.0

    with st.sidebar.expander("### Плотность (вещей/м²)"):
        storage_items_density = st.number_input(
            "Простое", 
            value=5, 
            step=1, 
            min_value=1,
            help="Плотность вещей для простого хранения (вещей/м²)."
        )
        loan_items_density = st.number_input(
            "Займы", 
            value=1, 
            step=1, 
            min_value=1,
            help="Плотность вещей для займов (вещей/м²)."
        )
        vip_items_density = st.number_input(
            "VIP", 
            value=2, 
            step=1, 
            min_value=1,
            help="Плотность вещей для VIP (вещей/м²)."
        )
        short_term_items_density = st.number_input(
            "Краткосрочное", 
            value=4, 
            step=1, 
            min_value=1,
            help="Плотность вещей для краткосрочного хранения (вещей/м²)."
        )

    with st.sidebar.expander("### Финансы (ежемесячные)"):
        salary_expense = st.number_input(
            "Зарплата (руб./мес.)", 
            value=240000, 
            step=10000, 
            min_value=0,
            help="Общие затраты на персонал в месяц."
        )
        miscellaneous_expenses = st.number_input(
            "Прочие (руб./мес.)", 
            value=50000, 
            step=5000, 
            min_value=0,
            help="Прочие ежемесячные расходы."
        )
        depreciation_expense = st.number_input(
            "Амортизация (руб./мес.)", 
            value=20000, 
            step=5000, 
            min_value=0,
            help="Ежемесячная амортизация оборудования."
        )
        marketing_expenses = st.number_input(
            "Маркетинг (руб./мес.)", 
            value=30000, 
            step=5000, 
            min_value=0,
            help="Затраты на маркетинг в месяц."
        )
        insurance_expenses = st.number_input(
            "Страхование (руб./мес.)", 
            value=10000, 
            step=1000, 
            min_value=0,
            help="Страховые платежи в месяц."
        )
        taxes = st.number_input(
            "Налоги (руб./мес.)", 
            value=50000, 
            step=5000, 
            min_value=0,
            help="Налоговые отчисления в месяц."
        )
        utilities_expenses = st.number_input(
            "Коммуналка (руб./мес.)", 
            value=20000, 
            step=5000, 
            min_value=0,
            help="Коммунальные услуги в месяц."
        )
        maintenance_expenses = st.number_input(
            "Обслуживание (руб./мес.)", 
            value=15000, 
            step=5000, 
            min_value=0,
            help="Затраты на обслуживание склада в месяц."
        )

    with st.sidebar.expander("### Финансы (единовременные)"):
        one_time_setup_cost = st.number_input(
            "Настройка (руб.)", 
            value=100000, 
            step=5000, 
            min_value=0,
            help="Единовременные затраты на настройку."
        )
        one_time_equipment_cost = st.number_input(
            "Оборудование (руб.)", 
            value=200000, 
            step=5000, 
            min_value=0,
            help="Единовременные затраты на оборудование."
        )
        one_time_other_costs = st.number_input(
            "Другие (руб.)", 
            value=50000, 
            step=5000, 
            min_value=0,
            help="Другие единовременные расходы."
        )
        one_time_legal_cost = st.number_input(
            "Юридические (руб.)", 
            value=20000, 
            step=5000, 
            min_value=0,
            help="Юридические единовременные расходы."
        )
        one_time_logistics_cost = st.number_input(
            "Логистика (руб.)", 
            value=30000, 
            step=5000, 
            min_value=0,
            help="Единовременные логистические затраты."
        )

    with st.sidebar.expander("### Расширенные параметры и прогнозирование"):
        disable_extended = st.checkbox(
            "🚫 Отключить расширенные параметры", 
            value=False,
            help="Если включено, расширенные параметры будут проигнорированы."
        )
        if not disable_extended:
            time_horizon = st.slider(
                "🕒 Горизонт прогноза (мес.)", 
                1, 
                24, 
                6,
                help="Период планирования в месяцах."
            )
            monthly_rent_growth = st.number_input(
                "📈 Рост аренды (%/мес.)", 
                value=1.0, 
                step=0.5, 
                min_value=0.0,
                help="Процентный рост аренды ежемесячно."
            ) / 100.0
            default_probability = st.number_input(
                "⚠️ Вероятность невозврата (%)", 
                value=5.0, 
                step=1.0, 
                min_value=0.0, 
                max_value=100.0,
                help="Вероятность невозврата по займам."
            ) / 100.0
            liquidity_factor = st.number_input(
                "💧 Ликвидность", 
                value=1.0, 
                step=0.1, 
                min_value=0.1,
                help="Коэффициент ликвидности для анализа."
            )
            safety_factor = st.number_input(
                "🛡 Запас", 
                value=1.2, 
                step=0.1, 
                min_value=0.1,
                help="Коэффициент запаса для анализа."
            )
            loan_grace_period = st.number_input(
                "⏳ Льготный период (мес.)", 
                value=0, 
                step=1, 
                min_value=0,
                help="Льготный период по займам."
            )
            monthly_income_growth = st.number_input(
                "📈 Рост доходов (%/мес.)", 
                value=0.0, 
                step=0.5,
                help="Предполагаемый рост доходов в %/мес."
            ) / 100.0
            monthly_expenses_growth = st.number_input(
                "📉 Рост расходов (%/мес.)", 
                value=0.0, 
                step=0.5,
                help="Предполагаемый рост расходов в %/мес."
            ) / 100.0
        else:
            time_horizon = 1
            monthly_rent_growth = 0.0
            default_probability = 0.0
            liquidity_factor = 1.0
            safety_factor = 1.2
            loan_grace_period = 0
            monthly_income_growth = 0.0
            monthly_expenses_growth = 0.0

        forecast_method = st.selectbox(
            "📊 Метод прогнозирования", 
            ["Базовый", "ML (линейная регрессия)", "Симуляция Монте-Карло"],
            help="Выберите метод прогноза."
        )
        if forecast_method == "Симуляция Монте-Карло":
            monte_carlo_simulations = st.number_input(
                "🎲 Симуляций Монте-Карло", 
                value=100, 
                step=10, 
                min_value=10,
                help="Число симуляций для Монте-Карло."
            )
            monte_carlo_deviation = st.number_input(
                "🔀 Отклонения (0.1 = ±10%)", 
                value=0.1, 
                step=0.01, 
                min_value=0.01,
                help="Отклонение для Монте-Карло симуляций."
            )
            monte_carlo_seed = st.number_input(
                "🔑 Seed", 
                value=42, 
                step=1,
                help="Зерно для случайных чисел (Монте-Карло)."
            )
        else:
            monte_carlo_simulations = 100
            monte_carlo_deviation = 0.1
            monte_carlo_seed = 42

        enable_ml_settings = False
        if forecast_method == "ML (линейная регрессия)":
            enable_ml_settings = st.checkbox(
                "🤖 Включить расширенный ML-прогноз", 
                value=False,
                help="Дополнительные настройки для ML-прогноза."
            )

# Основная логика
params = WarehouseParams(
    total_area=total_area,
    rental_cost_per_m2=rental_cost_per_m2,
    useful_area_ratio=useful_area_ratio,
    mode=mode,
    storage_share=storage_share,
    loan_share=loan_share,
    vip_share=vip_share,
    short_term_share=short_term_share,
    storage_area_manual=storage_area_manual,
    loan_area_manual=loan_area_manual,
    vip_area_manual=vip_area_manual,
    short_term_area_manual=short_term_area_manual,
    storage_fee=storage_fee,
    shelves_per_m2=shelves_per_m2,  # Изменено
    short_term_daily_rate=short_term_daily_rate,
    vip_extra_fee=vip_extra_fee,
    item_evaluation=item_evaluation,
    item_realization_markup=item_realization_markup,
    average_item_value=average_item_value,
    loan_interest_rate=loan_interest_rate,
    realization_share_storage=realization_share_storage,
    realization_share_loan=realization_share_loan,
    realization_share_vip=realization_share_vip,
    realization_share_short_term=realization_share_short_term,
    storage_items_density=storage_items_density,
    loan_items_density=loan_items_density,
    vip_items_density=vip_items_density,
    short_term_items_density=short_term_items_density,
    salary_expense=salary_expense,
    miscellaneous_expenses=miscellaneous_expenses,
    depreciation_expense=depreciation_expense,
    marketing_expenses=marketing_expenses,
    insurance_expenses=insurance_expenses,
    taxes=taxes,
    utilities_expenses=utilities_expenses,
    maintenance_expenses=maintenance_expenses,
    one_time_setup_cost=one_time_setup_cost,
    one_time_equipment_cost=one_time_equipment_cost,
    one_time_other_costs=one_time_other_costs,
    one_time_legal_cost=one_time_legal_cost,
    one_time_logistics_cost=one_time_logistics_cost,
    time_horizon=time_horizon,
    monthly_rent_growth=monthly_rent_growth,
    default_probability=default_probability,
    liquidity_factor=liquidity_factor,
    safety_factor=safety_factor,
    loan_grace_period=loan_grace_period,
    monthly_income_growth=monthly_income_growth,
    monthly_expenses_growth=monthly_expenses_growth,
    forecast_method=forecast_method,
    monte_carlo_simulations=monte_carlo_simulations,
    monte_carlo_deviation=monte_carlo_deviation,
    monte_carlo_seed=monte_carlo_seed,
    enable_ml_settings=enable_ml_settings
)

is_valid, error_message = validate_inputs(params)
if is_valid:
    areas = calculate_areas(params)
    for k, v in areas.items():
        setattr(params, k, v)

    # Изменения начинаются здесь
    # Вычисляем дополнительную площадь, занимаемую полками
    SHELF_AREA_PER_SHELF = 0.1  # м² на одну полку, можно настроить по необходимости
    number_of_shelves = params.shelves_per_m2 * params.storage_area  # Количество полок
    shelves_area = number_of_shelves * SHELF_AREA_PER_SHELF
    params.storage_area += shelves_area  # Увеличиваем площадь под хранение
    params.total_area += shelves_area  # Увеличиваем общую площадь склада
    # Изменения заканчиваются здесь

    items_dict = {
        "stored_items": calculate_items(params.storage_area, params.shelves_per_m2, params.storage_items_density),
        "total_items_loan": calculate_items(params.loan_area, params.shelves_per_m2, params.loan_items_density),
        "vip_stored_items": calculate_items(params.vip_area, params.shelves_per_m2, params.vip_items_density),
        "short_term_stored_items": calculate_items(params.short_term_area, params.shelves_per_m2, params.short_term_items_density)
    }
    base_financials = calculate_financials(params, disable_extended=False)
    profit_margin, profitability = calculate_additional_metrics(
        base_financials["total_income"], base_financials["total_expenses"], base_financials["profit"]
    )
    roi_val = calculate_roi(base_financials["total_income"], base_financials["total_expenses"])

    # Формируем список денежных потоков для IRR
    initial_investment = -(
        params.one_time_setup_cost +
        params.one_time_equipment_cost +
        params.one_time_other_costs
    )
    cash_flows = [initial_investment] + [base_financials["profit"]] * params.time_horizon
    irr_val = calculate_irr(cash_flows)  # Используем обновлённую функцию
    print(f"Расчитанный IRR: {irr_val}%")

    bep_val = calculate_total_bep(base_financials, params)

    # Загрузка ML-модели, если включены настройки ML
    ml_model = None
    if params.enable_ml_settings:
        model_path = "ml_model.pkl"
        if os.path.exists(model_path):
            try:
                ml_model = joblib.load(model_path)
                st.success("ML-модель успешно загружена.")
            except Exception as e:
                st.error(f"Ошибка загрузки ML-модели: {e}")
        else:
            st.warning("ML-модель не найдена. Пожалуйста, обучите модель.")

    # Вызов функций отображения вкладок
    tab1, tab2_, tab3_, tab4_ = st.tabs(["📊 Общие результаты", "📈 Прогнозирование", "🔍 Точка безубыточности", "📋 Детализация"])
    display_tab1(tab1, base_financials, profit_margin, profitability, roi_val, irr_val, params)
    display_tab2(tab2_, base_financials, False, params.enable_ml_settings, params.forecast_method, params, ml_model)
    display_tab3(tab3_, base_financials, False, False, params)
    display_tab4(tab4_, items_dict, base_financials, params, False, irr_val)
else:
    st.error(f"Ошибка ввода данных: {error_message}")
