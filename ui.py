# ui.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import generate_download_link, generate_excel_download
from calculations import (
    calculate_additional_metrics, calculate_roi, calculate_irr, 
    calculate_total_bep, monte_carlo_simulation, calculate_financials, min_loan_amount_for_bep
)
from ml_models import train_ml_model, predict_with_model
import joblib
import os

def display_tab1(tab, base_financials, profit_margin, profitability, roi, irr, params):
    """
    Отображает вкладку "Общие результаты".
    
    :param tab: Объект вкладки.
    :param base_financials: Словарь с финансовыми показателями.
    :param profit_margin: Маржа прибыли.
    :param profitability: Рентабельность.
    :param roi: ROI.
    :param irr: IRR.
    :param params: Объект WarehouseParams с параметрами склада.
    """
    with tab:
        st.header("📊 Общие результаты")
        st.write("Ниже представлены ключевые показатели текущей конфигурации хранения.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Общий доход (руб.)", f"{base_financials['total_income']:,.2f}")
        col2.metric("Общие расходы (руб.)", f"{base_financials['total_expenses']:,.2f}")
        col3.metric("Прибыль (руб.)", f"{base_financials['profit']:,.2f}")

        col4, col5 = st.columns(2)
        col4.metric("Маржа прибыли (%)", f"{profit_margin:.2f}%")
        col5.metric("Рентабельность (%)", f"{profitability:.2f}%")

        col6, col7 = st.columns(2)
        col6.metric("ROI (%)", f"{roi:.2f}%")
        col7.metric("IRR (%)", f"{irr:.2f}%")

        st.write("---")
        st.subheader("Мин. сумма займа для BEP займов")
        min_loan = min_loan_amount_for_bep(params, base_financials)
        if np.isinf(min_loan):
            st.write("**Бесконечность** - невозможно покрыть расходы при текущих условиях.")
        else:
            st.write(f"**{min_loan:,.2f} руб.** - минимальная сумма займа, необходимая для покрытия расходов.")

        # Диаграмма для наглядности доходов и расходов
        df_plot = pd.DataFrame({
            "Категория": ["Доход", "Расход"],
            "Значение": [base_financials['total_income'], base_financials['total_expenses']]
        })
        fig = px.bar(df_plot, x="Категория", y="Значение", title="Доходы и расходы", text="Значение",
                   color="Категория", color_discrete_map={"Доход": "green", "Расход": "red"})
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis={'title': 'Рубли'}, xaxis={'title': ''})
        st.plotly_chart(fig, use_container_width=True)

def display_tab2(tab, base_financials, disable_extended, enable_ml_settings, selected_forecast_method, params, ml_model=None):
    """
    Отображает вкладку "Прогнозирование".
    
    :param tab: Объект вкладки.
    :param base_financials: Словарь с финансовыми показателями.
    :param disable_extended: Флаг отключения расширенных параметров.
    :param enable_ml_settings: Флаг включения настроек ML.
    :param selected_forecast_method: Выбранный метод прогнозирования.
    :param params: Объект WarehouseParams с параметрами склада.
    :param ml_model: Загруженная ML-модель или None.
    """
    with tab:
        st.header("📈 Прогнозирование")
        st.write("Измените метод прогнозирования в боковой панели для обновления графиков.")

        if selected_forecast_method == "Базовый":
            st.info("Базовый прогноз: линейный рост доходов и расходов.")
            df_projection = pd.DataFrame({
                "Месяц": range(1, params.time_horizon + 1),
                "Доходы": np.linspace(
                    base_financials["total_income"],
                    base_financials["total_income"] * (1 + params.monthly_income_growth * params.time_horizon),
                    params.time_horizon
                ),
                "Расходы": np.linspace(
                    base_financials["total_expenses"],
                    base_financials["total_expenses"] * (1 + params.monthly_expenses_growth * params.time_horizon),
                    params.time_horizon
                )
            })
            df_projection["Прибыль"] = df_projection["Доходы"] - df_projection["Расходы"]
            df_long = df_projection.melt(id_vars="Месяц", value_vars=["Доходы", "Расходы", "Прибыль"],
                                         var_name="Показатель", value_name="Значение")
            fig = px.line(df_long, x="Месяц", y="Значение", color="Показатель", title="Базовый прогноз",
                          markers=True)
            fig.update_layout(yaxis={'title': 'Рубли'}, xaxis={'title': 'Месяц'})
            st.plotly_chart(fig, use_container_width=True)

        elif selected_forecast_method == "ML (линейная регрессия)":
            if enable_ml_settings:
                st.info("ML-прогноз: Обучение и прогнозирование с использованием линейной регрессии.")
                if ml_model:
                    future_months = list(range(params.time_horizon + 1, params.time_horizon + 7))  # Прогноз на 6 месяцев вперёд
                    # Здесь предполагается, что ML-модель принимает список месяцев и возвращает прогнозные доходы
                    predictions = predict_with_model(ml_model, future_months)
                    df_ml = pd.DataFrame({
                        "Месяц": future_months,
                        "Прогноз Доходы": predictions
                    })
                    st.write("Прогнозируемые доходы на следующие 6 месяцев с использованием ML-модели:")
                    st.dataframe(df_ml.style.format({"Прогноз Доходы": "{:,.2f} руб."}))
                    
                    # Визуализация
                    fig_ml = px.line(df_ml, x="Месяц", y="Прогноз Доходы", title="Прогноз Доходов с использованием ML-модели",
                                     markers=True)
                    fig_ml.update_layout(yaxis={'title': 'Рубли'}, xaxis={'title': 'Месяц'})
                    st.plotly_chart(fig_ml, use_container_width=True)
                else:
                    st.warning("ML-модель не обучена. Пожалуйста, обучите модель сначала.")
            else:
                st.info("ML-прогнозирование отключено.")
                st.write("Включите расширенные настройки для использования ML-прогноза.")

        else:  # Симуляция Монте-Карло
            st.info("Симуляция Монте-Карло для анализа разброса доходов и расходов.")
            df_mc = monte_carlo_simulation(
                base_financials["total_income"],
                base_financials["total_expenses"],
                params.time_horizon,
                params.monte_carlo_simulations,
                params.monte_carlo_deviation,
                params.monte_carlo_seed,
                params.monthly_income_growth,
                params.monthly_expenses_growth
            )
            st.dataframe(df_mc.style.format("{:,.2f}"))
            df_long = df_mc.melt(id_vars="Месяц",
                                 value_vars=["Средний Доход", "Средний Расход", "Средняя Прибыль"],
                                 var_name="Показатель", value_name="Значение")
            fig_mc = px.line(df_long, x="Месяц", y="Значение", color="Показатель", title="Монте-Карло: Средние значения",
                             markers=True)
            fig_mc.update_layout(yaxis={'title': 'Рубли'}, xaxis={'title': 'Месяц'})
            st.plotly_chart(fig_mc, use_container_width=True)

def display_tab3(tab, base_financials, disable_extended, enable_multi_sensitivity, params):
    """
    Отображает вкладку "Точка безубыточности".
    
    :param tab: Объект вкладки.
    :param base_financials: Словарь с финансовыми показателями.
    :param disable_extended: Флаг отключения расширенных параметров.
    :param enable_multi_sensitivity: Флаг включения многофакторного анализа.
    :param params: Объект WarehouseParams с параметрами склада.
    """
    with tab:
        st.header("🔍 Точка безубыточности (BEP)")
        st.write("Рассчитаем общую BEP, где доходы покрывают расходы.")

        bep_income = calculate_total_bep(base_financials, params)
        current_income = base_financials["total_income"]

        if bep_income == float('inf'):
            st.write("**Бесконечность** - невозможно покрыть расходы при текущих условиях.")
        else:
            st.write(f"**Необходимый доход для BEP:** {bep_income:,.2f} руб.")
            if current_income >= bep_income:
                st.success("Текущий доход **покрывает** расходы (BEP достигнут или превышен).")
            else:
                deficit = bep_income - current_income
                st.error(f"Текущий доход **не покрывает** расходы. Необходимо увеличить доход на **{deficit:,.2f} руб.**.")

        st.write("---")
        st.subheader("График сравнения текущего дохода и BEP")

        df_bep = pd.DataFrame({
            "Категория": ["Текущий Доход", "Необходимый Доход для BEP"],
            "Значение": [current_income, bep_income]
        })
        fig_bep = px.bar(df_bep, x="Категория", y="Значение", title="Сравнение текущего дохода и BEP", text="Значение",
                        color="Категория", color_discrete_map={"Текущий Доход": "blue", "Необходимый Доход для BEP": "orange"})
        fig_bep.update_traces(textposition="outside")
        fig_bep.update_layout(yaxis={'title': 'Рубли'}, xaxis={'title': ''})
        st.plotly_chart(fig_bep, use_container_width=True)

        st.write("---")
        st.subheader("Графики чувствительности")

        def build_bep_df(params, param_key, base_val):
            """
            Строит DataFrame для анализа чувствительности по заданному параметру.
            
            :param params: Объект WarehouseParams с параметрами склада.
            :param param_key: Ключ параметра для анализа.
            :param base_val: Базовое значение параметра.
            :return: DataFrame с результатами.
            """
            vals = np.linspace(base_val * 0.5 if base_val > 0 else 0.1, base_val * 1.5 if base_val > 0 else 1.5, 50)
            profits = []
            bep_incomes = []
            orig_val = getattr(params, param_key)
            for v in vals:
                setattr(params, param_key, v)
                fin = calculate_financials(params, disable_extended)
                bep_income = calculate_total_bep(fin, params)
                profits.append(fin["profit"])
                bep_incomes.append(bep_income)
            setattr(params, param_key, orig_val)
            return pd.DataFrame({param_key: vals, "Прибыль": profits, "Необходимый доход для BEP": bep_incomes})

        # Параметры для анализа
        analysis_params = {
            "storage_fee": "Тариф простого",
            "vip_extra_fee": "Доп. тариф VIP",
            "short_term_daily_rate": "Тариф краткосрочного"
        }

        for key, label in analysis_params.items():
            df_sensitivity = build_bep_df(params, key, getattr(params, key))
            fig_sensitivity = px.line(df_sensitivity, x=key, y=["Прибыль", "Необходимый доход для BEP"], 
                                      labels={key: label, "value": "Рубли", "variable": "Показатель"},
                                      title=f"{label}: Чувствительность", markers=True)
            fig_sensitivity.update_layout(yaxis={'title': 'Рубли'}, xaxis={'title': label})
            fig_sensitivity.update_traces(mode='lines+markers')
            st.plotly_chart(fig_sensitivity, use_container_width=True)

def display_tab4(tab, items, base_financials, params, disable_extended, irr_val):
    """
    Отображает вкладку "Детализация".
    
    :param tab: Объект вкладки.
    :param items: Словарь с количеством вещей.
    :param base_financials: Словарь с финансовыми показателями.
    :param params: Объект WarehouseParams с параметрами склада.
    :param disable_extended: Флаг отключения расширенных параметров.
    :param irr_val: Значение IRR.
    """
    from calculations import calculate_additional_metrics, calculate_roi, calculate_irr
    with tab:
        st.header("📋 Детализация")
        st.write("Подробная информация о площади, прибыли по типам, а также итоговый отчёт для скачивания.")

        # Метрики по видам хранения
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Простое (м²)", f"{params.storage_area:,.2f} м²")
        col2.metric("VIP (м²)", f"{params.vip_area:,.2f} м²")
        col3.metric("Краткосрочное (м²)", f"{params.short_term_area:,.2f} м²")
        col4.metric("Займы (м²)", f"{params.loan_area:,.2f} м²")

        st.write("---")
        # Таблица распределения площади
        df_storage = pd.DataFrame({
            "Тип хранения": ["Простое", "VIP", "Краткосрочное", "Займы"],
            "Площадь (м²)": [params.storage_area, params.vip_area, params.short_term_area, params.loan_area],
            "Доход (руб.)": [
                base_financials["storage_income"],
                base_financials["vip_income"],
                base_financials["short_term_income"],
                base_financials["loan_income_after_realization"]
            ]
        })
        st.dataframe(df_storage.style.format({"Площадь (м²)": "{:,.2f}", "Доход (руб.)": "{:,.2f}"}))

        st.write("---")
        # Таблица прибыли и расходов
        df_profit = pd.DataFrame({
            "Тип хранения": ["Простое", "VIP", "Краткосрочное", "Займы"],
            "Доход (руб.)": [
                base_financials["storage_income"],
                base_financials["vip_income"],
                base_financials["short_term_income"],
                base_financials["loan_income_after_realization"]
            ],
            "Ежемесячные расходы (руб.)": [
                params.storage_area * params.rental_cost_per_m2,
                params.vip_area * params.rental_cost_per_m2,
                params.short_term_area * params.rental_cost_per_m2,
                params.loan_area * params.rental_cost_per_m2
            ],
            "Прибыль (руб.)": [
                base_financials["storage_income"] - (params.storage_area * params.rental_cost_per_m2),
                base_financials["vip_income"] - (params.vip_area * params.rental_cost_per_m2),
                base_financials["short_term_income"] - (params.short_term_area * params.rental_cost_per_m2),
                base_financials["loan_income_after_realization"] - (params.loan_area * params.rental_cost_per_m2)
            ]
        })
        numeric_cols = ["Доход (руб.)", "Ежемесячные расходы (руб.)", "Прибыль (руб.)"]
        for col in numeric_cols:
            df_profit[col] = pd.to_numeric(df_profit[col], errors="coerce")

        def highlight_negative(s):
            return ['background-color: #ffcccc' if v < 0 else '' for v in s]

        st.dataframe(
            df_profit.style.apply(highlight_negative, subset=["Прибыль (руб.)"])
            .format({"Доход (руб.)": "{:,.2f}", "Ежемесячные расходы (руб.)": "{:,.2f}", "Прибыль (руб.)": "{:,.2f}"})
        )

        st.write("---")
        # Итоговые финансовые показатели
        profit_margin, profitability = calculate_additional_metrics(
            base_financials["total_income"], base_financials["total_expenses"], base_financials["profit"]
        )
        roi_val = calculate_roi(base_financials["total_income"], base_financials["total_expenses"])
        irr_val_display = irr_val  # Уже передано из main.py
        bep_income = calculate_total_bep(base_financials, params)
        current_income = base_financials["total_income"]

        df_results = pd.DataFrame({
            "Показатель": [
                "Общий доход",
                "Общие расходы",
                "Прибыль",
                "Маржа прибыли (%)",
                "Рентабельность (%)",
                "ROI (%)",
                "IRR (%)",
                "Доход от маркетинга",
                "Мин. сумма займа (руб.)",
                "Единовременные расходы (руб.)",
                "Налоги (руб./мес.)",
                "Страхование (руб./мес.)",
                "Коммуналка (руб./мес.)",
                "Обслуживание (руб./мес.)",
                "Необходимый доход для BEP (руб.)",
                "Текущий доход (руб.)"
            ],
            "Значение": [
                base_financials["total_income"],
                base_financials["total_expenses"],
                base_financials["profit"],
                profit_margin,
                profitability,
                roi_val,
                irr_val_display,
                base_financials["marketing_income"],
                min_loan_amount_for_bep(params, base_financials),
                params.one_time_expenses,
                params.taxes,
                params.insurance_expenses,
                params.utilities_expenses,
                params.maintenance_expenses,
                bep_income,
                current_income
            ]
        })

        st.markdown(generate_download_link(df_results, filename="results.csv", link_text="Скачать CSV"))
        st.markdown(generate_excel_download(df_results, filename="results.xlsx", link_text="Скачать Excel"))

        st.info("Скачайте расширенный отчёт для дальнейшего анализа.")
