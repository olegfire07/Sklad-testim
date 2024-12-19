# calculations.py

import numpy as np
import pandas as pd
import logging
import numpy_financial as npf  # Убедитесь, что пакет установлен

from data_model import WarehouseParams  # Импортируем WarehouseParams

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_irr(cash_flows):
    logging.info("Вызвана функция calculate_irr из calculations.py")
    try:
        irr = npf.irr(cash_flows)
        if irr is not None and not np.isnan(irr):
            return irr * 100  # Преобразуем в процентное значение
        else:
            logging.warning("IRR не может быть рассчитан: результат NaN или None.")
            return 0.0
    except Exception as e:
        logging.error(f"Ошибка при расчёте IRR: {e}")
        return 0.0

def calculate_areas(params: WarehouseParams):
    """
    Рассчитывает распределение площадей по видам хранения.

    :param params: Объект WarehouseParams с параметрами склада.
    :return: Словарь с рассчитанными площадями.
    """
    logging.info("Начало расчёта распределения площадей.")
    usable_area = params.total_area * params.useful_area_ratio
    if params.mode == "Автоматический":
        total_share = params.storage_share + params.loan_share + params.vip_share + params.short_term_share
        if total_share < 1e-9:
            storage_area = loan_area = vip_area = short_term_area = 0.0
            logging.warning("Сумма долей видов хранения равна нулю. Все площади установлены в 0.")
        else:
            f_storage = params.storage_share / total_share if total_share > 0 else 0
            f_loan = params.loan_share / total_share if total_share > 0 else 0
            f_vip = params.vip_share / total_share if total_share > 0 else 0
            f_short = params.short_term_share / total_share if total_share > 0 else 0

            storage_area = usable_area * f_storage
            loan_area = usable_area * f_loan
            vip_area = usable_area * f_vip
            short_term_area = usable_area * f_short

            logging.info(f"Распределение площадей - Простое: {storage_area:.2f} м², Займы: {loan_area:.2f} м², VIP: {vip_area:.2f} м², Краткосрочное: {short_term_area:.2f} м².")
    else:
        # При ручном вводе пользователь задаёт площади вручную
        total_manual = params.storage_area_manual + params.loan_area_manual + params.vip_area_manual + params.short_term_area_manual
        if total_manual > usable_area and total_manual > 0:
            factor = usable_area / total_manual
            storage_area = params.storage_area_manual * factor
            loan_area = params.loan_area_manual * factor
            vip_area = params.vip_area_manual * factor
            short_term_area = params.short_term_area_manual * factor
            logging.warning("Сумма вручную введённых площадей превышает полезную площадь. Пропорциональное снижение площадей.")
        else:
            storage_area = params.storage_area_manual
            loan_area = params.loan_area_manual
            vip_area = params.vip_area_manual
            short_term_area = params.short_term_area_manual
            logging.info(f"Распределение площадей (ручной ввод) - Простое: {storage_area:.2f} м², VIP: {vip_area:.2f} м², Краткосрочное: {short_term_area:.2f} м², Займы: {loan_area:.2f} м².")

    return {
        "usable_area": usable_area,
        "storage_area": storage_area,
        "loan_area": loan_area,
        "vip_area": vip_area,
        "short_term_area": short_term_area
    }

def calculate_items(area, shelves, density):
    """
    Рассчитывает количество вещей для данного типа хранения.

    :param area: Площадь (м²)
    :param shelves: Количество полок на м²
    :param density: Плотность вещей на полку
    :return: Количество вещей
    """
    items = area * shelves * density
    logging.info(f"Рассчитано {items} вещей для площади {area} м², полок {shelves} м² и плотности {density} вещей/полку.")
    return items

def calculate_financials(params: WarehouseParams, disable_extended: bool):
    """
    Расчёт финансовых показателей: доходы, расходы, прибыль.

    :param params: Объект WarehouseParams с параметрами склада.
    :param disable_extended: Флаг отключения расширенных параметров.
    :return: Словарь с финансовыми показателями.
    """
    logging.info("Начало расчёта финансовых показателей.")

    storage_items = calculate_items(params.storage_area, params.shelves_per_m2, params.storage_items_density)
    loan_items = calculate_items(params.loan_area, params.shelves_per_m2, params.loan_items_density)
    vip_items = calculate_items(params.vip_area, params.shelves_per_m2, params.vip_items_density)
    short_term_items = calculate_items(params.short_term_area, params.shelves_per_m2, params.short_term_items_density)

    # Доход от хранения (мес.)
    storage_income = params.storage_area * params.storage_fee
    vip_income = params.vip_area * (params.storage_fee + params.vip_extra_fee)
    short_term_income = params.short_term_area * params.short_term_daily_rate * 30

    # Доход от займов
    loan_evaluated_value = params.average_item_value * params.item_evaluation * loan_items
    daily_loan_interest_rate = params.loan_interest_rate / 100.0 if params.loan_interest_rate > 0 else 0
    loan_income = loan_evaluated_value * daily_loan_interest_rate * (1 - params.default_probability) * 30 if params.loan_interest_rate > 0 else 0.0

    # Реализация (наценка от реализации)
    storage_realization_items = storage_items * params.realization_share_storage
    loan_realization_items = loan_items * params.realization_share_loan
    vip_realization_items = vip_items * params.realization_share_vip
    short_term_realization_items = short_term_items * params.realization_share_short_term

    storage_realization = params.average_item_value * params.item_evaluation * storage_realization_items * (params.item_realization_markup / 100)
    loan_realization = params.average_item_value * params.item_evaluation * loan_realization_items * (params.item_realization_markup / 100)
    vip_realization = params.average_item_value * params.item_evaluation * vip_realization_items * (params.item_realization_markup / 100)
    short_term_realization = params.average_item_value * params.item_evaluation * short_term_realization_items * (params.item_realization_markup / 100)

    realization_income = storage_realization + loan_realization + vip_realization + short_term_realization
    marketing_income = 0.0  # Можете добавить реальные расчёты, если есть

    total_income = storage_income + short_term_income + realization_income + loan_income + vip_income + marketing_income

    # Ежемесячные расходы
    monthly_rent = params.total_area * params.rental_cost_per_m2
    total_monthly_expenses = (monthly_rent + params.salary_expense + params.miscellaneous_expenses +
                              params.depreciation_expense + params.marketing_expenses + params.insurance_expenses +
                              params.taxes + params.utilities_expenses + params.maintenance_expenses)

    # Единовременные расходы
    total_one_time = (params.one_time_setup_cost + params.one_time_equipment_cost +
                      params.one_time_other_costs + params.one_time_legal_cost + params.one_time_logistics_cost)
    params.one_time_expenses = total_one_time

    if params.time_horizon > 0:
        profit = total_income - total_monthly_expenses - params.one_time_expenses / params.time_horizon
    else:
        profit = total_income - total_monthly_expenses

    daily_storage_fee = params.storage_fee / 30 if params.storage_fee > 0 else 0.0

    logging.info("Расчёт финансовых показателей завершён.")

    return {
        "total_income": total_income,
        "total_expenses": total_monthly_expenses,
        "profit": profit,
        "storage_income": storage_income,
        "short_term_income": short_term_income,
        "realization_income": realization_income,
        "loan_income_after_realization": loan_income,
        "vip_income": vip_income,
        "marketing_income": marketing_income,
        "daily_storage_fee": daily_storage_fee,
        "storage_realization": storage_realization,
        "loan_realization": loan_realization,
        "vip_realization": vip_realization,
        "short_term_realization": short_term_realization,
        "storage_items": storage_items,
        "loan_items": loan_items,
        "vip_items": vip_items,
        "short_term_items": short_term_items
    }

def calculate_total_bep(financials: dict, params: WarehouseParams):
    """
    Рассчитывает общую точку безубыточности, где доходы покрывают расходы.

    :param financials: Словарь с финансовыми показателями.
    :param params: Объект WarehouseParams с параметрами склада.
    :return: Необходимый доход для покрытия расходов.
    """
    logging.info("Начало расчёта общей точки безубыточности (BEP).")
    total_income = financials["total_income"]
    total_expenses = financials["total_expenses"] + (params.one_time_expenses / params.time_horizon if params.time_horizon > 0 else 0)
    
    if total_income == 0:
        logging.warning("Общий доход равен нулю, BEP невозможно достичь.")
        return float('inf')  # Бесконечная точка безубыточности
    
    # BEP_income = total_expenses (для прибыли = 0)
    bep_income = total_expenses
    
    logging.info(f"Расчет общей BEP: необходимый доход {bep_income:.2f} руб.")
    return bep_income

def calculate_additional_metrics(total_income, total_expenses, profit):
    """
    Рассчитывает дополнительные финансовые метрики.

    :param total_income: Общий доход.
    :param total_expenses: Общие расходы.
    :param profit: Прибыль.
    :return: Кортеж (маржа прибыли, рентабельность).
    """
    profit_margin = (profit / total_income * 100) if total_income != 0 else 0
    profitability = (profit / total_expenses * 100) if total_expenses != 0 else 0
    logging.info(f"Маржа прибыли: {profit_margin:.2f}%, Рентабельность: {profitability:.2f}%")
    return profit_margin, profitability

def calculate_roi(total_income, total_expenses):
    """
    Рассчитывает ROI (Return on Investment).

    :param total_income: Общий доход.
    :param total_expenses: Общие расходы.
    :return: ROI в процентах.
    """
    try:
        roi = ((total_income - total_expenses) / total_expenses * 100) if total_expenses != 0 else float('inf')
        logging.info(f"Расчет ROI: {roi:.2f}%")
        return roi
    except ZeroDivisionError:
        logging.error("Расчет ROI невозможен: общие расходы равны нулю.")
        return float('inf')

def monte_carlo_simulation(base_income, base_expenses, time_horizon, simulations, deviation, seed, monthly_income_growth, monthly_expenses_growth):
    """
    Выполняет симуляцию Монте-Карло для доходов и расходов.

    :param base_income: Базовый доход.
    :param base_expenses: Базовые расходы.
    :param time_horizon: Горизонт планирования (мес.).
    :param simulations: Количество симуляций.
    :param deviation: Отклонение (например, 0.1 = ±10%).
    :param seed: Зерно для генератора случайных чисел.
    :param monthly_income_growth: Ежемесячный рост доходов.
    :param monthly_expenses_growth: Ежемесячный рост расходов.
    :return: DataFrame со средними значениями доходов, расходов и прибыли.
    """
    logging.info("Начало симуляции Монте-Карло.")
    np.random.seed(seed)
    months = np.arange(1, time_horizon + 1)
    
    income_growth = (1 + monthly_income_growth) ** months
    expense_growth = (1 + monthly_expenses_growth) ** months
    
    # Генерация случайных коэффициентов отклонения
    income_factors = np.random.uniform(1 - deviation, 1 + deviation, (simulations, time_horizon))
    expense_factors = np.random.uniform(1 - deviation, 1 + deviation, (simulations, time_horizon))
    
    # Векторизованные расчеты доходов и расходов
    incomes = base_income * income_growth * income_factors
    expenses = base_expenses * expense_growth * expense_factors
    profits = incomes - expenses

    avg_incomes = incomes.mean(axis=0)
    avg_expenses = expenses.mean(axis=0)
    avg_profit = profits.mean(axis=0)
    
    df = pd.DataFrame({
        "Месяц": months,
        "Средний Доход": avg_incomes,
        "Средний Расход": avg_expenses,
        "Средняя Прибыль": avg_profit
    })
    logging.info("Симуляция Монте-Карло завершена.")
    return df

def min_loan_amount_for_bep(params: WarehouseParams, fin: dict):
    """
    Расчёт минимальной суммы займа для покрытия расходов.

    :param params: Объект WarehouseParams с параметрами склада.
    :param fin: Словарь с финансовыми показателями.
    :return: Минимальная сумма займа.
    """
    logging.info("Расчет минимальной суммы займа для BEP.")
    if params.time_horizon > 0:
        total_exp = fin["total_expenses"] + params.one_time_expenses / params.time_horizon
    else:
        total_exp = fin["total_expenses"]
    loan_items = fin["loan_items"]
    if loan_items <= 0:
        logging.warning("Нет займаемых вещей, минимальная сумма займа устанавливается в 0.")
        return 0.0
    daily_loan_interest_rate = params.loan_interest_rate / 100.0 if params.loan_interest_rate > 0 else 0.0001
    # Минимальная сумма займа на одну вещь, чтобы покрыть расходы:
    min_loan_value = (total_exp / loan_items) / (daily_loan_interest_rate * 30)
    logging.info(f"Минимальная сумма займа: {min_loan_value:.2f} руб.")
    return max(min_loan_value, 0.0)
