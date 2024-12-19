# data_model.py

from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class WarehouseParams:
    """
    Класс для хранения параметров склада.
    """
    total_area: float
    rental_cost_per_m2: float
    useful_area_ratio: float
    mode: str  # "Автоматический" или "Ручной"
    storage_share: float
    loan_share: float
    vip_share: float
    short_term_share: float
    storage_area_manual: float
    loan_area_manual: float
    vip_area_manual: float
    short_term_area_manual: float
    storage_fee: float
    shelves_per_m2: float
    short_term_daily_rate: float
    vip_extra_fee: float
    item_evaluation: float
    item_realization_markup: float
    average_item_value: float
    loan_interest_rate: float
    realization_share_storage: float
    realization_share_loan: float
    realization_share_vip: float
    realization_share_short_term: float
    storage_items_density: float
    loan_items_density: float
    vip_items_density: float
    short_term_items_density: float
    salary_expense: float
    miscellaneous_expenses: float
    depreciation_expense: float
    marketing_expenses: float
    insurance_expenses: float
    taxes: float
    utilities_expenses: float
    maintenance_expenses: float
    one_time_setup_cost: float
    one_time_equipment_cost: float
    one_time_other_costs: float
    one_time_legal_cost: float
    one_time_logistics_cost: float
    time_horizon: int
    monthly_rent_growth: float
    default_probability: float
    liquidity_factor: float
    safety_factor: float
    loan_grace_period: int
    monthly_income_growth: float
    monthly_expenses_growth: float
    forecast_method: str
    monte_carlo_simulations: int
    monte_carlo_deviation: float
    monte_carlo_seed: int
    enable_ml_settings: bool
    one_time_expenses: float = 0.0
    usable_area: float = 0.0
    storage_area: float = 0.0
    loan_area: float = 0.0
    vip_area: float = 0.0
    short_term_area: float = 0.0
    payback_period: float = 0.0

def validate_inputs(params: WarehouseParams) -> Tuple[bool, str]:
    """
    Проверяет корректность введённых данных.

    :param params: Объект WarehouseParams с параметрами склада.
    :return: Кортеж (bool, str), где bool - результат проверки, str - сообщение об ошибке.
    """
    if params.total_area <= 0:
        return False, "Общая площадь должна быть больше нуля."
    if not (0 < params.useful_area_ratio <= 1):
        return False, "Доля полезной площади должна быть между 0 и 1."
    if params.mode == "Автоматический":
        total_share = params.storage_share + params.loan_share + params.vip_share + params.short_term_share
        if abs(total_share) < 1e-9:
            return False, "Сумма долей видов хранения должна быть больше нуля."
        if total_share > 1.0:
            return False, "Сумма долей видов хранения не должна превышать 100%."
    else:
        total_manual_area = params.storage_area_manual + params.loan_area_manual + params.vip_area_manual + params.short_term_area_manual
        usable_area = params.total_area * params.useful_area_ratio
        if total_manual_area == 0:
            return False, "Сумма вручную введённых площадей должна быть больше нуля."
        if total_manual_area > usable_area:
            return False, f"Сумма вручную введённых площадей ({total_manual_area} м²) превышает полезную площадь ({usable_area} м²)."
    return True, ""
