# test_calculations.py

import unittest
import numpy_financial as npf  # Убедитесь, что пакет установлен
from calculations import (
    calculate_items,
    calculate_total_bep,
    calculate_financials,
    calculate_areas,
    calculate_irr  # Импортируем из calculations.py
)
from data_model import WarehouseParams

class TestCalculations(unittest.TestCase):
    
    def test_calculate_items(self):
        self.assertEqual(calculate_items(10, 3, 5), 150)
        self.assertEqual(calculate_items(0, 3, 5), 0)
        self.assertEqual(calculate_items(10, 0, 5), 0)
        self.assertEqual(calculate_items(10, 3, 0), 0)

    def test_calculate_total_bep(self):
        params = WarehouseParams(
            total_area=100,
            rental_cost_per_m2=10,
            useful_area_ratio=0.5,
            mode="Автоматический",
            storage_share=0.25,
            loan_share=0.25,
            vip_share=0.25,
            short_term_share=0.25,
            storage_area_manual=0.0,
            loan_area_manual=0.0,
            vip_area_manual=0.0,
            short_term_area_manual=0.0,
            storage_fee=15,
            shelves_per_m2=3,
            short_term_daily_rate=6,
            vip_extra_fee=10,
            item_evaluation=0.8,
            item_realization_markup=20.0,
            average_item_value=15000,
            loan_interest_rate=0.317,
            realization_share_storage=0.5,
            realization_share_loan=0.5,
            realization_share_vip=0.5,
            realization_share_short_term=0.5,
            storage_items_density=5,
            loan_items_density=1,
            vip_items_density=2,
            short_term_items_density=4,
            salary_expense=240000,
            miscellaneous_expenses=50000,
            depreciation_expense=20000,
            marketing_expenses=30000,
            insurance_expenses=10000,
            taxes=50000,
            utilities_expenses=20000,
            maintenance_expenses=15000,
            one_time_setup_cost=100000,
            one_time_equipment_cost=200000,
            one_time_other_costs=50000,
            one_time_legal_cost=20000,
            one_time_logistics_cost=30000,
            time_horizon=6,
            monthly_rent_growth=0.01,
            default_probability=0.05,
            liquidity_factor=1.0,
            safety_factor=1.2,
            loan_grace_period=0,
            monthly_income_growth=0.0,
            monthly_expenses_growth=0.0,
            forecast_method="Базовый",
            monte_carlo_simulations=100,
            monte_carlo_deviation=0.1,
            monte_carlo_seed=42,
            enable_ml_settings=False
        )
        
        # Рассчитаем и установим площади
        areas = calculate_areas(params)
        for k, v in areas.items():
            setattr(params, k, v)
        
        # Теперь рассчитаем финансовые показатели
        financials = calculate_financials(params, disable_extended=False)
        bep = calculate_total_bep(financials, params)
        expected_bep = financials["total_expenses"] + (params.one_time_expenses / params.time_horizon)
        self.assertAlmostEqual(bep, expected_bep, places=2)
    
    def test_validate_inputs(self):
        from data_model import validate_inputs
        params = WarehouseParams(
            total_area=100,
            rental_cost_per_m2=10,
            useful_area_ratio=0.5,
            mode="Автоматический",
            storage_share=0.25,
            loan_share=0.25,
            vip_share=0.25,
            short_term_share=0.25,
            storage_area_manual=0.0,
            loan_area_manual=0.0,
            vip_area_manual=0.0,
            short_term_area_manual=0.0,
            storage_fee=15,
            shelves_per_m2=3,
            short_term_daily_rate=6,
            vip_extra_fee=10,
            item_evaluation=0.8,
            item_realization_markup=20.0,
            average_item_value=15000,
            loan_interest_rate=0.317,
            realization_share_storage=0.5,
            realization_share_loan=0.5,
            realization_share_vip=0.5,
            realization_share_short_term=0.5,
            storage_items_density=5,
            loan_items_density=1,
            vip_items_density=2,
            short_term_items_density=4,
            salary_expense=240000,
            miscellaneous_expenses=50000,
            depreciation_expense=20000,
            marketing_expenses=30000,
            insurance_expenses=10000,
            taxes=50000,
            utilities_expenses=20000,
            maintenance_expenses=15000,
            one_time_setup_cost=100000,
            one_time_equipment_cost=200000,
            one_time_other_costs=50000,
            one_time_legal_cost=20000,
            one_time_logistics_cost=30000,
            time_horizon=6,
            monthly_rent_growth=0.01,
            default_probability=0.05,
            liquidity_factor=1.0,
            safety_factor=1.2,
            loan_grace_period=0,
            monthly_income_growth=0.0,
            monthly_expenses_growth=0.0,
            forecast_method="Базовый",
            monte_carlo_simulations=100,
            monte_carlo_deviation=0.1,
            monte_carlo_seed=42,
            enable_ml_settings=False
        )
        
        is_valid, error_message = validate_inputs(params)
        self.assertTrue(is_valid)
        
        # Тестирование ошибки при отрицательной площади
        params.total_area = -10
        is_valid, error_message = validate_inputs(params)
        self.assertFalse(is_valid)
        self.assertEqual(error_message, "Общая площадь должна быть больше нуля.")
    
    def test_calculate_irr(self):
        print("Запуск теста calculate_irr")
        cash_flows = [-100000, 30000, 40000, 50000]
        irr = calculate_irr(cash_flows)
        expected_irr = npf.irr(cash_flows) * 100  # Теперь npf определён
        print(f"Расчитанный IRR: {irr}%")
        print(f"Ожидаемый IRR: {expected_irr}%")
        self.assertAlmostEqual(irr, expected_irr, places=2)


if __name__ == '__main__':
    unittest.main()
