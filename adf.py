# -*- coding: utf-8 -*-
"""
@author: Pozsgai Emil Csanád
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import norm
from sklearn.utils.validation import check_is_fitted
import warnings
warnings.filterwarnings("ignore")

# Globális random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. Adatok betöltése
data = pd.read_excel('Excel_base-c1.xlsx', header=0) #ALT2 Validáció volt ALT3 - rosszab eredmények
metadata = data.iloc[:, :4]  # Az első négy oszlop a változók leíró adatai
time_series_data = data.iloc[:, 4:]  # Idősor adatok az ötödik oszloptól
time_series_data.index = metadata.iloc[:, 1]  # Változók neveinek beállítása indexként

# Up-sampling az éves adatokból havi adatokra, ideiglenes transzponálással
upsampled_data = pd.DataFrame()

for var_name in time_series_data.index:
    series = time_series_data.loc[var_name]
    if not series.isnull().all():
        # Eredeti éves index létrehozása
        original_index = pd.date_range(start='2010', periods=len(series), freq='A')  # Éves időbélyegek
        # Cél index havi felbontáshoz
        target_index = pd.date_range(start='2010-01-01', end='2022-12-31', freq='M')
        
        # Adatok áthelyezése az új indexbe
        interpolated = pd.Series(index=target_index, dtype='float64')
        interpolated.loc[original_index] = series.values
        
        # Eloszlás illesztése az eredeti adatokhoz
        valid_values = series.dropna()
        dist_mean, dist_std = norm.fit(valid_values)  # Normális eloszlás paramétereinek becslése
        
        # Hiányzó értékek kitöltése az illesztett eloszláson alapulva
        missing_index = interpolated[interpolated.isnull()].index
        simulated_values = norm.rvs(loc=dist_mean, scale=dist_std, size=len(missing_index), random_state=RANDOM_STATE)
        interpolated.loc[missing_index] = simulated_values
        
        # Hozzáadás az upsampled_data DataFrame-hez
        upsampled_data[var_name] = interpolated

# Visszatranszponálás, hogy a változók soronként legyenek
upsampled_data = upsampled_data.T

# Normalizáció Min-Max skálázással az upsampled adatokra
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(
    scaler.fit_transform(upsampled_data),
    index=upsampled_data.index,
    columns=upsampled_data.columns
)

# A célváltozó és a magyarázó változók frissítése
target_variable = 'Lakások átlagára, ezer Ft/m2'
feature_variables = [var for var in normalized_data.index if var != target_variable]

# Stacionarizált adatok tárolása
stationalized_data = {}
adf_results = []
adf_results_detailed = []  # Az összes átalakítási lépés tárolása
non_stationary_variables = []  # Nem stacionárissá vált változók tárolása

# 3. ADF teszt és több átalakítás a legjobb stacionaritás elérése érdekében
diff_num_user = 2  # Első körben maximális differenciálási lépések száma
P_VALUE_USERDEF = 0.10

for var_name in [target_variable] + feature_variables:
    series = normalized_data.loc[var_name]
    diff_count = 0
    best_p_value = 1  # Kezdetben a legrosszabb p-érték
    best_series = None
    best_transformation = "Nincs"  # Kezdeti érték, ha nincs transzformáció
    best_adf_stat = None  # A legjobb ADF statisztika érték
    best_lag = None  # A legjobb lag érték
    stationary = False  # Stacionaritás állapotának nyomon követése

    # Alapértelmezett ADF teszt és differenciálás
    while diff_count <= diff_num_user:
        adf_test = adfuller(series, autolag='AIC')
        p_value = adf_test[1]
        lag_value = adf_test[2]
        adf_stat = adf_test[0]

        adf_results_detailed.append({
            'Változó': var_name,
            'Átalakítás': f'{diff_count}. differenciálás',
            'ADF Statisztika': round(adf_stat, 2),
            'Kritikus érték 1%': round(adf_test[4]['1%'], 2),
            'Kritikus érték 5%': round(adf_test[4]['5%'], 2),
            'Kritikus érték 10%': round(adf_test[4]['10%'], 2),
            'p-érték': round(p_value, 4),
            'Lag': lag_value,
            'Stacionáris-e': "Igen" if p_value < P_VALUE_USERDEF else "Nem"
        })

        if p_value < P_VALUE_USERDEF:
            stationary = True
            best_p_value = p_value
            best_series = series
            best_transformation = f'{diff_count}. differenciálás'
            best_adf_stat = adf_stat
            best_lag = lag_value
            break

        if p_value < best_p_value:
            best_p_value = p_value
            best_series = series
            best_transformation = f'{diff_count}. differenciálás'
            best_adf_stat = adf_stat
            best_lag = lag_value

        # Differenciálás
        series = series.diff().dropna()
        diff_count += 1

    # Egyéb transzformációk, ha az első differenciálások nem segítettek
    if not stationary:
        transformations = [
            ('Négyzetgyök', lambda x: np.sqrt(x)),
            ('Logaritmus', lambda x: np.log(x + 1)),
            ('Mozgóátlag', lambda x: x.rolling(window=3).mean().dropna())
        ]
        for name, func in transformations:
            try:
                transformed_series = func(normalized_data.loc[var_name])
                adf_test = adfuller(transformed_series, autolag='AIC')
                p_value = adf_test[1]
                lag_value = adf_test[2]
                adf_stat = adf_test[0]

                adf_results_detailed.append({
                    'Változó': var_name,
                    'Átalakítás': name,
                    'ADF Statisztika': round(adf_stat, 2),
                    'p-érték': round(p_value, 4),
                    'Lag': lag_value,
                    'Stacionáris-e': "Igen" if p_value < P_VALUE_USERDEF else "Nem"
                })

                if p_value < P_VALUE_USERDEF:
                    stationary = True
                    best_p_value = p_value
                    best_series = transformed_series
                    best_transformation = name
                    best_adf_stat = adf_stat
                    best_lag = lag_value
                    break

                if p_value < best_p_value:
                    best_p_value = p_value
                    best_series = transformed_series
                    best_transformation = name
                    best_adf_stat = adf_stat
                    best_lag = lag_value

            except Exception as e:
                print(f"{name} transzformáció hiba: {e}")

    # Ha az egyéb transzformációk sem segítettek, harmadik differenciálás
    if not stationary and diff_count == diff_num_user + 1:
        series = series.diff().dropna()
        adf_test = adfuller(series, autolag='AIC')
        p_value = adf_test[1]
        lag_value = adf_test[2]
        adf_stat = adf_test[0]

        adf_results_detailed.append({
            'Változó': var_name,
            'Átalakítás': f'{diff_num_user + 1}. differenciálás',
            'ADF Statisztika': round(adf_stat, 2),
            'p-érték': round(p_value, 4),
            'Lag': lag_value,
            'Stacionáris-e': "Igen" if p_value < P_VALUE_USERDEF else "Nem"
        })

        if p_value < P_VALUE_USERDEF:
            stationary = True
            best_p_value = p_value
            best_series = series
            best_transformation = f'{diff_num_user + 1}. differenciálás'
            best_adf_stat = adf_stat
            best_lag = lag_value

    # Ha egyik módszer sem segített, használjuk a legjobb p-értékű sorozatot
    if best_series is not None:
        stationalized_data[var_name] = best_series
        adf_results.append({
            'Változó': var_name,
            'ADF Statisztika': round(best_adf_stat, 2) if best_adf_stat is not None else None,
            'p-érték': round(best_p_value, 4),
            'Lag': best_lag,
            'Stacionáris-e': "Igen" if best_p_value < P_VALUE_USERDEF else "Nem teljesen, de legjobb transzformáció",
            'Legjobb transzformáció': best_transformation
        })
    else:
        non_stationary_variables.append(var_name)
        adf_results.append({
            'Változó': var_name,
            'ADF Statisztika': None,
            'p-érték': None,
            'Lag': None,
            'Stacionáris-e': "Nem",
            'Legjobb transzformáció': "Nincs"
        })

# Eredmények adatkeretbe rendezése
adf_results_df = pd.DataFrame(adf_results)
adf_results_df.to_excel("ADF_RESULTS_results_ORIGINAL.xlsx", index=False)
adf_results_df_detailed = pd.DataFrame(adf_results_detailed)
adf_results_df.to_excel("ADF_RESULTS_DETAILED_results_ORIGINAL.xlsx", index=False)
