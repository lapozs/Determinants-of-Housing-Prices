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
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
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
data = pd.read_excel('Excel_base_FINAL_ALT2-c1.xlsx', header=0) #ALT2 Validáció volt ALT3 - rosszab eredmények
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
adf_results_df_detailed = pd.DataFrame(adf_results_detailed)

# Változók legjobb kombinációjának keresése
min_features = 3  # Minimális változószám a kombinációban
max_features = 7  # Maximális változószám a kombinációban - elhagyható lenne

# Modell beépítése és hiperparaméter optimalizálás közvetlenül a modellekhez
results = []
# Tároló a fitted modellekhez
fitted_models = []

# XGBoost paraméterek (optimalizálva)
param_grid_xgb = {
    'n_estimators': [50, 100],                # Kevesebb iteráció
    'max_depth': [3, 4],                      # Sekélyebb fák
    'learning_rate': [0.05, 0.1],             # Lassabb tanulás
    'colsample_bytree': [0.7, 0.8],           # Kevesebb változó a fákon
    'subsample': [0.7, 0.8],                  # Csökkentett sor mintavétel
}

# Random Forest paraméterek (optimalizálva)
param_grid_rf = {
    'n_estimators': [50, 100],                # Gyorsabb modellépítés
    'max_depth': [5, 10],                     # Egyszerűbb fák
}

# Lasso paraméterek (optimalizálva)
param_grid_lasso = {
    'alpha': [0.01, 0.1],                     # Finomabb skála
    'max_iter': [500, 1000]                    # Limitált iteráció
}

param_grid_ridge = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['auto', 'sag', 'lsqr', 'saga'],
    'tol': [1e-4, 1e-3, 1e-2],
    'max_iter': [1000, 5000, 10000],  # Növelt iterációs határ
}

# SVM paraméterek (optimalizálva)
param_grid_svm = {
    'C': [0.1, 1],                            # Kisebb regularizáció
    'epsilon': [0.05, 0.1],                   # Kisebb eltérések
    'kernel': ['linear', 'rbf'],              # Gyakrabban használt kerneltípusok
}

# Decision Tree (Bayesian Tree) paraméterek (optimalizálva)
param_grid_tree = {
    'max_depth': [3, 5],                      # Egyszerűbb fa mélységek
    'min_samples_split': [2],                  # Többféle osztási szabály
}

# Bayesian Linear Regression paraméterek (optimalizálva)
param_grid_bayesian = {
    'alpha_1': [1e-3, 1e-1],                  # Finomabb priorok
    'alpha_2': [1e-3, 1e-1],                  # Finomabb priorok
}


# Összes kombinációk száma
total_combinations = sum(len(list(combinations(feature_variables, feature_count - 1))) 
                         for feature_count in range(min_features, max_features + 1))
current_combination = 1  # Számláló az aktuális kombinációhoz

# A változók legjobb lag értékeinek tárolása
lag_values = {row['Változó']: row['Lag'] for index, row in adf_results_df.iterrows()}

# Kombinációk létrehozása min_features-től max_features-ig
for feature_count in range(min_features, max_features + 1):
    combos = list(combinations(feature_variables, feature_count - 1))  # Aktuális feature_count kombinációk
    for combo in combos:
        print(f"\n--- Kombináció {current_combination}/{total_combinations} | {round(current_combination/total_combinations * 100, 3)} % ---")
        print(f"Választott kombináció: {combo}")
        
        selected_vars = [target_variable] + list(combo)
        model_data = pd.DataFrame({var: stationalized_data[var] for var in selected_vars})
        
        current_combination += 1  # Növeld a kombinációs számlálót

    # Válasszuk ki a végleges adatokat a stacionarizált adatok közül
    final_data = model_data.dropna()
    model_data.dropna(how='all', inplace=True)  # Teljesen üres sorok törlése
    model_data.interpolate(method='linear', inplace=True)  # Lineáris interpoláció
    model_data.fillna(method='ffill', inplace=True)  # Előző érték alapján töltés
    model_data.fillna(method='bfill', inplace=True)  # Következő érték alapján töltés

    # Ellenőrizzük, hogy nem üresek a tanító és cél adatok
    if final_data.empty or len(final_data.columns) < 2:
        continue

    # Train-test split időbeli sorrend alapján
    split_point = int(len(final_data) * 0.8)
    train_data, test_data = final_data.iloc[:split_point], final_data.iloc[split_point:]
    X_train, y_train = train_data.drop(columns=[target_variable]), train_data[target_variable]
    X_test, y_test = test_data.drop(columns=[target_variable]), test_data[target_variable]
    
    # ARIMAX modell (SARIMAX kiterjesztése) hiperparaméter optimalizálással
    try:
        # Paraméterek definiálása a rácshoz
        param_grid_arimax = {
            'order': [(p, 1, q) for p in range(0, 3) for q in range(0, 3)],  # ARIMA p és q értékek
            'seasonal_order': [(P, 1, Q, 12) for P in range(0, 3) for Q in range(0, 3)]  # Szezonális ARIMA (12 hónap)
        }
        best_arimax_rmse = float('inf')
        best_arimax_r2 = None
        best_arimax_order = None
        best_arimax_seasonal_order = None
    
        for order in param_grid_arimax['order']:
            for seasonal_order in param_grid_arimax['seasonal_order']:
                try:
                    # SARIMAX modell illesztése
                    arimax_model = SARIMAX(
                        y_train,
                        exog=X_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    arimax_fit = arimax_model.fit(disp=False)
    
                    # Előrejelzés
                    arimax_forecast = arimax_fit.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
    
                    # RMSE és R² számítása
                    arimax_rmse = np.sqrt(mean_squared_error(y_test, arimax_forecast))
                    arimax_r2 = r2_score(y_test, arimax_forecast)
    
                    # Ha jobb, mint az eddigi legjobb, mentse az eredményeket
                    if arimax_rmse < best_arimax_rmse:
                        best_arimax_rmse = arimax_rmse
                        best_arimax_r2 = arimax_r2
                        best_arimax_order = order
                        best_arimax_seasonal_order = seasonal_order
    
                except Exception as e:
                    # Hiba esetén lépjen tovább a következő paraméterekre
                    print(f"ARIMAX modell hiba: order={order}, seasonal_order={seasonal_order}. Üzenet: {e}")
                    continue
    except Exception as e:
        print(f"ARIMAX Model Hiba: {e}")

    # 6. VAR Modell
    
    # A VAR modell számára változónkénti lag értékek
    max_lag = max(lag_values[var] for var in selected_vars if var in lag_values)

    try:
        var_model = VAR(train_data)
        var_fitted = var_model.fit(max_lag)  # Változónkénti maximális lag
        var_forecast = var_fitted.forecast(train_data.values[-var_fitted.k_ar:], steps=len(test_data))
        var_rmse = np.sqrt(mean_squared_error(y_test, var_forecast[:, 0]))
        var_r2 = r2_score(y_test, var_forecast[:, 0])
    except Exception as e:
        print(f"VAR modell hiba: {e}")
        var_rmse, var_r2 = None, None
       
    # Bayesiánus döntési fa (DecisionTreeRegressor) hiperparaméter optimalizálással
    bayesian_tree_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    bayesian_tree_grid_search = GridSearchCV(bayesian_tree_model, param_grid_tree, cv=3, n_jobs=-1, verbose=0)
    bayesian_tree_grid_search.fit(X_train, y_train)
    best_bayesian_tree_model = bayesian_tree_grid_search.best_estimator_
    bayesian_tree_forecast = best_bayesian_tree_model.predict(X_test)
    bayesian_tree_rmse = np.sqrt(mean_squared_error(y_test, bayesian_tree_forecast))
    bayesian_tree_r2 = r2_score(y_test, bayesian_tree_forecast)
    
    # XGBoost Modell GPU-val és hiperparaméter optimalizálással
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE) #device = "cuda"
    xgb_grid_search = GridSearchCV(xgb_model, param_grid_xgb, cv=3, n_jobs=-1, verbose=0)
    xgb_grid_search.fit(X_train, y_train)
    best_xgb_model = xgb_grid_search.best_estimator_
    xgb_forecast = best_xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_forecast))
    xgb_r2 = r2_score(y_test, xgb_forecast)
    
    # Random Forest modell hiperparaméter optimalizálással
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid_search = GridSearchCV(rf_model, param_grid_rf, cv=3, n_jobs=-1, verbose=0)
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    rf_forecast = best_rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_forecast))
    rf_r2 = r2_score(y_test, rf_forecast)
       
    # Ridge modell hiperparaméter optimalizálással
    ridge_model = Ridge(random_state=RANDOM_STATE)
    ridge_grid_search = GridSearchCV(ridge_model, param_grid_ridge, cv=3, n_jobs=-1, verbose=0)
    ridge_grid_search.fit(X_train, y_train)
    best_ridge_model = ridge_grid_search.best_estimator_
    ridge_forecast = best_ridge_model.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_forecast))
    ridge_r2 = r2_score(y_test, ridge_forecast)

    # Eredmények hozzáadása minden modellhez az aktuális változókombinációval
    results.append({'Változók': combo, 'Model': 'VAR', 'RMSE': var_rmse, 'R²': var_r2})
    results.append({'Változók': combo, 'Model': 'ARIMAX', 'RMSE': arimax_rmse, 'R²': arimax_r2})
    results.append({'Változók': combo, 'Model': 'Bayesiánus Döntési Fa', 'RMSE': bayesian_tree_rmse, 'R²': bayesian_tree_r2})
    results.append({'Változók': combo, 'Model': 'XGBoost', 'RMSE': xgb_rmse, 'R²': xgb_r2})
    results.append({'Változók': combo, 'Model': 'Random Forest', 'RMSE': rf_rmse, 'R²': rf_r2})
    results.append({'Változók': combo, 'Model': 'Ridge', 'RMSE': ridge_rmse, 'R²': ridge_r2})
    
    
    fitted_models.append(('VAR', var_fitted, var_r2))
    fitted_models.append(('ARIMAX', arimax_fit, arimax_r2, X_train.columns.tolist()))
    fitted_models.append(('Bayesiánus Döntési Fa', best_bayesian_tree_model, bayesian_tree_r2))
    fitted_models.append(('XGBoost', best_xgb_model, xgb_r2))
    fitted_models.append(('Random Forest', best_rf_model, rf_r2))
    fitted_models.append(('Ridge', best_ridge_model, ridge_r2, X_train.columns.tolist()))


results_df = pd.DataFrame(results)  # A results listát DataFrame-be konvertáljuk

# Legjobb kombinációk kiválasztása az R² alapján
best_combinations = results_df.sort_values(by='R²', ascending=False).groupby('Model').head(3)

# Eredmények DataFrame-be rendezése és megjelenítése
results_df = pd.DataFrame(results)

# Az eredményeket egy Excel fájlba is elmenthetjük
adf_results_df.to_excel("ADF_RESULTS_results.xlsx", index=False)
results_df.to_excel("model_FINAL_results.xlsx", index=False)
best_combinations.to_excel("model_FINAL_best_combos.xlsx", index=False)

# Válasszuk ki a legjobban teljesítő modellt
best_model_row = results_df.sort_values(by="R²", ascending=False).iloc[0]
best_model_type = best_model_row['Model']

# Hatás irányának kategorizálása
def categorize_influence(value):
    if value > 0.05:
        return "pozitív"
    elif value < -0.05:
        return "negatív"
    else:
        return "semleges"
    
# Standardizált koefficiensek számítása
def calculate_standardized_coefficients(coefficients, feature_stds):
    """
    Standardizált koefficiensek számítása a tényezők szórásának felhasználásával.
    """
    return coefficients / feature_stds 

def extract_arimax_coefficients(model_instance, feature_names, order, seasonal_order):
    total_params = len(model_instance.params)
    ar_params = order[0]
    ma_params = order[2]
    seasonal_ar_params = seasonal_order[0]
    seasonal_ma_params = seasonal_order[2]
    num_exog = len(feature_names)

    start_idx = 1 + ar_params + ma_params + seasonal_ar_params + seasonal_ma_params
    end_idx = start_idx + num_exog

    # Hosszellenőrzés
    if end_idx > total_params:
        raise ValueError(f"Nem elegendő paraméter: Várható {end_idx}, de csak {total_params} található.")
    
    coefficients = model_instance.params[start_idx:end_idx]
    
    if len(coefficients) != len(feature_names):
        raise ValueError(f"Koefficiensek száma ({len(coefficients)}) nem egyezik a feature_names hosszával ({len(feature_names)}).")
    
    return coefficients

# Szűrjük ki azokat az ARIMAX modelleket, amelyeknek nincs érvényes R² értékük
valid_fitted_arimax = [model for model in fitted_models if model[0] == 'ARIMAX' and model[2] is not None]
top_3_arimax_models = sorted(valid_fitted_arimax, key=lambda x: x[2], reverse=True)[:3]

# ARIMAX modellek elemzése
for i, (model_name, model_instance, r2_score, feature_names) in enumerate(top_3_arimax_models, start=1):
    try:
        # Paraméterek kivonása
        coefficients = extract_arimax_coefficients(model_instance, feature_names, order=model_instance.model.order, seasonal_order=model_instance.model.seasonal_order)

        if coefficients is None:
            raise ValueError("A paraméterek kivonása sikertelen.")

        abs_coefficients = np.abs(coefficients)
        total_importance = abs_coefficients.sum()

        # Fontossági tényezők DataFrame létrehozása
        importance_df = pd.DataFrame({
            'Változó': feature_names,
            'Súly (%)': (abs_coefficients / total_importance) * 100, # Súlyok számítása a standardizált koefficiensek alapján
            'Hatás iránya': [categorize_influence(coef) for coef in coefficients],
            'Koefficiens érték (coef)': coefficients
        }).sort_values(by='Súly (%)', ascending=False)

        # Mentés Excel-be
        importance_df.to_excel(f"Top_ARIMAX_Model_{i}_Importance.xlsx", index=False)
        print(f"Top_ARIMAX_Model_{i}_Importance.xlsx mentve.")

    except Exception as e:
        print(f"Hiba az ARIMAX modell {i} elemzésekor: {e}")

from pingouin import partial_corr
import pandas as pd

# 90% R² feletti modellek szűrése
high_r2_models = [
    model for model in fitted_models
    if model[2] is not None and model[2] > 0.9 and model[0] in ['ARIMAX', 'Ridge']
]

# Eredmények tárolása
partial_corr_results = []

# Részleges korreláció számítása az ARIMAX és Ridge modellekre
for model_name, model_instance, r2_score, feature_names in high_r2_models:
    try:
        # Normalizált adatokat használjuk a részleges korreláció számításához
        for predictor in feature_names:
            covariates = [var for var in feature_names if var != predictor]

            # Részleges korreláció kiszámítása
            partial_corr_result = partial_corr(
                data=normalized_data.T,  # Az adatok transzponálva
                x=predictor,
                y=target_variable,
                covar=covariates
            )

            # Eredmény mentése
            partial_corr_results.append({
                'Model': model_name,
                'Változó': predictor,
                'R²': r2_score,
                'Részleges korreláció': partial_corr_result['r'].values[0],
                'P-érték': partial_corr_result['p-val'].values[0]
            })

    except Exception as e:
        print(f"Hiba a {model_name} modell részleges korrelációjának számításakor: {e}")

# Eredmények táblázatba rendezése
partial_corr_df = pd.DataFrame(partial_corr_results)

# Szűrés külön ARIMAX és Ridge modellekre
arimax_corr_df = partial_corr_df[partial_corr_df['Model'] == 'ARIMAX']
ridge_corr_df = partial_corr_df[partial_corr_df['Model'] == 'Ridge']

# Excel mentés
arimax_corr_df.to_excel("ARIMAX_Partial_Correlations.xlsx", index=False)
#ridge_corr_df.to_excel("Ridge_Partial_Correlations.xlsx", index=False)

