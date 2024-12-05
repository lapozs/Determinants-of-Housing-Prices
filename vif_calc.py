# -*- coding: utf-8 -*-
"""
@author: Pozsgai Emil Csanád
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Adatok betöltése
file_path = 'Excel_base008-c1.xlsx'
data = pd.read_excel(file_path, header=0)

# Az adatok elkülönítése
metadata = data.iloc[:, :4]  # Az első négy oszlop metaadatok
time_series_data = data.iloc[:, 4:]  # Az 5. oszloptól kezdődnek az idősoros adatok

# Magyarázó változók meghatározása (minden idősoros adatot magyarázó változónak tekintünk)
features = time_series_data.astype(float)

# Változónevek hozzáadása a features indexéhez
features.index = metadata.iloc[:, 1]

# 2. Tisztítás: konstans és nem numerikus oszlopok eltávolítása
features_transposed = features.T  # Transzponálás a VIF számításhoz
features_transposed = features_transposed.loc[:, (features_transposed != features_transposed.iloc[0]).any()]  # Konstans oszlopok eltávolítása
features_transposed = features_transposed.dropna(axis=1)  # Hiányzó adatok eltávolítása

# Ellenőrizzük, hogy maradtak-e használható oszlopok
if features_transposed.empty:
    raise ValueError("Nincsenek használható magyarázó változók a VIF számításhoz.")

# 3. Iteratív VIF számítás
def calculate_vif_with_reduction(df):
    vif_df = pd.DataFrame()
    while True:
        vif_values = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        vif_df = pd.DataFrame({'Változó': df.columns, 'VIF': vif_values})
        max_vif = vif_df['VIF'].max()

        # Ha az összes VIF kisebb, mint egy küszöbérték, kilépünk
        if max_vif < 100:
            break

        # Azonosítjuk a legmagasabb VIF-értékű oszlopot, és eltávolítjuk
        max_vif_var = vif_df.loc[vif_df['VIF'].idxmax(), 'Változó']
        print(f"Eltávolításra kerül a magas VIF-értékű változó: {max_vif_var} (VIF = {max_vif})")
        df = df.drop(columns=[max_vif_var])

    return vif_df, df

# 4. VIF számítás és csökkentés
vif_df, reduced_features = calculate_vif_with_reduction(features_transposed)

# 5. Eredmények mentése Excel fájlba
vif_df['Változó'] = vif_df['Változó'].astype(str)
vif_df.to_excel("vif_summary.xlsx", index=False)

print("\nMultikollinearitás csökkentése után a változók VIF értékei:")
print(vif_df)
print("\nEredmények mentve a 'vif_summary.xlsx' fájlba.")
