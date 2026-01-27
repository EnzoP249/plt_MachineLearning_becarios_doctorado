# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:51:37 2026

@author: Enzo
"""

###############################################################################
# PROYECTO PARA IMPLEMENTAR UN MODELO DE MACHINE LEARNING A UN BASE DE DATOS
# SOBRE BECARIOS DE PROGRAMAS DE DOCTORADO
###############################################################################

###############################################################################
# OBJETIVO DEL PROYECTO: DISEÑAR E IMPLEMENTAR UN ALGORITMO DE MACHINE LEARNING
# PARA LA ESTIMACIÓN DE REGULARIDADES EMPIRICAS EN LA BRECHA DE PRODUCTIVIDAD
# CIENTÍFICA DE BECARIOS DE PROGRAMAS DE DOCTORADO
###############################################################################

# Se importan las librerías que serán usadas
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import gaussian_kde


# Se construye una función que aborde la conversión de int en str para un procesamiento óptimizado
def int_to_str(value):
    return str(value)


# Especifica el diccionario de conversión en el parámetro converters
converters = {"codigo_scopus": int_to_str}

# Se agrega un archivo, en formato excel, que contiene la información sobre becarios de programas de doctorado
# El archivo se convierte en un dataframe para su explotación

becario = pd.read_excel("bd_becarios_doctorado.xlsx", sheet_name="Sheet1", header=0, converters=converters)
becario.shape
becario.columns

# Se identifican los becarios únicos
becario["Entidad Ejecutora/Subvencionado"].nunique()

# se realiza una distribución por género
becario.GÉNERO.value_counts()
becario.GÉNERO.value_counts(normalize=True).round(4)*100

# Insight = Se puede sostener que la distribución de becarios por género presenta una varianza relativamente baja

# Ahora bien, para fines del modelo predictivo, se construye un nuevo dataframe considerando solo los becarios que tengan codigo scopus
model_becario = becario.dropna(subset=["codigo_scopus"])

# Se construye la variable target
model_becario["y_pre"] = model_becario["pub_antes_beca_4"] / 4
model_becario["y_fund"] = (
    model_becario["pub_durante_beca"] + model_becario["pub_dur_rezago"] / (model_becario["periodo_beca"] + model_becario["periodo_rezago"])
    ) 


model_becario["target_brecha"] = model_becario["y_fund"] - model_becario["y_pre"]

# Analizo la distribución de la variable area
model_becario.area.value_counts(normalize=True).round(4)*100

# Se renombre la variable que almacena información sobre el pais en donde se realiza la subvención
model_becario.rename(columns=({"PAÍS EN DONDE SE REALIZA LA SUBVENCIÓN":"pais_subvencion"}), inplace=True)
model_becario.pais_subvencion.value_counts(normalize=False)


# Se construye el dataframe que alimentara a mi modelo de machine learning
model_becario.columns
model_becario = model_becario[["target_brecha", "area", "pais_subvencion"]]

# Antes de implementar el modelo de machine learning se realiza un análisis de las variables consideradas
# Se realiza un histograma de frecuencias de la variable brecha

data = model_becario["target_brecha"].dropna()

# Histograma (normalizado)
plt.hist(data, bins=20, density=True, alpha=0.6)

# KDE
kde = gaussian_kde(data)

x_vals = np.linspace(data.min(), data.max(), 1000)
plt.plot(x_vals, kde(x_vals))

# Etiquetas
plt.xlabel("Brecha de productividad científica")
plt.ylabel("Densidad")

# Mostrar
plt.show()

# Se analiza la distribución de las variables categóricas que servirán como predictoras
# Se analiza la variable área de conocimiento
model_becario.area.value_counts()
df_area = model_becario.area.value_counts()
df_area = df_area.to_frame()
df_area.reset_index(inplace=True)
df_area.rename(columns=({"area":"Área", "count":"Número_becarios"}), inplace=True)

# Configurar el tamaño de la figura
plt.figure(figsize=(14, 8))

# Crear una paleta de colores en tonos de azul inverso
palette = sns.color_palette("viridis", len(df_area["Número_becarios"]))

# Crear las barras del gráfico
bars = plt.bar(df_area["Área"], df_area["Número_becarios"], color=palette)

# Etiquetas de los ejes
plt.xlabel("Área", fontsize=18)
plt.ylabel("Número de becarios", fontsize=14)

# Agregar etiquetas de valores a cada barra con mayor tamaño de fuente
for bar in bars:
    plt.annotate(f'{bar.get_height():.0f}', 
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                 xytext=(0, 5),  # Desplazamiento de la etiqueta
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=18)

# Ajustar los márgenes para que las etiquetas no se corten
plt.xticks(rotation=45, fontsize=14)  # Rotar etiquetas del eje X si es necesario
plt.yticks(fontsize=14)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Ahora, se analiza la variable país en donde realizó sus estudio
model_becario.pais_subvencion.value_counts()
df_pais = model_becario.pais_subvencion.value_counts()
df_pais = df_pais.to_frame()
df_pais.reset_index(inplace=True)
df_pais.rename(columns=({"count":"Número_becarios"}), inplace=True)
# Se considera el top 5 de paises
df_pais = df_pais[df_pais["Número_becarios"]>=9]

# Configurar el tamaño de la figura
plt.figure(figsize=(14, 8))

# Crear una paleta de colores en tonos de azul inverso
palette = sns.color_palette("magma", len(df_pais["Número_becarios"]))

# Crear las barras del gráfico
bars = plt.bar(df_pais["pais_subvencion"], df_pais["Número_becarios"], color=palette)

# Etiquetas de los ejes
plt.xlabel("País", fontsize=18)
plt.ylabel("Número de becarios", fontsize=14)

# Agregar etiquetas de valores a cada barra con mayor tamaño de fuente
for bar in bars:
    plt.annotate(f'{bar.get_height():.0f}', 
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                 xytext=(0, 5),  # Desplazamiento de la etiqueta
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=18)

# Ajustar los márgenes para que las etiquetas no se corten
plt.xticks(rotation=45, fontsize=14)  # Rotar etiquetas del eje X si es necesario
plt.yticks(fontsize=14)
plt.tight_layout()


###############################################################################
# Ahora bien, se procede a implementar el modelo de machine learning
###############################################################################

###############################################################################
# ESTRATEGIA COMPARTIDA POR CLAUDE DE ANTROPHIC - Esquema híbrido
###############################################################################

# =========================================================
# 0) Semillas globales (Python / NumPy / hash)
# =========================================================
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# =========================
# 1) Configuración de datos
# =========================
target = "target_brecha"
cat_cols = ["area", "pais_subvencion"]

X = model_becario.drop(columns=[target]).copy()
y = model_becario[target].copy()

# Asegura orden estable de filas (evita variación accidental)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# CatBoost requiere categóricas como str
for c in cat_cols:
    X[c] = X[c].astype(str)

# =========================
# 2) Repeated K-Fold (n=100)
# =========================

# Se define el diseño experimental
k = 5
n_repeats = 5

# Semillas fijas por repetición → mismas particiones en cada corrida
seeds = [GLOBAL_SEED + i for i in range(n_repeats)]

# Guardamos métricas por fold y OOF por repetición
all_rows = []
oof_preds_by_rep = np.zeros((n_repeats, len(X)), dtype=float) # Se incializa una estructura

for rep_idx, seed in enumerate(seeds): # bucle asociado con las repeticiones
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed) # Se genera la partición del dataset

    # Recorremos folds
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):  # bucle asociado con cada fold o pliegue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        test_pool  = Pool(X_te, y_te, cat_features=cat_cols)

        # =========================
        # Modelo (determinista en CPU)
        # =========================
        model = CatBoostRegressor( # En total, y siguiendo la lógica de mi ejercicio, se construiran 25 modelos de CatBoost
            loss_function="RMSE", 
            eval_metric="RMSE", 
            
            # Se establece la capacidad del modelo

            iterations=1500, # Número de árboles
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,

            # Semilla fija por repetición
            random_seed=seed,

            # Bootstrap bayesiano (puede ser determinista si controlas seed + 1 thread)
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,

            # Early stopping por fold
            od_type="Iter",
            od_wait=150,

            # CLAVE para replicabilidad bit-a-bit
            thread_count=1,

            # Para evitar ruido en salida
            verbose=False,

            # Asegura CPU
            task_type="CPU"
        )

        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        preds = model.predict(test_pool)

        # OOF de ESTA repetición (cada obs se predice 1 vez por repetición)
        oof_preds_by_rep[rep_idx, test_idx] = preds

        # Métricas por fold (fuera de entrenamiento)
        mae  = mean_absolute_error(y_te, preds)
        rmse = mean_squared_error(y_te, preds, squared=False)
        r2   = r2_score(y_te, preds)

        all_rows.append({
            "repeat": rep_idx + 1,
            "seed": seed,
            "fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "best_iter": model.get_best_iteration()
        })

metrics_df = pd.DataFrame(all_rows)

print("\nResumen global (todos los folds y repeticiones):")
print(metrics_df[["MAE", "RMSE", "R2", "best_iter"]].agg(["mean", "std", "min", "max"]))


# =========================
# 3) OOF Global (promedio)
# =========================
oof_pred = oof_preds_by_rep.mean(axis=0)

oof_mae  = mean_absolute_error(y, oof_pred)
oof_rmse = mean_squared_error(y, oof_pred, squared=False)
oof_r2   = r2_score(y, oof_pred)

print("\nOOF Global (promedio sobre repeticiones):")
print(f"OOF MAE : {oof_mae:.4f}")
print(f"OOF RMSE: {oof_rmse:.4f}")
print(f"OOF R^2 : {oof_r2:.4f}")

metrics_df.to_csv("catboost_repeated_kfold_bayesian_metrics_n100.csv", index=False)


# =========================================================
# 4) Modelo final (100% datos) - síntesis interpretativa
# =========================================================
full_pool = Pool(X, y, cat_features=cat_cols)

# Heurística robusta: mediana del best_iter
final_iters = int(metrics_df["best_iter"].median())
final_iters = max(final_iters, 50)

final_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=final_iters,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5.0,
    random_seed=GLOBAL_SEED,

    bootstrap_type="Bayesian",
    bagging_temperature=1.0,

    # CLAVE replicabilidad
    thread_count=1,
    task_type="CPU",

    verbose=200
)

final_model.fit(full_pool)
final_model.save_model("catboost_brecha_final_bayesian_n100.cbm")


# Importancia de variables (modelo final)
fi = final_model.get_feature_importance(full_pool)

imp = (
    pd.DataFrame({
        "feature": X.columns,
        "importance": fi
    })
    .sort_values("importance", ascending=False)
)

print("\nImportancia de variables (modelo final):")
print(imp)


# =========================================================
# 5) Regularidades empíricas (OOF) por país y por área
# =========================================================
df_oof = X.copy()
df_oof["target_real"] = y
df_oof["oof_pred"] = oof_pred

oof_by_country = (
    df_oof.groupby("pais_subvencion")["oof_pred"]
          .agg(["mean", "count"])
          .sort_values("mean", ascending=False)
)

oof_by_area = (
    df_oof.groupby("area")["oof_pred"]
          .agg(["mean", "count"])
          .sort_values("mean", ascending=False)
)

print("\nOOF promedio por país:")
print(oof_by_country)

print("\nOOF promedio por área:")
print(oof_by_area)


# ==========================================================
# 6) Análisis combinatorio (país × área) + heatmap
# ==========================================================
tabla_interaccion = (
    df_oof
    .groupby(["pais_subvencion", "area"])
    .agg(
        mean_oof=("oof_pred", "mean"),
        count=("oof_pred", "size")
    )
    .reset_index()
    .sort_values("mean_oof", ascending=False)
)

tabla_interaccion_filtrada = tabla_interaccion[tabla_interaccion["count"] >= 3]

heatmap_data = tabla_interaccion_filtrada.pivot(
    index="area",
    columns="pais_subvencion",
    values="mean_oof"
)

plt.figure(figsize=(10, 5))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="inferno"
)
plt.xlabel("País de la beca")
plt.ylabel("Área de conocimiento")
plt.tight_layout()
plt.show()







