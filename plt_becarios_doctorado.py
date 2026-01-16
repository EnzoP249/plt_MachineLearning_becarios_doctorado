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
# PARA ESTABLECER UN MODELO PREDICTIVO
###############################################################################

# Se importan las librerías que serán usadas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

model_becario.to_excel("jajaja.xlsx")







































