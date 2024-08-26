# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:01:48 2024

@author: USER
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning: Scikit-Learn
#Datos listos para construir modelos
#Particionamiento del conjunto de datos: train/test
#Iniciamos la clase a modelar
#Ajustamos el modelo: Clase a modelar ´Subconjunto de datos de entrenamiento
#Calculo de algunos indicadores de calidad:
        #R2 (SCORE)
        #Construccion de pronostico + KPIS
        #
        
#Partición de los datos
from sklearn.model_selection import train_test_split
#Seleccionar la naturaleza del modelo a ajustar: Modelo de regresion lineal
from sklearn.linear_model import LinearRegression
#Submodulo metrics
from sklearn import metrics

# Dataset
datos = pd.read_csv("https://raw.githubusercontent.com/robintux/Datasets4StackOverFlowQuestions/master/marketing.csv")
datos.info()
datos.describe().round(2).T
#Valores Faltantes
datos.isnull().sum()

#Definamos las variables a utilizar para construir el modelo
y=datos['sales'];y
x=datos.drop('sales',axis=1);x

#Construyamos un modelo de regresion lineal

#Particionamiento del conjunto de datos: train/test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Instancias dla clase LinearRegression:
    #   Coef1 * youtube + Coef2 * facebook + Coef3 * newspaper + b = sales

#Ajustamos el modelo instanciado usando los datos de entrenamiento
modelo = LinearRegression().fit(x_train,y_train)

#R2
r2_model = LinearRegression().fit(x_train,y_train).score(x_train,y_train);r2_model

#Pronosticos de la variable dependiente
y_pronostico= modelo.predict(x_test);y_pronostico

# Calculo un indicador de calidad : MAPE
MAPE = metrics.mean_absolute_percentage_error(y_test, y_pronostico)*100

print("""
Modelo de Regresion Lineal (marketing)
  R2 : %.3f
  MAPE : %.3f
""" %(r2_model, MAPE))


def stability_rey() :
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

    #Instancias dla clase LinearRegression:
        #   Coef1 * youtube + Coef2 * facebook + Coef3 * newspaper + b = sales

    #Ajustamos el modelo instanciado usando los datos de entrenamiento
    modelo = LinearRegression().fit(x_train,y_train)

    #R2
    r2_model = LinearRegression().fit(x_train,y_train).score(x_train,y_train);r2_model

    #Pronosticos de la variable dependiente
    y_pronostico= modelo.predict(x_test);y_pronostico

    # Calculo un indicador de calidad : MAPE
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_pronostico)*100
    #OUTPUT
    return (r2_model,MAPE)
    

#Ejecutemos nuestra funcion Stability_rey muchas veces (800) para boservar que tan
#estable es la obtencion de indicadores de caldad para estos modelos ajustados

ListaR2=[]
ListaMAPE=[]

for i in range(800):
    r2_model,MAPE = stability_rey()
    ListaR2.append(r2_model)
    ListaMAPE.append(MAPE)

#Mostremos la distribucion de R2 score
plt.figure(figsize=(12,8))
plt.scatter(list(range(len(ListaR2))),ListaR2)
plt.title('Score: R2 [Regresion Lineal]')
plt.ylabel('R2')
plt.show()

#Mostremos la distribucion de MAPE
plt.figure(figsize=(12,8))
plt.scatter(list(range(len(ListaMAPE))),ListaMAPE)
plt.title('Score: MAPE [Regresion Lineal]')
plt.ylabel('MAPE')
plt.show()

#Histograma de resultados de SCORE MAPE
plt.hist(ListaMAPE)

#Forma Inferencial: Test de normalidad
from scipy.stats import shapiro
shapiro(ListaMAPE)
#cOMO EL P VALOR ES MEJOR QUE 0.05 se acpeta la 1
#La listaMAPE no proviene de una distribucion normal










###############################################################################
#Arboles de decisión

from sklearn.tree import DecisionTreeRegressor

#Documentacion
help(DecisionTreeRegressor())

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)
#Ajustamos el modelo usando el subconjunto de entrenamiento
modelo2=DecisionTreeRegressor().fit(xtrain,ytrain)

#Score
r2_modelo2=modelo2.score(xtrain,ytrain)

#Calculo de pronostico
y_pred = modelo2.predict(xtest)

#KPIS
MAPE_model2=metrics.mean_absolute_percentage_error(ytest,y_pred)

print('''
      Modelo: Arbol de decisión [Regresión]
      Score: %.3f
      MAPE : %.3f
      '''%(r2_modelo2,MAPE_model2))    

#aNALICEMOS LA ESTABILIDAD DE LOS MODELOS DE TIPO ARBOL DE DECISIÓN

def stability() :
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)
    #Ajustamos el modelo usando el subconjunto de entrenamiento
    modelo2=DecisionTreeRegressor().fit(xtrain,ytrain)
    
    #Score
    r2_modelo2=modelo2.score(xtrain,ytrain)
    
    #Calculo de pronostico
    y_pred = modelo2.predict(xtest)
    
    #KPIS
    MAPE_model2=metrics.mean_absolute_percentage_error(ytest,y_pred)*100

    #OUTPUT
    return (r2_modelo2,MAPE_model2)    

# Ejecutemos varias veces (1500) nuestra funcion Stability_dt para analizar
# la estabilidad (volatilidad) de los indicadores de calidad

ListaR2_dt = []
ListaMAPE_dt = []

for k in range(3000):
  r2_modelo2, MAPE_model2 = stability()
  ListaR2_dt.append(r2_modelo2)
  ListaMAPE_dt.append(MAPE_model2)

# Visualicemos la distribucion de los score de estos modelos de tipo Arbol de decision
plt.figure(figsize = (15,5))
plt.scatter(x = range(len(ListaR2_dt)) , y = ListaR2_dt)


# Visualicemos la distribucion de los MAPE obtenidos (1500 experimentos)
plt.figure(figsize = (15,5))
plt.scatter(x = range(len(ListaMAPE_dt)), y = ListaMAPE_dt)

# Calculemos la varianza para estos modelos de tipo arbol de decision
np.var(ListaMAPE_dt)



# Analicemos los resultado obtenidos (score, mape) modificando el argumento criterion
# de la clase DecisionTreeRegressor
#

Lista_R2_criterion = []
Lista_MAPE_criterion = []

for arg_criterion in ["squared_error", "friedman_mse", "absolute_error", "poisson"]:
  Model_dt = DecisionTreeRegressor(criterion = arg_criterion)
  Lista_R2 = []
  ListaMAPE = []

  for k in range(3000):
    X_train, X_test, ytrain, ytest = train_test_split(x,y,train_size = 0.80)
    Model_dt.fit(X_train, ytrain)
    R2_model_dt = Model_dt.score(X_train, ytrain)
    y_pred_dt = Model_dt.predict(X_test)
    MAPE_dt = metrics.mean_absolute_percentage_error(ytest, y_pred_dt)*100
    Lista_R2.append(R2_model_dt)
    ListaMAPE.append(MAPE_dt)
  Lista_R2_criterion.append(Lista_R2)
  Lista_MAPE_criterion.append(ListaMAPE)
  
# Analicemos la varianza de cada una de las listas que compone a Lista_MAPE_criterion

var_squared_error = np.var(Lista_MAPE_criterion[0])
var_friedman_mse = np.var(Lista_MAPE_criterion[1])
var_absolute_error = np.var(Lista_MAPE_criterion[2])
var_poisson = np.var(Lista_MAPE_criterion[3])

print("""
Varianzas en funcion dell argumento criterion:
  squared_error : %.3f
  friedman_mse  : %.3f
  absolute_error : %.3f
  poisson : %.3f
""" % (var_squared_error, var_friedman_mse, var_absolute_error, var_poisson))

fig, axes = plt.subplots(2, 2, figsize=(30, 15))
# Mostremos la distribucion de MAPE cuando criterion = squared_error
axes[0,0].scatter(x = range(len(Lista_MAPE_criterion[0])), y = Lista_MAPE_criterion[0])
axes[0,0].set_title("Criterion : squared_error - Varianza " + str(var_squared_error)[:5] )
#Mostremos la distribucion de MAPE cuando criterion = friedman_mse
axes[0,1].scatter(x = range(len(Lista_MAPE_criterion[1])), y = Lista_MAPE_criterion[1])
axes[0,1].set_title("Criterion : friedman_mse - Varianza " + str(var_friedman_mse)[:5] )
# Mostremos la distribucion de MAPE cuando criterion = absolute_error
axes[1,0].scatter(x = range(len(Lista_MAPE_criterion[2])), y = Lista_MAPE_criterion[2])
axes[1,0].set_title("Criterion : absolute_error - Varianza " + str(var_absolute_error)[:5] )
# Mostremos la distribucion de MAPE cuando criterion = poisson
axes[1,1].scatter(x = range(len(Lista_MAPE_criterion[3])), y = Lista_MAPE_criterion[3])
axes[1,1].set_title("Criterion : poisson - Varianza " + str(var_poisson)[:5] )
plt.show()