# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 19:47:06 2024

@author: USER
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn                 import metrics

#Datos
os.chdir('F:/00000db/')
df= pd.read_csv('hotel.csv')
#Cleaning
df.columns
df.info()
#Cuales tienen null
df.isnull().sum().sort_values(ascending=False)
df.isnull().sum().sort_values(ascending=False)*100/df.shape[0]
#Eliminemos las columnas con valores faltantes
df.drop(['company', 'agent','country','children'], axis=1, inplace=True)
df.info()

data1= df.select_dtypes(include=['float64','int64'])

#Definamos las variables independientes y dependientes
y=data1['is_canceled']
x=data1.drop('is_canceled',axis=1)

#Observemos la distribucion de valores en la variable dependiente
y.value_counts()

#Particionando
xtrain,xtest,ytrain,ytest= train_test_split(x,y,train_size=0.88,stratify=y)
#Instanciamos las clases a modelar
logistic = LogisticRegression()
tree = DecisionTreeClassifier()

#Ajustamos los moldelos instanciados usando los subconjuntos de entrenamiento
logistic.fit(xtrain, ytrain)
tree.fit(xtrain, ytrain)

#SCORE
r2_logistic = logistic.score(xtrain, ytrain)
r2_tree     =  tree.score(xtrain, ytrain)

#Pronostico

#Calculemos pronosticos (de la variable dependiente), usando los modelos 
#AJUSTADOS ay el subconjut de test de la variable independiente
y_pronostico_logistic = logistic.predict(xtest)
y_pronostico_tree     =  tree.predict(xtest)

#KPIS
acc_logistic = metrics.accuracy_score(ytest, y_pronostico_logistic)
acc_tree  = metrics.accuracy_score(ytest, y_pronostico_tree)

#Layout

print(''' 
Modelo: Regresion Logistica
    R2:     %.3f    
Accuracy:   %.3f   
Modelo: Decission Tree
    R2:    %.3f     
Accuracy:  %.3f         
      '''%(r2_logistic,acc_logistic,r2_tree,acc_tree))


#Empaquetemos el codigo de la celda anterior en una funcion que permita analizar la estabilidad de los modelos

def train1():
    #Particionando
    xtrain,xtest,ytrain,ytest= train_test_split(x,y,train_size=0.88,stratify=y)
    #Instanciamos las clases a modelar
    logistic = LogisticRegression()
    tree = DecisionTreeClassifier()

    #Ajustamos los moldelos instanciados usando los subconjuntos de entrenamiento
    logistic.fit(xtrain, ytrain)
    tree.fit(xtrain, ytrain)

    #SCORE
    r2_logistic = logistic.score(xtrain, ytrain)
    r2_tree     =  tree.score(xtrain, ytrain)

    #Pronostico

    #Calculemos pronosticos (de la variable dependiente), usando los modelos 
    #AJUSTADOS ay el subconjut de test de la variable independiente
    y_pronostico_logistic = logistic.predict(xtest)
    y_pronostico_tree     =  tree.predict(xtest)

    #KPIS
    acc_logistic = metrics.accuracy_score(ytest, y_pronostico_logistic)
    acc_tree  = metrics.accuracy_score(ytest, y_pronostico_tree)
    return r2_logistic,acc_logistic,r2_tree,acc_tree
#Ejecutando
train1()

#Realicemos 1500 ejecuciones de mi funcion train1( para observar como se comportan
#los indicadores de calidad para diferentes prciones de mi dataset)
lista_r2_logistic= []
lista_r2_tree= []
lista_acc_logistic= []
lista_acc_tree= []

for i in range(100):
    (r2_logistic,acc_logistic,r2_tree,acc_tree)=train1()
    lista_r2_logistic.append(r2_logistic)
    lista_r2_tree.append(r2_tree)
    lista_acc_logistic.append(acc_logistic)
    lista_acc_tree.append(acc_tree)
    
var_log_r2 = np.var(lista_r2_logistic)
var_log_acc = np.var(lista_acc_logistic)
var_tree_r2 = np.var(lista_r2_tree)
var_tree_acc = np.var(lista_acc_tree)


fig, axes = plt.subplots(2, 2, figsize=(20, 15))
#Modelo de Regresion Logistica  - R2
axes[0,0].scatter(x = range(len(lista_r2_logistic)), y = lista_r2_logistic)
axes[0,0].set_title("Modelo de Regresion Logistica  - R2" + str(var_log_r2) )
#Modelo de Regresion Logistica - Accuracy
axes[0,1].scatter(x = range(len(lista_acc_logistic)), y = lista_acc_logistic)
axes[0,1].set_title("Modelo de Regresion Logistica - Accuracy " + str(var_log_acc) )
# Modelo Arbol de decision  - R2
axes[1,0].scatter(x = range(len(lista_r2_tree)), y = lista_r2_tree)
axes[1,0].set_title("Modelo Arbol de decision  - R2" + str(var_tree_r2) )
# Modelo Arbol de decision - Accurac
axes[1,1].scatter(x = range(len(lista_acc_tree)), y = lista_acc_tree)
axes[1,1].set_title("Modelo Arbol de decision - Accuracy " + str(var_tree_acc) )
plt.show()


#Prueba
arg_criterion=['gini','entropy','log_loss']

ListaR2_c_L = []
ListaAcc_c_L = []
ListaR2_c_DT = []
ListaAcc_c_DT = []
for c in arg_criterion:
    ListaR2_crit = []
    ListaAcc_crit = []
    ListaR2_crit_dt = []
    ListaAcc_crit_dt = []
    for exp in range(10):
        r2_logistic,acc_logistic,r2_tree,acc_tree = train_test_split(x,y,train_size = 0.80)
        ListaR2_crit.append(r2_logistic)
        ListaAcc_crit.append(acc_logistic)
        ListaR2_crit_dt.append(r2_tree)
        ListaAcc_crit_dt.append(acc_tree)  
        ListaR2_c_L.append(ListaR2_crit)
        ListaAcc_c_L.append(ListaAcc_crit)
        ListaR2_c_DT.append(ListaR2_crit_dt)
        ListaAcc_c_DT.append(ListaAcc_crit_dt)
 
#Una vez terminando el analisis de la celda anterior
#Y siendo conciente de los valores que pueden tomar estos indicadores de caldiad
#Conclir de que los resktados obtenido no son suficientemente satisfactorios

#Luego: Realiar un barrido de hiperparametros
from sklearn.model_selection import GridSearchCV

#Voy a considerar al los siguientes parametors
Dict_HP_DT ={
   'criterion' :  ['gini','entropy','log_loss'],
    'splitter' : ['best','randon'],
  'max_depth' : [5,10,15,25,30,45,55,75,99,120],
  'ccp_alpha' : np.linspace(0.0001,20,200)
}


#Instanciamos el modelo base
MOD_DT = DecisionTreeClassifier()


#Configuramos el barrido de HP
MOD_DT_GS = GridSearchCV(estimator=MOD_DT, param_grid=Dict_HP_DT,cv=4,scoring='accuracy',verbose=3,n_jobs=1)
#Procedimiento pesado: Ajuste DELGS
MOD_DT_GS.fit(xtrain,ytrain)



