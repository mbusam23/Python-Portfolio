# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:17:16 2023

@author: MBUSTINZA
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns

os.getcwd()
os.chdir('C:\\Users\\mbustinza\\Downloads\\Database')

##Download from Kgagle
world_happiness = pd.read_csv(r'world_happiness.csv')

import matplotlib as plt
world_happiness.plt(happiness_score,life_exp)

''''
plt.scatter(world_happiness["happiness_score"],world_happiness["life_exp"])
sns.scatterplot(world_happiness["happiness_score"],world_happiness["life_exp"])
''''

sns.scatterplot(x='life_exp', y='happiness_score', data=world_happiness)
sns.lmplot(x='life_exp', y='happiness_score', data=world_happiness, ci=None)

# Create scatterplot of happiness_score vs life_exp with trendline
sns.lmplot(x='life_exp', y='happiness_score', data=world_happiness, ci=None)

# Correlation between life_exp and happiness_score
cor = world_happiness["life_exp"].corr(world_happiness["happiness_score"])
print(cor)



#Para que sirven las conversiones de estadistica?
    #Sirve para cambiar la sesgadez de las variables 
    #Tb mejora la correlacion ya que hace que no tengan  
    #una relacion lineal, por la naturaleza de las variables contiunas donde su media y mode estan mal comportadas


# LA CORRELACION NO IMPLICA CAUSALIDAD
    #Si x esta correlacionado con Y, no significa que X cause Y

#Ejemplo A
    # Variable A Consumo de cafe
    # Variable Y Cancer de pulmon
    # Variable Oculta TABAQUISMO

    #Resulta que A esta asociado con Y
    #Sin embargo no sifnifica que la cause
    #Se infiere que la cuasa pero es pq la variable A esta asociada tb con 
        #el consumo de tabaco. QUE ESTA SI CAUSA CANCER DE PILMON

#Ejemplo B
    # Variable A  FECHAS FESTIVAS COMO NAVIDAD
    # Variable Y  VENTAS EN RETAIL
    # Variable Oculta PROMOCIONES ESPECIALES
    
    # NO PODRIAMOS SABER el incremoento de las ventas explicadas
        #por las fiesta festivas ya que pueden deverse tb a las promociones especiales


# Scatterplot of gdp_per_cap and life_exp
sns.scatterplot(x='gdp_per_cap', y='life_exp', data=world_happiness)
cor1 = world_happiness['gdp_per_cap'].corr(world_happiness['life_exp'])
print(cor1)


# Scatterplot of happiness_score vs. gdp_per_cap
sns.scatterplot(x="happiness_score",y="gdp_per_cap",data=world_happiness)
cor = world_happiness["happiness_score"].corr(world_happiness["gdp_per_cap"])
print(cor)

# Create log_gdp_per_cap column
world_happiness['log_gdp_per_cap'] = np.log(world_happiness["gdp_per_cap"])

# Scatterplot of happiness_score vs. log_gdp_per_cap
sns.scatterplot(x='log_gdp_per_cap', y='happiness_score', data=world_happiness)
plt.show()

# Calculate correlation
cor = world_happiness["log_gdp_per_cap"].corr(world_happiness["happiness_score"])
print(cor)



# Scatterplot of grams_sugar_per_day and happiness_score
sns.scatterplot(x='grams_sugar_per_day',y='happiness_score', data= world_happiness)
plt.show()

# Correlation between grams_sugar_per_day and happiness_score
cor = world_happiness['grams_sugar_per_day'].corr(world_happiness['happiness_score'])
print(cor)


# EL AUMENTO DEL LOG PIB PER CAPITA ESTÁ ASOCIADO CON UNA PUNTUACIÓN DE FELICIDAD MÁS ALTA


