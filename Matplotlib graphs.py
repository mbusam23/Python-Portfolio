# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:00:19 2024

@author: USER
"""

'''Visualización de datos'''

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
os.chdir('path')
data= pd.read_csv('notas.csv', encoding='latin')
data['nota'].fillna(0,inplace=True);data

extrap=pd.DataFrame({'extra':[1,2,3,4,5,6]})

data2=data
data2['nota']=np.add(data['nota'],extrap['extra'])

'''Graficas tipo 1'''
#Grafico de barra
plt.figure(figsize=(8,5))
plt.style.use('_mpl-gallery')
plt.plot(data2['nombres'],data2['nota'])
plt.bar(data['nombres'],data['nota'],
        linewidth=1, edgecolor='black')
plt.xlabel('Nombres', loc='center')
plt.xticks(rotation=90)
plt.legend()
plt.show()

#Grafico de lineas
#plt.plot(data['nombres'],data['nota'])

'''_________________________________________________________________________'''
'''_________________________GRAFICO DE LINEAS_______________________________'''

'''Formas de graficar'''
peru=pd.DataFrame({'consumo':[11,12.6,9,14,14],
                   'inversion': [5.4,5.3,4,5.7,3.7],
                   'pib'    :[13.1,15.3,11.3,16.5,17.8],
    });peru

'''Graficas tipo 2'''

fig,ax = plt.subplots()
#ax.figure(figsize=(8,5))
ax.plot(peru['pib'],peru['consumo'], label='Consumo')
ax.plot(peru['pib'],peru['inversion'], label='Inversion')

ax.set_xlabel('pib')
ax.set_ylabel('Consumo & Inversion')
ax.set_title('Grafico1')
ax.legend()

ax.grid()
plt.show()



'''Grafico 3'''
# Define el tamaño de la figura y crea los subplots
fig,axs=plt.subplots(nrows=2,ncols=2, figsize=(12,8))
#Forma de hacer un subplots 1
axs[0][0].plot(peru['pib'],peru['inversion'])
#Inserta un texto en el grafico
axs[0][0].text(14,4.25,'Texto1', bbox=dict(facecolor='red',alpha=0.5))
#Forma de titulo
fig.axes[2].set_title('Grafico')
#Modificar los numeros de los ejes
axs[0][0].set_xlim(11,18)
axs[0][0].set_ylim(3.9,6)
#Forma de hacer un subplot forma 2
fig.axes[3].plot(peru['pib'],peru['consumo'])
## Ajusta el espacio entre subgráficos si es necesario
plt.tight_layout()


'''_________________________________________________________________________'''
'''______________________________SCATTER PLOT_______________________________'''


x_scatter=np.random.randint(0,100,30)
Y_scatter=np.random.randint(0,100,30)

plt.figure(figsize=(10,5))
plt.scatter(x_scatter, Y_scatter,label='datos' )
plt.tight_layout()

#Calcula la línea de tendencia (regresión lineal)
pf=np.polyfit(x_scatter,Y_scatter,1) # Ajuste lineal (grado 1)
ft=np.poly1d(pf)                    # Crea una función a partir de los coeficientes

plt.plot(x_scatter, ft(x_scatter), color='red', label='Línea de tendencia')
plt.legend()


'''_________________________________________________________________________'''
'''______________________________BAR PLOT___________________________________'''
plt.figure(figsize=(10,5))
plt.bar(['2020','2021','2022'],[10,15,6])
plt.title('Ventas por año del producto A')

'''_________________________________________________________________________'''
'''______________________________HIST PLOT___________________________________'''

hist_Data=np.random.normal(0,1,1000)
plt.figure(figsize=(10,5))
plt.xlim(-4,4)
plt.ylim(0,180)
plt.hist(hist_Data)

'''_________________________________________________________________________'''
'''____________________________grafico de pastel____________________________'''
colores = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']  # Colores personalizados

plt.figure(figsize=(10,5))
plt.pie([30,30,10,40,60],labels=['Perros','Gatos','Ciervos','Caballos','Canguros'],autopct='%1.1f%%',colors=colores)
plt.title('Distribucion de animales exoticos')













