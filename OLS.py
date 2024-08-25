
import pandas as pd

#Importando datos

ruta='F:/DAB Training/4. Python - DAB/Python para Economistas - Lambda/sesion_5/'
data=pd.read_csv(ruta+'Advertising.csv')

'''Analisis Grafico'''
import matplotlib.pyplot as plt

plt.hist(data['Sales'],bins=50)

plt.scatter(data['TV'], data['Sales'], color='green')


#Grafico de dispersion separados
plt.scatter(data['TV'], data['Sales'], color='green', label='graf1')
plt.scatter(data['Radio'], data['Sales'], color='blue',label='graf2')
plt.scatter(data['Newspaper'], data['Sales'], color='red',label='graf3')
plt.legend()
plt.show()

'''Modelamiento'''
import statsmodels.formula.api as sma

lm=sma.ols(formula='Sales ~ TV', data=data).fit()
lm.summary() #Resumen del modeol
print(lm.summary())

#Ver a detalle los parametros, R,etc
lm.params
lm.cov_HC0 #Matriz de covarianzas inicial
lm.rsquared #coef R cuadrado
lm.f_test('TV')


pred_sales = lm.predict(data["TV"])

plt.scatter(data=data, x='TV', y='Sales', color='Green')
plt.plot(data['TV'], pred_sales,color='red')
plt.title('Grafico de disperción y proyección')
plt.show()


'''Modelamiento 2'''
#data.columns.tolist()
lm2 = sma.ols(formula="Sales ~ TV + Radio + Newspaper", data = data).fit()
lm2.summary()



'''Validación de supuesto'''
'''Test econometricos'''

import statsmodels.stats.api as sms

#Bondad de ajuste
#Se busca valores mayores de R2
lm2.rsquared, lm2.rsquared_adj
#Criterio de Akaike o AIC
#Se busca un coeficeinte de AIC menores
lm2.aic
#Criterio de Akaike
lm2.bic

'''MULTICOLINEALIDAD'''
    #TV - RADIO Newspaper
ra=sma.ols(formula='TV ~Radio + Newspaper ', data= data).fit()
VIF= 1/(1-ra.rsquared)
VIF 
    #RADIO - TV
ra2=sma.ols(formula='Radio~ TV + Newspaper ', data= data).fit()
VIF2= 1/(1-ra2.rsquared)
VIF2   
    #Newspaper Radio
ra3=sma.ols(formula='Newspaper~TV + Radio', data= data).fit()
VIF3= 1/(1-ra3.rsquared)
VIF3

'''Test de Linealidad'''
'''Test de Wald '''
test1=sms.linear_reset(lm2)
test1


'''Heteroscedasticidad'''
#H0 HOMOCEDASTICIDAD 
#H1 HETEROCEDASTICAD
test2=sms.het_breuschpagan(lm2.resid,lm2.model.exog)
test2
#1. Multiplicadores de lagrange (LM Statistic)
#2. pvalue del LM
#3. F estadistico
#P value of F test


'''NORMALIDAD'''
'''JARQUE - BERA'''
plt.hist(lm2.resid, bins=30)

test3=sms.jarque_bera(lm2.resid)
test3
#1. JB Coeficiente
#2. pvalue del chi^2
#3. Sesgo
#4. Kurtosis>



'''sklearn'''

from sklearn.linear_model import LinearRegression

x= data[['TV','Radio','Newspaper']]
y= data['Sales']


rl = LinearRegression().fit(x,y)
#Los parametros del modelo
rl.intercept_, rl.coef_
    




























