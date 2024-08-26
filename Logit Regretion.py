# -*- coding: utf-8 -*-
"""clase 5"""

"""LOGIT"""

import os
import pandas as pd
os.chdir('Path')
binary=pd.read_csv('binary.csv')

'''Estimacion'''
import statsmodels.formula.api as smf

def crearDummies(df, var_name):
    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = pd.concat([df, dummy ], axis = 1)
    return df
df=crearDummies(binary, "rank")

#Estimacion
nlml= smf.logit('admit~gre+ gpa+rank_2+rank_3+rank_4', data=df).fit(method='newton')

print(nlml.summary())

nlml2=smf.logit('admit~gre+ gpa+rank_2+rank_3+rank_4', data=df).fit(method='bfgs', maxiter=50)
print(nlml2.summary())


nlml2.normalized_cov_params

'''De donde sale el error estandar'''
nlml2.normalized_cov_params #la raiz cuadrada  de la diagonal son 
#los errores estandares
'''De donde sale el estadistico z'''
import math
z= nlml2.params[0]/math.sqrt(nlml2.normalized_cov_params['Intercept']['Intercept'])
z
z2= nlml2.params[1]/math.sqrt(nlml2.normalized_cov_params['rank_2[T.True]']['rank_2[T.True]'])
z2
'''De donde sale el P valor'''
from scipy import stats
import numpy as np
# 2*( 1- Normalized(ABS(z)) )
np.round(2*(1-stats.norm.cdf(np.abs(z))),3)

'''R de Mc Fadden'''
rmf= 1-(nlml2.llf/nlml2.llnull) ; rmf

'''COEFICENTE DE PREDICCION'''
'''Matriz de confusion'''
score= nlml2.predict(linear=True); score
logit_pred_manual=np.exp(score)/(1+np.exp(score))
logit_pred_manual

probs=nlml2.predict()
