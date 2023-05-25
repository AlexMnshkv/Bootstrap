#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

plt.style.use('ggplot')

from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import re
from io import BytesIO
import requests
import json
from urllib.parse import urlencode
import gspread
import pingouin as pg
from pingouin import multivariate_normality
import math as math
import scipy as scipy
import scipy.stats as stats
from df2gspread import df2gspread as d2g
from oauth2client.service_account import ServiceAccountCredentials 
df=pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-a-/Stat_less9/hw_bootstrap.csv', sep=';')


# In[2]:


df


# In[3]:


df['value']=df.value.str.replace(',','.')
df['value']=df.value.astype(float)
df


# In[4]:


df_control=df.query('experimentVariant=="Control"')
df_control


# In[5]:


sns.distplot(df_control.value, hist=True, kde=False,color = 'blue',
             hist_kws={'edgecolor':'black'})


# In[6]:


df_test=df.query('experimentVariant=="Treatment"')
df_test


# In[7]:


sns.distplot(df_test.value, hist=True, kde=False,color = 'blue',
             hist_kws={'edgecolor':'black'})


# In[8]:


logo_test=np.log(df_test.value)


# In[9]:


logo_control=np.log(df_control.value)


# In[10]:


sns.distplot(logo_test, hist=True, kde=False,color = 'blue',
             hist_kws={'edgecolor':'black'})


# In[11]:


df_control.value


# In[12]:


import statsmodels.api as sm
import matplotlib.pyplot as plt

fig = sm.qqplot(df_control.value)
plt.show()


# In[13]:


import statsmodels.api as sm
import matplotlib.pyplot as plt

fig = sm.qqplot(df_test.value)
plt.show()


# In[14]:


from scipy import stats
stats.shapiro(df_control.value)


# In[15]:


a=df_test.query('value < 500')
a


# In[16]:


fig = sm.qqplot(a.value)
plt.show()


# In[17]:


df_control.value.describe()


# In[18]:


df_test.value


# In[19]:


df_test.value.describe()


# In[20]:


#t-тест
from scipy import stats
stats.ttest_ind(df_control.value, df_test.value)

# p-value меньше 0.05, поэтому мы отклоняем нулевую гипотезу и делаем вывод, что средние в группах значимо различаются.
# Результаты статзначимы.


# In[21]:


#t-test для выборки без выбросов
from scipy import stats
stats.ttest_ind(df_control.value, a.value)


# In[22]:


#U-тест

stats.mannwhitneyu(df_control.value, df_test.value)

# тест Манна — Уитни показал, что различия не статзначимы


# In[23]:


#t-test log

stats.ttest_ind(logo_control, logo_test)


# In[24]:


#U-тест log
import scipy.stats as stats
stats.mannwhitneyu(logo_control, logo_test, alternative='less')


# In[25]:


#Bootstrap
def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1), 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            len(data_column_1), 
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1)-statistic(samples_2)) # mean() - применяем статистику
        
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[26]:


booted_data = get_bootstrap(df_control.value, df_test.value)
booted_data


# In[27]:


booted_data["p_value"]

# p-value меньше 0.05, делаем вывод, что средние в группах значимо различаются.
# Результаты статзначимы


# In[28]:


booted_data["quants"] 


# Выводы: t-тест удобно использовать на большом количестве экспериментов
# t-тест чувствительный к шумам и выбросам
# Если признак распределен нормально, то t-test будет работать нормально!
# 
# К тесту Манна-Уитни особых требований к выборке нет. Его можно использовать когда распределение не является нормальным.
# Манн-Уитни проверяет только равенства распределений. Этот критерий не подходит для сравнения средних или медиан. 
# 
# Бутстрэп можно использовать когда выборка репрезентативна и когда есть математические ограничения на использование t-testа.
# 
# В нашем случае распределения являются нормальными с выбросами. Поэтому я бы использовала бутсрэп

# In[ ]:





# In[ ]:




