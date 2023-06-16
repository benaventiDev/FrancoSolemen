import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
file_name = '/home/benaventi/PycharmProjects/pythonProject/ingreso-1.csv'
df = pd.read_csv(file_name)
# Asigna 1 a los ingresos > 1 millón, 0 en caso contrario
df['HighIncome'] = np.where(df['ingreso'] > 1000, 1, 0)

# Calcula la proporción de altos ingresos y su intervalo de confianza
prop = df['HighIncome'].mean()
ci = stats.norm.interval(0.90, loc=prop, scale=np.sqrt(prop*(1-prop)/len(df)))


print("a) ", ci)
print("\n")

sns.boxplot(x='titulo', y='ingreso', data=df)
plt.title("Franco Cañoles")
plt.show()

#**************************C**************************
titulado = df[df['titulo'] == 1]['ingreso']
no_titulado = df[df['titulo'] == 0]['ingreso']
t_stat, p_val = stats.ttest_ind(titulado, no_titulado)
print("c) t: ", t_stat,", p" , p_val)
#Interpretar resultados basados en el valor p

#**************************D**************************
X = df[['educacion', 'experiencia', 'edad', 'titulo']]
X = sm.add_constant(X)  # Añade una constante (intersección) al modelo
y = df['ingreso']

model = sm.OLS(y, X)
results = model.fit()

#**************************E**************************
rsquared = results.rsquared  # R^2, porcentaje de la variabilidad explicado por el modelo

#**************************F**************************
#**************************G**************************
#**************************H**************************
#**************************I**************************
#**************************J**************************
#**************************K**************************
#**************************L**************************

