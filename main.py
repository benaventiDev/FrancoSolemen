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

print("d) ", results.summary(), "\nS")
#**************************E**************************
rsquared = results.rsquared  # R^2, porcentaje de la variabilidad explicado por el modelo
print("e) El porcentaje de variabilidad del ingreso mensual explicado por este modelo es ", rsquared)
#**************************F**************************
new_data = pd.DataFrame({'const':1, 'educacion':14, 'experiencia':6, 'edad':40, 'titulo':1}, index=[0])
prediction = results.predict(new_data)

print("f) El ingreso mensual de una persona que tiene 14 años de educacion y 6 de experiencia es ", prediction)
#**************************G**************************

#**************************H**************************
X = df[['educacion', 'experiencia', 'edad', 'titulo']]
y = df['ingreso']
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
significativas = results.pvalues[results.pvalues < 0.05]
print("h) Las variables significativas son:")
print(significativas)
#**************************I**************************
residuos = results.resid
plt.hist(residuos, bins=30, edgecolor='black')
plt.title('Histograma de los residuos: Franco Cañoles')
plt.show()


# Test de normalidad (por ejemplo, test de Shapiro-Wilk)
stat, p = stats.shapiro(residuos)


#**************************J**************************
# Gráfica de residuos vs predicción
plt.scatter(results.predict(), residuos)
plt.xlabel('Predicción')
plt.ylabel('Residuos')
plt.show()

# Test de Breusch-Pagan
names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test = sms.het_breuschpagan(residuos, results.model.exog)
lzip(names, test)


#**************************K**************************
durbinWatson = durbin_watson(residuos)
print("k) ", durbinWatson, ". Ya que el valor se encentra cerca de 2, hay poca evidencia de autocorrelacion en los datos.")
#**************************L**************************
# Matriz de correlación (mapa de calor)
corrMatrix = X.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Cálculo del VIF
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

