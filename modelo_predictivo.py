import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, BaggingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet, RANSACRegressor, HuberRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from scipy.stats import boxcox
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

# Carga el carchivo .xlsx
df = pd.read_excel('/content/drive/MyDrive/val.xlsx')

# Define los datos a usar
columnas_produccion = ['Variable 1', 'Variable 2', 'Variable 5', 'Variable 6', 'Variable 9 (Objetivo)', 'Variable 10', 'Variable 11']
data = df[columnas_produccion]

columnas_totales = ['Municipio', 'Año', 'Variable 1', 'Variable 2', 'Variable 5', 'Variable 6', 'Variable 9 (Objetivo)', 'Variable 10', 'Variable 11']
datos = df[columnas_totales]

# Limpia los datos
data = data.replace(' ', '', regex=True)
data = data.replace(',', '.', regex=True)

data = data.astype(float)

# Define la cantidad de municipios de entrenamiento
num_municipios_entrenamiento = 80

# Define municipios aleatorios
municipios_unicos = datos['Municipio'].unique()
np.random.shuffle(municipios_unicos)

# Define los municipios de entrenamiento y prueba
municipios_entrenamiento = municipios_unicos[:num_municipios_entrenamiento]
municipios_prueba = municipios_unicos[num_municipios_entrenamiento:]

train = datos[datos['Municipio'].isin(municipios_entrenamiento)]
test = datos[datos['Municipio'].isin(municipios_prueba)]

data_train = data.loc[train.index]
data_test = data.loc[test.index]

# Define la clase para la tranformación Box-Cox
class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, lmbda=0.15):
        self.features = features
        self.lmbda = lmbda

    def fit(self, X, y=None):
        return self

    def boxcox1p(self, x, lmbda):
        if lmbda != 0:
            return (x + 1)**lmbda - 1
        else:
            return np.log(x + 1)

    def transform(self, X, y=None):
        X = X.copy()
        for feature in self.features:
            X.loc[:, feature] = self.boxcox1p(X[feature], self.lmbda)
        return X


# Procesamiento de la Data
categorical_features = ['Municipio']
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

polynomial_transformer = PolynomialFeatures(degree=1)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)],
    remainder=StandardScaler())

# Aplica tranformación Box-Cox
num_features = ['Variable 1', 'Variable 2', 'Variable 5', 'Variable 6', 'Variable 10', 'Variable 11']
boxcox_transformer = BoxCoxTransformer(features=num_features)

# Selecciona las mejoras características
k = 7
feature_selector = SelectKBest(f_regression, k=k)

# Inicia el modelo
model1 = GradientBoostingRegressor()
model2 = RandomForestRegressor()
model3 = ElasticNet(max_iter=50000, l1_ratio=1)
model4 = XGBRegressor()
bagging_model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10)

# Stackea multiples modelos
stacked_models = StackingRegressor(estimators=[('gb', model1), ('rf', model2), ('en', model3), ('xgb', model4), ('bagging', bagging_model)])

# Define el pipeline
pipe = Pipeline(steps=[('boxcox_transformer', boxcox_transformer),
                       ('preprocessor', preprocessor),
                       ('polynomial_transformer', polynomial_transformer),
                       ('feature_selector', feature_selector),
                       ('model', stacked_models)])

# Ajusta los hiperparámetros dle modelo
parameters = {
    'model__gb__max_depth': [1, 2, 3, 4],
    'model__gb__min_samples_split': [2, 3, 5, 7, 10],
    'model__rf__n_estimators': [10, 30, 50, 70, 100, 150, 200],
    'model__rf__max_features': ['sqrt', 'log2', None],
    'model__en__alpha': np.logspace(-5, 0, num=20),
    'model__en__l1_ratio': np.linspace(0, 1, num=20),
    'model__xgb__learning_rate': np.logspace(-3, 0, num=10),
    'model__xgb__max_depth': [3, 5, 7, 9],
    'model__xgb__n_estimators': [50, 100, 150, 200],
    'model__xgb__min_child_weight': [1, 3, 5],
    'model__xgb__gamma': [0, 0.1, 0.2],
    'model__xgb__subsample': [0.5, 0.7, 0.9],
    'model__xgb__colsample_bytree': [0.5, 0.7, 0.9]
}


group_kfold = GroupKFold(n_splits=10)

# Rendimiento de busqueda bayesiana con cross validation
search = BayesSearchCV(pipe, parameters, n_iter=50, scoring='neg_mean_absolute_error', cv=group_kfold, n_jobs=-1)
search.fit(train.drop('Variable 9 (Objetivo)', axis=1), train['Variable 9 (Objetivo)'], groups=train['Municipio'])

print("Best parameters found: ", search.best_params_)
print("Lowest MAE found: ", np.abs(search.best_score_))

pipe.set_params(**search.best_params_)
pipe.fit(train.drop('Variable 9 (Objetivo)', axis=1), train['Variable 9 (Objetivo)'])

predictions = pipe.predict(test.drop('Variable 9 (Objetivo)', axis=1))
mae = mean_absolute_error(test['Variable 9 (Objetivo)'], predictions)
print("MAE: ", mae)

# Hacer predicciones para cada municipio
predictions = []
for municipio in municipios_unicos:
    municipio_data = datos[datos['Municipio'] == municipio]
    municipio_data = municipio_data[columnas_totales]

    x_test = []
    for i in range(7, len(municipio_data)):
        x_test.append(municipio_data.iloc[i-7:i, :])
    x_test = pd.concat(x_test)

    if x_test.shape[0] > 0:
        predicted_values = pipe.predict(x_test.drop('Variable 9 (Objetivo)', axis=1))

        for i in range(min(len(predicted_values), 25)):
            year = 2006 + i
            value = predicted_values[i]
            predictions.append([municipio, year, value])
    else:
        print(f"No hay suficientes datos para hacer una predicción para {municipio}.")

# Convierte las predicciones en un DataFrame
predictions_df = pd.DataFrame(predictions)

predictions_df = pd.DataFrame(predictions, columns=['Municipio', 'Año', 'Valor'])
predictions_df.to_excel('predicciones.xlsx', index=False)

import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict

best_params = OrderedDict([('model__en__alpha', 1.8329807108324375e-05),
                           ('model__en__l1_ratio', 1.0),
                           ('model__gb__max_depth', 2),
                           ('model__gb__min_samples_split', 5),
                           ('model__rf__max_features', 'log2'),
                           ('model__rf__n_estimators', 50),
                           ('model__xgb__colsample_bytree', 0.9),
                           ('model__xgb__gamma', 0.2),
                           ('model__xgb__learning_rate', 0.046415888336127774),
                           ('model__xgb__max_depth', 3),
                           ('model__xgb__min_child_weight', 3),
                           ('model__xgb__n_estimators', 100),
                           ('model__xgb__subsample', 0.5)])

params_df = pd.DataFrame.from_dict(best_params, orient='index', columns=['Value'])
print(params_df)


from sklearn import set_config

pipe = Pipeline(steps=[('boxcox_transformer', boxcox_transformer),
                       ('preprocessor', preprocessor),
                       ('polynomial_transformer', polynomial_transformer),
                       ('feature_selector', feature_selector),
                       ('model', stacked_models)])


set_config(display='diagram')
print(pipe)

from sklearn import set_config

set_config(display='diagram')

# Asume que 'pipe' es tu pipeline de aprendizaje automático
print(pipe)

# Asumamos que 'search' es tu objeto GridSearchCV o BayesSearchCV después de ajustarlo.
# cv_results_ contiene información detallada acerca de la búsqueda.
results = search.cv_results_

# Supongamos que estamos interesados en el parámetro 'model__rf__n_estimators'
param = 'param_model__rf__n_estimators'
param_values = results[param].data
mean_test_scores = results['mean_test_score']

plt.figure(figsize=(12, 6))
plt.title("Validación cruzada vs Número de estimadores")
plt.xlabel("Número de estimadores")
plt.ylabel("Score de validación cruzada")
plt.plot(param_values, mean_test_scores, marker='o')
plt.grid()
plt.show()

# Almacenar los resultados en una variable
results = pd.DataFrame(search.cv_results_)

print(results)

# Valores NaN
print(data_train_transformed.isnull().sum())

# Valores Infinitos
print(np.isinf(data_train_transformed).sum())

# Verificar si hay valores no positivos
print((data_train <= 0).sum())

for feature in num_features:
    print(f"{feature} tiene {data_train[feature].nunique()} valores únicos.")

# Generar un gráfico de densidad para cada característica numérica
for feature in num_features:
    plt.figure(figsize=(12, 6))
    sns.histplot(data_train[feature], kde=True)
    plt.title(f'Distribución de {feature} antes de la transformación de Box-Cox')
    plt.show()

print(data_train.head())
print(data_train_transformed.head())

boxcox_transformer = BoxCoxTransformer(features=num_features)
boxcox_transformer.fit(data_train)
data_train_transformed = boxcox_transformer.transform(data_train)

print(data_train_transformed.describe())

sns.boxplot(x=data_train['Variable 9 (Objetivo)'])
plt.show()

sns.boxplot(data=data_train_transformed)
plt.show()

# Generar histogramas con diferentes bins para cada característica numérica
bins_values = [10, 20, 50, 100]

for feature in num_features:
    for bins in bins_values:
        plt.figure(figsize=(12, 6))
        plt.hist(data_train[feature], bins=bins, alpha=0.5, label='Original')
        plt.hist(data_train_transformed[feature], bins=bins, alpha=0.5, label='Transformado')
        plt.title(f'Distribución de {feature} antes y después de la transformación de Box-Cox con {bins} bins')
        plt.legend()
        plt.show()

# Generar gráficos de densidad para cada característica numérica
for feature in num_features:
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data_train[feature], label='Original')
    sns.kdeplot(data_train_transformed[feature], label='Transformado')
    plt.title(f'Distribución de {feature} antes y después de la transformación de Box-Cox')
    plt.legend()
    plt.show()

from sklearn.preprocessing import MinMaxScaler

# Escalar los datos antes de graficar
scaler = MinMaxScaler()
data_train_scaled = pd.DataFrame(scaler.fit_transform(data_train), columns=data_train.columns)
data_train_transformed_scaled = pd.DataFrame(scaler.transform(data_train_transformed), columns=data_train_transformed.columns)

# Generar un gráfico de densidad para cada característica numérica
for feature in num_features:
    plt.figure(figsize=(12, 6))
    plt.hist(data_train_scaled[feature], bins=30, alpha=0.5, label='Original')
    plt.hist(data_train_transformed_scaled[feature], bins=30, alpha=0.5, label='Transformado')
    plt.title(f'Distribución de {feature} antes y después de la transformación de Box-Cox')
    plt.legend()
    plt.show()

print(results.keys())

# Extraer los valores de 'model__gb__max_depth'
param_values = [dic['model__gb__max_depth'] for dic in results['params'] if 'model__gb__max_depth' in dic]

mean_test_scores = results['mean_test_score']

plt.figure(figsize=(12, 6))
plt.title("Validación cruzada vs Profundidad máxima")
plt.xlabel("Profundidad máxima")
plt.ylabel("Score de validación cruzada")
plt.plot(param_values, mean_test_scores, marker='o')
plt.grid()
plt.show()

# Imprime ell coeficiente de determinación R^2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Calcula los residuos
predictions = pipe.predict(train.drop('Variable 9 (Objetivo)', axis=1))
residuos = train['Variable 9 (Objetivo)'] - predictions

# Calcula los valores predichos
valores_predichos = predictions

# Ajusta una regresión lineal entre los residuos y los valores predichos
reg = LinearRegression().fit(valores_predichos.reshape(-1, 1), residuos)

# Calcula el R^2 de la regresión
r2 = r2_score(residuos, reg.predict(valores_predichos.reshape(-1, 1)))

print("El coeficiente de determinación R^2 es:", r2)

# Histogramas de las variables numéricas
data[num_features].hist(bins=30, figsize=(10, 7))

# Matriz de correlación
corr_matrix = data[num_features].corr()
sns.heatmap(corr_matrix, annot=True)

# Primero, necesitamos preprocesar los datos de entrenamiento
train_preprocessed = preprocessor.fit_transform(train.drop('Variable 9 (Objetivo)', axis=1))

# Luego, ajustamos el modelo a los datos preprocesados
rf.fit(train_preprocessed, train['Variable 9 (Objetivo)'])

# Obtenemos las importancias de las características
importances = rf.feature_importances_

# Ahora necesitamos obtener los nombres de las características después del preprocesamiento.
# Para las características categóricas, OneHotEncoder las convierte en varias características binarias.
feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
# Añadimos las características numéricas a la lista
feature_names = np.concatenate([feature_names, num_features])

# Verificamos si el número de características después del preprocesamiento coincide con la longitud de 'importances'
if len(feature_names) == len(importances):
    # Si coinciden, podemos hacer un gráfico de barras de las importancias de las características
    plt.barh(feature_names, importances)
else:
    print(f"La cantidad de nombres de características ({len(feature_names)}) no coincide con la cantidad de importancias ({len(importances)})")

# Definimos la vairbales importantes del RandomForest

rf = RandomForestRegressor()
rf.fit(train_preprocessed, train['Variable 9 (Objetivo)'])

importances = rf.feature_importances_

# Imprimimos la importancia de las variables

for feature, importance in zip(train.columns, importances):
    print(f"La importancia de {feature} es: {importance}")

# Entrena tu pipeline
pipe.fit(train.drop('Variable 9 (Objetivo)', axis=1), train['Variable 9 (Objetivo)'])

# Obtén los nombres de las características después del preprocesamiento
preprocessed_features = []
for transformer in pipe.named_steps['preprocessor'].transformers_:
    # Para las características categóricas
    # Para las características categóricas
    if transformer[0] == 'cat':
      categories = transformer[1].categories_[0]
      feature_name = categorical_features[0]
      preprocessed_features.extend([f"{feature_name}_{category}" for category in categories])

    # Para las características numéricas
    else:
        preprocessed_features.extend(num_features)

# Entrena tu RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(pipe.transform(train.drop('Variable 9 (Objetivo)', axis=1)), train['Variable 9 (Objetivo)'])

# Obtiene las importancias de las características
importances = rf.feature_importances_

# Imprime las importancias de las características
for feature, importance in zip(preprocessed_features, importances):
    print(f"La importancia de {feature} es: {importance}")

# Imprimimos la garafica de residuales

predictions = pipe.predict(train.drop('Variable 9 (Objetivo)', axis=1))
residuals = train['Variable 9 (Objetivo)'] - predictions
plt.scatter(predictions, residuals)
plt.xlabel('Predicciones')
plt.ylabel('Residuales')

# Imprimimos la importancia de las varibales

for feature in num_features:
    correlation = data[feature].corr(data['Variable 9 (Objetivo)'])
    print(f'Correlación de {feature} con la variable objetivo: {correlation}')

# Definimos e imprimimos el MAE promedio

from sklearn.model_selection import cross_val_score, RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=5)
scores = cross_val_score(pipe, train.drop('Variable 9 (Objetivo)', axis=1), train['Variable 9 (Objetivo)'], scoring='neg_mean_absolute_error', cv=rkf)
print('MAE promedio: ', np.mean(np.abs(scores)))

# Reemplaza 'Variable 11' por la variable predictora que deseas visualizar
sns.scatterplot(x='Variable 10', y='Variable 9 (Objetivo)', data=data)
plt.show()

# Imprimimos la gráfica de curva de aprendizaje

from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    pipe, train.drop('Variable 9 (Objetivo)', axis=1), train['Variable 9 (Objetivo)'], train_sizes=np.linspace(0.1, 1.0, 5))

plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores.mean(axis=1), 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()

# Rendimiento en el conjunto de entrenamiento
train_predictions = pipe.predict(train.drop('Variable 9 (Objetivo)', axis=1))
train_mae = mean_absolute_error(train['Variable 9 (Objetivo)'], train_predictions)
print("Train MAE: ", train_mae)

# Rendimiento en el conjunto de prueba
test_predictions = pipe.predict(test.drop('Variable 9 (Objetivo)', axis=1))
test_mae = mean_absolute_error(test['Variable 9 (Objetivo)'], test_predictions)
print("Test MAE: ", test_mae)

# Prueba de adfuller para la variable objetivo de entrada

from statsmodels.tsa.stattools import adfuller

# Realiza la prueba ADF en tus datos de entrada
result = adfuller(train['Variable 9 (Objetivo)'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# Prueba de adfuller para la variable objetivo de salida
# Aplicar la prueba ADF a la columna 'Valor'
predicted_values = predictions_df['Valor']
result = adfuller(predicted_values)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# Realiza la prueba de Breusch-Pagan para los datos residuales

from statsmodels.compat import lzip
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

# Primero necesitamos obtener los residuos del modelo de entrenamiento.
residuals = train['Variable 9 (Objetivo)'] - pipe.predict(train.drop('Variable 9 (Objetivo)', axis=1))

# Preprocesa las características
transformed_features = pd.DataFrame(preprocessor.transform(train.drop('Variable 9 (Objetivo)', axis=1)))

# Realiza la prueba de Breusch-Pagan
bp_test = sms.het_breuschpagan(residuals, transformed_features)

labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))


import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Cargar el archivo GeoJSON
datos_geo = gpd.read_file('/content/drive/MyDrive/hgomunicipal.geojson')

# Calcular la mediana de la predicción por municipio desde 2023 hasta 2030
df_predicciones['Año'] = df_predicciones['Año'].astype(int)
predicciones_2023_2030 = df_predicciones[(df_predicciones['Año'] >= 2023) & (df_predicciones['Año'] <= 2030)]
medianas_predicciones = predicciones_2023_2030.groupby('Municipio')['Valor'].median()

# Combinar el GeoDataFrame con las medianas de las predicciones
datos_geo = datos_geo.merge(medianas_predicciones, left_on='NOMBRE', right_index=True)

# Crear un mapa de colores con un solo color (azul en este ejemplo)
mapa_colores = mcolors.LinearSegmentedColormap.from_list("", ["white","red"])

# Crear una instancia de normalización
norm = mcolors.Normalize(vmin=datos_geo['Valor'].min(), vmax=datos_geo['Valor'].max())

# Graficar el mapa coroplético con el mapa de colores y la normalización especificados
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
datos_geo.plot(column='Valor', cmap=mapa_colores, norm=norm, linewidth=0.8, ax=ax, edgecolor='0.8')

# Remover los ejes
ax.axis('off')

# Agregar una barra de colores
sm = plt.cm.ScalarMappable(cmap=mapa_colores, norm=norm)
fig.colorbar(sm)

plt.title('Predicción por municipio de la producción media de 2023 a 2030 (cabezas de ganado bovino)')
plt.show()



