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
df = pd.read_excel('val.xlsx') # Modificar en donde coloques la ruta del archivo

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
