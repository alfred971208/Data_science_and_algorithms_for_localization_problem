import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Carga los datos desde el archivo de Excel
xls = pd.ExcelFile('/content/predicciones (9).xlsx')

# Carga cada hoja en un dataframe
df_hoja1 = pd.read_excel(xls, 'Sheet1')
df_hoja2 = pd.read_excel(xls, 'Hoja1')

# Muestra las primeras filas de cada dataframe
df_hoja1.head(), df_hoja2.head()

# Renombra la columna 'Valor' en cada dataframe para mayor claridad
df_hoja1.rename(columns={'Valor': 'Valor_Predicho'}, inplace=True)
df_hoja2.rename(columns={'Valor': 'Valor_Real'}, inplace=True)

# Combina los dos dataframes en 'Municipio' y 'Año'
df_combinado = pd.merge(df_hoja1, df_hoja2, on=['Municipio', 'Año'])

# Muestra las primeras filas del dataframe combinado
df_combinado.head()

# Calcula el Error Absoluto Medio (MAE)
mae = mean_absolute_error(df_combinado['Valor_Real'], df_combinado['Valor_Predicho'])

# Calcula el Error Cuadrado Medio (MSE)
mse = mean_squared_error(df_combinado['Valor_Real'], df_combinado['Valor_Predicho'])

# Calcula el Error Porcentual Absoluto Medio (MAPE)
mape = np.mean(np.abs((df_combinado['Valor_Real'] - df_combinado['Valor_Predicho']) / df_combinado['Valor_Real'])) * 100

# Calcula el Coeficiente de Determinación (R²)
r2 = r2_score(df_combinado['Valor_Real'], df_combinado['Valor_Predicho'])

mae, mse, mape, r2

# Grafica los valores reales y los valores pronosticados
plt.figure(figsize=(14, 6))

# Selecciona un municipio aleatorio para la visualización
municipio_aleatorio = df_combinado['Municipio'].sample(1).values[0]
datos_a_graficar = df_combinado[df_combinado['Municipio'] == municipio_aleatorio]

plt.plot(datos_a_graficar['Año'], datos_a_graficar['Valor_Real'], label='Valores Reales', marker='o')
plt.plot(datos_a_graficar['Año'], datos_a_graficar['Valor_Predicho'], label='Valores Pronosticados', marker='x')

plt.title('Comparación de Valores Reales y Pronosticados para el Municipio: ' + municipio_aleatorio)
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()

# Cargar ambos conjuntos de datos desde las hojas del archivo Excel
df_hoja1 = pd.read_excel("/content/predicciones (9).xlsx", 'Sheet1')
df_hoja2 = pd.read_excel("/content/predicciones (9).xlsx", 'Hoja1')

# Renombrar la columna 'Valor' en cada dataframe para mayor claridad
df_hoja1.rename(columns={'Valor': 'Valor_Predicho'}, inplace=True)
df_hoja2.rename(columns={'Valor': 'Valor_Real'}, inplace=True)

# Combinar los dos dataframes basándose en 'Municipio' y 'Año'
df_combinado = pd.merge(df_hoja1, df_hoja2, on=['Municipio', 'Año'])

# Seleccionar un municipio aleatorio para la visualización
municipio_aleatorio = df_combinado['Municipio'].sample(1).values[0]
datos_a_graficar = df_combinado[df_combinado['Municipio'] == municipio_aleatorio]

# Aplicar el filtro HP para obtener la tendencia de los valores reales y pronosticados
_, tendencia_real = sm.tsa.filters.hpfilter(datos_a_graficar['Valor_Real'], lamb=1600)

# Extraer datos pronosticados hasta el 2030 para el municipio seleccionado
datos_pronosticados_2030 = df_hoja1[(df_hoja1['Municipio'] == municipio_aleatorio) & (df_hoja1['Año'] <= 2030)]

# Aplicar el filtro HP para obtener la tendencia de los valores pronosticados hasta el 2030
_, tendencia_predicho_2030 = sm.tsa.filters.hpfilter(datos_pronosticados_2030['Valor_Predicho'], lamb=1600)

# Establecer el rango de años desde el inicio del conjunto de datos hasta el 2030
rango_anos = range(df_combinado['Año'].min(), 2031)

# Graficar
plt.figure(figsize=(14, 6))

# Valores reales y su tendencia
plt.plot(datos_a_graficar['Año'], datos_a_graficar['Valor_Real'], label='Valores Reales', marker='o')
plt.plot(datos_a_graficar['Año'], tendencia_real, label='Tendencia Reales (Filtro HP)', linestyle='--')

# Valores pronosticados hasta 2030 y su tendencia
plt.plot(datos_pronosticados_2030['Año'], datos_pronosticados_2030['Valor_Predicho'], label='Valores Pronosticados', marker='x')
plt.plot(datos_pronosticados_2030['Año'], tendencia_predicho_2030, label='Tendencia Pronosticados (Filtro HP)', linestyle=':')

# Ajustes del gráfico
plt.title('Comparación de Valores Reales y Pronosticados (hasta 2030) y sus Tendencias para el Municipio: ' + municipio_aleatorio)
plt.xlabel('Año')
plt.ylabel('Cabezas de ganado')
plt.xticks(rango_anos)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Carga ambos conjuntos de datos desde las hojas del archivo Excel
df_hoja1 = pd.read_excel("/content/predicciones (9).xlsx", 'Sheet1')
df_hoja2 = pd.read_excel("/content/predicciones (9).xlsx", 'Hoja1')

# Renombra la columna 'Valor' en cada dataframe para mayor claridad
df_hoja1.rename(columns={'Valor': 'Valor_Predicho'}, inplace=True)
df_hoja2.rename(columns={'Valor': 'Valor_Real'}, inplace=True)

# Combina los dos dataframes basándose en 'Municipio' y 'Año'
df_combinado = pd.merge(df_hoja1, df_hoja2, on=['Municipio', 'Año'])

# Selecciona un municipio aleatorio para la visualización
municipio_aleatorio = df_combinado['Municipio'].sample(1).values[0]
datos_a_graficar = df_combinado[df_combinado['Municipio'] == municipio_aleatorio]

# Aplica el filtro HP para obtener la tendencia de los valores reales y pronosticados
_, tendencia_real = sm.tsa.filters.hpfilter(datos_a_graficar['Valor_Real'], lamb=1600)

# Extrae datos pronosticados hasta el 2030 para el municipio seleccionado
datos_pronosticados_2030 = df_hoja1[(df_hoja1['Municipio'] == municipio_aleatorio) & (df_hoja1['Año'] <= 2030)]

# Aplica el filtro HP para obtener la tendencia de los valores pronosticados hasta el 2030
_, tendencia_predicho_2030 = sm.tsa.filters.hpfilter(datos_pronosticados_2030['Valor_Predicho'], lamb=1600)

# Establece el rango de años desde el inicio del conjunto de datos hasta el 2030
rango_anos = range(df_combinado['Año'].min(), 2031)

# Grafica
plt.figure(figsize=(14, 6))

# Valores reales y su tendencia
plt.plot(datos_a_graficar['Año'], datos_a_graficar['Valor_Real'], label='Valores Reales', marker='o')
plt.plot(datos_a_graficar['Año'], tendencia_real, label='Tendencia Reales (Filtro HP)', linestyle='--')

# Valores pronosticados hasta 2030 y su tendencia
plt.plot(datos_pronosticados_2030['Año'], datos_pronosticados_2030['Valor_Predicho'], label='Valores Pronosticados', marker='x')
plt.plot(datos_pronosticados_2030['Año'], tendencia_predicho_2030, label='Tendencia Pronosticados (Filtro HP)', linestyle=':')

# Ajustes del gráfico
plt.title('Comparación de Valores Reales y Pronosticados (hasta 2030) y sus Tendencias para el Municipio: ' + municipio_aleatorio)
plt.xlabel('Año')
plt.ylabel('Cabezas de ganado')
plt.xticks(rango_anos)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

