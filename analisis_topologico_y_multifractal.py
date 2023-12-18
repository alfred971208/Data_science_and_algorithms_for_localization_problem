import pandas as pd
import numpy as np
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from scipy.fft import fft
import pywt
import matplotlib.pyplot as plt
from MFDFA import MFDFA

pd.options.display.float_format= '{:,.4f}'.format

# Carga de datos
df = pd.read_excel('Copia de val.xlsx') # Modificar en donde coloques la ruta del archivo
df = df.drop('Unnamed: 0', axis = 1)

df

# Análisis topológico

time_series = df['Cabezas de ganado bovino (Objetivo)'].values

# Crea una incrustación de Takens de la serie de tiempo para la reconstrucción del espacio de fases
embedding_dimension = 3  # Ajusta esta dimensión basada en tus datos
embedding_time_delay = 1  # Ajusta este delay basado en tus datos

# Inicializa la clase SingleTakensEmbedding con los parámetros adecuados
embedding = SingleTakensEmbedding(time_delay=embedding_time_delay, dimension=embedding_dimension)
time_series_embedded = embedding.fit_transform(time_series.reshape(-1, 1))  # Se asegura de que time_series sea 2D

# La entrada para VietorisRipsPersistence.fit_transform debe ser un arreglo 3D
time_series_embedded_3D = time_series_embedded[None, :, :]

# Calcula la persistencia homológica
homology_dimensions = [0, 1]  # Ajusta las dimensiones de homología que te interesan
persistence = VietorisRipsPersistence(homology_dimensions=homology_dimensions)
diagrams = persistence.fit_transform(time_series_embedded_3D)

# Visualiza los diagramas de persistencia
plot_diagram(diagrams[0], plotly_params={"layout": {"title": "Diagrama de Persistencia"}})

# Análisis de mulifractalidad 

# Renombra las columnas para el análisis
renamed_columns_new = {
    'Producción en canal (Ton)': 'Produccion_en_canal_Ton',
    'Producción en pie (Ton)': 'Produccion_en_pie_Ton',
    'Precio promedio en canal ($/Kg)': 'Precio_promedio_canal_Kg',
    'Precio promedio en pie ($/Kg)': 'Precio_promedio_pie_Kg',
    'Valor de la producción en canal (Miles $)': 'Valor_produccion_canal',
    'Valor de la producción en pie (Miles $)': 'Valor_produccion_pie',
    'Peso promedio en canal (Kg)': 'Peso_promedio_canal_Kg',
    'Peso promedio en pie (Kg)': 'Peso_promedio_pie_Kg',
    'Cabezas de ganado bovino (Objetivo)': 'Cabezas_ganado_bovino',
    'Población': 'Poblacion',
    'Población varón 15-64 años': 'Poblacion_varon_15_64'
}

new_data_renamed = df.rename(columns=renamed_columns_new)

# Agrega la data por año para la serie de tiempo del análisis
time_series_data_new = new_data_renamed.groupby('Año')['Cabezas_ganado_bovino'].sum()

# Grafica la serie de tiempo
plt.figure(figsize=(12, 6))
plt.plot(time_series_data_new, marker='o', linestyle='-')
plt.title('Serie Temporal de Cabezas de Ganado Bovino por Año')
plt.xlabel('Año')
plt.ylabel('Total de Cabezas de Ganado Bovino')
plt.grid(True)
plt.show()

# Grafica la gráfica log-log para la escala de invarianza 
log_years_new = np.log(time_series_data_new.index.astype(float))
log_values_new = np.log(time_series_data_new.values.astype(float))

plt.figure(figsize=(12, 6))
plt.scatter(log_years_new, log_values_new)
plt.title('Gráfico Log-Log de la Serie Temporal')
plt.xlabel('Log(Año)')
plt.ylabel('Log(Total de Cabezas de Ganado Bovino)')
plt.grid(True)
plt.show()

# Renombra las columnas para el análisis
renamed_columns = {
    'Producción en canal (Ton)': 'Produccion_en_canal_Ton',
    'Producción en pie (Ton)': 'Produccion_en_pie_Ton',
    'Precio promedio en canal ($/Kg)': 'Precio_promedio_canal_Kg',
    'Precio promedio en pie ($/Kg)': 'Precio_promedio_pie_Kg',
    'Valor de la producción en canal (Miles $)': 'Valor_produccion_canal',
    'Valor de la producción en pie (Miles $)': 'Valor_produccion_pie',
    'Peso promedio en canal (Kg)': 'Peso_promedio_canal_Kg',
    'Peso promedio en pie (Kg)': 'Peso_promedio_pie_Kg',
    'Cabezas de ganado bovino (Objetivo)': 'Cabezas_ganado_bovino',
    'Población': 'Poblacion',
    'Población varón 15-64 años': 'Poblacion_varon_15_64'
}

data_new_renamed = df.rename(columns=renamed_columns)

# Agrega la data por año para la serie de tiempo del análisis
time_series_data = data_new_renamed.groupby('Año')['Cabezas_ganado_bovino'].sum()

# Prepara para el análisis de espectro y la descomposición de la varianza 
# Extrae los valores para la serie de tiempo
y = time_series_data.values

# Genera el análisis de espectro ocupando FFT 
yf = fft(y)
xf = np.fft.fftfreq(len(y), 1)  # Assuming 1 year interval

# Grafica el espectro de furier 
plt.figure(figsize=(12, 6))
plt.plot(xf, np.abs(yf))
plt.title("Espectro de Fourier")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# Descomposición de la varianza ocupando Wavelets
coeffs = pywt.wavedec(y, 'db1', level=4)  # Using Daubechies wavelet

# Grafica el coeficiente de Wavelet
plt.figure(figsize=(12, 6))
for i, coef in enumerate(coeffs):
    plt.subplot(len(coeffs), 1, i + 1)
    plt.plot(coef)
    plt.title(f"Detalle nivel {i+1}")
plt.tight_layout()
plt.show()

def calculate_hurst_exponent(time_series):
    # Convierte la siere de tiempo a una matriz de numpy sí todavía no está preparado 
    ts = np.array(time_series)

    # Calcula la media de la serie de tiempo 
    mean_ts = np.mean(ts)

    # Crear una matriz de desviaciones acumuladas de la media
    cum_dev = np.cumsum(ts - mean_ts)

    # Calcula el rango de las desviaciones acumuladas 
    R = np.max(cum_dev) - np.min(cum_dev)

    # Calcula la desviación estandar de las series de tiempo 
    S = np.std(ts)

    # Calcula el exponente de Hurst
    H = np.log(R / S) / np.log(len(ts))

    return H

# Calcula el exponente de Hurst para las series de tiempo 
hurst_exponent = calculate_hurst_exponent(time_series_data.values)
hurst_exponent

renamed_columns_latest = {
    'Producción en canal (Ton)': 'Produccion_en_canal_Ton',
    'Producción en pie (Ton)': 'Produccion_en_pie_Ton',
    'Precio promedio en canal ($/Kg)': 'Precio_promedio_canal_Kg',
    'Precio promedio en pie ($/Kg)': 'Precio_promedio_pie_Kg',
    'Valor de la producción en canal (Miles $)': 'Valor_produccion_canal',
    'Valor de la producción en pie (Miles $)': 'Valor_produccion_pie',
    'Peso promedio en canal (Kg)': 'Peso_promedio_canal_Kg',
    'Peso promedio en pie (Kg)': 'Peso_promedio_pie_Kg',
    'Cabezas de ganado bovino (Objetivo)': 'Cabezas_ganado_bovino',
    'Población': 'Poblacion',
    'Población varón 15-64 años': 'Poblacion_varon_15_64'
}

data_latest_renamed = df.rename(columns=renamed_columns_latest)

# Agrupa los datos por año para el análisis de series temporales
time_series_data_latest = data_latest_renamed.groupby('Año')['Cabezas_ganado_bovino'].sum()

time_series_data_latest.head()  # Muestra los primeros datos para verificar

# la serie temporal
y = time_series_data_latest.values

# Define las escalas
lag = np.unique(np.logspace(0.5, 3, 100).astype(int))

# Define el valor de q para un análisis monofractal (en este caso q=2)
q = 2

# Ordena de la tendencia polinómica en MFDFA
order = 1

# Realiza MFDFA
lag, dfa = MFDFA(y, lag=lag, q=q, order=order)

# Visualización en escala log-log
plt.loglog(lag, dfa, 'o', label='Serie Temporal: MFDFA q=2')

# Ajuste lineal para encontrar la pendiente (Hurst exponent)
coefficients = np.polyfit(np.log(lag)[4:20], np.log(dfa[4:20]), 1)
H_hat = coefficients[0]  # Esto es la pendiente

# Mostrando el exponente de Hurst estimado
# Se asegura de que H_hat es un número y no un array
if isinstance(H_hat, np.ndarray):
    H_hat = H_hat[0]

print('Estimated H = {:.3f}'.format(H_hat))

plt.xlabel('Lag')
plt.ylabel('DFA')
plt.legend()
plt.show()
