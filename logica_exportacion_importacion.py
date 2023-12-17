import pandas as pd
import numpy as np
import math
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

# Agrega el archivo con los datos de las predicciones
df = pd.read_excel('/content/drive/MyDrive/val 2.xlsx')

# Define la columna de prodicción promedio
# Relación entre oferta y demanda simplemente es la resta entre el promedio de la producción de carne menos el promedio del consumo de carne
columnas_produccion = ['Relación entre oferta y demanda']
data = df[columnas_produccion]

columnas_totales = ['Municipio', 'Relación entre oferta y demanda']
datos = df[columnas_totales]

# Limpia los datos
data = data.replace(' ', '', regex=True)
data = data.replace(',', '.', regex=True)

data = data.astype(float)

df

# Carga el archivo GeoJSON
geo_data = gpd.read_file('/content/drive/MyDrive/hgomunicipal.geojson')

# Fusiona el geodataframe con los datos
geo_data = geo_data.merge(datos, left_on='NOMBRE', right_on='Municipio')

# Define el rango de colores basado en los valores de los datos
data_min = geo_data['Relación entre oferta y demanda'].min()
data_max = geo_data['Relación entre oferta y demanda'].max()

# Define el mapa de colores del colorbar y la normalización
cmap = mcolors.LinearSegmentedColormap.from_list("", ["black", "white", "red"])
norm = mcolors.TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)

# Dibuja el mapa coroplético con el mapa de colores especificado y la normalización
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
geo_data.plot(column='Relación entre oferta y demanda', cmap=cmap, norm=norm, linewidth=0.8, ax=ax, edgecolor='0.8')

# Elimina los ejes
ax.axis('off')

# Crea un colorbar con el mapa de colores especificado y la normalización
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', format=FuncFormatter(lambda x, _: f'{int(x)}'))
cbar.set_ticks([data_min, 0, data_max])
cbar.ax.set_yticklabels([f'{int(-1*(data_min))}', '0', f'{int(data_max)}'], fontsize=10)

# Agrega etiquetas al colorbar
cbar.ax.text(2.5, 0, 'Posibles importaciones           Posibles exportaciones', rotation='vertical', ha='center', va='center', fontsize=10, color='black')
cbar.ax.text(-1, 0, 'Cabezas de ganado bovino', rotation='vertical', ha='center', va='center', fontsize=10, color='black')

plt.title('Relación entre oferta y demanda por municipio')
plt.show()
