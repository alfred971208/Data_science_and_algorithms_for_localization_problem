import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as mso
import geopandas as gpd
import matplotlib.colors as mcolor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix
from matplotlib.ticker import FormatStrFormatter
from statsmodels.tsa.stattools import adfuller

pd.options.display.float_format= '{:,.4f}'.format

# Carga de datos
df = pd.read_excel('Copia de val.xlsx') # Modificar en donde coloques la ruta del archivo
df = df.drop('Unnamed: 0', axis = 1)

df

# Cargando bibliotecas y datos 
data = pd.read_excel("Copia de val.xlsx") # Modificar en donde coloques la ruta del archivo
X = data.drop(columns=["Cabezas de ganado bovino (Objetivo)", "Municipio", "Unnamed: 0"])
y = data["Cabezas de ganado bovino (Objetivo)"]

# LDA 
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
result = pd.DataFrame(X_lda, columns=['LDA1', 'LDA2'])
result['Municipio'] = data['Municipio']

# Aplicando KMeans con k=4 
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(X)
result['Cluster'] = clusters

# Visualización
plt.figure(figsize=(15, 10))
sns.scatterplot(x="LDA1", y="LDA2", hue="Cluster", data=result, palette="Set2", s=100, style="Cluster")
plt.title("Análisis de Clúster de Municipios en Espacio LDA")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Aplicar PCA  y obtener los tres primeros componentes
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
result_3d = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
result_3d['Cluster'] = clusters

# Visualización en 3D 
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
colors = sns.color_palette("Set2", n_colors=k_optimal)

for i in range(k_optimal):
    cluster_data = result_3d[result_3d['Cluster'] == i]
    ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], cluster_data['PCA3'], c=[colors[i]], label=f'Cluster {i}', s=60)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title("Análisis de Clúster de Municipios en Espacio 3D (PCA)")
ax.legend()
plt.show()

# Impresión del tipo de datos
df.dtypes

# Tratamiento de los datos
columnas_produccion = ['Producción en canal (Ton)', 'Producción en pie (Ton)', 'Precio promedio en canal ($/Kg)', 'Precio promedio en pie ($/Kg)', 'Valor de la producción en canal (Miles $)', 'Valor de la producción en pie (Miles $)', 'Peso promedio en canal (Kg)', 'Peso promedio en pie (Kg)', 'Cabezas de ganado bovino (Objetivo)', 'Población', 'Población varón 15-64 años']
data = df[columnas_produccion]

columnas_totales = ['Municipio', 'Año', 'Producción en canal (Ton)', 'Producción en pie (Ton)', 'Precio promedio en canal ($/Kg)', 'Precio promedio en pie ($/Kg)', 'Valor de la producción en canal (Miles $)', 'Valor de la producción en pie (Miles $)', 'Peso promedio en canal (Kg)', 'Peso promedio en pie (Kg)', 'Cabezas de ganado bovino (Objetivo)', 'Población', 'Población varón 15-64 años']
datos = df[columnas_totales]

data = data.replace(' ', '', regex=True)
data = data.replace(',', '.', regex=True)

data = data.astype(float)

# Impresión del tipo de datos tratados
data.dtypes

# Visualización de datos faltantes
mso.matrix(df)

# Impresión de columnas
df[['Producción en canal (Ton)', 'Producción en pie (Ton)', 'Precio promedio en canal ($/Kg)', 'Precio promedio en pie ($/Kg)', 'Valor de la producción en canal (Miles $)', 'Valor de la producción en pie (Miles $)', 'Peso promedio en canal (Kg)', 'Peso promedio en pie (Kg)', 'Cabezas de ganado bovino (Objetivo)', 'Población', 'Población varón 15-64 años']].describe(), # Impresión de faltantes en columnas
pd.DataFrame(df.isna().sum(), columns=['Missings']) 

# Genera la matriz de dispersión
scatter_matrix(df, figsize=(45, 45))

# Guarda la imagen con una resolución más alta
plt.savefig('scatter_matrix.png', dpi=200)

# Imprimimos la relevancia de las varibales
for feature in data:
    correlation = data[feature].corr(data['Cabezas de ganado bovino (Objetivo)'])
    print(f'Correlación de {feature} con la variable objetivo: {correlation}'), # Matriz de correlación
corr = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True)
plt.title('Matriz de correlación')
plt.show()

# Gráficas de series de tiempo
for i in columnas_produccion:
  plt.figure(figsize=(14,7))
  sns.lineplot(data = df, y=i, x='Año', linewidth=.8)
  plt.title('Serie de tiempo para ' + i)
  plt.xlabel('Año')
  plt.ylabel(i)
  plt.show()

# Gráficas de densidad
for i in columnas_produccion:
  plt.figure(figsize=(14,7))
  sns.kdeplot(data = df, x=i, hue = 'Año', fill=True, alpha=.5, linewidth=0, palette = 'viridis')
  plt.title('Gráfica de densidad para ' + i)
  plt.xlabel(i)
  plt.ylabel('Densidad')
  plt.show()

# Gráficas de correlación entre 'Cabezas de ganado bovino (Objetivo)' y cada variable
for i in columnas_produccion:
  g = sns.jointplot(data = df, y='Cabezas de ganado bovino (Objetivo)', x = i, hue = 'Año', alpha=.5, linewidth=0, palette = 'viridis')

  # Desactivar notación científica
  g.ax_joint.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
  g.ax_joint.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

  # Ajustar los límites del eje y y x
  g.ax_joint.set_ylim(0, max(df['Cabezas de ganado bovino (Objetivo)']))
  g.ax_joint.set_xlim(0, max(df[i]))

  # Añadir un poco de margen
  g.ax_joint.margins(0.1)

  plt.subplots_adjust(top=0.9)
  plt.suptitle('Correlación entre Cabezas de ganado bovino (Objetivo) y ' + i)
  plt.xlabel(i)
  plt.ylabel('Cabezas de ganado bovino (Objetivo)')
  plt.tight_layout()  # Ajustar el tamaño de la gráfica
  plt.show()

# Prueba de Dickey-Fuller aumentada para cada variable
for i in columnas_produccion:
  x = df[i].values
  x = np.nan_to_num(x)
  result = adfuller(x)
  print('Resultados de la prueba de Dickey-Fuller para ' + i + ':')
  print('Estadística ADF:', result[0])
  print('p-valor:', result[1])
  print('Valores Críticos:')
  for key, value in result[4].items():
    print('\t{}: {:.3f}'.format(key, value))

# Gráficas de retraso para cada variable
from pandas.plotting import lag_plot
for i in columnas_produccion:
  plt.figure(figsize=(5,5))
  lag_plot(df[i])
  plt.title('Gráfica de retraso para ' + i)
  plt.xlabel('Valor en t')
  plt.ylabel('Valor en t+1')
  plt.show()

# Crear un colormap para asociar cada año a un color
years = df['Año'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
color_map = {year: color for year, color in zip(years, colors)}

# Función para generar gráficas de retraso coloreadas por año para cada variable
def colored_lag_plot(data, columns, years, color_map):
    for column in columns:
        plt.figure(figsize=(7,7))
        for year in years[:-1]:  # Excluimos el último año porque no tiene un t+1
            subset = data[data['Año'] == year]
            next_year_subset = data[data['Año'] == year + 1]
            plt.scatter(subset[column], next_year_subset[column].values,
                        color=color_map[year], label=year, s=10)

        plt.title(f'Gráfica de retraso para {column}')
        plt.xlabel(f'Valor en t ({column})')
        plt.ylabel(f'Valor en t+1 ({column})')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

# Generar las gráficas de retraso coloreadas por año para cada variable
colored_lag_plot(df, columnas_produccion, years, color_map)

# Mapa coroplético para la variable 'Cabezas de ganado bovino (Objetivo)'

# Cargar los datos geográficos
geo_data = gpd.read_file('hgomunicipal.geojson') # Modificar en donde coloques la ruta del archivo

# Calcular la mediana de la producción para cada municipio desde 2006 hasta 2021
df['Año'] = df['Año'].astype(int)
data_2006_2021 = df[(df['Año'] >= 2006) & (df['Año'] <= 2021)]
median_data = data_2006_2021.groupby('Municipio')['Cabezas de ganado bovino (Objetivo)'].median()

# Fusionar los datos geográficos con los datos de producción
geo_data = geo_data.merge(median_data, left_on='NOMBRE', right_index=True)

# Crear un mapa de colores y una instancia de normalización
cmap = mcolors.LinearSegmentedColormap.from_list("", ["white","red"])
norm = mcolors.Normalize(vmin=geo_data['Cabezas de ganado bovino (Objetivo)'].min(), vmax=geo_data['Cabezas de ganado bovino (Objetivo)'].max())

# Crear un mapa coroplético con el mapa de colores y la normalización especificados
plt.figure(figsize=(10, 6))
geo_data.plot(column='Cabezas de ganado bovino (Objetivo)', cmap=cmap, norm=norm, linewidth=0.8, edgecolor='0.8')

# Remover los ejes
plt.axis('off')

# Añadir una barra de colores
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm)

plt.title('Producción media por municipio de 2006 a 2021 (Cabezas de ganado bovino)')
plt.show(), # Definir las columnas no numéricas
non_numeric_columns = ['Municipio', 'Año', 'Mes']

# Calcular los cambios porcentuales mes a mes para cada variable
for column in df.columns:
    if column not in non_numeric_columns:
        df[f'{column}_cambio_pct'] = df.groupby('Municipio')[column].pct_change()

# Definir las columnas de producción (variables originales)
columnas_produccion = [col for col in df.columns if col not in non_numeric_columns and "_cambio_pct" not in col]

# Definir paletas y estilos
colors = plt.cm.tab20c(np.linspace(0, 1, 10))  # 10 colores distintos como ejemplo
line_styles = ['-', '--', '-.', ':']  # Estilos de línea: sólido, punteado, dash-dot, puntos

# Crear un diccionario que asocie cada municipio con una combinación de color y estilo de línea
municipios = df['Municipio'].unique()
municipio_styles = {}
for idx, municipio in enumerate(municipios):
    municipio_styles[municipio] = (colors[idx % len(colors)], line_styles[idx % len(line_styles)])

# Función modificada para graficar usando las combinaciones
def plot_percent_changes_with_styles(data, columns):
    for column in columns:
        plt.figure(figsize=(15, 8))

        for municipio in municipios:
            subset = data[data['Municipio'] == municipio]
            color, line_style = municipio_styles[municipio]
            plt.plot(subset['Año'], subset[f'{column}_cambio_pct'], label=municipio, color=color, linestyle=line_style, alpha=0.6)

        plt.title(f'Cambios porcentuales a lo largo del tiempo para {column}')
        plt.xlabel('Año')
        plt.ylabel('Cambio porcentual')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

# Generar las gráficas intertemporales
plot_percent_changes(df, columnas_produccion)
