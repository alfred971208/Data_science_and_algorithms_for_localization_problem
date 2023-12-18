from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import folium

def geo_a_cartesiano(lat, lon):
    R = 6371  # radio de la Tierra en kilómetros
    x = R * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = R * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = R * np.sin(np.radians(lat))
    return x, y, z

def cartesiano_a_geo(x, y, z):
    R = 6371  # radio de la Tierra en kilómetros
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z / R))
    return lat, lon

# Datos proporcionados
data = pd.DataFrame({
    'Municipio': ['Acatlán', 'Agua Blanca de Iturbide', 'Ajacuba', 'Alfajayucan', 'Almoloya', 'Atlapexco', 'Chapantongo', 'Chapulhuacán', 'Chilcuautla', 'Eloxochitlán', 'Huautla', 'Huazalingo', 'Huehuetla', 'Huichapan', 'Jacala de Ledezma', 'Juárez Hidalgo', 'La Misión', 'Lolotla', 'Metepec', 'Molango de Escamilla', 'Nicolás Flores', 'Nopala de Villagrán', 'Omitlán de Juárez', 'Pacula', 'Pisaflores', 'San Agustín Metzquititlán', 'San Bartolo Tutotepec', 'San Felipe Orizatlán', 'Santiago de Anaya', 'Tasquillo', 'Tecozautla', 'Tenango de Doria', 'Tepetitlán', 'Tlahuiltepa', 'Villa de Tezontepec', 'Xochicoatlán'],
    'Latitud': [20.14599753, 20.34998156, 20.0926803, 20.41015828, 19.70312738, 21.01745106, 20.28559483, 21.15787458, 20.3314391, 20.74687461, 21.03188243, 20.98056421, 20.46035677, 20.37553905, 21.00849043, 20.78338131, 21.10140801, 20.84163503, 20.23881199, 20.78658031, 20.76804674, 20.25195194, 20.16956885, 21.0502993, 21.19493683, 20.53306527, 20.3991586, 21.17106016, 20.38273787, 20.55204235, 20.53415978, 20.33845235, 20.18700312, 20.92389198, 19.879986, 20.77704277],
    'Longitud': [-98.43835281, -98.35651984, -99.12210399, -99.34946458, -98.4032803, -98.34829642, -99.41319277, -98.90435814, -99.23167454, -98.8093052, -98.28668517, -98.50774232, -98.07859535, -99.6510293, -99.18853709, -98.82918225, -99.123012, -98.71711415, -98.32208348, -98.7306035, -99.15056418, -99.64433781, -98.6485979, -99.29636159, -99.00590219, -98.63880134, -98.20205941, -98.60758905, -98.96390502, -99.31248643, -99.6349425, -98.2267008, -99.38024786, -98.95017064, -98.81930736, -98.67966839],
    'Peso': [1947.515445, 1488.42062, 564.8016176, 1126.671252, 715.7856113, 70.18432667, 2075.752896, 2076.831214, 286.4689418, 535.8706135, 1531.505859, 45.81143524, 1136.262735, 1357.398389, 1217.834983, 317.6627911, 1735.792089, 593.5610713, 1375.535971, 769.5643214, 743.7061007, 2378.867366, 11.74892797, 615.0936316, 1776.225709, 25.68872557, 1635.505363, 3193.503324, 428.250717, 506.6549098, 1068.321568, 393.5998613, 693.901223, 236.2399636, 526.1847575, 164.6085326]
})

# Convertir lat, lon a x, y, z
data['x'], data['y'], data['z'] = geo_a_cartesiano(data['Latitud'], data['Longitud'])

# Extraer las coordenadas y pesos
X = data[['x', 'y', 'z']].values
pesos = data['Peso'].values

# Normalizar las características
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Determinar el número óptimo de clusters usando los métodos de Elbow y Silhouette
rango_n_clusters = list(range(2,11))
elbow = []
puntajes_silhouette = []

for n_clusters in rango_n_clusters:
    # Inicializar el agrupador
    agrupador = KMeans(n_clusters=n_clusters, random_state=0)

    # Ajustar los datos y hacer predicciones
    etiquetas_cluster = agrupador.fit_predict(X_escalado)

    # Calcular la puntuación promedio de silhouette
    promedio_silhouette = silhouette_score(X_escalado, etiquetas_cluster)
    puntajes_silhouette.append(promedio_silhouette)

    # Calcular la suma total dentro del cluster de cuadrados (método de Elbow)
    total_dentro_SS = agrupador.inertia_
    elbow.append(total_dentro_SS)

# Dibujar el gráfico del método de Elbow
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(rango_n_clusters, elbow, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método de Elbow')

# Dibujar el gráfico de los puntajes de Silhouette
plt.subplot(1,2,2)
plt.plot(rango_n_clusters, puntajes_silhouette, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Puntaje de Silhouette')
plt.title('Método de Puntaje de Silhouette')
plt.show()

# Aplicar clustering jerárquico
ac = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='ward')
ac.fit(X_escalado)

# Dibujar el dendrograma correspondiente
plt.figure(figsize=(10, 5))
Z = linkage(pdist(X_escalado), 'ward')
dendrogram(Z)

# Añadir anotaciones al dendrograma
plt.title('Dendrograma de clustering jerárquico')
plt.xlabel('Índice de la Muestra')
plt.ylabel('Distancia')

# Anotar la altura en la que se decide hacer el corte
max_d = 50 # Ajustar este valor según el dendrograma
plt.axhline(y=max_d, c='k', ls='--', lw=0.5)
plt.text(0.5, max_d, 'Corte para 10 clusters', va='center')

# Anotar los clusters formados
num_clusters = 14 # Ajustar este valor según tus necesidades
for i in range(num_clusters):
    # Dibujar una línea horizontal en la altura de la fusión
    height = Z[-i-1, 2]
    plt.axhline(y=height, c='k', ls='--', lw=0.5)

    # Añadir una anotación para el cluster
    plt.text(Z[-i-1, 0], height, f'Cluster {i+1}', va='center')

plt.show()

# Ejecutar el algoritmo KMeans
clusters = 14 # o cualquier valor que hayas determinado a partir de los métodos Elbow o Silhouette
kmeans = KMeans(n_clusters=clusters, n_init=10000) # se ejecutará 10000 veces con diferentes semillas aleatorias
kmeans.fit(X_escalado, sample_weight=pesos)

# Añadir etiquetas de cluster al dataframe
data['cluster'] = kmeans.labels_

# Dibujar los clusters en 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=data['cluster'], s=pesos/500)

# Dibujar los centroides
centroides_escalados = kmeans.cluster_centers_
centroides = escalador.inverse_transform(centroides_escalados)
ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], s=300, c='red')

plt.show()

# Convertir los centroides de los clusters de vuelta a lat, lon
centroides_latlon = np.array([cartesiano_a_geo(x, y, z) for x, y, z in centroides])

# Crear un mapa centrado en la ubicación media
mapa = folium.Map(location=[data['Latitud'].mean(), data['Longitud'].mean()], zoom_start=7)

# Añadir puntos al mapa
for lat, lon, peso, cluster in zip(data['Latitud'], data['Longitud'], data['Peso'], data['cluster']):
    folium.CircleMarker([lat, lon], radius=peso/500, color='blue', fill=True, fill_opacity=0.6).add_to(mapa)

# Añadir los centroides de los clusters al mapa
for lat, lon in centroides_latlon:
    folium.Marker(location=(lat, lon), icon=folium.Icon(color='red')).add_to(mapa)

mapa
