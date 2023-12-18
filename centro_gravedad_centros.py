import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from geopy.distance import geodesic
from scipy.optimize import minimize
import statistics
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Datos proporcionados
data = pd.DataFrame({
    'Municipio': ['Acatlán', 'Agua Blanca de Iturbide', 'Ajacuba', 'Alfajayucan', 'Almoloya', 'Atlapexco', 'Chapantongo', 'Chapulhuacán', 'Chilcuautla', 'Eloxochitlán', 'Huautla', 'Huazalingo', 'Huehuetla', 'Huichapan', 'Jacala de Ledezma', 'Juárez Hidalgo', 'La Misión', 'Lolotla', 'Metepec', 'Molango de Escamilla', 'Nicolás Flores', 'Nopala de Villagrán', 'Omitlán de Juárez', 'Pacula', 'Pisaflores', 'San Agustín Metzquititlán', 'San Bartolo Tutotepec', 'San Felipe Orizatlán', 'Santiago de Anaya', 'Tasquillo', 'Tecozautla', 'Tenango de Doria', 'Tepetitlán', 'Tlahuiltepa', 'Villa de Tezontepec', 'Xochicoatlán'],
    'Latitud': [20.14599753, 20.34998156, 20.0926803, 20.41015828, 19.70312738, 21.01745106, 20.28559483, 21.15787458, 20.3314391, 20.74687461, 21.03188243, 20.98056421, 20.46035677, 20.37553905, 21.00849043, 20.78338131, 21.10140801, 20.84163503, 20.23881199, 20.78658031, 20.76804674, 20.25195194, 20.16956885, 21.0502993, 21.19493683, 20.53306527, 20.3991586, 21.17106016, 20.38273787, 20.55204235, 20.53415978, 20.33845235, 20.18700312, 20.92389198, 19.879986, 20.77704277],
    'Longitud': [-98.43835281, -98.35651984, -99.12210399, -99.34946458, -98.4032803, -98.34829642, -99.41319277, -98.90435814, -99.23167454, -98.8093052, -98.28668517, -98.50774232, -98.07859535, -99.6510293, -99.18853709, -98.82918225, -99.123012, -98.71711415, -98.32208348, -98.7306035, -99.15056418, -99.64433781, -98.6485979, -99.29636159, -99.00590219, -98.63880134, -98.20205941, -98.60758905, -98.96390502, -99.31248643, -99.6349425, -98.2267008, -99.38024786, -98.95017064, -98.81930736, -98.67966839],
    'Peso': [1947.515445, 1488.42062, 564.8016176, 1126.671252, 715.7856113, 70.18432667, 2075.752896, 2076.831214, 286.4689418, 535.8706135, 1531.505859, 45.81143524, 1136.262735, 1357.398389, 1217.834983, 317.6627911, 1735.792089, 593.5610713, 1375.535971, 769.5643214, 743.7061007, 2378.867366, 11.74892797, 615.0936316, 1776.225709, 25.68872557, 1635.505363, 3193.503324, 428.250717, 506.6549098, 1068.321568, 393.5998613, 693.901223, 236.2399636, 526.1847575, 164.6085326]
})

# Definir la función objetivo para minimizar
def objective_function(coords):
    lat, lon = coords
    total_distance = 0
    for idx, row in data.iterrows():
        total_distance += row['Peso'] * geodesic((lat, lon), (row['Latitud'], row['Longitud'])).miles**2
    return total_distance

# Coordenadas iniciales (podríamos empezar por el centro de gravedad calculado en coordenadas cartesianas)
initial_coords = [20.5, -98.5]  # Estas son coordenadas aproximadas para el centro de México

# Utilizar el algoritmo de Nelder-Mead para minimizar la función objetivo
res = minimize(objective_function, initial_coords, method='Nelder-Mead')

# Las coordenadas optimizadas son el resultado de la optimización
optimized_coords = res.x

# Calcular la media de las latitudes y las longitudes para centrar el mapa
mediaLong = statistics.mean(data['Longitud'])
mediaLat = statistics.mean(data['Latitud'])

# Crear un objeto de mapa base Map()
mapa = folium.Map(location=[mediaLat, mediaLong], zoom_start = 9)

# Crear una capa de mapa de calor
mapa_calor = HeatMap(list(zip(data['Latitud'], data['Longitud'], data['Peso'])), min_opacity=0.2, max_val=data['Peso'].max(),radius=50, blur=50, max_zoom=1)

# Crear el marcador de Centro de Gravedad
tooltip = 'Centro de gravedad'
folium.Marker(optimized_coords, popup="Centro de gravedad", tooltip="Centro de gravedad").add_to(mapa)

# Adherir la capa de mapa de calor al mapa principal
mapa_calor.add_to(mapa)

mapa
