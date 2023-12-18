import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from geopy.distance import geodesic
from scipy.optimize import minimize
import statistics
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

data = pd.DataFrame({
    'Municipio': ['Atlapexco', 'Chapulhuacán', 'Eloxochitlán', 'Huautla', 'Huazalingo', 'Huehuetla', 'Juárez Hidalgo', 'Lolotla', 'Molango de Escamilla', 'San Agustín Metzquititlán', 'San Bartolo Tutotepec', 'San Felipe Orizatlán', 'Tenango de Doria', 'Tlahuiltepa', 'Xochicoatlán'],
    'Latitud': [21.01745106, 21.15787458, 20.74687461, 21.03188243, 20.98056421, 20.46035677, 20.78338131, 20.84163503, 20.78658031, 20.53306527, 20.3991586, 21.17106016, 20.33845235, 20.92389198, 20.77704277],
    'Longitud': [-98.34829642, -98.90435814, -98.8093052, -98.28668517, -98.50774232, -98.07859535, -98.82918225, -98.71711415, -98.7306035, -98.63880134, -98.20205941, -98.60758905, -98.2267008, -98.95017064, -98.67966839],
    'Peso': [70.18432667, 2076.831214, 535.8706135, 1531.505859, 45.81143524, 1136.262735, 317.6627911, 593.5610713, 769.5643214, 25.68872557, 1635.505363, 3193.503324, 393.5998613, 236.2399636, 164.6085326]
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
