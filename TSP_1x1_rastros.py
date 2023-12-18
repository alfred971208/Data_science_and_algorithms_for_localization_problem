from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, cos, sin, asin, sqrt
import numpy as np
import folium

instalaciones = [
    {"Municipio": "San Bartolo Tutotepec", "Latitud": 20.3991586, "Longitud": -98.20205941},
    {"Municipio": "Huejutla de Reyes", "Latitud": 21.13959058, "Longitud": -98.42044493},
    {"Municipio": "Molango de Escamilla", "Latitud": 20.78658031, "Longitud": -98.7306035},
    {"Municipio": "Calnali", "Latitud": 20.8974302121529, "Longitud": -98.5835981618066},
    {"Municipio": "Tlanchinol", "Latitud": 20.9895151177999, "Longitud": -98.6602832194761}
]

clientes = [
    {"Municipio": "Chapantongo", "Latitud": 20.28559483, "Longitud": -99.41319277},
    {"Municipio": "Chapulhuacán", "Latitud": 21.15787458, "Longitud": -98.90435814},
    {"Municipio": "San Bartolo Tutotepec", "Latitud": 20.3991586, "Longitud": -98.20205941},
    {"Municipio": "Jacala de Ledezma", "Latitud": 21.00849043, "Longitud": -99.18853709},
    {"Municipio": "Huejutla de Reyes", "Latitud": 21.13959058, "Longitud": -98.42044493},
    {"Municipio": "Molango de Escamilla", "Latitud": 20.78658031, "Longitud": -98.7306035},
    {"Municipio": "Alfajayucan", "Latitud": 20.41015828, "Longitud": -99.34946458},
    {"Municipio": "Huichapan", "Latitud": 20.37553905, "Longitud": -99.6510293},
    {"Municipio": "Villa de Tezontepec", "Latitud": 19.879986, "Longitud": -98.81930736},
    {"Municipio": "Francisco I. Madero", "Latitud": 20.24540323, "Longitud": -99.08881409},
    {"Municipio": "Acatlán", "Latitud": 20.14599753, "Longitud": -98.43835281},
    {"Municipio": "Nicolás Flores", "Latitud": 20.76804674, "Longitud": -99.15056418},
    {"Municipio": "San Felipe Orizatlán", "Latitud": 21.17106016, "Longitud": -98.60758905},
    {"Municipio": "Tecozautla", "Latitud": 20.53415978, "Longitud": -99.6349425},
    {"Municipio": "Metztitlán", "Latitud": 20.5948676, "Longitud": -98.7642084}
]
# Función para crear los datos del problema.
def haversine(loc1, loc2):
    lon1, lat1 = loc1["Longitud"], loc1["Latitud"]
    lon2, lat2 = loc2["Longitud"], loc2["Latitud"]
    # Convertir grados a radianes.
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Fórmula de haversine.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radio de la tierra en kilómetros.
    return c * r

def create_data_model(locations):
    data = {}
    data['distance_matrix'] = [[0]*len(locations) for _ in range(len(locations))]
    for i in range(len(locations)):
        for j in range(len(locations)):
            data['distance_matrix'][i][j] = int(haversine(locations[i], locations[j]))
    return data


def main():
    best_instalacion = None  # Variable para almacenar la mejor instalación
    min_distance = float('inf')  # Inicializar la distancia mínima como infinito

    for i in range(len(instalaciones)):
        total_distance = 0
        for j in range(len(clientes)):
            locations = [instalaciones[i], clientes[j], instalaciones[i]]
            # Crear el modelo de datos.
            data = create_data_model(locations)
            # Crear el problema de enrutamiento.
            manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), 1, 0)
            routing = pywrapcp.RoutingModel(manager)
            def distance_callback(from_index, to_index):
                return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            # Resolver el problema.
            solution = routing.SolveWithParameters(search_parameters)
            # Calcular la distancia total de la ruta.
            if solution:
                total_distance += solution.ObjectiveValue()

        # Actualizar la mejor instalación si se encuentra una distancia menor.
        if total_distance < min_distance:
            min_distance = total_distance
            best_instalacion = instalaciones[i]

        # Imprimir la instalación y la distancia total de la ruta.
        print(f"Instalación: {instalaciones[i]['Municipio']}, Distancia total de la ruta: {total_distance} km")

    if best_instalacion is not None:
        print(f"\nMejor instalación: {best_instalacion['Municipio']}, Distancia total de la ruta: {min_distance} km")
    else:
        print("No se encontró una mejor instalación.")
    # Crear un mapa centrado en la mejor instalación
    mapa = folium.Map(location=[best_instalacion["Latitud"], best_instalacion["Longitud"]], zoom_start=10)

    # Agregar marcadores para las instalaciones
    for instalacion in instalaciones:
        distancia = haversine(best_instalacion, instalacion)
        folium.Marker(
            location=[instalacion["Latitud"], instalacion["Longitud"]],
            tooltip=f"Distancia: {distancia:.2f} km"
        ).add_to(mapa)

    # Agregar marcadores para los clientes
    for cliente in clientes:
        folium.Marker(
            location=[cliente["Latitud"], cliente["Longitud"]],
            tooltip=cliente["Municipio"]
        ).add_to(mapa)

    # Conectar la mejor instalación con los clientes utilizando líneas
    for cliente in clientes:
        folium.PolyLine(
            locations=[
                [best_instalacion["Latitud"], best_instalacion["Longitud"]],
                [cliente["Latitud"], cliente["Longitud"]]
            ],
            color="blue",
            weight=2,
            opacity=0.7
        ).add_to(mapa)

    # Mostrar el mapa
    display(mapa)

main()
