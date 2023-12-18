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
    best_distance = float('inf')
    best_route = []
    best_location = None
    # Iterar sobre cada instalación.
    for i in range(len(instalaciones)):
        locations = [instalaciones[i]] + clientes
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
        # Comprobar si la solución es mejor que la mejor encontrada hasta ahora.
        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            if solution.ObjectiveValue() < best_distance:
                best_distance = solution.ObjectiveValue()
                best_route = route
                best_location = locations[0]
            print(f"Instalación: {locations[0]['Municipio']}, Distancia total de la ruta: {solution.ObjectiveValue()} km")
            m = folium.Map(location=[locations[0]['Latitud'], locations[0]['Longitud']], zoom_start=10)
            for i in route:
                location = locations[i]
                folium.Marker([location['Latitud'], location['Longitud']], popup=location['Municipio']).add_to(m)
            for i in range(len(route) - 1):
                loc1 = locations[route[i]]
                loc2 = locations[route[i+1]]
                folium.PolyLine([[loc1['Latitud'], loc1['Longitud']], [loc2['Latitud'], loc2['Longitud']]], color='blue').add_to(m)
            m.save(f'ruta_{locations[0]["Municipio"]}.html')
            display(m)

    print(f"Mejor instalación: {best_location['Municipio']}, Distancia total de la ruta: {best_distance} km")

main()
