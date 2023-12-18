from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, cos, sin, asin, sqrt
from sklearn_extra.cluster import KMedoids
import numpy as np
import folium

instalaciones = [
    {"Municipio": "Chapantongo", "Latitud": 20.28559483, "Longitud": -99.41319277, "Producción": 6602.24979},
    {"Municipio": "Chapulhuacán", "Latitud": 21.15787458, "Longitud": -98.90435814, "Producción": 6602.24979},
    {"Municipio": "San Bartolo Tutotepec", "Latitud": 20.3991586, "Longitud": -98.20205941, "Producción": 6602.24979},
    {"Municipio": "Jacala de Ledezma", "Latitud": 21.00849043, "Longitud": -99.18853709, "Producción": 6602.24979},
    {"Municipio": "Huejutla de Reyes", "Latitud": 21.13959058, "Longitud": -98.42044493, "Producción": 6602.24979},
    {"Municipio": "Molango de Escamilla", "Latitud": 20.78658031, "Longitud": -98.7306035, "Producción": 6602.24979},
    {"Municipio": "Alfajayucan", "Latitud": 20.41015828, "Longitud": -99.34946458, "Producción": 6602.24979},
    {"Municipio": "Huichapan", "Latitud": 20.37553905, "Longitud": -99.6510293, "Producción": 6602.24979},
    {"Municipio": "Villa de Tezontepec", "Latitud": 19.879986, "Longitud": -98.81930736, "Producción": 6602.24979},
    {"Municipio": "Francisco I. Madero", "Latitud": 20.24540323, "Longitud": -99.08881409, "Producción": 6602.24979},
    {"Municipio": "Acatlán", "Latitud": 20.14599753, "Longitud": -98.43835281, "Producción": 6602.24979},
    {"Municipio": "Nicolás Flores", "Latitud": 20.76804674, "Longitud": -99.15056418, "Producción": 6602.24979},
    {"Municipio": "San Felipe Orizatlán", "Latitud": 21.17106016, "Longitud": -98.60758905, "Producción": 6602.24979},
    {"Municipio": "Tecozautla", "Latitud": 20.53415978, "Longitud": -99.6349425, "Producción": 6602.24979},
    {"Municipio": "Metztitlán", "Latitud": 20.5948676, "Longitud": -98.7642084, "Producción": 6602.24979}
]

clientes = [
    {"Municipio": "Actopan", "Latitud": 20.26918743, "Longitud": -98.94286206, "Demanda": 1339.69499},
    {"Municipio": "Apan", "Latitud": 19.71003178, "Longitud": -98.45236722, "Demanda": 2471.989578},
    {"Municipio": "Atitalaquia", "Latitud": 20.05929791, "Longitud": -99.22112947, "Demanda": 557.1169201},
    {"Municipio": "Atotonilco de Tula", "Latitud": 20.00039212, "Longitud": -99.21807773, "Demanda": 2713.440279},
    {"Municipio": "Emiliano Zapata", "Latitud": 19.65736929, "Longitud": -98.54730232, "Demanda": 449.384075},
    {"Municipio": "Epazoyucan", "Latitud": 20.01811094, "Longitud": -98.63604826, "Demanda": 602.1081288},
    {"Municipio": "Francisco I. Madero", "Latitud": 20.24540323, "Longitud": -99.08881409, "Demanda": 1237.102658},
    {"Municipio": "Huejutla de Reyes", "Latitud": 21.13959058, "Longitud": -98.42044493, "Demanda": 4979.113861},
    {"Municipio": "Ixmiquilpan", "Latitud": 20.4874612, "Longitud": -99.21584985, "Demanda": 2838.446607},
    {"Municipio": "Mineral de La Reforma", "Latitud": 20.07243478, "Longitud": -98.69588487, "Demanda": 15063.60747},
    {"Municipio": "Mineral del Monte", "Latitud": 20.14055695, "Longitud": -98.67147452, "Demanda": 408.9543905},
    {"Municipio": "Mixquiahuala de Juárez", "Latitud": 20.22971536, "Longitud": -99.21397966, "Demanda": 1416.818176},
    {"Municipio": "Pachuca de Soto", "Latitud": 20.12699971, "Longitud": -98.73005493, "Demanda": 22317.94682},
    {"Municipio": "Progreso de Obregón", "Latitud": 20.24868014, "Longitud": -99.18942359, "Demanda": 1158.828115},
    {"Municipio": "San Agustín Tlaxiaca", "Latitud": 20.1157998, "Longitud": -98.88675497, "Demanda": 1756.706179},
    {"Municipio": "San Salvador", "Latitud": 20.28495895, "Longitud": -99.0154511, "Demanda": 1244.233287},
    {"Municipio": "Santiago Tulantepec de Lugo Guerrero", "Latitud": 20.04050747, "Longitud": -98.35736774, "Demanda": 1837.631258},
    {"Municipio": "Tepeapulco", "Latitud": 19.7858894, "Longitud": -98.5530995, "Demanda": 2769.985565},
    {"Municipio": "Tepeji del Río de Ocampo", "Latitud": 19.90548931, "Longitud": -99.34182441, "Demanda": 2968.837365},
    {"Municipio": "Tezontepec de Aldama", "Latitud": 20.19289391, "Longitud": -99.27271678, "Demanda": 1921.076482},
    {"Municipio": "Tizayuca", "Latitud": 19.84126223, "Longitud": -98.98151696, "Demanda": 7102.832774},
    {"Municipio": "Tlahuelilpan", "Latitud": 20.13082664, "Longitud": -99.23456837, "Demanda": 389.8091821},
    {"Municipio": "Tlanalapa", "Latitud": 19.81786009, "Longitud": -98.60380072, "Demanda": 278.6898165},
    {"Municipio": "Tlaxcoapan", "Latitud": 20.09155632, "Longitud": -99.2213648, "Demanda": 353.5982297},
    {"Municipio": "Tolcayuca", "Latitud": 19.95673828, "Longitud": -98.9216934, "Demanda": 847.0972813},
    {"Municipio": "Tula de Allende", "Latitud": 20.07145808, "Longitud": -99.34480352, "Demanda": 4899.476703},
    {"Municipio": "Tulancingo de Bravo", "Latitud": 20.10754951, "Longitud": -98.38196463, "Demanda": 9277.88531},
    {"Municipio": "Zacualtipán de Ángeles", "Latitud": 20.64546431, "Longitud": -98.6535594, "Demanda": 2064.969542},
    {"Municipio": "Zapotlán de Juárez", "Latitud": 19.97414785, "Longitud": -98.8619187, "Demanda": 695.7076297},
    {"Municipio": "Zempoala", "Latitud": 19.91562419, "Longitud": -98.66811157, "Demanda": 3070.658113}
]

# Función para crear los datos del problema.
def haversine(loc1, loc2):
    lon1, lat1 = loc1['Longitud'], loc1['Latitud']
    lon2, lat2 = loc2['Longitud'], loc2['Latitud']
    # Convertir grados a radianes.
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Fórmula de haversine.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radio de la tierra en kilómetros.
    return c * r

def create_data_model(locations, depot):
    # Tu función create_data_model, pero ahora tiene un nuevo parámetro 'depot'
    # Las distancias desde y hacia todas las demás instalaciones se establecen en un valor muy alto para que no sean seleccionadas

    data = {}
    data['distance_matrix'] = [[0]*len(locations) for _ in range(len(locations))]
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i < len(instalaciones) and i != depot:  # Esta es una instalación que no es el depósito
                data['distance_matrix'][i][j] = 99999999
            else:
                data['distance_matrix'][i][j] = int(haversine(locations[i], locations[j]))
    # Convertir la matriz de distancias a un array de NumPy.
    data['distance_matrix'] = np.array(data['distance_matrix'])
    return data

def main():

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    client_locations = np.array([[c["Latitud"], c["Longitud"]] for c in clientes])
    depot_locations = np.array([[i["Latitud"], i["Longitud"]] for i in instalaciones])

    # Construye una lista de todas las ubicaciones como diccionarios
    all_locations_dict = instalaciones + clientes

    # Construye la matriz de distancia
    distance_matrix = np.array([[haversine(loc1, loc2) for loc2 in all_locations_dict] for loc1 in all_locations_dict])


    # Realiza el agrupamiento
    kmedoids = KMedoids(n_clusters=3, random_state=0).fit(distance_matrix)

    # Asigna los clientes a los grupos
    client_groups = {i: [] for i in range(3)}
    for i, label in enumerate(kmedoids.labels_[len(depot_locations):]):
        client_groups[label].append(clientes[i])

    for group_id, group in client_groups.items():
        print(f"Clúster {group_id + 1}:")
        total_demand = 0
        for client in group:
            print(f"\tMunicipio: {client['Municipio']}")
            total_demand += client['Demanda']
        print(f"\tDemanda total del clúster: {total_demand}")
        average_daily_demand = total_demand / 365 / 48
        print(f"\tDemanda diaria promedio por clúster: {average_daily_demand:.2f}")


    for depot in range(len(instalaciones)):
        total_distance = 0
        mapa = folium.Map(location=[20.503020, -98.212204], zoom_start=7)

        # Agregar un marcador para la instalación
        folium.Marker(
            location=[instalaciones[depot]["Latitud"], instalaciones[depot]["Longitud"]],
            icon=folium.Icon(color='blue'),
            popup=instalaciones[depot]["Municipio"]
        ).add_to(mapa)

        print(f'Instalación: {instalaciones[depot]["Municipio"]}')
        for group_id, group in enumerate(client_groups.values()):
            # Agregar un marcador para cada cliente en el grupo
            for client in group:
                folium.Marker(
                    location=[client["Latitud"], client["Longitud"]],
                    icon=folium.Icon(color='green'),
                    popup=client["Municipio"]
                ).add_to(mapa)



            num_vehicles = 1  # Solo un vehículo por grupo
            locations = [instalaciones[depot]] + group
            data = create_data_model(locations, 0)


            manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                   num_vehicles,
                                                   0)
            routing = pywrapcp.RoutingModel(manager)

            index_to_node = np.array([manager.IndexToNode(i) for i in range(manager.GetNumberOfIndices())])

            def distance_callback(from_index, to_index):
                return data['distance_matrix'][index_to_node[from_index]][index_to_node[to_index]]

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                index = routing.Start(0)
                itinerary = []
                vehicle_distance = 0

                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    next_node_index = manager.IndexToNode(solution.Value(routing.NextVar(index)))
                    vehicle_distance += haversine(locations[node_index], locations[next_node_index])
                    itinerary.append(locations[node_index])
                    index = solution.Value(routing.NextVar(index))
                itinerary.append(locations[manager.IndexToNode(index)])

                total_distance += vehicle_distance

                print(f'\tVehículo: {group_id + 1}, Distancia recorrida: {vehicle_distance} kilómetros')

                feature_group = folium.FeatureGroup(f'Grupo {group_id + 1}')
                route_color = colors[group_id % len(colors)]
                folium.PolyLine(locations=[[i["Latitud"], i["Longitud"]] for i in itinerary], color=route_color, weight=2.5, opacity=1).add_to(feature_group)
                feature_group.add_to(mapa)

        # Después de agregar todas las capas, agregar un control de capas al mapa.
        folium.LayerControl().add_to(mapa)

        print(f'\tDistancia total de recorrida: {total_distance} kilómetros')

        # Guardar y mostrar el mapa
        mapa.save(f'Instalacion_{depot+1}.html')
        display(mapa)

if __name__ == '__main__':
    main()
