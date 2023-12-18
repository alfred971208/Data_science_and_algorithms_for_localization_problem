from deap import base, creator, tools, algorithms
import random
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Tus datos
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

# Constantes
COST_MULTIPLIER = 1000

# Crear el tipo de fitness y el tipo de individuo
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Definición de las funciones
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # radio de la Tierra en km
    dlon = radians(lon2) - radians(lon1)
    dlat = radians(lat2) - radians(lat1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def calcular_aptitud(individual):
    distancia_total, demanda_total, costo_total = calcular_distancia_demanda_costo(individual)
    return distancia_total, costo_total

def calcular_distancia_demanda_costo(individual):
    distancia_total = 0
    demanda_total = [0] * len(instalaciones)
    capacidad_total = [instalacion["Producción"] for instalacion in instalaciones]
    costo_total = 0

    for i, cliente in enumerate(clientes):
        instalacion_seleccionada = instalaciones[individual[i] % len(instalaciones)]
        distancia_total += haversine(instalacion_seleccionada["Latitud"],
                                    instalacion_seleccionada["Longitud"],
                                    cliente["Latitud"],
                                    cliente["Longitud"])
        demanda_total[individual[i] % len(instalaciones)] += cliente["Demanda"]

    for i in range(len(instalaciones)):
        if demanda_total[i] > capacidad_total[i]:
            costo_total += (demanda_total[i] - capacidad_total[i]) * COST_MULTIPLIER
    return distancia_total, demanda_total, costo_total

def proximidad_geo():
    X = [[cliente["Latitud"], cliente["Longitud"]] for cliente in clientes]
    kmeans = KMeans(n_clusters=len(instalaciones), n_init=10).fit(X)
    individual = kmeans.labels_
    return individual.tolist()

def crear_individuo():
    return proximidad_geo()

def custom_mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(0, len(instalaciones) - 1)
    return individual,

def custom_mate(ind1, ind2):
    for i in range(len(ind1)):
        if abs(instalaciones[ind1[i]]["Latitud"] - instalaciones[ind2[i]]["Latitud"]) < 0.05 and abs(instalaciones[ind1[i]]["Longitud"] - instalaciones[ind2[i]]["Longitud"]) < 0.05:
            if clientes[i]["Demanda"] <= instalaciones[ind2[i]]['Producción'] and clientes[i]["Demanda"] <= instalaciones[ind1[i]]['Producción']:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, crear_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate, indpb=0.05)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", calcular_aptitud)

tamanio_poblacion = 500
probabilidad_cruce = 0.9
probabilidad_mutacion = 0.01
generaciones = 1000

def init_population():
    return toolbox.population(n=tamanio_poblacion)

def evaluate_population(poblacion):
    fitnesses = list(map(toolbox.evaluate, poblacion))
    for ind, fit in zip(poblacion, fitnesses):
        ind.fitness.values = fit
    return poblacion

def perform_evolution(poblacion):
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "std", "min", "max"

    # Evaluar la población inicial
    poblacion = evaluate_population(poblacion)

    # Actualizar la logbook y el Frente de Pareto con la población inicial
    record = stats.compile(poblacion)
    logbook.record(gen=0, **record)
    hof.update(poblacion)

    # Comenzar la evolución
    for gen in range(1, generaciones + 1):
        offspring = algorithms.varOr(poblacion, toolbox, lambda_=tamanio_poblacion, cxpb=probabilidad_cruce, mutpb=probabilidad_mutacion)

        # Evaluar los individuos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Actualizar la población
        poblacion[:] = toolbox.select(offspring + poblacion, k=tamanio_poblacion)

        # Actualizar la logbook y el Frente de Pareto
        hof.update(poblacion)
        record = stats.compile(poblacion)
        logbook.record(gen=gen, **record)

    return poblacion, hof, logbook

def main():
    poblacion = init_population()
    poblacion, hof, logbook = perform_evolution(poblacion)

    # Imprimir el mejor individuo encontrado
    mejor_ind = tools.selBest(poblacion, 1)[0]
    print("Mejor individuo es ", mejor_ind, " con distancia total: ", mejor_ind.fitness.values[0], "y costo total: ", mejor_ind.fitness.values[1])

    # Imprimir la mejor solución encontrada
    print("La mejor solución es:")
    for i, cliente in enumerate(clientes):
        print("Cliente", i, "asignado a la instalación", mejor_ind[i])

    # Mostrar las rutas de la solución
    for i, cliente in enumerate(clientes):
        print(f"Cliente {cliente['Municipio']} atendido por {instalaciones[mejor_ind[i]]['Municipio']}")

    # Grafica la evolución
    gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
    avg_dist = [a[0] for a in avg]
    avg_cost = [a[1] for a in avg]
    max_dist = [m[0] for m in max_]
    max_cost = [m[1] for m in max_]
    plt.figure()
    plt.semilogy(gen, avg_dist, label="Promedio Distancia")
    plt.semilogy(gen, avg_cost, label="Promedio Costo")
    plt.semilogy(gen, max_dist, label="Máxima Distancia")
    plt.semilogy(gen, max_cost, label="Máximo Costo")
    plt.xlabel("Generación")
    plt.ylabel("Valor")
    plt.legend()
    plt.show()


    return poblacion, logbook, hof
def visualizar_resultados(mejor_ind):
    # Crear un mapa centrado en la ubicación media de todas las instalaciones y clientes
    lat_media = sum(inst["Latitud"] for inst in instalaciones + clientes) / len(instalaciones + clientes)
    lon_media = sum(inst["Longitud"] for inst in instalaciones + clientes) / len(instalaciones + clientes)
    mapa = folium.Map(location=[lat_media, lon_media], zoom_start=6)

    # Agregar marcadores para las instalaciones
    for i, inst in enumerate(instalaciones):
        folium.Marker([inst["Latitud"], inst["Longitud"]], popup=f"Instalación {i}", icon=folium.Icon(color="green")).add_to(mapa)

    # Agregar marcadores para los clientes y líneas para representar las asignaciones de clientes a instalaciones
    for i, cliente in enumerate(clientes):
        folium.Marker([cliente["Latitud"], cliente["Longitud"]], popup=f"Cliente {i}").add_to(mapa)
        folium.PolyLine([(cliente["Latitud"], cliente["Longitud"]), (instalaciones[mejor_ind[i]]["Latitud"], instalaciones[mejor_ind[i]]["Longitud"])], color="blue").add_to(mapa)

    # Mostrar el mapa
    return mapa

if __name__ == "__main__":
    poblacion, logbook, hof = main()
    mejor_ind = hof[0]
    mapa = visualizar_resultados(mejor_ind)
    display(mapa)
