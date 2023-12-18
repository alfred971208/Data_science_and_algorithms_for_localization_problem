from deap import base, creator, tools, algorithms
import random
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Tus datos
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
