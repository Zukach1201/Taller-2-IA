import math
import random

from optimization.problem import SmartGridOptimizationProblem
from optimization.result import Configuration, OptimizationResult


def configuration_score(
    problem: SmartGridOptimizationProblem, configuration: Configuration
) -> float:
    """
    Combina cobertura, redundancia y exposición en un puntaje a maximizar.

    Tips:
    - Use problem.score_components(configuration); ya retorna cobertura,
      redundancia y exposición en ese orden.
    """
    cobertura, redundancia, exposicion = problem.score_components(configuration)
    return cobertura - redundancia - exposicion


def hill_climbing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    max_iterations: int = 500,
) -> OptimizationResult:
    """
    Ejecuta ascenso de colina con mejora estricta.

    Debe examinar todos los vecinos, seleccionar el de mayor puntaje y
    conservar el orden entregado por el problema para desempatar. La búsqueda
    termina cuando no existe una mejora estricta o se alcanza el límite.

    Tips:
    - problem.neighbors(current) retorna vecinos válidos en el orden que debe
      usarse para desempatar.
    - Cada llamada a configuration_score(...) cuenta como una evaluación.
    - Inicialice los historiales con la configuración inicial y agregue solo las
      mejoras aceptadas antes de retornar el OptimizationResult.
    """
    current_config = initial_configuration
    current_score = configuration_score(problem, current_config)
    evaluations = 1  # Evaluación de la configuración inicial
    
    history = [current_config]
    iterations = 0
    mejora_encontrada = True

    while iterations < max_iterations and mejora_encontrada:
        neighbors = problem.neighbors(current_config)
        
        if not neighbors:
            mejora_encontrada = False
        else:
            best_neighbor = None
            best_neighbor_score = float('-inf')

            for neighbor in neighbors:
                score = configuration_score(problem, neighbor)
                evaluations += 1
                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = neighbor

            if best_neighbor_score > current_score:
                current_config = best_neighbor
                current_score = best_neighbor_score
                history.append(current_config)
                iterations += 1
            else:
                mejora_encontrada = False

    return OptimizationResult(
        best_configuration=current_config,
        best_score=current_score,
        evaluations=evaluations,
        iterations=iterations,
        history=history,
    )


def cooling_schedule(initial_temperature: float, cooling_rate: float, iteration: int) -> float:
    """
    Retorna el programa geométrico T(t) = T0 * alpha**t.

    Esta función se invoca desde simulated_annealing en cada iteración.
    """
    return initial_temperature * (cooling_rate ** iteration)


def simulated_annealing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    initial_temperature: float = 20.0,
    cooling_rate: float = 0.97,
    max_iterations: int = 500,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta recocido simulado para un problema de maximización.

    Debe proponer un vecino aleatorio por iteración, aceptar siempre las
    mejoras y aplicar exp(delta / temperature) en los demás casos. El estado
    actual y el mejor estado encontrado deben conservarse por separado.

    Tips:
    - Seleccione el candidato con rng.choice(problem.neighbors(current)) y use
      exclusivamente rng para conservar la reproducibilidad.
    - Obtenga la temperatura con cooling_schedule(...) y calcule la aceptación
      con delta = puntaje_candidato - puntaje_actual y math.exp(...).
    - Mantenga separados el estado actual y el mejor encontrado; registre el
      estado actual después de cada intento, incluso si se rechaza.
    - Detenga la ejecución cuando la temperatura alcance minimum_temperature.
    """
    rng = rng or random.Random()
    minimum_temperature = 1e-9


    current_config = initial_configuration
    current_score = configuration_score(problem, current_config)
    
    best_config = current_config
    best_score = current_score

    evaluations = 1  
    history = [current_config]
    iterations = 0

    temperature = cooling_schedule(initial_temperature, cooling_rate, iterations)

    while iterations < max_iterations and temperature > minimum_temperature:
        neighbors = problem.neighbors(current_config)
        
        if neighbors:
            # En este el vecino es aleatorio si no estoy mal
            candidate = rng.choice(neighbors)
            candidate_score = configuration_score(problem, candidate)
            evaluations += 1

            delta = candidate_score - current_score


            if delta > 0 or rng.random() < math.exp(delta / temperature):
                current_config = candidate
                current_score = candidate_score

                if current_score > best_score:
                    best_config = current_config
                    best_score = current_score

        history.append(current_config)
        iterations += 1
        temperature = cooling_schedule(initial_temperature, cooling_rate, iterations)

    return OptimizationResult(
        best_configuration=best_config,
        best_score=best_score,
        evaluations=evaluations,
        iterations=iterations,
        history=history,
    )


def one_point_crossover(
    parent1: Configuration, parent2: Configuration, rng: random.Random
) -> tuple[Configuration, Configuration]:
    """
    Realiza un cruce de un punto y retorna dos descendientes.

    La reparación de la cantidad de módulos se realiza posteriormente.

    Tips:
    - Seleccione con rng un corte interior, entre las posiciones 1 y len-1.
    - Cada descendiente combina el prefijo de un padre con el sufijo del otro.
    - Retorne tuplas y no repare aquí los descendientes.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud")
    if len(parent1) < 2:
        return parent1, parent2

    # Aquí decidimos hacer el corte interior aleatorio, así garantizamos que ambos padres aporten material genetico
    punto_de_corte = rng.randint(1, len(parent1) - 1)

    # Cada hijo toma el prefijo de un padre y el sufijo del otro
    primer_descendiente = parent1[:punto_de_corte] + parent2[punto_de_corte:]
    segundo_descendiente = parent2[:punto_de_corte] + parent1[punto_de_corte:]
    return primer_descendiente, segundo_descendiente


def swap_mutation(
    individual: Configuration, mutation_probability: float, rng: random.Random
) -> Configuration:
    """
    Aplica mutación por intercambio con la probabilidad indicada.

    Cuando ocurre una mutación, intercambia un bit activo y uno inactivo para
    conservar la cantidad de módulos instalados.

    Tips:
    - Use rng.random() para decidir si se aplica la mutación.
    - Identifique por separado los índices activos e inactivos y seleccione uno
      de cada grupo con rng.choice(...).
    - Si alguno de los dos grupos está vacío, no hay un intercambio posible.
    - Retorne una tupla nueva; no modifique el individuo recibido.
    """
    # Esta es la mutacion, que solo ocurre con la probabilidad indicada por cromosoma
    if rng.random() >= mutation_probability:
        return tuple(individual)

    indices_modulos_activos = [posicion for posicion, bit in enumerate(individual) if bit]
    indices_modulos_inactivos = [posicion for posicion, bit in enumerate(individual) if not bit]

    # Sin ambos grupos no existe intercambio posible y el cromosoma queda sin cambios
    if not indices_modulos_activos or not indices_modulos_inactivos:
        return tuple(individual)

    posicion_a_apagar = rng.choice(indices_modulos_activos)
    posicion_a_encender = rng.choice(indices_modulos_inactivos)

    #Aquí intercambiamos un bit activo por uno inactivo. Lo que buscábamos era conservar la cantidad de módulos. 
    cromosoma_mutado = list(individual)
    cromosoma_mutado[posicion_a_apagar] = 0
    cromosoma_mutado[posicion_a_encender] = 1
    return tuple(cromosoma_mutado)


def genetic_algorithm(
    problem: SmartGridOptimizationProblem,
    population_size: int = 40,
    generations: int = 100,
    mutation_probability: float = 0.05,
    elite_size: int = 2,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta un algoritmo genético generacional.

    Debe integrar la población inicial, la selección por torneo, el cruce, la
    reparación, la mutación y el elitismo entregados por el proyecto. Retorna
    el mejor individuo encontrado durante toda la ejecución.

    Tips:
    - Use problem.initial_population(...), problem.tournament_select(...) y
      problem.repair_configuration(...) para las operaciones ya entregadas.
    - Aplique one_point_crossover(...) antes de reparar y swap_mutation(...)
      después de la reparación.
    - Conserve los mejores individuos por elitismo y registre en los historiales
      el mejor global de cada generación.
    """
    rng = rng or random.Random()
    if population_size < 2:
        raise ValueError("La población debe tener al menos dos individuos")
    if generations < 0:
        raise ValueError("El número de generaciones no puede ser negativo")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= elite_size <= population_size:
        raise ValueError("elite_size debe estar entre 0 y population_size")

    poblacion_actual = problem.initial_population(population_size, rng)
    puntajes_poblacion = [
        configuration_score(problem, cromosoma) for cromosoma in poblacion_actual
    ]
    evaluaciones_totales = population_size

    # El mejor global se guarda aparte para no perderlo si una generacion empeora
    indice_mejor_inicial = max(
        range(population_size), key=lambda posicion: puntajes_poblacion[posicion]
    )
    mejor_configuracion_global = poblacion_actual[indice_mejor_inicial]
    mejor_puntaje_global = puntajes_poblacion[indice_mejor_inicial]

    history = [mejor_configuracion_global]
    score_history = [mejor_puntaje_global]
    generaciones_ejecutadas = 0

    for _ in range(generations):
        # Aqui aplique Elitismo, que básicamente son los mejores individuos pasan intactos y encabezan la nueva poblacion
        orden_por_aptitud = sorted(
            range(len(poblacion_actual)),
            key=lambda posicion: puntajes_poblacion[posicion],
            reverse=True,
        )
        nueva_poblacion = [
            poblacion_actual[posicion] for posicion in orden_por_aptitud[:elite_size]
        ]

        while len(nueva_poblacion) < population_size:
            primer_padre = problem.tournament_select(
                poblacion_actual, puntajes_poblacion, rng
            )
            segundo_padre = problem.tournament_select(
                poblacion_actual, puntajes_poblacion, rng
            )
            descendientes = one_point_crossover(primer_padre, segundo_padre, rng)

            # Este es el orden obligatorio por descendiente: primero reparar y luego mutar
            for descendiente in descendientes:
                if len(nueva_poblacion) >= population_size:
                    break
                descendiente_reparado = problem.repair_configuration(descendiente, rng)
                nueva_poblacion.append(
                    swap_mutation(descendiente_reparado, mutation_probability, rng)
                )

        # Acá es donde hacemos todo el reemplazo generacional completo, es decir, los hijos sustituyen a la poblacion anterior
        poblacion_actual = nueva_poblacion
        puntajes_poblacion = [
            configuration_score(problem, cromosoma) for cromosoma in poblacion_actual
        ]
        evaluaciones_totales += population_size

        indice_mejor_generacion = max(
            range(len(poblacion_actual)),
            key=lambda posicion: puntajes_poblacion[posicion],
        )
        if puntajes_poblacion[indice_mejor_generacion] > mejor_puntaje_global:
            mejor_puntaje_global = puntajes_poblacion[indice_mejor_generacion]
            mejor_configuracion_global = poblacion_actual[indice_mejor_generacion]

        generaciones_ejecutadas += 1
        history.append(mejor_configuracion_global)
        score_history.append(mejor_puntaje_global)

    return OptimizationResult(
        best_configuration=mejor_configuracion_global,
        best_score=mejor_puntaje_global,
        evaluations=evaluaciones_totales,
        iterations=generaciones_ejecutadas,
        history=history,
        score_history=score_history,
    )

