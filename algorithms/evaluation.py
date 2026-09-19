import math

from world.game_state import GameState


def base_evaluation_function(state: GameState) -> float:
    """
    Retorna la evaluación base entregada para desarrollar el punto 4.

    Esta función no forma parte del código que debe modificar el estudiante y
    permite probar Minimax antes de desarrollar la heurística del punto 5.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    return float(state.get_score())


def evaluation_function(state: GameState) -> float:
    """
    Evalúa un estado desde la perspectiva del defensor MAX.

    Debe conservar las utilidades terminales de la evaluación base y diseñar
    una valoración no trivial para estados de corte. Minimax y alfa-beta usan
    esta misma función al comparar sus decisiones en el punto 5.

    Tips:
    - Los estados terminales ya se resuelven antes del bloque TODO; diseñe allí
      únicamente la valoración de estados no terminales.
    - Consulte state.defender_position, state.intruder_position,
      state.pending_terminals, state.get_score() y state.get_legal_actions(0).
    - state.layout.distance(start, goal) calcula y almacena en caché la distancia
      real por el mapa respetando los muros.
    - Maneje conjuntos vacíos y distancias infinitas, y mantenga todo estado no
      terminal estrictamente entre -1000 y +1000.
    """
    if state.is_win() or state.is_lose():
        return base_evaluation_function(state)

    distancia_inalcanzable = state.layout.height * state.layout.width
    puntaje_acumulado = state.get_score()
    n_terminales_pendientes = len(state.pending_terminals)

    # Distancia real por el mapa al terminal pendiente mas cercano, evitando infinitos
    distancias_a_terminales = [
        state.layout.distance(state.defender_position, terminal)
        for terminal in state.pending_terminals
    ]
    distancias_finitas = [d for d in distancias_a_terminales if math.isfinite(d)]
    distancia_al_terminal_cercano = min(distancias_finitas, default=distancia_inalcanzable)

    distancia_intruso_defensor = state.layout.distance(
        state.intruder_position, state.defender_position
    )
    if not math.isfinite(distancia_intruso_defensor):
        distancia_intruso_defensor = distancia_inalcanzable

    riesgo_captura_inmediata = 1.0 if distancia_intruso_defensor <= 1 else 0.0
    movilidad_defensor = len(state.get_legal_actions(0))

    # El margen de seguridad satura: mas alla de seis pasos el intruso ya no condiciona
    margen_seguridad = min(distancia_intruso_defensor, 6)

    valor_heuristico_crudo = (
        puntaje_acumulado
        - 100.0 * n_terminales_pendientes
        - 10.0 * distancia_al_terminal_cercano
        + 8.0 * margen_seguridad
        - 60.0 * riesgo_captura_inmediata
        + 2.0 * movilidad_defensor
    )

    # tanh comprime de forma monotona y garantiza el rango abierto exigido (-1000, 1000)
    return 999.0 * math.tanh(valor_heuristico_crudo / 500.0)
