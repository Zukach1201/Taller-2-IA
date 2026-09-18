from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    #Separamos la función recursiva con la que la llama para que el código se vea más limpio y ordenado.
    #Encuentra la función get_action más abajo.

    def _minimax_value(self, state: GameState, depth: int, agent_index: int) -> float:
        """
        Calcula el valor Minimax recursivamente según las especificaciones del taller.
        """
        self.nodes_evaluated += 1

        # Caso base: profundidad agotada o estado final (victoria/derrota)
        if depth == 0 or state.is_win() or state.is_lose():
            result_value = evaluation_function(state)
        else:
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                result_value = evaluation_function(state)
            else:
                next_agent = (agent_index + 1) % state.get_num_agents()
                next_depth = depth - 1

                if agent_index == 0:  # MAX (Defensor)
                    max_val = float('-inf')
                    for action in legal_actions:
                        successor = state.generate_successor(agent_index, action)
                        val = self._minimax_value(successor, next_depth, next_agent)
                        if val > max_val:
                            max_val = val
                    result_value = max_val
                else:  # MIN (Intruso)
                    min_val = float('inf')
                    for action in legal_actions:
                        successor = state.generate_successor(agent_index, action)
                        val = self._minimax_value(successor, next_depth, next_agent)
                        if val < min_val:
                            min_val = val
                    result_value = min_val

        return result_value

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        self.nodes_evaluated = 0

        # Contar la raíz como estado procesado
        self.nodes_evaluated += 1

        legal_actions = state.get_legal_actions(0)
        best_action = None

        if legal_actions:
            best_value = float('-inf')

            for action in legal_actions:
                successor = state.generate_successor(0, action)
                value = self._minimax_value(successor, self.depth - 1, 1)

                if value > best_value:
                    best_value = value
                    best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        # TODO: Add your code here
        raise NotImplementedError("Punto 5: implemente AlphaBetaAgent.get_action")
