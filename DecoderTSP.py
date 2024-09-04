from brkga_mp_ipr.types import BaseChromosome
import numpy as np

class TSPDecoder:
    def __init__(self, numVertices, matrizCustos, prazos, arestas):
        self.numVertices = numVertices
        self.matrizCustos = matrizCustos
        self.prazos = prazos
        self.arestas = set(arestas)  
        self.M = 100000  # Valor absurdo q tava no PLI

    def decode(self, chromosome: BaseChromosome, rewrite: bool = False) -> float:
        # Força a rota começar e terminar em 0
        permutation = [0] + sorted(range(1, self.numVertices), key=lambda k: chromosome[k]) + [0]
        
        total_cost = 0
        time_elapsed = 0
        t = [0] * self.numVertices  # Tempos de chegada

        visited = set()
        visited.add(0)  # Começa em 0

        for k in range(len(permutation) - 1):
            u = permutation[k]
            v = permutation[k + 1]

            # Verifica se a aresta existe
            if (u, v) not in self.arestas:
                return float('inf')# Penalidade absurda

            # Calcula o tempo de chegada ao vértice v
            time_elapsed += self.matrizCustos[u][v]
            t[v] = time_elapsed

            # Respeitar as restrições de prazo
            if v != 0 and t[v] > self.prazos[v]:
                return float('inf')  # Penalidade absurda

            total_cost += self.matrizCustos[u][v]

            visited.add(v)

            # Restrições MTZ para eliminação de subrotas
            if v != 0:  # Ignora o vértice 0
                if t[u] + self.matrizCustos[u][v] - t[v] > self.M * (1 - int((u, v) in self.arestas)):
                    return float('inf')  # Penalidade por não atender à eliminação de subrotas

        # Garantir que a rota visite todos os vértices exatamente uma vez e retorne a 0
        if len(visited) != self.numVertices or permutation[0] != 0 or permutation[-1] != 0:
            return float('inf')

        return total_cost
