import glob
import numpy as np
from brkga_mp_ipr.enums import Sense
from brkga_mp_ipr.algorithm import BrkgaMpIpr
from brkga_mp_ipr.types_io import load_configuration
from brkga_mp_ipr.types import BaseChromosome

nomeInstancias = glob.glob('Instancias/000_teste.txt')

def leituraInstancia(nomeInstancia):
    arqv = open(nomeInstancia, "r")

    numVertices = int(arqv.readline())
    matrizCustos = np.empty((numVertices, numVertices), dtype=int)
    
    for i in range(numVertices):
        linha = list(map(int, arqv.readline().split()))
        for j in range(numVertices):
            matrizCustos[i][j] = linha[j]
    
    arqv.readline()  # Linha vazia entre a matriz e os prazos
    prazos = list(map(int, arqv.readline().split()))

    arestas = [(i, j) for i in range(numVertices) for j in range(numVertices) if matrizCustos[i][j] != 0]

    return numVertices, matrizCustos, arestas, prazos


class TSPDeadlineDecoder:
    def __init__(self, numVertices, matrizCustos, prazos):
        self.numVertices = numVertices
        self.matrizCustos = matrizCustos
        self.prazos = prazos
        self.M = [[max(prazos[i] + matrizCustos[i][j], 0) for j in range(numVertices)] for i in range(numVertices)]

    def decode(self, chromosome: BaseChromosome, rewrite: bool = False) -> float:
        permutation = [0] + sorted(range(1, len(chromosome)), key=lambda k: chromosome[k]) + [0]
        
        total_cost = 0
        time_elapsed = 0
        t = [0] * self.numVertices  # Tempos de chegada
        x = np.zeros((self.numVertices, self.numVertices))  # Variáveis de decisão

        # Implementando restrições de fluxo de entrada e saída dos vértices
        for j in range(self.numVertices):
            incoming = sum([x[i][j] for i in range(self.numVertices) if self.matrizCustos[i][j] > 0])
            if incoming != 1:
                return float('inf')

        for i in range(self.numVertices):
            outgoing = sum([x[i][j] for j in range(self.numVertices) if self.matrizCustos[i][j] > 0])
            if outgoing != 1:
                return float('inf')

        # Percorrendo a permutação para calcular custos e verificar restrições
        for k in range(len(permutation) - 1):
            u = permutation[k]
            v = permutation[k + 1]

            # Verifica se a aresta existe
            if self.matrizCustos[u][v] <= 0:
                return float('inf')

            x[u][v] = 1  # Atualiza a variável de decisão para a aresta utilizada
            time_elapsed += self.matrizCustos[u][v]
            t[v] = time_elapsed

            # Respeitar as restrições de prazo
            if t[v] > self.prazos[v]:
                return float('inf')

            # Eliminação de subrotas
            if u != 0 and v != 0:
                if t[u] + self.matrizCustos[u][v] - t[v] > self.M[u][v] * (1 - x[u][v]):
                    return float('inf')

            total_cost += self.matrizCustos[u][v]

        # Verificar se todos os vértices foram visitados
        for i in range(self.numVertices):
            if sum([x[i][j] for j in range(self.numVertices)]) != 1:
                return float('inf')
            if sum([x[j][i] for j in range(self.numVertices)]) != 1:
                return float('inf')

        return total_cost

def main():
    arqv = open("testes.txt", "w")

    for instancia in nomeInstancias:
        numVertices, custos, arestas, prazos = leituraInstancia(instancia)

        print(f"\n\nResolução do Caixeiro Viajante com Prazos para a instância: {instancia}")
        arqv.write(f"\n\nResolução do Caixeiro Viajante com Prazos para a instância: {instancia}\n")

        # Criar o decodificador
        decoder = TSPDeadlineDecoder(numVertices, custos, prazos)

        # Carregar a configuração do BRKGA
        brkga_params, _ = load_configuration("config.conf")

        # Configuração do BRKGA
        brkga = BrkgaMpIpr(
            decoder=decoder,
            sense=Sense.MINIMIZE,
            seed=1234,  # Semente para o gerador de números aleatórios
            chromosome_size=numVertices,
            params=brkga_params
        )

        brkga.initialize()

        # Evolução do BRKGA por um número fixo de gerações
        num_generations = 100
        brkga.evolve(num_generations)

        best_cost = brkga.get_best_fitness()
        best_chromosome = brkga.get_best_chromosome()

        print(f"Melhor custo encontrado: {best_cost}")
        arqv.write(f"Melhor custo encontrado: {best_cost}\n")
        
        # Mostrar a melhor rota encontrada
        best_permutation = sorted(range(len(best_chromosome)), key=lambda k: best_chromosome[k])
        print(f"Melhor rota: {best_permutation}")
        arqv.write(f"Melhor rota: {best_permutation}\n")

    arqv.close()

if __name__ == "__main__":
    main()
