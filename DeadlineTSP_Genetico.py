import glob
from random import random, seed
import sys
import numpy as np
from brkga_mp_ipr.enums import Sense
from brkga_mp_ipr.algorithm import BrkgaMpIpr
from brkga_mp_ipr.types_io import load_configuration
import time

from DecoderTSP import TSPDecoder

nomeInstancias = glob.glob("NovasInstancias/*.txt")

def leituraInstancia(nomeInstancia):
    arqv = open(nomeInstancia, "r")

    numVertices = int(arqv.readline())
    matrizCustos = np.empty((numVertices, numVertices), dtype=int)
    
    for i in range(numVertices):
        linha = list(map(int, arqv.readline().split()))
        for j in range(numVertices):
            matrizCustos[i][j] = linha[j]
            # print("leitura", i,",", j, matrizCustos)
    arqv.readline()  # Linha vazia entre a matriz e os prazos
    prazos = list(map(int, arqv.readline().split()))

    arestas = [(i, j) for i in range(numVertices) for j in range(numVertices) if matrizCustos[i][j] != 0]

    return numVertices, matrizCustos, arestas, prazos




class StopRule:
    GENERATIONS = 0
    TARGET = 1
    IMPROVEMENT = 2

def main():
    arqv = open("testes.txt", "w")
    if len(sys.argv) < 3:
        print("Usage: python main_minimal.py <seed> <num_generations> ")
        sys.exit(1)

    ########################################
    # Read the command-line arguments and the instance
    ########################################

    seed = int(sys.argv[1])
    num_generations  = int(sys.argv[2])
    stop_rule = StopRule.GENERATIONS  

    for instancia in nomeInstancias:
        numVertices, custos, arestas, prazos = leituraInstancia(instancia)

        print(f"\n\nResolução do Caixeiro Viajante com Prazos para a instância: {instancia}")
        arqv.write(f"\n\nResolução do Caixeiro Viajante com Prazos para a instância: {instancia}\n")

    
        decoder = TSPDecoder(numVertices, custos, prazos,arestas)

        brkga_params, _ = load_configuration("config.conf")

        brkga = BrkgaMpIpr(
            decoder=decoder,
            sense=Sense.MINIMIZE,
            seed=  seed,  
            chromosome_size=numVertices,
            params=brkga_params
        )

        brkga.initialize()

        iter_without_improvement = 0
        best_cost = float('inf')
        target_cost = 0 

  
        start_time = time.time()
        maximum_time = 600  
        stop_argument = 50  # Número de iterações sem melhora

        for iteration in range(num_generations):
            brkga.evolve(1)
            current_best_cost = brkga.get_best_fitness()

            if current_best_cost < best_cost:
                best_cost = current_best_cost
                iter_without_improvement = 0  # Reset contador se houve melhora

                found_cost = time.time() - start_time
            else:
                iter_without_improvement += 1

            print(f"Iteração {iteration}: {iter_without_improvement} iterações sem melhoria")

            # Verificar se atende a algum critério de parada
            if time.time() - start_time > maximum_time:
                print("Critério de tempo atingido. Parando a execução.")
                break  # Critério de tempo

            if stop_rule == StopRule.IMPROVEMENT or iter_without_improvement >= stop_argument: #devia ser um and ao inves de um or
                print(f"Critério de {stop_argument} iterações sem melhoria atingido. Parando a execução.")
                break  # Critério de iterações sem melhoria

            if stop_rule == StopRule.TARGET or best_cost <= target_cost:#devia ser um and ao inves de um or
                print(f"Critério de atingir o custo alvo {target_cost} atingido. Parando a execução.")
                break  # Critério de atingir o custo alvo


        print(f"Melhor custo encontrado: {best_cost}")
        arqv.write(f"Melhor custo encontrado: {best_cost}\n")
        
        # Mostrar a melhor rota encontrada
        best_chromosome = brkga.get_best_chromosome()
        best_permutation = sorted(range(len(best_chromosome)), key=lambda k: best_chromosome[k])
        print(f"Melhor rota: {best_permutation}")
        arqv.write(f"Melhor rota: {best_permutation}\n")

        print(f"Tempo de execução até encontrar o melhor custo: {found_cost}")
        arqv.write(f"Tempo de execução até encontrar o melhor custo: {found_cost}")

    arqv.close()

if __name__ == "__main__":
    main()