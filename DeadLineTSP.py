import glob
import numpy as np
from pulp import *
import time

nomeInstancias = glob.glob('Instancias\*.txt')

def leituraInstancia(nomeInstancia):
    arqv = open(nomeInstancia, "r")

    numVertices = int(arqv.readline())
    matrizCustos = np.empty((numVertices, numVertices), dtype=int)
    
    for i in range (numVertices):
        linha = list(map(int, arqv.readline().split()))
        for j in range (numVertices):
            matrizCustos[i][j] = linha[j]
    
    arqv.readline()
    prazos = list(map(int, arqv.readline().split()))

    arestas = [(i, j) for i in range(numVertices) for j in range(numVertices) if matrizCustos[i][j] != 0]

    return numVertices, matrizCustos, arestas, prazos

def main():
    arqv = open("testes.txt", "w")
    tempoLimite = 2

    for instancia in nomeInstancias:
        numVertices, custos, arestas, prazos = leituraInstancia(instancia)

        print(f"\n\nResolucao do Caixeiro Viajante com Prazos para a instancia: {instancia}")
        arqv.write(f"\n\nResolucao do Caixeiro Viajante com Prazos para a instancia: {instancia}\n")

        #Inicializando o LP
        dl_tsp = LpProblem("Caxeiro_Viajante_com_Prazos", LpMinimize)

        #Variaveis de decisao
        x = LpVariable.dicts("x", arestas, cat = "Binary")
        t = LpVariable.dicts("t", [i for i in range(numVertices)], lowBound = 0, upBound = None, cat = "Integer")

        #Funcao objetivo
        dl_tsp += lpSum([custos[i][j] * x[i, j] for (i, j) in arestas])

        #Restricoes de fluxo de entrada e saida dos vertices
        for j in range(numVertices):
            dl_tsp += lpSum([x[i, j] for (i, u) in arestas if u == j]) == 1

        for i in range(numVertices):
            dl_tsp += lpSum([x[i, j] for (u, j) in arestas if u == i]) == 1

        #Restricao de eliminação de subrotas
        M = [[max(prazos[i] + custos[i][j], 0) for j in range(numVertices)] for i in range(numVertices)]
        #M = 100000

        for (i, j) in arestas:
            if j > 0:
                dl_tsp += t[i] + custos[i][j] - t[j] <= M[i][j] * (1 - x[i, j])

        #Restricao de prazo
        for i in range(numVertices):
            if prazos[i] > 0:
                dl_tsp += t[i] <= prazos[i]
        
        #Resolvendo 
        resolucao = dl_tsp.solve(PULP_CBC_CMD(timeLimit=1200, msg=False))
        
        if LpStatus[resolucao] == "Optimal":
            print(f"Status do problema: {LpStatus[resolucao]}")
            arqv.write(f"Status do problema: {LpStatus[resolucao]}\n")

            for var in dl_tsp.variables():
                if var.varValue > 0:
                    print(f"{var.name} = {var.varValue}")
                    arqv.write(f"{var.name} = {var.varValue}\n")

            print(f"Tempo total de percurso = {value(dl_tsp.objective)}")
            arqv.write(f"Tempo total de percurso = {value(dl_tsp.objective)}\n")
        elif LpStatus[resolucao] == "Not Solved":
            print("Nenhuma solucao foi encontrada dentro do tempo limite de 20min.")
            arqv.write("Nenhuma solucao foi encontrada dentro do tempo limite de 20min.\n")

            print(f"Status do problema: {LpStatus[resolucao]}")
            arqv.write(f"Status do problema: {LpStatus[resolucao]}\n")

            for var in dl_tsp.variables():
                if var.varValue > 0:
                    print(f"{var.name} = {var.varValue}")
                    arqv.write(f"{var.name} = {var.varValue}\n")

            print(f"Tempo total de percurso = {value(dl_tsp.objective)}")
            arqv.write(f"Tempo total de percurso = {value(dl_tsp.objective)}\n")
        else:
            print("Nenhuma solucaoo foi encontrada.")
            arqv.write("Nenhuma solucao foi encontrada.\n")


main()