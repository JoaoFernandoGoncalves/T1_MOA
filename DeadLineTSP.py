import glob
import numpy as np
from pulp import *

nomeInstancias = glob.glob('Instancias\*.txt')

def leituraInstancia(nomeInstancia):
    arqv = open(nomeInstancia, "r")

    numVertices = int(arqv.readline())
    matrizCustos = np.empty((21,21))
    
    for i in range (numVertices):
        linha = list(map(int, arqv.readline().split()))
        for j in range (numVertices):
            matrizCustos[i][j] = linha[j]
    
    arqv.readline()
    prazos = list(map(int, arqv.readline().split()))

    arestas = [(i, j) for i in range(numVertices) for j in range(numVertices) if matrizCustos[i][j] != 0]

    return numVertices, matrizCustos, arestas, prazos

#TESTE

numVertices, custos, arestas, prazos = leituraInstancia(nomeInstancias[0])

#Inicializando o LP
dl_tsp = LpProblem("Caxeiro_Viajante_com_Prazos", LpMinimize)

#Variaveis de decisao
x = LpVariable.dicts("x", arestas, cat = "Binary")
t = LpVariable.dicts("t", [i for i in range(numVertices)], lowBound = 1, upBound = numVertices, cat = "Continuous")

#Funcao objetivo
dl_tsp += lpSum([custos[i][j] * x[i, j] for (i, j) in arestas])

#Restricoes de fluxo de saida dos vertices
for j in range(numVertices):
    dl_tsp += lpSum([x[i, j] for (i, u) in arestas if u == j]) == 1

for i in range(numVertices):
    dl_tsp += lpSum([x[i, j] for (u, j) in arestas if u == i]) == 1

#Restricao de eliminação de subrotas
M = 100000
for (i, j) in arestas:
    if i > 0:
        dl_tsp += t[i] + custos[i][j] - t[j] <= M * (1 - x[i , j])

#Restricao de prazo
for i in range(numVertices):
    dl_tsp += t[i] <= prazos[i]

#Resolvendo
resolucao = dl_tsp.solve()
print(f"Status do problema: {LpStatus[resolucao]}")

#Mostra as variaveis
for var in dl_tsp.variables():
    if var.varValue > 0:
        print(f"{var.name} = {var.varValue}")

#Mostra a funcao objetivo
print(f"Tempo total de percuso = {value(dl_tsp.objective)}")
