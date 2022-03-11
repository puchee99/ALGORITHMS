#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:53:11 2020

@author: puche99
"""
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

""" FUNCIONS PRINCIPALS"""
def add_node(graph, node):
    #afegeix node a un graf
    graph[node] = {}
    
def del_node(graph,node):
    #elimina node a un graf
    assert node in list(graph.keys()), "Node not in graph"
    del graph[node]
    for val in list(graph.keys()):
        for n in list(graph[val].keys()):
            if node == n:
                del graph[val][n]
    
def ret_nodes(graph):
    #retorna nodes del  graf
    return list(graph.keys())

def ret_arist(graph, node):
    #elimina aresta de un graf
    assert node in list(graph.keys()), "Node not in graph"
    return list(grafo[node].keys())

def calculate_euclidean(C1, C2):
    #calcula la distancia entre dos nodes
    return np.round((((C1[0]-C2[0])**2)+((C1[1]-C2[1])**2))**(1/2),2)


""" FUNCIONS LLEGIR DOCUMENT i CREACIÓ ESTRUCTURA """
def origen_destino(file):
    #llegeix el node inicial i el final
    f = open(file,"r+")
    lines = f.readlines()
    origen = lines[1].split()
    destino = lines[-1].split()
    return (int(origen[0]),int(origen[1])), (int(destino[0]),int(destino[1]))

def to_visit(file):
    #llegeix els nodes a visitar
    l = []
    f = open(file,"r+")
    for x in f.readlines()[1:]:
        line = x.split()
        l.append((int(line[0]), int(line[1])))
    return l

def create_graph(doc_name, path):
    #crea el graf
    graph = {}
    file = open(path+doc_name+".txt","r+")
    for x in file.readlines()[1:]:
        line = x.split()
        origen = (int(float(line[0])), int(float(line[1])))
        destino = (int(float(line[2])), int(float(line[3])))
        if origen in graph.keys():
            graph[origen][destino] = calculate_euclidean(origen,destino)
        else:
            graph[origen] = {destino : calculate_euclidean(origen,destino)}
        if destino in graph.keys():
            graph[destino][origen] = calculate_euclidean(origen,destino)
        else:
            graph[destino] = {origen : calculate_euclidean(origen,destino)}    
    file.close()
    return graph
  
""" CALCULA EL COST DE UN CAMÍ JA REALITZAT """
def calculate_path_dist(grafo, path):
    #calcula la distancia de un cami ja establert
    total = 0
    for i in range(len(path)-1):
        total += grafo[path[i]][path[i+1]]
    return total


""" DJIKSTRA """
def Min_Dist_Djikstra(dist, visited): 
    #agafa el index del node amb la distancia minima
    min_val = np.inf
    global N
    for v in range(N): 
        if dist[v] < min_val and visited[v] == False: 
            min_val= dist[v] 
            min_index = v 
    return min_index 

def Djikstra(graph, start,condition = None):  
    #calcula Djikstra
    global N
    all_nodes = list(graph.keys())
    nodes = list(graph.keys())
    nodes.remove(start)
    nodes.insert(0,start)
    dist = [np.infty]*N
    dist[0] = 0
    visited = [False]*N
    path = {x:[] for x in range(N)}
    for count in range(N):
        u = Min_Dist_Djikstra(dist, visited)
        visited[u] = True
        for v in range(N):
            if nodes[v] in list(graph[nodes[u]].keys()):
                if graph[nodes[u]][nodes[v]] > 0 and visited[v] == False and \
                dist[v] > dist[u] + graph[nodes[u]][nodes[v]]:
                    dist[v] = np.round(dist[u] + graph[nodes[u]][nodes[v]],2)
                    if condition != None:
                        path[v].append(nodes[u])
                        [path[v].append(val) for val in path[u] if val != []]
    if condition != None:
        indexs = [all_nodes.index(x) for x in condition]
        minimum = dist.index(min([x for x in dist if dist.index(x) in indexs]))
        if minimum == 0 and path[1][0]!=start:
            minimum = 1
    else:
        minimum = dist.index(min(dist[1:]))
        if minimum < list(graph.keys()).index(start):
            minimum -=1
    return dist, minimum, path[minimum][:-1]


""" GREEDY"""
def Greedy(graph, origen, destino,to_vis = None): 
    # calcula greedy
    all_nodes = list(graph.keys())
    if to_vis == None:
        to_vis = list(graph.keys())
    else:
        to_vis = copy.copy(to_vis)
    path = [origen]
    to_vis.remove(origen)
    to_vis.remove(destino)
    while to_vis != []:
        result = Djikstra(graph, origen, to_vis)
        nex = all_nodes[result[1]]
        for val in reversed(result[2]):
            path.append(val)
        to_vis.remove(nex)
        origen = nex
        path.append(origen)
    
    result = Djikstra(graph, origen, [destino])
    nex = all_nodes[result[1]]
    for val in reversed(result[2]):
        path.append(val)
    path.append(destino)
    return path

""" BACKTRACKING"""
cost_max = [np.inf]   
result_BTP = []    
def Backtracking_Pure(graph, v, to_visit, index, nexts, cost):
    #calcula backtraking pur
        for vei in nexts:
            v[index] = vei
            result = v[:index+1]
            new_cost = calculate_path_dist(graph, result)
            if new_cost <= cost:
                if [vei] == to_visit:
                    result_BTP.append(result)
                elif vei == to_visit[-1]:
                    #print("GOAL2")
                    Backtracking_Pure(graph, v, to_visit, index+1, list(graph[vei]), cost)
                elif vei in to_visit:
                    to_visit.remove(vei)
                    Backtracking_Pure(graph, v, to_visit, index+1, list(graph[vei]), cost)
                    to_visit.append(vei)
                else:
                    Backtracking_Pure(graph, v, to_visit, index+1, list(graph[vei]), cost)
            v[index] = False

result_BTG = []    
def Backtracking_Greedy(graph,all_nodes, v, to_visit, index, cost):  
    #calcula backtraking greedy
    result = Djikstra(graph, v[index-1], to_visit)
    next_node = all_nodes[result[1]]
    if next_node == v[index-1]:
        print(v)
        next_node = all_nodes[result[1]+1]
    for i,val in enumerate(reversed(result[2])):
        v[index+i] = val
    v[index+len(result[2])] = next_node
    new_cost = calculate_path_dist(graph, v[:index+len(result[2])])
    if new_cost <= cost:
        if [next_node] == to_visit:
            result_BTG.append(v[:index+len(result[2])+1])
        elif next_node in to_visit and next_node != to_visit[-1]:
            to_visit.remove(next_node)
            Backtracking_Greedy(graph,all_nodes, v, to_visit, index+len(result[2])+1, cost)
            to_visit.append(next_node)
        else:
            Backtracking_Greedy(graph,all_nodes, v, to_visit, index+len(result[2])+1, cost)
        v[index:index+len(result[2])] = [False]*len(result[2])  
    return

""" FUNCIONS AUXILIARS ByB"""
def create_path_list(grafo):
    # funció auxiliar per els calculs de ByB
    l = []
    for x in grafo:
        for node in grafo[x]:
            a = [node,x,grafo[x][node]]
            if a not in l:
                l.append(a)
    caminos = copy.deepcopy(l)
    for x in (l):
        aux = [x[1], x[0], x[2]]
        caminos.append(aux)
    caminos = sorted(caminos)
    return caminos


def podar(cola):
    #funció auxiliar ByB
    #print('COLA AL COMENÇAR LA PODA', cola)
    minim = np.inf
    for x in cola:
        if  (x[1] < minim):#and (len(x[0]) == 5):  #si camí és sol. completa
            minim = x[1]
    cont = -1
    for x in cola:
        cont += 1
        if (x[1] > minim):  #si cost camí és major que cost mínim, eliminem camí
            cola.pop(cont)
    #print('COLA A LACABAR LA PODA', cola)
    return(cola)


def ramificar(cami, caminos, dt, to_visit):
    #funció auxiliar ByB
    rami = list()

    for x in caminos:
        #print(" x {} \n cami {}".format(x,cami))
        if ((x[0] == cami[0][-1]) and (x[1] not in cami[0])) or \
            ((x[0] == cami[0][-1]) and (x[1] == dt)): #and (x[1] not in cami[0])
                    aux = cami[0] + x[1]
                    afegir = [aux, np.round(cami[1] + x[2],2)]
                    #print('RAMIFICACIÓ TROBADA', afegir)
                    if afegir not in rami:
                        rami.append(afegir)
    #print('RAMIFICACIÓ A AFEGIR A CAMÍ',rami)
    #print(rami)
    return(rami)

""" FUNCIO GLOBAL ByB"""
def Branch_Bound(cola, caminos, dest, to_visit):
    #ByB H1
    while (len(cola) != 0):
        #print('\nCOLA AL COMENÇAR EL WHILE', cola)
        rami = ramificar(cola[0], caminos, dest, to_visit)
        for x in rami:
            #print("RAMIIII",x,to_visit)
            if all(el in x[0] for el in to_visit) and x[0][-1] == dest:
                return [x]
        if (rami != []): cola.pop(0)
        else: return cola
        cola += rami
        cola.sort(key=lambda x: x[1])
        cola = podar(cola)
        #print('COLA A LACABAR EL WHILE', cola,"\n")

 
""" 
GRAFIQUES...
all_times = [] 
for i in range(1,5):
    test = "Tests_v1/"
    num_test = str(i)
    grafo = create_graph("Grafo"+num_test, test)
    N = len(list(grafo.keys()))
    O, D = origen_destino(test+"Visits"+num_test+".txt") 
    cities_to_visit = to_visit(test+"Visits"+num_test+".txt")
    start_greedy = time.time()
    result_Gr = Greedy(grafo,O,D,cities_to_visit) 
    time_greedy = time.time() - start_greedy
    all_times.append(time_greedy)
    print(i)
plt.plot([1,2,3,4],all_times, color="green",marker="o")
plt.xlabel("Algorithm")
plt.ylabel("Time (s)")
plt.title("Greedy")
"""  
"""CARGUEM LES DADES i PROVEM DJIKSTRA"""
all_times = []
test = "Tests_v1/"
num_test = "3"
grafo = create_graph("Grafo"+num_test, test)
N = len(list(grafo.keys()))
O, D = origen_destino(test+"Visits"+num_test+".txt") 
cities_to_visit = to_visit(test+"Visits"+num_test+".txt")
print("Ciudades a visitar --> ",cities_to_visit.copy(),"\n")

result_Dj = Djikstra(grafo, list(grafo.keys())[0])[0]
print("El resultado del algoritmo Djikstra es :--> ",result_Dj,"\n")


    
"""CALCULEM GREEDY"""
start_greedy = time.time()
result_Gr = Greedy(grafo,O,D,cities_to_visit) 
time_greedy = time.time() - start_greedy
all_times.append(time_greedy)
print("El resultado del algoritmo Greedy es :--> ",result_Gr,"\n")


"""PREPAREM DADES i CALCULEM BACKTRACKING PUR"""
v = [False for i in range(N)]
visit = cities_to_visit.copy()
v[0] = cities_to_visit[0]
visit.remove(v[0])
start_BTP = time.time()
Backtracking_Pure(grafo,v,visit,1,list(grafo[v[0]]), calculate_path_dist(grafo, result_Gr))
time_BTP = time.time() - start_BTP
all_times.append(time_BTP)
print("El resultado del algoritmo Backtracking Puro es :--> ",result_BTP,"\n")


"""PREPAREM DADES i CALCULEM BACKTRACKING GREEDY"""
v = [False for i in range(N)]
visit = cities_to_visit.copy()
v[0] = cities_to_visit[0]
visit.remove(v[0])
start_BTG = time.time()
Backtracking_Greedy(grafo, list(grafo.keys()), v, visit, 1, calculate_path_dist(grafo, result_Gr))
time_BTG = time.time() - start_BTG
all_times.append(time_BTG)
print("El resultado del algoritmo Backtracking Greedy es :--> ",result_BTG,"\n")

""" PREPAREM FORMAT PER CALCULAR ByB """
visit = cities_to_visit.copy()
paths = create_path_list(grafo)
paths2 = paths.copy()
paths3 = paths.copy()
val = grafo.keys()
new_g = {}
for i,x in enumerate(val):
    new_g[x] = chr(i+65).lower()
    new_g[chr(i+65).lower()] = x
for fil in paths:
    fil[0:2] = [new_g[fil[0]], new_g[fil[1]]]
for i,x in enumerate(visit):
    visit[i] = new_g[x]
    
cola = [[new_g[O],0]]

""" CALCULEM ByB H1 """
start_BB1 = time.time()
result_B = Branch_Bound(cola, paths, new_g[D], visit)
time_BB1 = time.time() - start_BB1
result_BB1 = [new_g[val] for val in result_B[0][0]]
all_times.append(time_BB1)
print("El resultado del algoritmo Brach & Bound H1 es :--> ",result_BB1,"\n")

cola = [[new_g[O],0]]

""" PREPAREM NOVES DADES i CALCULEM ByB H2 """
start_BB2 = time.time()
for i,x in enumerate(paths):
    res = Djikstra(grafo,new_g[paths2[i][0]],[new_g[paths2[i][0]]])
    paths2[i][2] = res[0][res[1]]
result_B2 = Branch_Bound(cola, paths2, new_g[D], visit[1:-1])
time_BB2 = time.time() - start_BB2
result_BB2 = [new_g[val] for val in result_B2[0][0]]
all_times.append(time_BB2)
print("El resultado del algoritmo Brach & Bound H2 es :--> ",result_BB2,"\n")

cola = [[new_g[O],0]]

""" PREPAREM NOVES DADES i CALCULEM ByB H3 """
start_BB3 = time.time()
for i,x in enumerate(paths):
    res = Djikstra(grafo,new_g[paths3[i][0]],[new_g[paths3[i][0]]])
    paths3[i][2] += res[0][res[1]]
result_B3 = Branch_Bound(cola, paths3, new_g[D], visit[1:-1])
time_BB3 = time.time() - start_BB3
result_BB3 = [new_g[val] for val in result_B3[0][0]]
all_times.append(time_BB3)
print("El resultado del algoritmo Brach & Bound H3 es :--> ",result_BB3,"\n")


plt.plot(["Greedy","BTP","BTG","BB1","BB2","BB3"],all_times, color="green",marker="o")
plt.xlabel("Algorithm")
plt.ylabel("Time (s)")