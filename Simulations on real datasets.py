import networkx as nx
import numpy as np
# from utils_simplagion_MC import *
import random as rd
import numpy as np
import json
import random
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
import pandas as pd
from scipy.stats import norm

#Markov chain function
def markovChain(t_max, G, G2, beta, fai, beta_D, mu, node_neighbors_dict, tri_neighbors_dict, tri_neighbors_dict2, theta1, theta2, i0):
    N = len(node_neighbors_dict)
    
    p = np.zeros(2000)
    p[rd.sample(list(G.nodes),int(N*i0))] = 1
    p_new = np.copy(p)

    l = np.zeros(2000)
    nodes = list(G.nodes)    
    for node in nodes:
        l[node] = np.clip(norm.rvs(loc=0.5, scale=0.3, size=1), 0, 1)    
    
    l2 = np.zeros(2000)
    nodes = list(G2.nodes)    
    for node in nodes:
        l2[node] = np.clip(norm.rvs(loc=0.5, scale=0.3, size=1), 0, 1) 

    #设置初始信息感知个体
    for n in G2.nodes(): 
        if random.random() < 0.1:
            G2.nodes[n]['as'] = 'A' #感知
        else:
            G2.nodes[n]['as'] = 'U' #不感知
  
    q = 1
    pTime = [np.mean(p)*2000/N]

    t = 1
    while t < t_max:
        # 信息传播过程
        for n in G2.nodes():
            nbs = G2.neighbors(n)
            if G2.nodes[n]['as'] == 'U' :
                for nb in nbs:
                    if G2.nodes[nb]['as'] == 'A' and l2[n] + l2[nb] > theta1 and rd.random() < 0.4:
                        G2.nodes[n]['as'] = 'A'

        for n in G2.nodes():
            if G2.nodes[n]['as'] == 'U':
                for j, k in tri_neighbors_dict2[n]:
                    if G2.nodes[j]['as'] == 'A' and G2.nodes[k]['as'] == 'A' and l2[n] + l2[j]+ l2[k] > theta2  and rd.random() < 0.5:
                        G2.nodes[n]['as'] = 'A'
        # 疾病传播过程
        for i in list(G.nodes): 
            if G2.nodes[i]['as'] == 'U' :           
                #Updating the q_i (infections) - d=1
                for j in node_neighbors_dict[i]:
                    if l[i] + l[j] >= theta1:
                        q *= (1.-beta*p[j])
                    
                #Updating the q_i (infections) - d=2
                for j, k in tri_neighbors_dict[i]:
                    if l[i] + l[j] + l[k] >= theta2:
                        q *= (1.-beta_D*p[j]*p[k])
                
                #Updating the vector
                p_new[i] = (1-q)*(1-p[i]) + (1.-mu)*p[i]

            else:
                for j in node_neighbors_dict[i]:
                    if l[i]+l[j] >= theta1:
                        q *= (1.-fai*beta*p[j])
                    
                #Updating the q_i (infections) - d=2
                for j, k in tri_neighbors_dict[i]:
                    if l[i]+l[j]+l[k] >= theta2:
                        q *= (1.-fai*beta_D*p[j]*p[k])                
                #Updating the vector
                p_new[i] = (1-q)*(1-p[i]) + (1.-mu)*p[i]

                #Resetting the i-th parameters
                q = 1
            
        p = np.copy(p_new)
        pTime.append(np.mean(p)*2000/N)

        for i in list(G2.nodes): 
            # for j in node_neighbors_dict[i]:
            pairs2 = 0.7*(sum(l2[j] for j in G2.neighbors(i))/G2.degree(i))
            # for j, k in tri_neighbors_dict[i]:
            triple2 = 0.3*(sum((l2[j]+ l2[k])for j, k in tri_neighbors_dict2[i])/(2*len(tri_neighbors_dict2[i])+1))
            l2[i] = 0.9*l2[i] + 0.1*(pairs2 + triple2)        
        
        for i in list(G.nodes): 
            # for j in node_neighbors_dict[i]:
            pairs = 0.7*(sum(l[j] for j in node_neighbors_dict[i])/G.degree(i))
            # for j, k in tri_neighbors_dict[i]:
            triple = 0.3*(sum((l[j]+ l[k])for j, k in tri_neighbors_dict[i])/(2*len(tri_neighbors_dict[i])+1))
            l[i] = 0.9*l[i] + 0.1*(pairs + triple)

        theta1 += 0.001
        theta2 += 0.002
        print(theta1)

        t += 1

    return pTime

def get_tri_neighbors_dict(triangles_list):
    tri_neighbors_dict = defaultdict(list)
    for i, j, k in triangles_list:
        tri_neighbors_dict[i].append((j,k))
        tri_neighbors_dict[j].append((i,k))
        tri_neighbors_dict[k].append((i,j))
    return tri_neighbors_dict
def get_tri_neighbors_dict2(triangles_list2):
    tri_neighbors_dict2 = defaultdict(list)
    for i, j, k in triangles_list2:
        tri_neighbors_dict2[i].append((j,k))
        tri_neighbors_dict2[j].append((i,k))
        tri_neighbors_dict2[k].append((i,j))
    return tri_neighbors_dict2
def import_sociopattern_simcomp_SCM(dataset_dir, dataset, n_minutes,):
    filename = dataset_dir+'aggr_'+str(n_minutes)+'min_cliques_thr1_'+dataset+'.json'
    # filename = dataset_dir + dataset + '_simplices'+'.json'

    cliques = json.load(open(filename,'r'))
    
    G, G2, node_neighbors_dict, triangles_list, triangles_list2 = create_simplicial_complex_from_cliques(cliques)
    # print(G)
    

    N = len(node_neighbors_dict.keys())
    avg_k1 = 1.*sum([len(v) for v in node_neighbors_dict.values()])/N
    avg_k2 = 3.*len(triangles_list)/N 
    #ass = nx.degree_assortativity_coefficient(facet_list_to_graph(cliques))

    return G, G2, node_neighbors_dict, triangles_list,triangles_list2, avg_k1, avg_k2

def create_simplicial_complex_from_cliques(cliques):
    
    G = nx.Graph()
    triangles_list = set() #will contain list of triangles (2-simplices)
    
    for c in cliques:
        d = len(c)
        
        if d==2:
            i, j = c
            G.add_edge(i, j)
        
        elif d==3:
            #adding the triangle as a sorted tuple (so that we don't get both ijk and jik for example)
            triangles_list.add(tuple(sorted(c)))
            #adding the single links
            for i, j in combinations(c, 2):
                G.add_edge(i, j)
            
        else: #d>3, but I only consider up to dimension 3
            #adding the triangles
            for i, j, k in combinations(c, 3):
                triangles_list.add(tuple(sorted([i,j,k])))

            #adding the single links
            for i, j in combinations(c, 2):
                G.add_edge(i, j)
                
    if nx.is_connected(G)==False:
        print('not connected')
                
    #Creating a dictionary of neighbors
    node_neighbors_dict = {}
    for n in G.nodes():
        node_neighbors_dict[n] = G[n].keys()

    #converting the triangle set of tuples into a triangle list of lists
    triangles_list = [list(tri) for tri in triangles_list]
    # print(node_neighbors_dict)
    G2 = G.copy()
    triangles_list2 = [list(tri) for tri in triangles_list]

    return  G, G2, node_neighbors_dict, triangles_list, triangles_list2


# Reading clean Sociopatterns data
dataset_dir = 'Data/Processed_data/'
dataset = 'Thiers13' #'InVS15','SFHH', 'LH10','LyonSchool','Thiers13','LH10',congress-bills,email-Eu
n_minutes = 15 #Aggregation time
# thr = 0.80 #fraction of removed cliques (0.80: retaining the 20% most weighted)

G, G2, node_neighbors_dict, triangles_list, triangles_list2, avg_k1, avg_k2 = import_sociopattern_simcomp_SCM(dataset_dir, dataset, n_minutes)
tri_neighbors_dict = get_tri_neighbors_dict(triangles_list)
tri_neighbors_dict2 = get_tri_neighbors_dict2(triangles_list2)

# mu = x
# lambda1s = x
# # lambdaD_targets = x
# lambdaD_targets = x
# I_percentage = x #initial conditions (% of infected)

# beta = x
# beta_Ds = x
# beta_D = x

# t_max = x
# fai = x
# theta1 = x
# theta2 = x
# i0 = x
# results:
# rho_markov = [markovChain()]

    