# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:58:54 2019

@author: yayou
"""

import networkx as nx
import numpy as np
import operator
import re

def graph_articles(citation_set, node_info, directed_or_not = 'n'):
    if directed_or_not == 'y':
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for node in node_info:
        if node[0] != 'ID':
            G.add_node(int(node[0]), year = node[1])
    for i in citation_set:
        if i[2] == 1 or i[2] == '1':
            if G.node[int(i[0])]['year'] <= G.node[int(i[1])]['year']:
                G.add_edge(i[0], i[1])
            else:
                G.add_edge(i[1], i[0])
    return G

def graph_authors(citation_set, node_info, IDs, directed_or_not = 'n'):
    
    reverse_index = dict()
    for i,a in enumerate(node_info):
        reverse_index[int(a[0])] = i
    
    if directed_or_not == 'y':
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    counter = 0
    for citation in citation_set:
        source = int(citation[0])
        target = int(citation[1])
                        
        if type(node_info[reverse_index[source],3]) != float:
            source_authors = re.sub(r',Jr.?', 'Jr',
                                re.sub(r',(?=[a-z])', '',
                                       re.sub(r'\([^)]*\)?', '',
                                              re.sub(r' ', '' , node_info[reverse_index[source],3])))).split(",")
        else:
            source_authors = []
            
        if type(node_info[reverse_index[target],3]) != float:
            target_authors = re.sub(r',Jr.?', 'Jr',
                                    re.sub(r',(?=[a-z])', '',
                                           re.sub(r'\([^)]*\)?', '',
                                                  re.sub(r' ', '' , node_info[reverse_index[target],3])))).split(",")
        else:
            target_authors = []
        
        #print(source_authors[:10])
        
        if citation[2] == '1' or citation[2] == 1:
            for auth1 in source_authors:
                if auth1 != '':
                    for auth2 in target_authors:
                        if auth2 != '':
                            G.add_edge(auth1, auth2)
        else:
            for auth1 in source_authors:
                if auth1 != '':
                    for auth2 in target_authors:
                        if auth2 != '':
                            G.add_node(auth1)
                            G.add_node(auth2)
               
        counter += 1
    
        if counter % 5000 == True:
            print(counter, "training examples processsed")
    
    return G

def graph_authors_weight(citation_set, node_info, IDs, directed_or_not = 'n'):
    if directed_or_not == 'y':
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    counter = 0
    for citation in citation_set:
        source = citation[0]
        target = citation[1]
        
        source_authors = [re.sub(r',Jr.?', 'Jr',
                            re.sub(r',(?=[a-z])', '',
                                   re.sub(r'\([^)]*\)?', '',
                                          re.sub(r' ', '' , element[3])))).split(",") for element in node_info if element[0]==source][0]
        target_authors = [re.sub(r',Jr.?', 'Jr',
                            re.sub(r',(?=[a-z])', '',
                                   re.sub(r'\([^)]*\)?', '',
                                          re.sub(r' ', '' , element[3])))).split(",") for element in node_info if element[0]==target][0]
        
        if citation[2] == '1' or citation[2] == 1:
            for auth1 in source_authors:
                if auth1 != '':
                    for auth2 in target_authors:
                        if auth2 != '':
                            if (auth1,auth2) in G.edges:
                                G.get_edge_data(auth1,auth2)['weight'] += 1
                            else:
                                G.add_edge(auth1, auth2, weight=1)
        else:
            for auth1 in source_authors:
                if auth1 != '':
                    for auth2 in target_authors:
                        if auth2 != '':
                            G.add_edge(auth1, auth2, weight=0)
               
        counter += 1
    
        if counter % 5000 == True:
            print(counter, "traininefgesgsegg examples processsed")
    
    return G

def compute_authors_affinity_for_article(citation_set, node_info, G=None):
    if G != G:
        G = graph_authors_weight(citation_set, node_info)
    
    affinities = []
    
    for citation in citation_set:
        source = citation[0]
        target = citation[1]
        
        source_authors = [element[3].split(",") for element in node_info if element[0]==source][0]
        target_authors = [element[3].split(",") for element in node_info if element[0]==target][0]
        
        feature = 0
        for auth1 in source_authors:
            for auth2 in target_authors:
                feature += G.get_edge_data(auth1, auth2)['weight']
        
        affinities.append(feature)
        
    return np.array(affinities)

def compute_page_rank_feature_for_articles(citation_set, G=None):
    if G != G:
        G = graph_articles(citation_set)
        bool = True
    else:
        bool = False
    
    pg_rk = nx.pagerank(G)
    
    pg_rk_features = []
    for citation in citation_set:
        pg_rk_features.append(one_page_rank_feature_for_articles(citation, pg_rk))
    
    if bool:
        return np.array(pg_rk_features), G
    else:
        return np.array(pg_rk_features)

def one_page_rank_feature_for_articles(citation, pg_rk):
    return pg_rk[int(citation[0])] + pg_rk[int(citation[1])]
    
def compute_page_club_feature_for_articles(citation_set, node_info, G=None):
    if G != G:
        G = graph_articles(citation_set, node_info, directed_or_not='y')
    
    pg_rk = nx.pagerank(G)
    sorted_pgr = sorted(pg_rk.items(), key= operator.itemgetter(1), reverse = True)
    #sp_keys = sorted_pgr.keys()
    sp_keys = [int(a) if a!='ID' else -1 for (a,b) in sorted_pgr]
    #print(sp_keys)
    sp_keys_seen = dict()
    for s in sp_keys:
        sp_keys_seen[s] = False
    
    #nbedgesin = [0]
    
    pageclub = []
    
    in_degs_tuple = list(G.in_degree())
    in_degs = [b for (a,b) in in_degs_tuple]
    k_in = sum(in_degs)/ float(len(in_degs))
    
    n = len(sp_keys)
    
    s=0
    i=1
    kincum=0
    koutcum=0
    
    
    
    undirG = G.to_undirected()
    
    for (k,v) in sorted_pgr:
        edges_i = undirG.edges(k)
        for (a,b) in edges_i:
            if k!= 'ID':
                if int(a) == int(k):
                    if sp_keys_seen[int(b)]:
                        s+=1
                else:
                    if sp_keys_seen[int(a)]:
                        s+=1        
        if k != 'ID':
            sp_keys_seen[int(k)]= True
        else:
            sp_keys_seen[-1] = True
        kincum += G.in_degree(k)
        koutcum += G.out_degree(k)
        if kincum>0 and koutcum >0:
            pageclub.append((s*k_in*n)/(kincum*koutcum))
            G.node[k]['pageclub'] = (s*k_in*n)/(kincum*koutcum)
        else:
            pageclub.append(1)
            G.node[k]['pageclub'] = 1
            
        i+=1
    
    features_edges = []
    for citation in training_set:
        ic0 = key_dict[int(citation[0])]
        ic1 = key_dict[int(citation[1])]
        features_edges.append(G.node[int(citation[0])]['pageclub'] + G.node[int(citation[1])]['pageclub'])

        
    return features_edges

def compute_page_rank_feature_for_authors(citation_set, node_info, G=None):
    if G != G:
        G = graph_authors(citation_set, node_info)
        bool = True
    else:
        bool = False
    
    pg_rk = nx.pagerank(G)
    pg_rk_features = []
    for citation in citation_set:
        source = citation[0]
        target = citation[1]
        
        source_authors = [element[3].split(",") for element in node_info if element[0]==source][0]
        target_authors = [element[3].split(",") for element in node_info if element[0]==target][0]
        
        feature = 0
        for auth in source_authors + target_authors:
            feature += pg_rk[auth]
        
        pg_rk_features.append(feature)
    
    if bool:
        return np.array(pg_rk_features), G
    else:
        return np.array(pg_rk_features)
    
def compute_rich_club_feature_for_articles(citation_set, G=None):
    if G != G:
        G = graph_articles(citation_set)
        bool = True
    else:
        bool = False
    
    rc_cl = nx.richclub.rich_club_coefficient(G)
    
    rc_cl_features = []
    for citation in citation_set:
        rc_cl_features.append(rc_cl[citation[0]] + rc_cl[citation[1]])
    
    if bool:
        return np.array(rc_cl_features), G
    else:
        return np.array(rc_cl_features)
    
def compute_rich_club_feature_for_authors(citation_set, node_info, G=None):
    if G != G:
        G = graph_authors(citation_set, node_info)
        bool = True
    else:
        bool = False
    
    rc_cl = nx.richclub.rich_club_coefficient(G)
    rc_cl_features = []
    for citation in citation_set:
        source = citation[0]
        target = citation[1]
        
        source_authors = [element[3].split(",") for element in node_info if element[0]==source][0]
        target_authors = [element[3].split(",") for element in node_info if element[0]==target][0]
        
        feature = 0
        for auth in source_authors + target_authors:
            feature += rc_cl[auth]
        
        rc_cl_features.append(feature)
    
    if bool:
        return np.array(rc_cl_features), G
    else:
        return np.array(rc_cl_features)
    
def compute_shorthest_path_feature_for_articles(citation_set,G=None):
    if G != G:
        G = graph_articles(citation_set)
        bool = True
    else:
        bool = False
    
    sht_pth_features = []
    for citation in citation_set:
        if citation[2] == '1':
            G.remove_edge(citation[0], citation[1])
        sht_pth_features.append(
                nx.shortest_path_length(G,citation[0], citation[1]) if nx.has_path(G, citation[0], citation[1]) else 30)
        if citation[2] == '1':
            G.add_edge(citation[0], citation[1])
    
    if bool:
        return np.array(sht_pth_features), G
    else:
        return np.array(sht_pth_features)