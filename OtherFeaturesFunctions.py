# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:06:24 2019

@author: yayou
"""


import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing
import nltk

import AllGraphFunctions as GF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import scipy



def initialize_nltk():
    nltk.download('punkt') # for tokenization
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    
    return stpwds, stemmer

def abstract_TFIDF(node_info, stpwds):
    corpus = [element[5] for element in node_info]
    vectorizer = TfidfVectorizer(stop_words="english")
    features_TFIDF = vectorizer.fit_transform(corpus)

    return features_TFIDF

def title_TFIDF(node_info, stpwds):
    corpus = [element[2] for element in node_info]
    vectorizer = TfidfVectorizer(stop_words="english")
    features_TFIDF = vectorizer.fit_transform(corpus)

    return features_TFIDF

def title_doc2voc(node_info, citation_set):
    token_title =    [TaggedDocument(words=nltk.tokenize.word_tokenize(element[2].lower()),tags=[str(element[0])]) for i,element in enumerate(node_inf)]
    model_title = Doc2Vec(size=128,
                alpha=0.025, 
                min_alpha=0.00025,
                min_count=1,
                dm =0)
    model_title.build_vocab(token_title)
    for epoch in range(100):
        model_title.train(token_title,
                    total_examples=model_title.corpus_count,
                    epochs=model_title.iter)

        model_title.alpha -= 0.0002

        model_title.min_alpha = model_title.alpha

    titlefeatures = []
    for citation in citation_set:
        d = scipy.spatial.distance.cosine(model_title.docvecs[str(citation[0])],model_title.docvecs[str(citation[1])])
        titlefeatures.append(d)
        
    return titlefeatures

def abstract_doc2voc(node_info,citation_set):
    token_abstract = [TaggedDocument(words=nltk.tokenize.word_tokenize(element[5].lower()),tags=[str(element[0])]) for i,element in enumerate(node_inf)]
    model_abstract = Doc2Vec(size=128,
                alpha=0.025, 
                min_alpha=0.00025,
                min_count=1,
                dm =0)
    model_abstract.build_vocab(token_abstract)

    for epoch in range(100):
        model_abstract.train(token_abstract,
                    total_examples=model_abstract.corpus_count,
                    epochs=model_abstract.iter)

        model_abstract.alpha -= 0.0002
        
        model_abstract.min_alpha = model_title.alpha
        
    abstractfeatures = []
    for citation in citation_set:
        d = scipy.spatial.distance.cosine(model_abstract.docvecs[str(citation[0])],model_abstract.docvecs[str(citation[1])])
        abstractfeatures.append(d)
        
    return abstractfeatures

def h_index(citation_set, node_info, G):
    all_auth = np.array([(re.sub(r',Jr.?', 'Jr',
                            re.sub(r',(?=[a-z])', '',
                                   re.sub(r'\([^)]*\)?', '',
                                          re.sub(r' ', '' , a[3])))).split(","),a[0]) if type(a[3]) == str else [] for a in node_inf])
    authors = defaultdict(list)
    for a,c in zip(all_auth,node_inf):
        if a != []:
            #print(a)
            (b,i) = a
            for auteur in b:
                authors[auteur].append(i)

    del authors['']
    
    authors_inv = defaultdict(list)
    for auth in authors:
        for art in authors[auth]:
            authors_inv[art].append(auth)
    
    authors_h = dict()
    for aut in authors.keys():
        liste = [G.in_degree(int(i)) for i in authors[aut]]
        liste.sort(reverse=True)
        i=1
        j=0
        while j<len(liste) and liste[j] >= i:
            i+=1
            j+=1
        
        authors_h[aut] = i-1
        
    for i in G:
    #print(i)
        auteurs = authors_inv[i]
        if auteurs==[]:
            G.node[int(i)]['h-index'] = 1
        else:
            hlist = [authors_h[j] for j in auteurs]
    #if len(hlist) == 0:
    #    G.node[i]['h-index'] = 1
            G.node[int(i)]['h-index'] = sum(hlist)/len(hlist)
        
    hindexfeatures = []
    for citation in training_set:
        a=0
        b=0
        if(G.has_node(int(citation[0]))):
            a = G.node[int(citation[0])]['h-index']
        if(G.has_node(int(citation[1]))):
            b = G.node[int(citation[1])]['h-index']
        hindexfeatures.append((a + b)/2)
    
    return hindexfeatures


    
def compute_TFIDF_abstract_feature(citation_set, node_info, IDs, stpwds):
        dist_abstract = []
        TFIDF_abstract = abstract_TFIDF(node_info, stpwds)
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
            index_source = IDs.index(source)
            index_target = IDs.index(target)
    
            dist_abstract.append(
                np.linalg.norm(
                    (TFIDF_abstract[index_source]-TFIDF_abstract[index_target]
                    ).toarray()
                )
            )
    
            counter += 1
            if counter % 5000 == True:
                print(counter, "training examples processsed")

        dist_abstract = np.array(dist_abstract)
        
        return dist_abstract
    
def compute_TFIDF_title_feature(citation_set, node_info, IDs, stpwds):
        dist_title = []
        TFIDF_title = title_TFIDF(node_info, stpwds)
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            index_source = IDs.index(source)
            index_target = IDs.index(target)
    
            dist_title.append(
                np.linalg.norm(
                    (TFIDF_title[index_source]-TFIDF_title[index_target]
                    ).toarray()
                )
            )
    
            counter += 1
            if counter % 5000 == True:
                print(counter, "training examples processsed")

        dist_title = np.array(dist_title)
        
        return dist_title
    
def compute_temp_diff(citation_set, node_info, IDs):
        temp_diff = []
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            source_info = [element for element in node_info if element[0]==source][0]
            target_info = [element for element in node_info if element[0]==target][0]
            
            temp_diff.append(
                    int(source_info[1]) - int(target_info[1])
                    )
            
            counter += 1
            if counter % 5000 == True:
                print(counter, "training examples processsed")
        
        temp_diff = np.array(temp_diff)
        
        return temp_diff
    
def compute_overlaping_titles(citation_set, node_info, IDs, stemmer, stpwds):
        overlap_title = []
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            source_info = [element for element in node_info if element[0]==source][0]
            target_info = [element for element in node_info if element[0]==target][0]
    
            
            source_title = source_info[2].lower().split(" ")
            
            source_title = [token for token in source_title if token not in stpwds]
            source_title = [stemmer.stem(token) for token in source_title]
    
            target_title = target_info[2].lower().split(" ")
            target_title = [token for token in target_title if token not in stpwds]
            target_title = [stemmer.stem(token) for token in target_title]
            
            overlap_title.append(
                    len(set(source_title).intersection(set(target_title)))
                    )
            
            counter += 1
            if counter % 5000 == True:
                print(counter, "training examples processsed")
        
        return np.array(overlap_title)
            
        
        
def compute_common_auth(citation_set, node_info, IDs):
        comm_auth = []
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            source_info = [element for element in node_info if element[0]==source][0]
            target_info = [element for element in node_info if element[0]==target][0]
            
            source_auth = source_info[3].split(",")
            target_auth = target_info[3].split(",")
            
            comm_auth.append(
                    len(set(source_auth).intersection(set(target_auth)))
                    )
            
            counter += 1
            if counter % 5000 == True:
                print(counter, "training examples processsed")
            
        comm_auth = np.array(comm_auth)
        
        return comm_auth
    
def articles_page_rank_feature(citation_set, node_info, G, IDs):
    return GF.compute_page_rank_feature_for_articles(citation_set, G)

def authors_page_rank_feature(citation_set, node_info, G, IDs):
    return GF.compute_page_rank_feature_for_authors(citation_set, node_info,G)
            
def page_club_feature(citation_set, node_info, G):
    return GF.compute_page_club_feature_for_articles(citation_set, node_info, G)
            
def shorthest_path_feature(citation_set, node_info, G):
    return GF.compute_shorthest_path_feature_for_articles(citation_set,G)

def compute_author_affinity(citation_set, node_info, G, IDs):
    return GF.compute_authors_affinity_for_article(citation_set,node_info,G)

def compute_same_journal_or_not(citation_set, node_info):
        same_journal = []
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            source_info = [element for element in node_info if element[0]==source][0]
            target_info = [element for element in node_info if element[0]==target][0]
            
            same_journal.append(
                    int(source_info[4] == target_info[4])
                    )
            
            counter += 1
            if counter % 5000 == True:
                print(counter, "training examples processsed")
        
        same_journal = np.array(same_journal)
        
        return same_journal


        
