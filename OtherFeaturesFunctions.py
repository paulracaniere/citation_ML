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
    
def articles_page_rank_feature(citation_set, node_info, IDs):
    return GF.compute_page_rank_feature_for_articles(citation_set, GF.graph_articles(citation_set, node_info))

def authors_page_rank_feature(citation_set, node_info, IDs):
    G = GF.graph_authors(citation_set, node_info, IDs)
    return GF.compute_page_rank_feature_for_authors(citation_set, node_info,G)
            
def page_club_feature(citation_set, node_info):
    return GF.compute_page_club_feature_for_articles(citation_set, node_info, GF.graph_articles(citation_set, node_info, directed_or_not = 'y'))
            
def shorthest_path_feature(citation_set, node_info):
    G = GF.graph_articles(citation_set, node_info)
    return GF.compute_shorthest_path_feature_for_articles(citation_set,G)

def compute_author_affinity(citation_set, node_info, IDs):
    G = GF.graph_authors_weight(citation_set, node_info, IDs)
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


        
