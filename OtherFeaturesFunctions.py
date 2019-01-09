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

import re
from collections import defaultdict

import operator

def initialize_nltk():
    '''
    Initialize the stopwords and stemmer
    '''
    
    nltk.download('punkt') # for tokenization
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    
    return stpwds, stemmer


def title_doc2voc(node_info, citation_set):
    '''
    Returns the feature extracted from the computation of the doc2vec model applied to the title data
    '''
    
    token_title =[
        TaggedDocument(
            words=nltk.tokenize.word_tokenize(element[2].lower()),
            tags=[str(element[0])]
        ) for i,element in enumerate(node_info)
    ]
    
    #building the doc2vec model
    model_title = Doc2Vec(
                size=128,
                alpha=0.025, 
                min_alpha=0.00025,
                min_count=1,
                dm =0
    )
    model_title.build_vocab(token_title)
    
    #training the model
    for epoch in range(20):
        model_title.train(
                    token_title,
                    total_examples=model_title.corpus_count,
                    epochs=model_title.iter
        )

        model_title.alpha -= 0.0002

        model_title.min_alpha = model_title.alpha

    #building the feature by computing the cosine distance between the arrays
    titlefeatures = []
    for citation in citation_set:
        d = scipy.spatial.distance.cosine(
            model_title.docvecs[str(citation[0])],
            model_title.docvecs[str(citation[1])]
        )
        titlefeatures.append(d)
        
    return titlefeatures

def abstract_doc2voc(node_info,citation_set):
    '''
    Returns the feature extracted from the computation of the doc2vec model applied to the title data
    '''
    
    token_abstract = [
        TaggedDocument(
            words=nltk.tokenize.word_tokenize(element[5].lower()),
            tags=[str(element[0])]
        ) for i,element in enumerate(node_info)
    ]
    
    #building the doc2vec model
    model_abstract = Doc2Vec(size=128,
                alpha=0.025, 
                min_alpha=0.00025,
                min_count=1,
                dm =0)
    model_abstract.build_vocab(token_abstract)

    #training the model
    for epoch in range(20):
        model_abstract.train(
                    token_abstract,
                    total_examples=model_abstract.corpus_count,
                    epochs=model_abstract.iter
        )

        model_abstract.alpha -= 0.0002
        
        model_abstract.min_alpha = model_abstract.alpha
        
    #building the feature
    abstractfeatures = []
    for citation in citation_set:
        d = scipy.spatial.distance.cosine(
            model_abstract.docvecs[str(citation[0])],
            model_abstract.docvecs[str(citation[1])]
        )
        abstractfeatures.append(d)
        
    return abstractfeatures

def h_index(citation_set, node_info, G):
    '''
    Returns the feature extracted from the h-index values of the papers' authors.
    '''
    #We clean the authors names and put them into a single array. We use the regex library.
    authors_array = np.array([(
        re.sub(r',Jr.?', 'Jr',
               re.sub(r',(?=[a-z])', '',
                      re.sub(r'\([^)]*\)?', '',
                             re.sub(r' ', '' , a[3])))).split(","),a[0]) if type(a[3]) == str else [] for a in node_info])
    
    #papers_by_author gives the id of the papers an author wrote
    papers_by_author = defaultdict(list)
    for a in authors_array:
        if a != []:
            (b,i) = a
            for author in b:
                papers_by_author[author].append(i)
    del papers_by_author['']
    
    #papers_by_author gives the id of the papers an author wrote
    authors_of_paper = defaultdict(list)
    for author in papers_by_author:
        for paper in papers_by_author[author]:
            authors_of_paper[paper].append(author)
    
    #compute the h-index for every author
    authors_h_index = dict()
    for author in papers_by_author.keys():
        nb_citations = [G.in_degree(int(i)) for i in papers_by_author[author]]
        nb_citations.sort(reverse=True)
        i=1
        j=0
        while j<len(nb_citations) and nb_citations[j] >= i:
            i+=1
            j+=1
        authors_h_index[author] = i-1
        
    #compute the average h-index of a paper as the average of its authors
    for i in G:
        authors_of_i = authors_of_paper[i]
        if authors_of_i==[]:
            G.node[int(i)]['h-index'] = 1
        else:
            h_index_of_i = [authors_h_index[j] for j in authors_of_i]
            G.node[int(i)]['h-index'] = sum(h_index_of_i)/len(h_index_of_i)
     
    #compute the h_index features of every edge in the set
    h_index_features = []
    for citation in citation_set:
        a=0
        b=0
        i0 = int(citation[0])
        i1 = int(citation[1])
        if(G.has_node(i0)):
            a = G.node[i0]['h-index']
        if(G.has_node(i1)):
            b = G.node[i1]['h-index']
        h_index_features.append((a + b)/2)
    
    return h_index_features


def abstract_TFIDF(node_info, stpwds, stemmer):
    '''
    Returns the TFIDF from the abstract data
    '''
 
    corpus = [' '.join([stemmer.stem(a) for a in nltk.tokenize.word_tokenize(element[5])]) for element in node_info]
    vectorizer = TfidfVectorizer(stop_words="english")
    features_TFIDF = vectorizer.fit_transform(corpus)

    return features_TFIDF

def title_TFIDF(node_info, stpwds, stemmer):
    '''
    Returns the TFIDF from the title data
    '''
    
    corpus = [' '.join([stemmer.stem(a) for a in nltk.tokenize.word_tokenize(element[2])]) for element in node_info]
    vectorizer = TfidfVectorizer(stop_words="english")
    features_TFIDF = vectorizer.fit_transform(corpus)

    return features_TFIDF
    
def compute_TFIDF_abstract_feature(citation_set, node_info, IDs, stpwds,  stemmer, TFIDF_abstract=None):
    '''
    Returns the feature extracted from the computation of the TFIDF applied on the abstract data
    '''
     
        dist_abstract = []
        if TFIDF_abstract != TFIDF_abstract:
            TFIDF_abstract = abstract_TFIDF(node_info, stpwds, stemmer)


        #Dictionary with the paper ID as the key and its index in node_info as the value
        #This help skip a loop afterwards while looking up at a paper's info, and is used in several other following functions
        
        reverse_index = dict()
        for i,a in enumerate(node_info):
            reverse_index[int(a[0])] = i

        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            dist_abstract.append((TFIDF_abstract[reverse_index[source]].dot(TFIDF_abstract[reverse_index[target]].T))[0,0])
    
            counter += 1
            if counter % 30000 == True:
                print(counter, "training examples processsed")

        dist_abstract = np.array(dist_abstract)
        
        return dist_abstract, TFIDF_abstract
    
def compute_TFIDF_title_feature(citation_set, node_info, IDs, stpwds, stemmer, TFIDF_title=None):
    '''
    Returns the feature extracted from the computation of the TFIDF applied on the title data
    '''
    
        dist_title = []
        
        if TFIDF_title != TFIDF_title:
            TFIDF_title = TFIDF_title = title_TFIDF(node_info, stpwds, stemmer)
        
        reverse_index = dict()
        for i,a in enumerate(node_info):
            reverse_index[int(a[0])] = i
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            dist_title.append((TFIDF_title[reverse_index[source]].dot(TFIDF_title[reverse_index[target]].T))[0,0])
    
            counter += 1
            if counter % 30000 == True:
                print(counter, "training examples processsed")

        dist_title = np.array(dist_title)
        
        return dist_title, TFIDF_title
    
def compute_temp_diff(citation_set, node_info, IDs):
    '''
    Returns the feature corresponding to the year difference between the papers on an edge
    '''
    
        temp_diff = []

        reverse_index = dict()
        for i,a in enumerate(node_info):
            reverse_index[int(a[0])] = i
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            source_info = node_info[reverse_index[source]]
            target_info = node_info[reverse_index[target]]
            
            temp_diff.append(
                    int(source_info[1]) - int(target_info[1])
                    )
            
            counter += 1
            if counter % 30000 == True:
                print(counter, "training examples processsed")
        
        temp_diff = np.array(temp_diff)
        
        return temp_diff
    
def compute_overlaping_titles(citation_set, node_info, IDs, stemmer, stpwds):
    '''
    Returns the feature of the number of intersecting words between the two papers' title
    '''
   
        overlap_title = []
        
        reverse_index = dict()
        for i,a in enumerate(node_info):
            reverse_index[int(a[0])] = i        
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            source_info = node_info[reverse_index[source]]
            target_info = node_info[reverse_index[target]]
            
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
            if counter % 30000 == True:
                print(counter, "training examples processsed")
        
        return np.array(overlap_title)
            
        
        
def compute_common_auth(citation_set, node_info, IDs):
    '''
    Returns the feature of the number of intersecting authors between the two papers
    '''
   
        comm_auth = []
        
        reverse_index = dict()
        for i,a in enumerate(node_info):
            reverse_index[int(a[0])] = i
        
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
            source_info = node_info[reverse_index[source]]
            target_info = node_info[reverse_index[target]]
            
            if type(source_info[3]) != float:
                source_auth = source_info[3].split(",")
            else:
                source_auth = []
                
            if type(target_info[3]) != float:
                target_auth = target_info[3].split(",")
            else:
                target_auth = []
            
            comm_auth.append(
                    len(set(source_auth).intersection(set(target_auth)))
                    )
            
            counter += 1
            if counter % 30000 == True:
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
    '''
    Returns the feature, an array of 1 or 0 whether the papers have been published in the same journal or not
    '''
   
        same_journal = []

        reverse_index = dict()
        for i,a in enumerate(node_info):
            reverse_index[int(a[0])] = i
            
        counter = 0
        for i in range(len(citation_set)):
            source = citation_set[i][0]
            target = citation_set[i][1]
    
         
            source_info = node_info[reverse_index[source]]
            target_info = node_info[reverse_index[target]]

            
            same_journal.append(
                    int(source_info[4] == target_info[4])
                    )
            
            counter += 1
            if counter % 30000 == True:
                print(counter, "training examples processsed")
        
        same_journal = np.array(same_journal)
        
        return same_journal

        
def compute_all_features_training(citation_set, node_info, IDs):
    '''
    Compute all of the training features and returns them concatenated
    '''
    
    stpwds, stemmer = initialize_nltk()
    
    graph_article = GF.graph_articles(citation_set)
    graph_authors = GF.graph_authors(citation_set, node_info, IDs)
    graph_authors_weight = GF.graph_authors_weight(citation_set, node_info, IDs)
    
    doc2voc_title = title_doc2voc(
            node_info, 
            citation_set
            ).reshape(len(citation_set),1)
    doc2voc_abstract = abstract_doc2voc(
            node_info, 
            citation_set
            ).reshape(len(citation_set),1)
    h_indexe = h_index(
            citation_set, 
            node_info, 
            graph_article
            ).reshape(len(citation_set),1)
    TFIDF_abstract, TFIDF_sparse_abstract = compute_TFIDF_abstract_feature(
            citation_set, 
            node_info, 
            IDs, 
            stpwds, 
            stemmer
            ).reshape(len(citation_set),1)
    TFIDF_title, TFIDF_sparse_title = compute_TFIDF_title_feature(
            citation_set, 
            node_info, 
            IDs, 
            stpwds, 
            stemmer
            ).reshape(len(citation_set),1)
    temp_diff = compute_temp_diff(
            citation_set, 
            node_info, 
            IDs
            ).reshape(len(citation_set),1)
    overlaping_title = compute_overlaping_titles(
            citation_set, 
            node_info, 
            IDs, 
            stemmer, 
            stpwds
            ).reshape(len(citation_set),1)
    common_auth = compute_common_auth(
            citation_set, 
            node_info, 
            IDs
            ).reshape(len(citation_set),1)
    article_page_rank = articles_page_rank_feature(
            citation_set, 
            node_info, 
            graph_article, 
            IDs
            ).reshape(len(citation_set),1)
    authors_page_rank = authors_page_rank_feature(
            citation_set, 
            node_info, 
            graph_authors, 
            IDs
            ).reshape(len(citation_set),1)
    page_club = page_club_feature(
            citation_set, 
            node_info, 
            graph_article
            ).reshape(len(citation_set),1)
    shorthest_path = shorthest_path_feature(
            citation_set, 
            node_info, 
            graph_article
            ).reshape(len(citation_set),1)
    author_affinity = compute_author_affinity(
            citation_set, 
            node_info, 
            graph_authors_weight, 
            IDs
            ).reshape(len(citation_set),1)
    same_journal_or_not = compute_same_journal_or_not(
            citation_set, 
            node_info
            ).reshape(len(citation_set),1)
    
    data = np.concatenate(
            doc2voc_title,
            doc2voc_abstract,
            h_indexe,
            TFIDF_abstract,
            TFIDF_title,
            temp_diff,
            overlaping_title,
            common_auth,
            article_page_rank,
            authors_page_rank,
            page_club,
            shorthest_path,
            author_affinity,
            same_journal_or_not
            )
    
    return data, graph_article, graph_authors, graph_authors_weight, TFIDF_sparse_abstract, TFIDF_sparse_title

def compute_all_features_testing(citation_set, node_info, IDs, graph_article, graph_authors, graph_authors_weight, TFIDF_sparse_abstract, TFIDF_sparse_title):
    '''
    Compute all of the testing features and returns them concatenated
    '''
    
    stpwds, stemmer = initialize_nltk()
    
    doc2voc_title = title_doc2voc(
            node_info, 
            citation_set
            ).reshape(len(citation_set),1)
    doc2voc_abstract = abstract_doc2voc(
            node_info, 
            citation_set
            ).reshape(len(citation_set),1)
    h_indexe = h_index(
            citation_set, 
            node_info, 
            graph_article
            ).reshape(len(citation_set),1)
    TFIDF_abstract, TFIDF_sparse_abstract = compute_TFIDF_abstract_feature(
            citation_set, 
            node_info, 
            IDs, 
            stpwds, 
            stemmer,
            TFIDF_sparse_abstract
            ).reshape(len(citation_set),1)
    TFIDF_title, TFIDF_sparse_title = compute_TFIDF_title_feature(
            citation_set, 
            node_info, 
            IDs, 
            stpwds, 
            stemmer,
            TFIDF_sparse_title
            ).reshape(len(citation_set),1)
    temp_diff = compute_temp_diff(
            citation_set, 
            node_info, 
            IDs
            ).reshape(len(citation_set),1)
    overlaping_title = compute_overlaping_titles(
            citation_set, 
            node_info, 
            IDs, 
            stemmer, 
            stpwds
            ).reshape(len(citation_set),1)
    common_auth = compute_common_auth(
            citation_set, 
            node_info, 
            IDs
            ).reshape(len(citation_set),1)
    article_page_rank = articles_page_rank_feature(
            citation_set, 
            node_info, 
            graph_article, 
            IDs
            ).reshape(len(citation_set),1)
    authors_page_rank = authors_page_rank_feature(
            citation_set, 
            node_info, 
            graph_authors, 
            IDs
            ).reshape(len(citation_set),1)
    page_club = page_club_feature(
            citation_set, 
            node_info, 
            graph_article
            ).reshape(len(citation_set),1)
    shorthest_path = shorthest_path_feature(
            citation_set, 
            node_info, 
            graph_article
            ).reshape(len(citation_set),1)
    author_affinity = compute_author_affinity(
            citation_set, 
            node_info, 
            graph_authors_weight, 
            IDs
            ).reshape(len(citation_set),1)
    same_journal_or_not = compute_same_journal_or_not(
            citation_set, 
            node_info
            ).reshape(len(citation_set),1)
    
    data = np.concatenate(
            doc2voc_title,
            doc2voc_abstract,
            h_indexe,
            TFIDF_abstract,
            TFIDF_title,
            temp_diff,
            overlaping_title,
            common_auth,
            article_page_rank,
            authors_page_rank,
            page_club,
            shorthest_path,
            author_affinity,
            same_journal_or_not
            )
    
    return data
    
    
    




    

        
