# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 19:29:59 2022

@author: Alanr
"""

import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k = 3):
        self.k = k
    
    def learning(self, X, C):
        self.X = X 
        self.C = C
        self.n_muestras = X.shape[1]
        
    def classification(self, Y):
        clases = []
        for i in range(Y.shape[1]):
            distances = np.empty(self.n_muestras)
            for n in range(self.n_muestras):
                distances[n] = euclidea(self.X[:, n], Y[:,i])
                
            k_distances = np.argsort(distances)
            k_label = self.C[k_distances[:self.k]]
            C = Counter(k_label).most_common(1)
            
            clases.append(C[0][0])
        return clases
            
def euclidea(x,y):
    return np.sqrt(np.sum((x-y)**2))

