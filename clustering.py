# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 12:42:42 2014

@author: James Ahad
"""
import numpy as np
import pyplot as pyp
from sklearn.decomposition import PCA
from sampleParameters import sample
class clusteredData:
    #blockSize = number of feature vectors in a block that is to be
    #            decomposed, defaults to 30
    #dimensionality = number of dimensions to cluster the data
    #                 defaults to 3
    def __init__(self,dimensionality=3):
        self.dims = dimensionality
        self.pca = PCA(self.dims,copy=False,whiten=True)
        self.pc_eigen=[]
        self.samples=[]
        self.delta = np.array([1,4])
        self.theta = np.array([4,8])
        self.alpha = np.array([8,12])
        self.beta = np.array([12,30])
    
    def appendSample(self,x,sampleRate):
        bands = np.vstack([self.delta,self.theta,self.alpha,self.beta])
        newSample = sample(x,sampleRate,bands)
        np.append(self.samples,newSample)
        return newSample
        
    def analyze(self,blockSize = None):
	   if(blockSize == None):
			featureVectors = self.samples[:].featureVectors()
	   else:
			featureVectors = self.samples[:-blockSize].featureVectors()
	   if self.pc_eigen.shape[0] == 1 and self.pc_eigen.shape[0] == 1:
            pc_history = self.pca.fit_transform(featureVectors)
	   else:
            np.vstack((pc_history,self.pca.fit_transform(featureVectors)))
        
        
        