# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 12:42:42 2014

@author: James Ahad
"""
from numpy import *
from pyplot import *
from sklearn.decomposition import PCA
import sampleParameters
class clusteredData:
    #blockSize = number of feature vectors in a block that is to be
    #            decomposed, defaults to 30
    #dimensionality = number of dimensions to cluster the data
    #                 defaults to 3
    def __init__(self,blockSize=30,dimensionality=3):
        self.blockSize = blockSize
        self.dims = dimensionality
        self.pca = PCA(self.dims,copy=False,whiten=True)
        self pc_history = array([0])
        self.samples=[]
        self.delta = array([1,4])
        self.theta = array([4,8])
        self.alpha = array([8,12])
        self.beta = array([12,30])
    
    def appendSample(x,samplerate):
        bands = matrix([delta,theta,alpha,beta])
        newSample = sample(x,samplerate,bands)
        append(samples,newSample)
        return newSample
        
    def analyze(start=-self.blockSize,ending=-1):
        featureVectors = samples[start:ending].featureVectors()
        if pc_history.shape[0] == 1 and pc_history.shape[0] == 1:
            pc_history = self.pca.fit_transform(featureVectors)
        else:
            vstack((pc_history,self.pca.fit_transform(featureVectors)))
        
        
        