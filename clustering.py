# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 12:42:42 2014

@author: James Ahad
"""

"""
TESTING NOTES (3/11/14)
Working:
    __init__
    appendSample
"""
import datetime as dt
import numpy as np
#import pyplot as pyp
from sklearn.decomposition import KernelPCA
from sampleParameters import sample


class clusteredData:
    # blockSize = number of feature vectors in a block that is to be
    #            decomposed, defaults to 30
    # dimensionality = number of dimensions to cluster the data
    #                 defaults to 3

    def __init__(self, dimensionality=3):
        self.dims = dimensionality
        self.pca = KernelPCA(self.dims)

        self.pc_eigen = []
        self.pc_tstamp = []

        self.samples = []
        self.sp_tstamp = []

        self.delta = np.array([1, 4])
        self.theta = np.array([4, 8])
        self.alpha = np.array([8, 12])
        self.beta = np.array([12, 30])

    def appendSample(self, x, sampleRate,timestamp=None):
        if(timestamp is None):
            self.sp_tstamp = np.append(self.sp_tstamp,dt.datetime.now())
        else:
            self.sp_tstamp = np.append(self.sp_tstamp,timestamp)
        bands = np.vstack([self.delta, self.theta, self.alpha, self.beta])
        newSample = sample(x, sampleRate, bands)
        self.samples = np.append(self.samples, newSample)
        return newSample

    def analyze(self, blockSize=None, timestamp=None):
        #First estabilish time stamp for beginning of analysis
        if(timestamp is None):
            self.pc_tstamp = np.append(self.pc_tstamp,dt.datetime.now())
        else:
            self.pc_tstamp = np.append(self.pc_tstamp,timestamp)

        #determine the blocksize to analyze
        if(blockSize is None):
            sampleSet = np.arange(self.samples.shape[0])
        else:
            sampleSet = np.arange(self.samples.shape[0]-blockSize,self.samples.shape[0])
        featureVectors = []
        for i in sampleSet:
            if featureVectors == []:
                featureVectors = self.samples[i].featureVector()
            else:
                featureVectors = np.vstack((featureVectors, self.samples[i].featureVector()))

        #Analyze the given block size and dump only the most recent eigenvector
        #if self.pc_eigen == []:
        self.pc_eigen = self.pca.fit_transform(featureVectors)#[-1,:]
        #else:
        #    self.pc_eigen = np.vstack((self.pc_eigen, self.pca.fit_transform(featureVectors)[-1,:]))
