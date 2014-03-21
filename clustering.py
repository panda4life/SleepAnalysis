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
    analyze
    initialLearn
    kmeansDist
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
        self.haveLearned = False

        self.pc_eigen = []
        self.pc_tstamp = []

        self.samples = []
        self.sp_tstamp = []

        self.delta = np.array([1, 4])
        self.theta = np.array([4, 8])
        self.alpha = np.array([8, 12])
        self.beta = np.array([12, 30])

    def appendSample(self, x, sampleRate, timestamp=None, category=-1):
        if(timestamp is None):
            self.sp_tstamp = np.append(self.sp_tstamp,dt.datetime.now())
        else:
            self.sp_tstamp = np.append(self.sp_tstamp,timestamp)
        bands = np.vstack([self.delta, self.theta, self.alpha, self.beta])
        newSample = sample(x, sampleRate, bands, category)
        self.samples = np.append(self.samples, newSample)
        return newSample

    def analyze(self):
        featureVectors = []
        for i in np.arange(0,self.samples.shape[0]):
            """ #pure band powers
            if featureVectors == []:
                featureVectors = self.samples[i].featureVector()
            else:
                featureVectors = np.vstack((featureVectors, self.samples[i].featureVector()))
            """
            #band ratios
            feaVec = self.samples[i].featureVector()
            bandRatioFeaVec = [feaVec[7]/feaVec[8],feaVec[6]/feaVec[7],feaVec[6]**2/feaVec[7]/feaVec[8]]
            if featureVectors == []:
                featureVectors = np.append(feaVec[0:6],bandRatioFeaVec[:])
            else:
                featureVectors = np.vstack((featureVectors, np.append(feaVec[0:6],bandRatioFeaVec[:])))

        #Analyze the given block size and dump only the most recent eigenvector
        #if self.pc_eigen == []:
        self.pc_eigen = self.pca.fit_transform(featureVectors)#[-1,:]
        #else:
        #    self.pc_eigen = np.vstack((self.pc_eigen, self.pca.fit_transform(featureVectors)[-1,:]))

    def initialLearn(self, categories = set()):
        if(len(self.samples)==0):
            print('Cannot learn with no learning set. Exiting')
            return
        if(self.haveLearned):
            print('Categories have already been parsed and learned')
            return
        if(len(categories) == 0): #Does not enter cats
            self.categories = categories
            for s in self.samples:
                self.categories.add(s.category)
        else:
            for s in self.samples:
                if s.category not in categories:
                    print('Category %s not in list' % self.samples[i].category)
                    return
            self.categories = categories

        self.analyze()
        self.haveLearned = True

    def kmeansDist(self, sampleIndex, category):
        catSet = []
        if(self.haveLearned == False):
            print('Dataset has not been learned, kmeans_dist cannot be computed. Exiting')
            return

        #the vector of indices of self.samples in which the samples are categorized as a given category
        catSet = [i for i,s in enumerate(self.samples) if(s.category==category)]
        nPriorSamples = len(catSet)

        #if there exist samples that have not been analyzed, do dim reduction    
        if(len(self.samples)>len(self.pc_eigen)):
            self.analyze()
        
        kmeans = 0
        for i in catSet:
            dist2 = 0
            for j in np.arange(0,self.dims):
                dist2 += (self.pc_eigen[i,j] - self.pc_eigen[sampleIndex,j])**2
            kmeans += (dist2**.5)/nPriorSamples
        return kmeans

    def classifyNewSamples(self):
        '''
        FIXME:
        Cut data set in half and remove annotations 
        Used learned set to categorize using kmean min distance
        '''
        if(len(self.sampleClass) == len(self.samples)):
            print('No new samples to classify. Exiting')
            return
        self.analyze()
        newSampleSet = np.arange(len(self.sampleClass),len(self.samples))
        for sampInd in newSampleSet:
            classification = None
            minDist = float('inf')
            for catInd in np.arange(0,len(self.categories)):
                kmeans = self.kmeans_dist(sampInd,catInd)
                if(minDist > kmeans):
                    minDist = kmeans
                    classification = self.categories[catInd]
            self.sampleClass = np.append(self.sampleClass,classification)

