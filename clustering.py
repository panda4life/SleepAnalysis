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

    def appendSample(self, x, sampleRate,timestamp=None):
        if(timestamp is None):
            self.sp_tstamp = np.append(self.sp_tstamp,dt.datetime.now())
        else:
            self.sp_tstamp = np.append(self.sp_tstamp,timestamp)
        bands = np.vstack([self.delta, self.theta, self.alpha, self.beta])
        newSample = sample(x, sampleRate, bands)
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

    def initialLearn(self, sampleSet, categoryVector, ncategories = -1, categories = []):
        if(len(self.samples)!=0):
            print('Cannot learn when data is already input, Exiting')
            return
        if(len(sampleSet) != len(categoryVector)):
            print('Each sample must have a matched category, Exiting')
            return
        self.sampleClass = categoryVector
        if(ncategories == -1):
            if(len(categories) == 0): #Does not enter ncats or cats
                ncategories = max(self.category)
                self.categories = np.arange(0,ncategories)
            else: #Does not enter ncats, but gives cats
                ncategories = len(categories)
                self.categories = categories
        elif(len(categories) == 0): #Enters ncats but does not give cats
            self.categories = np.arange(0,ncategories)
        else: #Enters both ncats and cats
            if(ncategories != len(categories)):
                print('Specified number of categories is not equal to the number of given categories, Exiting')
                return
            self.categories = categories
        for i in np.arange(0,len(sampleSet)):
            self.appendSample(sampleSet(i))
        self.haveLearned = True

    def kmeans_dist(self, sampleIndex, categoryIndex=-1, category=None):
        catSet = []
        if(self.haveLearned == False):
            print('Dataset has not been learned, kmeans_dist cannot be computed. Exiting')
            return
        if(category == None and categoryIndex == -1): #Did not specify any category
            print('Please specify classification to compute distance to. Exiting')
            return
        elif(categoryIndex == -1 and category != None): #Entered only category index
            catSet = np.where(self.sampleClass == self.categories(categoryIndex))
        elif(categoryIndex != -1 and category == None): #Entered only category
            catSet = np.where(self.sampleClass == category)
        elif(categoryIndex != -1 and category != None):
            if(self.categories(categoryIndex) != category): #Entered both but doesnt match
                print('Entered categoryIndex does not match entered category. Exiting')
                return
            else: #Entered both and matches, default to searching based on category
                catSet = np.where(self.sampleClass == category)
        if(sampleIndex > len(self.pc_eigen) or (sampleIndex < 0 and len(self.pc_eigen)<len(self.samples))):
            self.analyze()

        nPriorSamples = len(catSet)
        kmeans = 0;
        for i in catSet:
            dist2 = 0
            for j in np.arange(0,self.dims):
                dist2 += (self.pc_eigen[i,j] - self.pc_eigen[sampleIndex,j])**2
            kmeans += (dist2**.5)/nPriorSamples
        return kmeans

    def classifyNewSamples(self):
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