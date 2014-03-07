# -*- coding: utf-8 -*-
"""
Created on Thur Mar 06 22:01:22 2014

@author: James Ahad
"""

from numpy import *
from scipy import integrate as sc
class sample:
    #x = sample of sample size n
    #samplerate = sample rate of input sample
    #bands = frequency bands of data to analyze, 
    #        defaults one band of 0Hz-30Hz
    def __init__(self,x,samplerate,bands=array([[0,30]])):
        assert bands.shape[1] == 2
        self.sample = x
        self.sampleRate = sampleRate
        self.bands = bands
        self.hjorth = hjorth(x,1/self.samplerate)
        self.harmonicAllSignal = harmonic(self.sample,self.sampleRate)
        self.harmonicBanded = harmonic(self.sample,self.sampleRate,self.bands)
        self.bandPower = fft_bandpower(self.sample,self.sampleRate,self.bands)
        
    # set of functions to calculate the 3 hjorth parameters for a signal    
    def hjorth(x,dt):
        return array([activity(x), mobility(x,dt), complexity(x,dt)])
        
    def activity(x):
        return std(x)
    
    def mobility(x,dt):
        dev = diff(x,n=1,axis=0)/dt
        return (activity(dev)/activity(x))^.5

    def complexity(x,dt):
        dev = diff(x,n=1,axis=0)/dt
        return mobility(dev)/mobility(x)
    
    #Helper function to generate spectral power density
    def spectralPower(x):
        #We take the magnitude of the FFT and square its magnitude to 
        #find spectral coefficients that correspond to power        
        fft = fft(x) #V
        spectralCoeffs = absolute(fft)^2 #V^2 (approximates power)
        return spectralCoeffs
        
    # function to calculate harmonic parameters
    def harmonic(x,samplerate,bands=array([[0, 30]])):
        #returns numpy.matrix
        #rows are bands, and columns are each parameter (# of bands x 3 matrix)
        assert bands.shape[1] == 2        
        spectralCoeffs = spectralPower(x)
        freq = fftfreq(len(x),1/sampleRate)
        fc = []
        fsig = []
        cenPower = []
        for i in arange(bands.shape[0]):
            intReg = find(freq>=bands[i][0] and freq<=bands[i][1])
            freqInt = freq[intReg]
            specInt = spectralCoeffs[intReg]
            fc_temp = sc.cumtrapz(specInt*freqInt,freqInt)/sc.cumtrapz(specInt,freqInt)
            fsig_temp = (sc.cumtrapz((freqInt-fc_temp(-1))^2*specInt,freqInt)/sc.cumtrapz(specInt,freqInt))^.5
            cenPower_temp = spectralCoeffs[find_nearest(freq,fc_temp(-1))]
            fc = append(fc,fc_temp(-1))
            fsig = append(fsig,fsig_temp(-1))
            cenPower = append(cenPower,cenPower_temp(-1))
        harmResult = hstack((fc,fsig,cenPower)
        return harmResult.T
    
    # function to calculate FFT and power spectra
    def fft_bandpower(x,sampleRate,bands=array([[0, 30]])):
        #returns array
        #each band is its own column, corresponding to each row of "bands"
        assert bands.shape[1] == 2
        #We then take the cumulative integral of the spectral
        #power density so we can calculate the power of each
        #band
        freq = fftfreq(len(x),1/sampleRate)
        cumPower = sc.cumtrapz(spectralPower(x),freq)
        bandPower = []
        for i in arange(bands.shape[0]):
            intReg = find(freq>=bands[i][0] and freq<=bands[i][1])
            bandPower = append(bandPower,cumPower[intReg[-1]]-cumPower[intReg[0]])
        return bandPower
            
    #function to return the default feature vector    
    # <Hjorth (3 params), HarmonicAll(3 params), bandPower(# bands params)
    def featureVector():
        return append(self.hjorth,self.harmonicAllSignal,self.bandPower)
        
    