# -*- coding: utf-8 -*-
"""
Created on Thur Mar 06 22:01:22 2014

@author: James Ahad
"""

import numpy as np
from scipy import integrate as sc
class sample:
	#x = sample of sample size n
	#sampleRate = sample rate of input sample in Hz
	#bands = frequency bands of data to analyze, 
	#        defaults one band of 0Hz-30Hz
	def __init__(self,x,sampleRate,bands=np.array([[0,30]])):
		assert bands.shape[1] == 2
		self.x = x
		self.sampleRate = sampleRate
		self.dt = 1.0/self.sampleRate
		self.bands = bands

		''' FIXME: What can we calculate immediately, vs. put off until later?
			We should do certain calclations on initialization, like dt, since
			it is invariant across all functions.
			However, some functions need to be able to accept different parameters
			such as activity which uses two different arrays in the mobility function.
		'''

		# self.hjorth = self.hjorthCalc(x,1/self.sampleRate)
		# self.harmonicAllSignal = self.harmonic(self.sample,self.sampleRate)
		# self.harmonicBanded = self.harmonic(self.sample,self.sampleRate,self.bands)
		# self.bandPower = fft_bandpower(self.sample,self.sampleRate,self.bands)
		
	# set of functions to calculate the 3 hjorth parameters for a signal    
	def hjorth(self, x, dt):
		return np.array([self.activity(x), self.mobility(x, dt), self.complexity(x, dt)])
		
	def activity(self, x):
		return np.std(x)
	
	def mobility(self, x):
		dev = np.diff(x,n=1,axis=0)/self.dt
		return (self.activity(dev)/self.activity(x))**0.5

	def complexity(self, x):
		dev = np.diff(x,n=1,axis=0)/self.dt
		return self.mobility(dev)/self.mobility(x)
	
	#Helper function to generate spectral power density
	def spectralPower(self, x):
		#We take the magnitude of the FFT and square its magnitude to 
		#find spectral coefficients that correspond to power        
		fft = np.fft.fft(x) #V
		spectralCoeffs = np.absolute(fft)**2 #V^2 (approximates power)
		return spectralCoeffs
		
	def find_nearest(array,value):
		idx = (np.abs(array-value)).argmin()
		return array[idx]

	# function to calculate harmonic parameters
	def harmonic(self, x):
		#returns numpy.matrix
		#rows are bands, and columns are each parameter (# of bands x 3 matrix)
		assert self.bands.shape[1] == 2        
		spectralCoeffs = self.spectralPower(x)
		# freq = np.fftfreq(len(x),1/self.sampleRate)
		freq = np.fft.fftfreq(len(x), self.dt)
		fc = []
		fsig = []
		cenPower = []
		for i in np.arange(self.bands.shape[0]):
			intReg = (freq>=self.bands[i][0]) & (freq<=self.bands[i][1])
			freqInt = freq[intReg]
			specInt = spectralCoeffs[intReg]
			fc_temp = sc.cumtrapz(specInt*freqInt,freqInt)/sc.cumtrapz(specInt,freqInt)
			fsig_temp = (sc.cumtrapz((freqInt-fc_temp(-1))**2*specInt,freqInt)/sc.cumtrapz(specInt,freqInt))**0.5
			cenPower_temp = spectralCoeffs[find_nearest(freq,fc_temp(-1))]
			fc = np.append(fc,fc_temp(-1))
			fsig = np.append(fsig,fsig_temp(-1))
			cenPower = np.append(cenPower,cenPower_temp(-1))
		harmResult = np.hstack((fc,fsig,cenPower))
		return harmResult
	
	# function to calculate FFT and power spectra
	def fft_bandpower(self, x):
		print self.bands
		#returns array
		#each band is its own column, corresponding to each row of "bands"
		assert self.bands.shape[1] == 2
		#We then take the cumulative integral of the spectral
		#power density so we can calculate the power of each
		#band
		freq = np.fft.fftfreq(len(x), self.dt)
		cumPower = sc.cumtrapz(self.spectralPower(x),freq)
		bandPower = []
		for i in np.arange(self.bands.shape[0]):
			intReg = (freq>=self.bands[i][0]) & (freq<=self.bands[i][1])
			print intReg
			bandPower = np.append(bandPower,cumPower[intReg[-1]]-cumPower[intReg[0]])
		return bandPower
			
	#function to return the default feature vector    
	# <Hjorth (3 params), HarmonicAll(3 params), bandPower(# bands params)
	def featureVector(self):
		return np.append(self.hjorth,self.harmonicAllSignal,self.bandPower)

