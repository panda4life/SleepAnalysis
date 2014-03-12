# -*- coding: utf-8 -*-
"""
Created on Thur Mar 06 22:01:22 2014

@author: James Ahad
"""

"""
TESTING NOTES (3/11/14)
Working:
    __init__
    hjorth
    activity
    mobility
    complexity
    spectralPower
    harmonic
    fft_bandpower
    featureVector
    find_nearest
"""


import numpy as np
from scipy import integrate as sc
class sample:
	#x = sample of sample size n
	#sampleRate = sample rate of input sample in Hz
	#bands = frequency bands of data to analyze,
	#        defaults one band of 0Hz-30Hz
	def __init__(self,x,sampleRate,bands=np.array([[0,30]]),stageAnnotation=-1):
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
	def hjorth(self):
		return np.array([self.activity(), self.mobility(), self.complexity()])

	def activity(self):
		return np.std(self.x)

	def mobility(self):
		dev = sample(np.diff(self.x,n=1,axis=0)/self.dt,self.sampleRate)
		return (dev.activity()/self.activity())**0.5

	def complexity(self):
		dev = sample(np.diff(self.x,n=1,axis=0)/self.dt,self.sampleRate)
		return dev.mobility()/self.mobility()

	#Helper function to generate spectral power density
	def spectralPower(self):
		#We take the magnitude of the FFT and square its magnitude to
		#find spectral coefficients that correspond to power
		fft = np.fft.fft(self.x) #V
		spectralCoeffs = np.absolute(fft)**2 #V^2 (approximates power)
		return spectralCoeffs

	# function to calculate harmonic parameters
	def harmonic(self, bands=np.array([[0,30]])):
		#returns numpy.matrix
		#rows are bands, and columns are each parameter (# of bands x 3 matrix)
		assert bands.shape[1] == 2
		spectralCoeffs = self.spectralPower()
		# freq = np.fftfreq(len(x),1/self.sampleRate)
		freq = np.fft.fftfreq(len(self.x), self.dt)
		fc = []
		fsig = []
		cenPower = []
		for i in np.arange(bands.shape[0]):
			intReg = np.where(freq[np.where(freq<bands[i,1])]>bands[i,0])
			freqInt = freq[intReg]
			specInt = spectralCoeffs[intReg]
			fc_temp = sc.cumtrapz(freqInt*specInt,freqInt)/sc.cumtrapz(specInt,freqInt)
			fsig_temp = (sc.cumtrapz((freqInt-fc_temp[-1])**2*specInt,freqInt)/sc.cumtrapz(specInt,freqInt))**.5
			cenPower_temp = spectralCoeffs[find_nearest(freq,fc_temp[-1])]
			fc = np.append(fc,fc_temp[-1])
			fsig = np.append(fsig,fsig_temp[-1])
			cenPower = np.append(cenPower,cenPower_temp)
		harmResult = np.hstack((fc,fsig,cenPower))
		return harmResult

	# function to calculate FFT and power spectra
	def fft_bandpower(self):
		#returns array
		#each band is its own column, corresponding to each row of "bands"
		assert self.bands.shape[1] == 2
		#We then take the cumulative integral of the spectral
		#power density so we can calculate the power of each
		#band
		freq = np.fft.fftfreq(len(self.x), self.dt)
		cumPower = sc.cumtrapz(self.spectralPower(),freq)
		bandPower = []
		for i in np.arange(self.bands.shape[0]):
			start = find_nearest(freq,self.bands[i,0])
			ending = find_nearest(freq,self.bands[i,1])
			bandPower = np.append(bandPower,cumPower[ending]-cumPower[start])
		return bandPower

	#function to return the default feature vector
	# <Hjorth (3 params), HarmonicAll(3 params), bandPower(# bands params)
	def featureVector(self):
		return np.append(np.append([self.hjorth()],[self.harmonic()]),[self.fft_bandpower()])

#helper function that returns the index of the value in array "array" where the value is closest to "value"
def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx