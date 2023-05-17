"""
All the data structures used in VFPred
"""

import numpy as np

class EcgSignal(object):

	'''
		Class denoting ecg signals 		
	'''

	def __init__(self,signal,annotation,Fs,channel=None,source=None):

		self.signal = np.array(signal[:])		# signal as an array
		self.annotation = annotation			# annotation
		self.channel = channel					# channel number
		self.source = source					# name of database
		self.Fs=Fs								# sampling frequency

class Annotation(object):

	'''
		Class denoting beat anotations 		
	'''

	def __init__(self,index,label):

		self.index = index			# array of indices
		self.label = label			# array of labels

class Features(object):

	'''
		Extracted features from ecg signals 
	'''

	def __init__(self,Fs,label,file,episode,channel):

		self.Fs = Fs					# sampling frequency
		self.label=label				# annotation label 
		self.file=file					# file name (number actually)
		self.episode=episode			# episode number
		self.channel=channel			# channel number
	
