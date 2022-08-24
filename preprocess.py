
from scipy.signal import find_peaks
from data_structures import Annotation,EcgSignal
from extract_features import compute_rpeaks_features

import numpy as np 
import wfdb
import os






def downloadData():

	try:							# creating necessary directories
		os.mkdir('database')
		os.mkdir('database/cudb')
		os.mkdir('database/vfdb')
	except:
		pass 
	
	wfdb.dl_database('vfdb', dl_dir='database/vfdb')	# download mitMVAdb
	wfdb.dl_database('cudb', dl_dir='database/cudb')		# download cudb



def createAnnotationArray(indexArray,labelArray,hi,NSRsymbol):

	annotations = []

	for i in range(len(indexArray)):

		annotations.append(Annotation(index=indexArray[i],label=labelArray[i]))

	distributedAnnotations = createDistributedAnnotations(annotationArray=annotations,hi=hi,NSRsymbol=NSRsymbol)

	return distributedAnnotations


def createDistributedAnnotations(annotationArray,hi,NSRsymbol):

	labelArray=[]

	localLo = 0
	localHi = annotationArray[0].index
	currLabel = NSRsymbol

	## The following is similar to interval covering algorithms

	## We are assuming the first unannotated part to be NSR

	for i in range(localLo,localHi):

		labelArray.append(currLabel)


	## now for the other actual annotated segments

	for i in range(1,len(annotationArray)):			
													# interval
		localLo = annotationArray[i-1].index
		localHi = annotationArray[i].index
		currLabel = annotationArray[i-1].label

		for j in range(localLo,localHi):

			labelArray.append(currLabel)

	## for the last segment

	localLo = annotationArray[len(annotationArray)-1].index
	localHi = hi
	currLabel = annotationArray[len(annotationArray)-1].label

	for j in range(localLo, localHi):
		labelArray.append(currLabel)

	return labelArray				# point wise annotation array




def createCUDBAnnotation(annotationIndex,annotationArr,lenn):

	li = []							# initialize

	for i in range(lenn):						
		li.append('notVF')					

	st=-1
	en=-1


	for i in range(len(annotationArr)):

		if(annotationArr[i]=='N'):				

			li[i]='NSR'

	for i in range(len(annotationArr)):

		if(annotationArr[i]=='['):

			st = annotationIndex[i]

		if(annotationArr[i]==']'):

			en = annotationIndex[i]

			for j in range(st,en+1):

				li[j]='VF'
			st = -1
			en = -1

	if(st!=-1):

		for j in range(st,lenn):
			li[j] = 'VF'

	return np.array(li)





def processCUDBFile(files_annexe,Te,list_label,list_signal,list_channel,list_fs,list_source):

	signals, fields = wfdb.rdsamp(files_annexe) 		# collect the signal and metadata
	Fs=fields['fs']								# sampling frequency 	

	channel1Signal = []							# channel 1 signal

	for i in signals:
												# separating the two channels
		channel1Signal.append(i[0])

	channel1Signal = np.array(channel1Signal)	# converting lists to numpy arrays

	annotation = wfdb.rdann(files_annexe, 'atr')		# collecting the annotation
	annotIndex = annotation.sample				# annotation indices
	annotSymbol = annotation.symbol				# annotation symbols

							# creating the annotation array
	annotationArray = createCUDBAnnotation(annotationIndex=annotIndex,annotationArr=annotSymbol,lenn=len(channel1Signal))

	nSamplesIn1Sec = Fs					# computing samples in one episode
	nSamplesInEpisode = Te * Fs
	i=0									# episode counter

	while((i+nSamplesInEpisode)<len(channel1Signal)):		# loop through the whole signal

		j = i + nSamplesInEpisode

		VF = 0												# VF indices
		NSR = 0												# NSR indices
		notVF = 0											# Not VF indices
		Noise = 0											# Noise indices


		for k in range(i,j):

			if(annotationArray[k]=='VF'):
				VF+=1
			elif(annotationArray[k]=='NSR'):	# unnecessary
				NSR += 1
			else:
				notVF +=1

		if(Noise*3<nSamplesInEpisode):						# noisy episode

			if(2*VF>=nSamplesInEpisode):
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Only_Channel',source='cudb',Fs=Fs)
				
				list_label.append(ecgEpisode.annotation)
				list_signal.append(ecgEpisode.signal)
				list_channel.append(ecgEpisode.channel)
				list_fs.append(ecgEpisode.Fs)
				list_source.append(ecgEpisode.source)
			
			elif(2*NSR>=nSamplesInEpisode):
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='NSR',channel='Only_Channel',source='cudb',Fs=Fs)
				list_label.append(ecgEpisode.annotation)
				list_signal.append(ecgEpisode.signal)
				list_channel.append(ecgEpisode.channel)
				list_fs.append(ecgEpisode.Fs)
				list_source.append(ecgEpisode.source)
			
				
			else:
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='NotVF',channel='Only_Channel',source='cudb',Fs=Fs)
				list_label.append(ecgEpisode.annotation)
				list_signal.append(ecgEpisode.signal)
				list_channel.append(ecgEpisode.channel)
				list_fs.append(ecgEpisode.Fs)
				list_source.append(ecgEpisode.source)
			
	



		i += nSamplesIn1Sec								# sliding the window




def processMITMVADBFile(path_annexe,Te,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD):


	signals, fields = wfdb.rdsamp(path_annexe) # A 2d numpy array storing the physical signals from the record.


	 										# collect the signal and metadata
	Fs=fields['fs']							# sampling frequency 

	channel1Signal = []						# channel 1 signal
	channel2Signal = []						# channel 2 signal

	for i in signals:
											# separating the two channels
		channel1Signal.append(i[0])			
		channel2Signal.append(i[1])

	

	channel1Signal = np.array(channel1Signal)		# converting lists to numpy arrays
	channel2Signal = np.array(channel2Signal)

	annotation = wfdb.rdann(path_annexe, 'atr')			# collecting the annotation
	annotIndex = annotation.sample					# annotation indices
	annotSymbol = annotation.aux_note				# annotation symbols

	for i in range(len(annotSymbol)):

		annotSymbol[i] = annotSymbol[i].rstrip('\x00') # because the file contains \x00 

		if(annotSymbol[i]=='(N'):		# N = NSR
			annotSymbol[i]='(NSR'

		elif (annotSymbol[i] == '(VFIB'):	# VFIB = VF
			annotSymbol[i] = '(VF'




			# creating the annotation array
	annotationArr = createAnnotationArray(indexArray=annotIndex,labelArray=annotSymbol,hi=len(channel1Signal),NSRsymbol='(NSR') 
	


	nSamplesIn1Sec = Fs					# computing samples in one episode
	nSamplesInEpisode = Te * Fs	


	i=0									# episode counter

	while((i+nSamplesInEpisode)<len(channel1Signal)):			# loop through the whole signal

		j = i + nSamplesInEpisode

		VF = 0													# VF indices
		notVF = 0												# Not VF indices
		Noise =0												# Noise indices

		for k in range(i,j):

			if(annotationArr[k]=='(VF'):
				VF+=1
			else:						# anything other than VF
				notVF +=1

			if(annotationArr[k]=='(NOISE'):
				Noise += 1

		if(Noise*3<nSamplesInEpisode):
			k=k+1						# noisy episode

		
			
																			# saving channel 1 signal
			ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Channel1',source='MITMVAdb',Fs=Fs)
			
			list_label.append(ecgEpisode.annotation)
			list_signal.append(ecgEpisode.signal)
			list_channel.append(ecgEpisode.channel)
			list_fs.append(ecgEpisode.Fs)
			list_source.append(ecgEpisode.source)






            
			RDAmpM,RDAmpSD=compute_rpeaks_features(ecgEpisode.signal)
			list_feature_RDAmpM.append(RDAmpM)
			
			list_feature_RDAmpSD.append(RDAmpSD)

																			
																			# saving channel 2 signal
			ecgEpisode = EcgSignal(signal=channel2Signal[i:j], annotation='VF' if VF > notVF else 'NotVF', channel='Channel2', source='MITMVAdb', Fs=Fs)
			list_label.append(ecgEpisode.annotation)
			list_signal.append(ecgEpisode.signal)
			list_channel.append(ecgEpisode.channel)
			list_fs.append(ecgEpisode.Fs)
			list_source.append(ecgEpisode.source)


			RDAmpM,RDAmpSD=compute_rpeaks_features(ecgEpisode.signal)
			list_feature_RDAmpM.append(RDAmpM)
			
			list_feature_RDAmpSD.append(RDAmpSD)


		i += nSamplesIn1Sec								# sliding the window

	