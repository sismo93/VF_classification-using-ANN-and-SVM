import wfdb
import numpy as np 
from anotation import createAnnotationArray,createCUDBAnnotation
from data_structures import EcgSignal
from feature_extraction import inverte_signal, compute_rpeaks_features_vf
from filtre import filter_passe_HB,filtre_notch


def processMITMVADBFile(path_annexe,Te,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD,list_feature_qrsa,list_feature_qrsasd,list_s,list_r_peaks,list_q,list_mean_rr,list_sdsd,list_pnn50,list_rmssd,list_sdnn,list_lf_power,list_hf_power,list_lfhf_ratio,channel,list_pos_area,list_neg_area,list_vlf_power,list_ulf_power,list_lfnu,list_hfnu):



	signals, fields = wfdb.rdsamp(path_annexe) # A 2d numpy array storing the physical signals from the record.

	 										# collect the signal and metadata
	Fs=fields['fs']							# sampling frequency 

	channel1Signal = []						# channel 1 signal
	channel2Signal = []						# channel 2 signal
	for i in signals:						# separating the two channels
		channel1Signal.append(i[0])			
		channel2Signal.append(i[1])


	channel1Signal = np.array(channel1Signal)		# converting lists to numpy arrays
	channel2Signal = np.array(channel2Signal)

	annotation = wfdb.rdann(path_annexe, 'atr')		# collecting the annotation
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

	
	counter=0

	while((i+nSamplesInEpisode)<len(channel1Signal)):
			# loop through the whole signal

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

		if(Noise<len(channel1Signal[i:j])):
			k=k+1						# noisy episode
			
			counter+=1
            
			
		
			if channel==1:
																			# saving channel 1 signal
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Channel1',source='MITMVAdb',Fs=Fs)
				
				list_label.append(ecgEpisode.annotation)
				list_signal.append(ecgEpisode.signal)
				list_channel.append(ecgEpisode.channel)
				list_fs.append(ecgEpisode.Fs)
				list_source.append(ecgEpisode.source)

				#compute_RR_intervall(ecgEpisode.signal,ecgEpisode.Fs,"vfdb")
				
				
				record_filtered = filter_passe_HB(ecgEpisode.signal,ecgEpisode.Fs,'highpass', fc=0.5)
				record_filtered_final = filter_passe_HB(record_filtered,ecgEpisode.Fs,'lowpass', fc=100)
				record_filtered_final_notch = filtre_notch(record_filtered_final,ecgEpisode.Fs,60,20)

				


				
				RDAmpM,RDAmpSD,qrsa,qrsasd,r_peaks, q_points, s_points,mean_rr, rmssd, sdsd, sdnn, pnn50,lf_power, hf_power, vlf_power, ulf_power, lfhf_ratio, lfnu, hfnu,pos_area ,neg_area=compute_rpeaks_features_vf(record_filtered_final_notch,ecgEpisode.Fs,"vfdb",counter)
			
				list_feature_RDAmpM.append(RDAmpM)
				list_pnn50.append(pnn50)
				list_feature_RDAmpSD.append(RDAmpSD)

				list_feature_qrsa.append(qrsa)
				list_feature_qrsasd.append(qrsasd)
				list_r_peaks.append(r_peaks)
				list_q.append(q_points)
				list_s.append(s_points)

				list_mean_rr.append(mean_rr)
				list_sdsd.append(sdsd)
				list_rmssd.append(rmssd)
				list_sdnn.append(sdnn)

				list_lf_power.append(lf_power)
				list_hf_power.append(hf_power)
				list_lfhf_ratio.append(lfhf_ratio)
				list_pos_area.append(pos_area)
				list_neg_area.append(neg_area)
				list_vlf_power.append(vlf_power)
				list_ulf_power.append(ulf_power)
				list_lfnu.append(lfnu) 
				list_hfnu.append(hfnu)
				

				

			else:
																						# saving channel 1 signal
				ecgEpisode = EcgSignal(signal=channel2Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Channel2',source='MITMVAdb',Fs=Fs)
				
				list_label.append(ecgEpisode.annotation)
				list_signal.append(ecgEpisode.signal)
				list_channel.append(ecgEpisode.channel)
				list_fs.append(ecgEpisode.Fs)
				list_source.append(ecgEpisode.source)

				#compute_RR_intervall(ecgEpisode.signal,ecgEpisode.Fs,"vfdb")
				
				record_filtered = filter_passe_HB(ecgEpisode.signal,ecgEpisode.Fs,'highpass', fc=0.5)
				record_filtered_final = filter_passe_HB(record_filtered,ecgEpisode.Fs,'lowpass', fc=100)
				record_filtered_final_notch = filtre_notch(record_filtered_final,ecgEpisode.Fs,60,20)

				


				
				
				RDAmpM,RDAmpSD,qrsa,qrsasd,r_peaks, q_points, s_points,mean_rr, rmssd, sdsd, sdnn, pnn50,lf_power, hf_power, vlf_power, ulf_power, lfhf_ratio, lfnu, hfnu,pos_area ,neg_area=compute_rpeaks_features_vf(record_filtered_final_notch,ecgEpisode.Fs,"vfdb",counter)
				list_feature_RDAmpM.append(RDAmpM)
				list_pnn50.append(pnn50)
				list_feature_RDAmpSD.append(RDAmpSD)

				list_feature_qrsa.append(qrsa)
				list_feature_qrsasd.append(qrsasd)
				list_r_peaks.append(r_peaks)
				list_q.append(q_points)
				list_s.append(s_points)

				list_mean_rr.append(mean_rr)
				list_sdsd.append(sdsd)
				list_rmssd.append(rmssd)
				list_sdnn.append(sdnn)

				list_lf_power.append(lf_power)
				list_hf_power.append(hf_power)
				list_lfhf_ratio.append(lfhf_ratio)
				list_pos_area.append(pos_area)
				list_neg_area.append(neg_area)
				list_vlf_power.append(vlf_power)
				list_ulf_power.append(ulf_power)
				list_lfnu.append(lfnu) 
				list_hfnu.append(hfnu)

		



		i += nSamplesIn1Sec								# sliding the window

	










def processCUDBFile(path_annexe,Te,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD,list_feature_qrsa,list_feature_qrsasd,list_s,list_r_peaks,list_q,list_mean_rr,list_sdsd,list_pnn50,list_rmssd,list_sdnn,list_lf_power,list_hf_power,list_lfhf_ratio,channel,list_pos_area,list_neg_area,list_vlf_power,list_ulf_power,list_lfnu,list_hfnu):


	signals, fields = wfdb.rdsamp(path_annexe) # A 2d numpy array storing the physical signals from the record.

	 										# collect the signal and metadata
	Fs=fields['fs']							# sampling frequency 

	channel1Signal = signals[:, 0]


	annotation = wfdb.rdann(path_annexe, 'atr')# collecting the annotation
	annotIndex = annotation.sample				# annotation indices
	annotSymbol = annotation.symbol				# annotation symbols

							# creating the annotation array
	annotationArray = createCUDBAnnotation(annotationIndex=annotIndex,annotationArr=annotSymbol,lenn=len(channel1Signal))

	nSamplesIn1Sec = Fs					# computing samples in one episode
	nSamplesInEpisode = Te * Fs
	i=0									# episode counter
	counter=0
	while((i+nSamplesInEpisode)<len(channel1Signal)):		# loop through the whole signal

		j = i + nSamplesInEpisode

		VF = 0												# VF indices
		NSR = 0												# NSR indices
		notVF = 0											# Not VF indices
		Noise = 0											# Noise indices


		for k in range(i,j):

			if(annotationArray[k]=='VF'):
				VF+=1
			else:
				notVF+=1

		if(Noise < len(channel1Signal[i:j])/2):


			ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Channel1',source='cudb',Fs=Fs)
			
			list_label.append(ecgEpisode.annotation)
			list_signal.append(ecgEpisode.signal)
			list_channel.append(ecgEpisode.channel)
			list_fs.append(ecgEpisode.Fs)
			list_source.append(ecgEpisode.source)
			record_filtered = filter_passe_HB(ecgEpisode.signal,ecgEpisode.Fs,'highpass', fc=0.5)
			record_filtered_final = filter_passe_HB(record_filtered,ecgEpisode.Fs,'lowpass', fc=100)
			record_filtered_final_notch = filtre_notch(record_filtered_final,ecgEpisode.Fs,60,20)

			
			
			
			RDAmpM,RDAmpSD,qrsa,qrsasd,r_peaks, q_points, s_points,mean_rr, rmssd, sdsd, sdnn, pnn50,lf_power, hf_power, vlf_power, ulf_power, lfhf_ratio,lfnu, hfnu,pos_area ,neg_area=compute_rpeaks_features_vf(record_filtered_final_notch,ecgEpisode.Fs,"cudb",counter)
			list_feature_RDAmpM.append(RDAmpM)
			list_pnn50.append(pnn50)
			list_feature_RDAmpSD.append(RDAmpSD)

			list_feature_qrsa.append(qrsa)
			list_feature_qrsasd.append(qrsasd)
			list_r_peaks.append(r_peaks)
			list_q.append(q_points)
			list_s.append(s_points)

			list_mean_rr.append(mean_rr)
			list_sdsd.append(sdsd)
			list_rmssd.append(rmssd)
			list_sdnn.append(sdnn)

			list_lf_power.append(lf_power)
			list_hf_power.append(hf_power)
			list_lfhf_ratio.append(lfhf_ratio)
			list_pos_area.append(pos_area)
			list_neg_area.append(neg_area)
			list_vlf_power.append(vlf_power)
			list_ulf_power.append(ulf_power)
			list_lfnu.append(lfnu) 
			list_hfnu.append(hfnu)
	

			

		i += nSamplesIn1Sec								# sliding the window




