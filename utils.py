




import os 
import pandas as pd
from preprocess import processCUDBFile
from preprocess import processMITMVADBFile



def extract_file(database,path):

    dat_files_all = []

    for dirname, _, filenames in os.walk(path):
        dat_files = [ fi for fi in filenames if fi.endswith(".dat") ]
        for filename in dat_files:
            dat_files_all.append(filename)
    print( str(database) +' : '+str(len(dat_files_all)))
    return dat_files_all




def processDB(Te,file_name,database,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD):
	print('Processing files '+str(database))

	for i in file_name:	
		files=os.path.join("database",os.path.join(str(database),str(i)))
		

		if database=="vfdb":

			if (os.path.isfile(files)):
			
				fileNo=i.split(".")
				files_annexe=os.path.join("database",os.path.join(str(database),str(fileNo[0])))

				processMITMVADBFile(files_annexe,Te,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD)

		elif database=="cudb":

			if (os.path.isfile(files)):
			
				fileNo=i.split(".")
				files_annexe=os.path.join("database",os.path.join(str(database),str(fileNo[0])))
				print(files_annexe)

				processCUDBFile(files_annexe,Te,list_label,list_signal,list_channel,list_fs,list_source)

	return list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD


def create_Dataframe_location( ):

	try:							# creating the necessary directories
		os.mkdir('dataframe')
		os.mkdir('dataframe/cudb')
		os.mkdir('dataframe/vfdb')
	except:
		pass
				# process MITMVA db files


def to_dataframe(Te,database,file_name):

    df = pd.DataFrame(columns=['label','signal','channel','db','fs'])
    list_label=[]
    list_signal=[]
    list_channel=[]
    list_fs=[]
    list_source=[]
    list_feature_RDAmpM=[]
    list_feature_RDAmpSD=[]
    list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD=processDB(Te,file_name,database,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD)
    df['label']=list_label
    df['signal']=list_signal
    df['channel']=list_channel
    df['fs']=list_fs
    df['db']=list_source
    df['list_feature_RDAmpM']=list_feature_RDAmpM
    df['list_feature_RDAmpSD']=list_feature_RDAmpSD

    return df