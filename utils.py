import os 
import wfdb
import pandas as pd
from preprocessing import processCUDBFile,processMITMVADBFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def plot_roc_curve(X, y, n_splits, gamma, C, output_path):
    
    cv = StratifiedKFold(n_splits=n_splits)
    clf = SVC(kernel='rbf', gamma=gamma, C=C, probability=True)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    accuracies=[]

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            clf,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
        # Calculate and print accuracy per fold
        y_pred = clf.predict(X[test])
        accuracy = accuracy_score(y[test], y_pred)
        print(accuracy)
        accuracies.append(accuracy)

    # Calculate and print average accuracy
    print(f"Accuracy per folds:{accuracies}")
    average_accuracy = np.mean(accuracies)
    print(f"Average accuracy: {average_accuracy:.2f}")
        
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.savefig(f"{output_path}/cv={gamma}_C={C}.png")
    plt.show()






def downloadData():

	try:							# creating necessary directories
		os.mkdir('database')
		os.mkdir('database/cudb')
		os.mkdir('database/vfdb')
	except:
		pass 
	
	wfdb.dl_database('vfdb', dl_dir='database/vfdb')	# download mitMVAdb
	wfdb.dl_database('cudb', dl_dir='database/cudb')		# download cudb



def supprimer_lignes_na(df):
    return df.dropna(axis=0, how='any')

def extract_file(database,path):

    dat_files_all = []

    for dirname, _, filenames in os.walk(path):
        dat_files = [ fi for fi in filenames if fi.endswith(".dat") ]
        for filename in dat_files:
            dat_files_all.append(filename)
    print( str(database) +' : '+str(len(dat_files_all)))

    return dat_files_all




def to_dataframe(Te,database,file_name,channel):
    df = pd.DataFrame(columns=['label','signal','channel','db','fs'])
    list_label=[]
    list_signal=[]
    list_channel=[]
    list_fs=[]
    list_source=[]
    list_feature_RDAmpM=[]
    list_feature_RDAmpSD=[]
    list_feature_qrsa=[]
    list_feature_qrsasd=[]
    list_s=[]
    list_r_peaks=[]
    list_q=[]
    list_mean_rr=[]
    list_std_rr=[]
    list_rmssd=[]
    list_sdnn=[]
    list_sdsd=[]
    list_pnn50=[]
    list_lf_power=[]
    list_hf_power=[]
    list_lfhf_ratio=[]
    list_pos_area=[]
    list_neg_area=[]
    list_vlf_power=[]
    list_ulf_power=[]
    list_lfnu=[]
    list_hfnu=[]
    
    list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD,list_feature_qrsa,list_feature_qrsasd,list_s,list_r_peaks,list_q,list_mean_rr,list_sdsd,list_pnn50,list_rmssd,list_sdnn,list_lf_power,list_hf_power,list_lfhf_ratio,list_pos_area,list_neg_area,list_vlf_power,list_ulf_power,list_lfnu,list_hfnu=processDB(Te,file_name,database,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD,list_feature_qrsa,list_feature_qrsasd,list_s,list_r_peaks,list_q,list_mean_rr,list_sdsd,list_pnn50,list_rmssd,list_sdnn,list_lf_power,list_hf_power,list_lfhf_ratio,channel,list_pos_area,list_neg_area,list_vlf_power,list_ulf_power,list_lfnu,list_hfnu)
    df['label']=list_label
    df['signal']=list_signal
    df['channel']=list_channel
    df['fs']=list_fs
    df['db']=list_source
    df['RDAmpM']=list_feature_RDAmpM
    df['RDAmpSD']=list_feature_RDAmpSD
    df['QRsa']=list_feature_qrsa
    df['QRaSD']=list_feature_qrsasd
    df["S"]=list_s
    df["Q"]=list_q
    df["R"]=list_r_peaks
    df["mean_rr"]=list_mean_rr
    df["sdsd"]=list_sdsd
    df["pnn50"]=list_pnn50
    df["rmssd"]=list_rmssd
    df["sdnn"]=list_sdnn
    df["lf_power"]=list_lf_power
    df["hf_power"]=list_hf_power
    df["lfhf_ratio"]=list_lfhf_ratio
    df["pos_area"]=list_pos_area
    df["neg_area"]=list_neg_area
    df["vlf_power"]=list_vlf_power
    df["ulf_power"]=list_ulf_power
    df["lfnu"]=list_lfnu
    df["hfnu"]=list_hfnu
    return df


def processDB(Te,file_name,database,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD,list_feature_qrsa,list_feature_qrsasd,list_s,list_r_peaks,list_q,list_mean_rr,list_sdsd,list_pnn50,list_rmssd,list_sdnn,list_lf_power,list_hf_power,list_lfhf_ratio,channel,list_pos_area,list_neg_area,list_vlf_power,list_ulf_power,list_lfnu,list_hfnu):
	print('Processing files '+str(database))

	for i in file_name:	
		files=os.path.join("database",os.path.join(str(database),str(i)))
		

		if database=="vfdb":

			if (os.path.isfile(files)):
			
				fileNo=i.split(".")
				files_annexe=os.path.join("database",os.path.join(str(database),str(fileNo[0])))

				processMITMVADBFile(files_annexe,Te,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD,list_feature_qrsa,list_feature_qrsasd,list_s,list_r_peaks,list_q,list_mean_rr,list_sdsd,list_pnn50,list_rmssd,list_sdnn,list_lf_power,list_hf_power,list_lfhf_ratio,channel,list_pos_area,list_neg_area,list_vlf_power,list_ulf_power,list_lfnu,list_hfnu)

		elif database=="cudb":

			if (os.path.isfile(files)):
			
				fileNo=i.split(".")
				files_annexe=os.path.join("database",os.path.join(str(database),str(fileNo[0])))
				print(files_annexe)

				processCUDBFile(files_annexe,Te,list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD,list_feature_qrsa,list_feature_qrsasd,list_s,list_r_peaks,list_q,list_mean_rr,list_sdsd,list_pnn50,list_rmssd,list_sdnn,list_lf_power,list_hf_power,list_lfhf_ratio,channel,list_pos_area,list_neg_area,list_vlf_power,list_ulf_power,list_lfnu,list_hfnu)

	return list_label,list_signal,list_channel,list_fs,list_source,list_feature_RDAmpM,list_feature_RDAmpSD,list_feature_qrsa,list_feature_qrsasd,list_s,list_r_peaks,list_q,list_mean_rr,list_sdsd,list_pnn50,list_rmssd,list_sdnn,list_lf_power,list_hf_power,list_lfhf_ratio,list_pos_area,list_neg_area,list_vlf_power,list_ulf_power,list_lfnu,list_hfnu