from scipy.signal import find_peaks
from plot import peaks_hr
import wfdb 
import numpy as np 
from detection import detect_r_peaks,detect_q_s,hrv_frequency_analysis
import matplotlib.pyplot as plt

def inverte_signal(signal):
    new_signal=[]
    for i in signal:
        if i <0:
            new_signal.append(abs(i))
        elif i==0:
            new_signal.append(0)
        else:
            new_signal.append(-i)
    new_signal=np.array(new_signal)
    return new_signal



def inverte_signal1(signal, r_peaks):
    # Get the 5 smallest amplitude values of the signal
    smallest_amps = np.sort(signal)[:5]

    print(smallest_amps)

    # Get the mean amplitude of the R-peaks
    r_peak_amps = signal[r_peaks]
    mean_r_peak_amp = np.mean(r_peak_amps)

    # Check if the mean R-peak amplitude is greater than the largest absolute amplitude value
    if mean_r_peak_amp < abs(smallest_amps[-1]):
        # Invert the signal
        new_signal=[]
        for i in signal:
            if i < 0:
                new_signal.append(abs(i))
            elif i == 0:
                new_signal.append(0)
            else:
                new_signal.append(-i)
        new_signal=np.array(new_signal)
        print("Signal inverted.")
    else:
        new_signal = signal
        print("Signal not inverted.")

    # Plot the original and inverted signal
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(signal, color='black')
    axs[0].set_title('Original Signal')
    axs[1].plot(new_signal, color='red')
    axs[1].set_title('Inverted Signal')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.show()

    return new_signal




def plot_signal_with_area(signal, fs):
    # Compute the area under the curve for the positive and negative parts of the signal
    pos_area = np.sum(signal[signal > 0])/fs
    neg_area = np.sum(signal[signal < 0])/fs
    """
    # Set the figure size
    plt.figure(figsize=(15,4))

    # Plot the signal
    plt.plot(signal, color='darkblue')

    # Highlight the positive and negative areas under the curve
    plt.fill_between(range(len(signal)), 0, signal, where=signal>=0, color='blue', alpha=0.3)
    plt.fill_between(range(len(signal)), 0, signal, where=signal<=0, color='red', alpha=0.3)

    # Add a legend for the areas
    plt.legend(['Signal', f'Positive area: {pos_area:.2f}', f'Negative area: {neg_area:.2f}'], loc='upper left')

    # Set the axis labels and title
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Signal with Positive and Negative Areas Highlighted')
    plt.savefig(f"./Healthy_nn/pos_area:{pos_area}.png", format='png')
    plt.show()
    """
    return pos_area, neg_area

def compute_rpeaks_features_vf(signal,FS,db,counter):

    #signal=inverte_signal(signal)

    pos_area ,neg_area=plot_signal_with_area(signal,FS)
  
    r_peaks, mean_rr, rmssd, sdsd, sdnn, pnn50=detect_r_peaks(signal,FS)
    #r_peaks=r_peaks[1:-1]
    #reaa=inverte_signal1(signal,r_peaks)
    mean_qrs_area, mean_r_peak_amp,std_r_peak_amp,std_qrs_area,r_peaks, q_points, s_points=detect_q_s(signal,r_peaks,FS,counter)

    lf_power, hf_power, vlf_power, ulf_power, lfhf_ratio, lfnu, hfnu=hrv_frequency_analysis(signal,FS)


    return mean_qrs_area, mean_r_peak_amp,std_r_peak_amp,std_qrs_area,r_peaks, q_points, s_points,mean_rr, rmssd, sdsd, sdnn, pnn50,lf_power, hf_power, vlf_power, ulf_power, lfhf_ratio, lfnu, hfnu,pos_area ,neg_area
    















def unique(list1):
     
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    final_list=[]
    for i in unique_list:
        final_list.append(abs(i))
    
    return final_list