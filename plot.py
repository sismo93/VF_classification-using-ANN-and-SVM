import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def peaks_hr(sig, peak_inds,qrs_inds,point_x,point_Y, title,counter,saveto=None, figsize=(20, 10)):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    #hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
     
    N = sig.shape[0]
    
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    
    ax_left.plot(sig, color='#3979f0', label='Signal')
    #ax_left.plot(peak_inds, sig[peak_inds], 'x', 
                 #color='#8b0000', label='Peak', markersize=12)
    #ax_left.plot(point_x,point_Y, 'x', 
                 #color='green', label='Peak', markersize=12)
    #ax_left.plot(qrs_inds,sig[qrs_inds], 'x',
                 #color='blue', label='Peak', markersize=12)     
   

    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')

   
    
    if saveto is not None:
        plt.savefig("filenvn_"+str(counter), dpi=600)
    plt.show()






def print_r_peaks(sig, peak_inds, title, figsize=(20, 10), saveto=None):
    
    fig, ax_left = plt.subplots(figsize=figsize)
    
    ax_left.plot(sig, color='#3979f0', label='Signal')
    ax_left.plot(peak_inds, sig[peak_inds], 'rx', marker='x', 
                 color='#8b0000', label='Peak', markersize=12)
   
    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')
   
    ax_left.tick_params('y', colors='#3979f0')
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()





def compute_RR_intervall(ecg,fs,db,figsize=(20, 10),title="peak detection"):
    r_peaks, _ = find_peaks(ecg, distance=35,height=0.65)
    
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    #hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
     # r_peaks, _ = find_peaks(ecg, distance=15,height=0.25) for > 200
    N = ecg.shape[0]
    
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    
    ax_left.plot(ecg, color='#3979f0', label='Signal')
    ax_left.plot(r_peaks, ecg[r_peaks], 'rx', label='Peak', markersize=12)
      
    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')
    ax_right.set_ylabel('Heart rate (bpm)', color='m')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params('y', colors='#3979f0')
    ax_right.tick_params('y', colors='m')
    

    plt.show()
 