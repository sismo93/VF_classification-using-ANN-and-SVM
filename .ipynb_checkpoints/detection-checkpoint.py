
import matplotlib.pyplot as plt
import numpy as np 
from wfdb.processing import find_local_peaks
import scipy.signal as signal
from scipy.fft import fft






def hrv_frequency_analysis(sig, fs):
    # Find all local peaks
    peaks = find_local_peaks(sig, radius=int(0.2*fs))
    r_peaks = peaks
    
    # Compute RR intervals
    rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds
    
    # Compute frequency domain features
    freq, power = signal.welch(rr_intervals, fs=1.0/np.mean(rr_intervals), nperseg=min(256, len(rr_intervals)))
    lf_power = np.trapz(power[(freq >= 0.04) & (freq < 0.15)])
    hf_power = np.trapz(power[(freq >= 0.15) & (freq < 0.4)])
    vlf_power = np.trapz(power[(freq >= 0.0033) & (freq < 0.04)])
    ulf_power = np.trapz(power[freq < 0.0033])
    total_power = np.trapz(power)
    lfnu = lf_power / (total_power - vlf_power) * 100 if total_power != vlf_power else np.nan
    hfnu = hf_power / (total_power - vlf_power) * 100 if total_power != vlf_power else np.nan
    lfhf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
    """
    # Plot power spectral density
    plt.figure(figsize=(8,5))
    plt.plot(freq, power, 'b')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('HRV Frequency Analysis\nLF Power: {:.4f}, HF Power: {:.4f}, VLF Power: {:.4f}, ULF Power: {:.4f}, LF/HF Ratio: {:.4f}, LFnu: {:.4f}, HFnu: {:.4f}'.format(lf_power, hf_power, vlf_power, ulf_power, lfhf_ratio, lfnu, hfnu))
    
    # Plot vertical lines at the LF, VLF, ULF and HF cutoff frequencies
    plt.axvline(x=0.0033, color='r', linestyle='--', linewidth=1)
    plt.axvline(x=0.04, color='r', linestyle='--', linewidth=1)
    plt.axvline(x=0.15, color='r', linestyle='--', linewidth=1)
    plt.axvline(x=0.4, color='r', linestyle='--', linewidth=1)
    
    plt.show()
    """
    return lf_power, hf_power, vlf_power, ulf_power, lfhf_ratio, lfnu, hfnu



def detect_r_peaks(sig, fs):
    # Find all local peaks
    peaks = find_local_peaks(sig, radius=int(0.2*fs))
    
    r_peaks = peaks
    max_amplitude = np.max(sig)
    r_peaks = [p for p in r_peaks if sig[p] > 0.4 * max_amplitude]

    # Calculate time domain features
    rr_intervals = np.diff(r_peaks) / fs * 1000.0  # Convert to milliseconds
    mean_rr = np.mean(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    sdsd = np.std(np.diff(rr_intervals))
    sdnn = np.std(rr_intervals)
    pnn50 = np.sum(np.diff(rr_intervals) > 50) / len(rr_intervals) * 100  # pNN50 in percentage
    """
    # Plot the signal, R-peaks, and RR intervals
    plt.figure(figsize=(15,4))
    plt.plot(sig, color='darkblue')
    plt.plot(r_peaks, sig[r_peaks], 'go', markersize=5)
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.title('R-Peak Detection\nMean RR: {:.2f}ms, SDNN: {:.2f}ms, RMSSD: {:.2f}ms, SDSD: {:.2f}ms, pNN50: {:.2f}%'.format(mean_rr, sdnn, rmssd, sdsd, pnn50))
    
    # Plot RR intervals as horizontal lines
    for i in range(len(rr_intervals)):
        x_start = r_peaks[i]
        x_end = r_peaks[i+1] if i+1 < len(r_peaks) else x_start + int(mean_rr * fs / 1000.0)
        y_pos = sig[x_start]
        plt.plot([x_start, x_end], [y_pos, y_pos], 'b-', linewidth=2)
    
    # Plot vertical lines at the R-peaks
    for r_peak in r_peaks:
        plt.axvline(x=r_peak, color='r', linestyle='--', linewidth=1)
    plt.savefig(f"./Healthy_nn1/mean_rr:*{mean_rr}.png", format='png')
    plt.show()
    """
    return r_peaks, mean_rr, rmssd, sdsd, sdnn, pnn50



def detect_q_s(sig, r_peaks, fs,counter):
    # Compute the first derivative
    deriv1 = np.diff(sig)

    # Apply a moving average filter
    ma_size = int(0.05*fs)
    ma_filter = np.ones(ma_size)/ma_size
    deriv1_filt = np.convolve(deriv1, ma_filter, mode='same')

    # Square the filtered derivative
    deriv1_filt_sq = deriv1_filt**2

    # Apply another moving average filter
    ma_size = int(0.15*fs)
    ma_filter = np.ones(ma_size)/ma_size
    deriv1_filt_sq_filt = np.convolve(deriv1_filt_sq, ma_filter, mode='same')

    # Compute a threshold
    thresh = 0.08*np.max(deriv1_filt_sq_filt)

    # Find the Q and S points
    q_points = []
    s_points = []
    for r_peak in r_peaks:
        # Find the Q point
        i = r_peak - 1
        while i > 0 and deriv1_filt_sq_filt[i] > thresh:
            i -= 1
        q_points.append(i)

        # Find the S point
        i = r_peak + 1
        while i < len(sig)-1 and deriv1_filt_sq_filt[i] > thresh:
            i += 1
        s_points.append(i)

    #plot_qrs_points(sig, r_peaks, q_points, s_points,counter)
    #plot_qrs_area(sig, r_peaks, q_points, s_points, fs)
    mean_qrs_area, mean_r_peak_amp,std_r_peak_amp,std_qrs_area,r_peaks, q_points, s_points=plot_positive_qrs_area(sig, r_peaks, q_points, s_points, fs)


  
    return mean_qrs_area, mean_r_peak_amp,std_r_peak_amp,std_qrs_area,r_peaks, q_points, s_points
    
    #compute_qrs_metrics(sig, r_peaks, q_points, s_points)


def plot_qrs_points(sig, r_peaks, q_points, s_points, counter):
    # Plot the signal, R-peaks, Q-points, and S-points
    
    # Set the figure size
    plt.figure(figsize=(15,4))

    # Plot the signal
    plt.plot(sig, color='darkblue')

    # Highlight the signal between Q-points and S-points in green
    for i in range(len(q_points)):
        plt.axvspan(q_points[i], s_points[i], alpha=0.3, color='green')

    # Plot the R-peaks, Q-points, and S-points
    plt.plot(r_peaks, sig[r_peaks], 'go', markersize=5, label='R-Peaks')
    plt.plot(q_points, sig[q_points], 'bo', markersize=5, label='Q-Points')
    plt.plot(s_points, sig[s_points], 'ro', markersize=5, label='S-Points')

    plt.legend(loc='upper right')
    plt.ylabel('Amplitude')
    plt.title('Q, R and S Point Detection')
    plt.tight_layout()
    plt.savefig(f"./Healthy_nn1/{counter}.png", format='png')
    plt.show()

def plot_qrs_area(sig, r_peaks, q_points, s_points, fs):
    # Compute the QRS complex area and overlay it on the ECG signal plot
    plt.figure(figsize=(12,4))
    plt.plot(sig, color='black')
    for r, q, s in zip(r_peaks, q_points, s_points):
        qrs_area = np.trapz(sig[q:s+1], dx=1/fs)
        plt.fill_between(np.arange(q, s+1), sig[q:s+1], color='red', alpha=0.3)
        plt.text(r, sig[r], f"{qrs_area:.2f}", ha='center', va='top', fontsize=8)
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with QRS Complex Area Overlay')
    plt.show()




def plot_positive_qrs_area(sig, r_peaks, q_points, s_points, fs):
    # Plot the positive part of the QRS complex and calculate its area
    #plt.figure(figsize=(15,4))
    #plt.plot(sig, color='darkblue')

    qrs_areas = []
    max_qrs_area = 0
    max_qrs_range = ()
    r_peaks_amp = []

    for q, s in zip(q_points, s_points):
        # Extract the QRS complex
        qrs = sig[q:s+1]
        # Extract the positive part of the QRS complex
        pos_qrs = qrs[qrs > 0]
        # Get the indices of the positive part of the QRS complex
        pos_idx = np.where(qrs > 0)[0]
        # Calculate the area under the positive part of the QRS complex
        if len(pos_qrs) > 1:
            qrs_area = np.trapz(pos_qrs, dx=1/fs)
            # Get the indices relative to the original signal
            pos_idx_rel = pos_idx + q
            # Plot the data under the corresponding R-peak
            #plt.plot(np.arange(q, s+1), sig[q:s+1], color="magenta",alpha=0.65)
            # Plot the positive part of the QRS complex as a fill between Q and S points
            #plt.fill_between(pos_idx_rel, pos_qrs, y2=0, where=(q <= pos_idx_rel) & (pos_idx_rel <= s),color='red')
            # Find the maximum amplitude within the QRS complex and use it to determine the R-peak location
            max_amp = np.max(qrs)
            r = np.where(sig[q:s+1] == max_amp)[0][0] + q
            # Add the QRS area as text above the R-peak
            #plt.text(r, sig[r], f"qrs_area:{qrs_area:.2f}", ha='center', va='top', fontsize=8)
            # Add the Q-point and S-point as vertical lines
            #plt.scatter([q], [sig[q]], color='green', marker='o',s=65)
            #plt.scatter([s], [sig[s]], color='dodgerblue', marker='o',s=65)
            qrs_areas.append(qrs_area)
            r_peaks_amp.append(sig[r])
        
            # Add the R-peak amplitude as text next to the vertical line
            #plt.text(r, sig[r], f"Ramp:{sig[r]:.2f}", ha='center', va='bottom', fontsize=8)

    # Set the plot title and axis labels
    #plt.title('ECG Signal with Positive QRS Complex Area Overlay')
    #plt.xlabel('Sample Number')
    #plt.ylabel('Amplitude')

    # Calculate and plot the mean of the QRS complex areas and its standard deviation
    if len(qrs_areas) > 0:
        mean_qrs_area = np.mean(qrs_areas)
        std_qrs_area = np.std(qrs_areas)
        #plt.plot([0, 1], [mean_qrs_area, mean_qrs_area],color='white', marker=' ', label=f"Mean QRS Area: {mean_qrs_area:.4f} ± {std_qrs_area:.4f}")
    else:
        mean_qrs_area=0 
        std_qrs_area=0

    if len(r_peaks_amp) > 0:
        mean_r_peak_amp = np.mean(r_peaks_amp)
        std_r_peak_amp = np.std(r_peaks_amp)
        #plt.plot([0, 1], [mean_r_peak_amp, mean_r_peak_amp], color='white', marker=' ',label=f"Mean R-Peak Amplitude: {mean_r_peak_amp:.4f} ± {std_r_peak_amp:.4f}")
    else:
        mean_r_peak_amp=0
        std_r_peak_amp=0
    
    #plt.legend(loc='lower right')

    #plt.savefig(f"./Healthy_nn1/Qrsa:{mean_qrs_area}.png", format='png')
    #Show the plot
    #plt.show()
    

    return mean_qrs_area, mean_r_peak_amp,std_r_peak_amp,std_qrs_area,r_peaks, q_points, s_points



def compute_qrs_metrics(sig, r_peaks, q_points, s_points):


   # Compute the QRSA, QRSASD, RPamp and RPampSD
    qrsa = []
    qrsasd = []
    rpamp = []
    rpampsd = []

    for i in range(len(q_points)):
        # Compute QRSA and QRSASD
        qrsa_val = sum(sig[r_peaks[i]:s_points[i]+1])
        qrsa.append(qrsa_val)
        qrsasd_val = np.std(sig[r_peaks[i]:s_points[i]+1])
        qrsasd.append(qrsasd_val)

        # Compute RPamp and RPampSD
        rpamp_val = sig[r_peaks[i]] - sig[q_points[i]]
        rpamp.append(rpamp_val)
        rpampsd_val = np.std([sig[r_peaks[i]], sig[q_points[i]]])
        rpampsd.append(rpampsd_val)

    # Define mean values
    qrsa_mean = np.mean(qrsa)
    qrsasd_mean = np.mean(qrsasd)
    rpamp_mean = np.mean(rpamp)
    rpampsd_mean = np.mean(rpampsd)

    # Plot the histograms with mean values
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    # Define colors for histograms
    hist_colors = ['#4e79a7', '#f28e2b']

    # Define colors for mean lines
    mean_colors = ['#6baed6', '#fc8d62']

    # Plot QRSA histogram and mean line
    axs[0,0].hist(qrsa, bins=10, color=hist_colors[0])
    axs[0,0].axvline(x=qrsa_mean, color=mean_colors[0], label='Mean')
    axs[0,0].set_xlabel('QRSA')
    axs[0,0].set_ylabel('Frequency')
    axs[0,0].set_title('QRSA Histogram')
    axs[0,0].legend()

    # Plot QRSASD histogram and mean line
    axs[0,1].hist(qrsasd, bins=10, color=hist_colors[1])
    axs[0,1].axvline(x=qrsasd_mean, color=mean_colors[1], label='Mean')
    axs[0,1].set_xlabel('QRSASD')
    axs[0,1].set_ylabel('Frequency')
    axs[0,1].set_title('QRSASD Histogram')
    axs[0,1].legend()

    # Plot RPamp histogram and mean line
    axs[1,0].hist(rpamp, bins=10, color=hist_colors[0])
    axs[1,0].axvline(x=rpamp_mean, color=mean_colors[0], label='Mean')
    axs[1,0].set_xlabel('RP Amplitude')
    axs[1,0].set_ylabel('Frequency')
    axs[1,0].set_title('RP Amplitude Histogram')
    axs[1,0].legend()

    # Plot RPampSD histogram and mean line
    axs[1,1].hist(rpampsd, bins=10, color=hist_colors[1])
    axs[1,1].axvline(x=rpampsd_mean, color=mean_colors[1], label='Mean')
    axs[1,1].set_xlabel('RP Amplitude SD')
    axs[1,1].set_ylabel('Frequency')
    axs[1,1].set_title('RP Amplitude SD Histogram')
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()




    # Plot the histograms with mean values
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    # Define colors for histograms
    hist_colors = ['#4e79a7', '#f28e2b']

    # Define colors for mean lines
    mean_colors = ['#6baed6', '#fc8d62']

    # Plot QRSA histogram and mean line
    counts, bins, patches = axs[0,0].hist(qrsa, bins=10, color=hist_colors[0], alpha=0.8, edgecolor='black')
    for i, patch in enumerate(patches):
        axs[0,0].text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f"{int(counts[i])}", ha='center', va='bottom')
    axs[0,0].axvline(x=qrsa_mean, color=mean_colors[0], label='Mean')
    axs[0,0].set_xlabel('QRSA Value')
    axs[0,0].set_ylabel('Value Count')
    axs[0,0].set_title('QRSA Histogram')
    axs[0,0].legend()

    # Plot QRSASD histogram and mean line
    counts, bins, patches = axs[0,1].hist(qrsasd, bins=10, color=hist_colors[1], alpha=0.8, edgecolor='black')
    for i, patch in enumerate(patches):
        axs[0,1].text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f"{int(counts[i])}", ha='center', va='bottom')
    axs[0,1].axvline(x=qrsasd_mean, color=mean_colors[1], label='Mean')
    axs[0,1].set_xlabel('QRSASD Value')
    axs[0,1].set_ylabel('Value Count')
    axs[0,1].set_title('QRSASD Histogram')
    axs[0,1].legend()

    # Plot RPamp histogram and mean line
    counts, bins, patches = axs[1,0].hist(rpamp, bins=10, color=hist_colors[0], alpha=0.8, edgecolor='black')
    for i, patch in enumerate(patches):
        axs[1,0].text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f"{int(counts[i])}", ha='center', va='bottom')
    axs[1,0].axvline(x=rpamp_mean, color=mean_colors[0], label='Mean')
    axs[1,0].set_xlabel('RP Amplitude Value')
    axs[1,0].set_ylabel('Value Count')
    axs[1,0].set_title('RP Amplitude Histogram')
    axs[1,0].legend()




    counts, bins, patches = axs[1,1].hist(rpampsd, bins=10, color=hist_colors[1], alpha=0.8, edgecolor='black')
    for i, patch in enumerate(patches):
        axs[1,1].text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f"{int(bins[i])}", ha='center', va='bottom')
    axs[1,1].axvline(x=rpampsd_mean, color=mean_colors[1], label='Mean')
    axs[1,1].set_xlabel('RP Amplitude SD')
    axs[1,1].set_ylabel('Value')
    axs[1,1].set_title('RP Amplitude SD Histogram with Mean Line')
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()

















    """

def plot_positive_qrs_area(sig, r_peaks, q_points, s_points, fs):
    # Plot the positive part of the QRS complex and calculate its area
    plt.figure(figsize=(12,4))
    plt.plot(sig, color='black')

    qrs_areas = []
    max_qrs_area = 0
    max_qrs_range = ()
    r_peaks_amp = []

    for r, q, s in zip(r_peaks, q_points, s_points):
        # Extract the QRS complex
        qrs = sig[q:s+1]
        # Extract the positive part of the QRS complex
        pos_qrs = qrs[qrs > 0]
        # Get the indices of the positive part of the QRS complex
        pos_idx = np.where(qrs > 0)[0]
        # Calculate the area under the positive part of the QRS complex
        if len(pos_qrs) > 1:
            qrs_area = np.trapz(pos_qrs, dx=1/fs)
            # Get the indices relative to the original signal
            pos_idx_rel = pos_idx + q
            # Plot the positive part of the QRS complex as a red fill
            plt.fill_between(pos_idx_rel, pos_qrs, color='red', alpha=0.3)
            # Add the QRS area as text above the R-peak
            plt.text(r, sig[r], f"{qrs_area:.2f}", ha='center', va='top', fontsize=8)
            # Plot the data under the corresponding R-peak
            plt.plot(np.arange(q, s+1), sig[q:s+1], alpha=0.5)

            qrs_areas.append(qrs_area)

        r_peaks_amp.append(sig[r])

        # Plot the R-peak as a vertical line
        plt.axvline(r, color='blue', linestyle='--')
        # Add the R-peak amplitude as text next to the vertical line
        plt.text(r, sig[r], f"{sig[r]:.2f}", ha='center', va='bottom', fontsize=8)

    # Set the plot title and axis labels
    plt.title('ECG Signal with Positive QRS Complex Area Overlay')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    # Calculate and plot the mean of the QRS complex areas
    if len(qrs_areas) > 0:
        mean_qrs_area = np.mean(qrs_areas)
        plt.axhline(mean_qrs_area, color='green', linestyle='--', label=f"Mean QRS Area: {mean_qrs_area:.2f}")
        plt.legend(loc='upper right')
    # Highlight the range with the maximum QRS area
    if max(qrs_areas, default=0) > 0:
        max_qrs_idx = np.argmax(qrs_areas)
        max_qrs_range = (q_points[max_qrs_idx], s_points[max_qrs_idx])
        plt.axvspan(max_qrs_range[0], max_qrs_range[1], color='yellow', alpha=0.3)

    # Calculate and plot the mean of the R-peak amplitudes
    if len(r_peaks_amp) > 0:
        mean_r_peak_amp = np.mean(r_peaks_amp)
        plt.axhline(mean_r_peak_amp, color='cyan', linestyle='--', label=f"Mean R-Peak Amplitude: {mean_r_peak_amp:.2f}")
        plt.legend(loc='upper right')

    # Show the plot
    plt.show()

    return mean_qrs_area, mean_r_peak_amp





def plot_positive_qrs_area(sig, r_peaks, q_points, s_points, fs):
    # Plot the positive part of the QRS complex and calculate its area
    plt.figure(figsize=(12,4))
    plt.plot(sig, color='black')

    qrs_areas = []
    max_qrs_area = 0
    max_qrs_range = ()
    r_peaks_amp = []

    last_s_idx = -1
    for r, q, s in zip(r_peaks, q_points, s_points):
        # Check if the R peak falls between the last S point and the current Q point
        if s > last_s_idx and r < s:
            # Extract the QRS complex
            qrs = sig[q:s+1]
            # Extract the positive part of the QRS complex
            pos_qrs = qrs[qrs > 0]
            # Get the indices of the positive part of the QRS complex
            pos_idx = np.where(qrs > 0)[0]
            # Calculate the area under the positive part of the QRS complex
            if len(pos_qrs) > 1:
                qrs_area = np.trapz(pos_qrs, dx=1/fs)
                # Get the indices relative to the original signal
                pos_idx_rel = pos_idx + q
                # Plot the positive part of the QRS complex as a red fill
                plt.fill_between(pos_idx_rel, pos_qrs, color='red', alpha=0.3)
                # Add the QRS area as text above the R-peak
                plt.text(r, sig[r], f"{qrs_area:.2f}", ha='center', va='top', fontsize=8)
                # Plot the data under the corresponding R-peak
                plt.plot(np.arange(q, s+1), sig[q:s+1], alpha=0.5)

                qrs_areas.append(qrs_area)

            r_peaks_amp.append(sig[r])

            # Plot the R-peak as a vertical line
            plt.axvline(r, color='blue', linestyle='--')
            # Add the R-peak amplitude as text next to the vertical line
            plt.text(r, sig[r], f"{sig[r]:.2f}", ha='center', va='bottom', fontsize=8)

            # Update the last S index
            last_s_idx = s

    # Set the plot title and axis labels
    plt.title('ECG Signal with Positive QRS Complex Area Overlay')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    # Calculate and plot the mean of the QRS complex areas and its standard deviation
    if len(qrs_areas) > 0:
        mean_qrs_area = np.mean(qrs_areas)
        std_qrs_area = np.std(qrs_areas)
        plt.axhline(mean_qrs_area, color='green', linestyle='--', label=f"Mean QRS Area: {mean_qrs_area:.2f} ± {std_qrs_area:.2f}")
        plt.legend(loc='upper right')
    # Highlight the range with the maximum QRS area
    if max(qrs_areas, default=0) > 0:
        max_qrs_idx = np.argmax(qrs_areas)
        max_qrs_range = (q_points[max_qrs_idx], s_points[max_qrs_idx])
        plt.axvspan(max_qrs_range[0], max_qrs_range[1], color='yellow', alpha=0.3)

    # Calculate and plot the mean of the R-peak amplitudes
    if len(r_peaks_amp) > 0:
        mean_r_peak_amp = np.mean(r_peaks_amp)
        std_r_peak_amp = np.std(r_peaks_amp)
        plt.axhline(mean_r_peak_amp, color='cyan', linestyle='--', label=f"Mean R-Peak Amplitude: {mean_r_peak_amp:.2f} ± {std_r_peak_amp:.2f}")
        plt.legend(loc='upper right')

    # Show the plot
    plt.show()

    return mean_qrs_area, mean_r_peak_amp,std_r_peak_amp,std_qrs_area


"""