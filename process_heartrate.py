from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from sklearn.decomposition import FastICA

MIN_HR_HZ = 30.0 / 60.0
MAX_HR_HZ = 240.0 / 60.0
SEC_PER_MIN = 60

def plotSignals_3D(signal, fps, label):
	seconds = np.arange(0, signal.shape[0] / fps, 1.0 / fps)
	colors = ["r", "g", "b"]
	fig = plt.figure()
	fig.patch.set_facecolor('white')
	for i in range(3):
		plt.plot(seconds, signal[:,i], colors[i])
	plt.xlabel('Time (sec)', fontsize=17)
	plt.ylabel(label, fontsize=17)
	plt.tick_params(axis='x', labelsize=17)
	plt.tick_params(axis='y', labelsize=17)

def plotSpectrum_3D(freqs, powerSpec):
	idx = np.argsort(freqs)
	fig = plt.figure()
	fig.patch.set_facecolor('white')
	for i in range(3):
		plt.plot(freqs[idx], powerSpec[idx,i])
	plt.xlabel("Frequency (Hz)", fontsize=17)
	plt.ylabel("Power", fontsize=17)
	plt.tick_params(axis='x', labelsize=17)
	plt.tick_params(axis='y', labelsize=17)
	plt.xlim([0.75, 4])

def plotSignals_1D(signal, fps, label):
	seconds = np.arange(0, signal.shape[0] / fps, 1.0 / fps)
	fig = plt.figure()
	fig.patch.set_facecolor('white')
	plt.plot(seconds, signal, "r")
	plt.xlabel('Time (sec)', fontsize=17)
	plt.ylabel(label, fontsize=17)
	plt.tick_params(axis='x', labelsize=17)
	plt.tick_params(axis='y', labelsize=17)

def plotSpectrum_1D(freqs, powerSpec):
	fig = plt.figure()
	fig.patch.set_facecolor('white')
	plt.plot(freqs, powerSpec)
	plt.xlabel("Frequency (Hz)", fontsize=17)
	plt.ylabel("Power", fontsize=17)
	plt.tick_params(axis='x', labelsize=17)
	plt.tick_params(axis='y', labelsize=17)
	plt.xlim([0.75, 4])

# processes a buttersworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

	def butter_bandpass(lowcut, highcut, fs, order=5):
		nyq = 0.5 * fs
		low = lowcut / nyq
		high = highcut / nyq
		b, a = signal.butter(order, [low, high], btype='bandpass')
		return b, a

	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = signal.lfilter(b, a, data)
	return y

# processes data as a continuous segment, can be used to process each iteration
# of a sliding window approach
def process_window_pos(C, plot_window=False): 
	# Step 1: Spatial averaging
	if plot_window:
		f = np.arange(0,C.shape[1])
		plt.plot(f, C[0,:] , 'r', f,  C[1,:], 'g', f,  C[2,:], 'b')
		plt.title("Mean RGB - Sliding Window")
		plt.show()
	
	#Step 2 : Temporal normalization
	mean_color = np.mean(C, axis=1)
	diag_mean_color = np.diag(mean_color)
	diag_mean_color_inv = np.linalg.inv(diag_mean_color)
	
	Cn = np.matmul(diag_mean_color_inv,C)

	if plot_window:
		f = np.arange(0,Cn.shape[1])
		plt.plot(f, Cn[0,:] , 'r', f,  Cn[1,:], 'g', f,  Cn[2,:], 'b')
		plt.title("Temporal normalization - Sliding Window")
		plt.show()

	#Step 3: 
	projection_matrix = np.array([[0,1,-1],[-2,1,1]])
	S = np.matmul(projection_matrix,Cn)

	if plot_window:
		f = np.arange(0,S.shape[1])
		#plt.ylim(0,100000)
		plt.plot(f, S[0,:] , 'c', f,  S[1,:], 'm')
		plt.title("Projection matrix")
		plt.show()

	#Step 4:
	#2D signal to 1D signal
	std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
	P = np.matmul(std,S)
	if plot_window:
		f = np.arange(0,len(P))
		plt.plot(f, P, 'k')
		plt.title("Alpha tuning")
		plt.show()

	normalized_window = (P - np.mean(P))/np.std(P)

	if plot_window:
		f = np.arange(0,len(P))
		plt.plot(f, normalized_window, 'k')
		plt.title("Alpha tuning normalized")
		plt.show()

	return normalized_window

def fft(bvp, fps, plot):
	maxPwrSrc = np.abs(np.fft.fft(bvp))**2
	freqs = np.fft.fftfreq(bvp.shape[0], 1.0 / fps)

	# Find heart rate
	validIdx = np.where((freqs >= MIN_HR_HZ) & (freqs <= MAX_HR_HZ))
	validPwr = maxPwrSrc[validIdx]
	validFreqs = freqs[validIdx]
	maxPwrIdx = np.argmax(validPwr)
	hr = validFreqs[maxPwrIdx]

	if plot:
		plotSignals_1D(bvp, fps, "Normalized color intensity")
		plotSpectrum_1D(freqs, maxPwrSrc)

	return hr

# processes RGB signal via POS projection and FFT
def process_pos(signal_rgb, fps, filter=False, plot=False):

	bvp = process_window_pos(signal_rgb.T, plot)

	if plot:
		f = np.arange(0,len(bvp))
		plt.plot(f, bvp, 'k')
		plt.title("Before butterworth")
		plt.show()

	hr = fft(bvp, fps, plot)

	if not filter: return hr

	bvp_filtered = butter_bandpass_filter(bvp, hr-0.25, hr+0.25, fps, order=1)

	if plot:
		f = np.arange(0,len(bvp_filtered))
		plt.plot(f, bvp_filtered, 'k')
		plt.title("After butterworth")
		plt.show()

	return hr

# processes RGB signal via ICA and FFt
def process_ica(signal_rgb, fps, filter=False, plot=False):
	# Normalize across the window to have zero-mean and unit variance
	mean = np.mean(signal_rgb, axis=0)
	std = np.std(signal_rgb, axis=0)
	normalized = (signal_rgb - mean) / std

	# Separate into three source signals using ICA
	ica = FastICA()
	srcSig = ica.fit_transform(normalized)

	# Find power spectrum
	powerSpec = np.abs(np.fft.fft(srcSig, axis=0))**2
	freqs = np.fft.fftfreq(signal_rgb.shape[0], 1.0 / fps)

	# Find heart rate
	maxPwrSrc = np.max(powerSpec, axis=1)
	validIdx = np.where((freqs >= MIN_HR_HZ) & (freqs <= MAX_HR_HZ))
	validPwr = maxPwrSrc[validIdx]
	validFreqs = freqs[validIdx]
	maxPwrIdx = np.argmax(validPwr)
	hr = validFreqs[maxPwrIdx]

	if plot:
		plotSignals_3D(normalized, fps, "Normalized color intensity")
		plotSignals_3D(srcSig, fps, "Source signal strength")
		plotSpectrum_3D(freqs, powerSpec)

	return hr

# process 1D signal via FFT
def process_fft(signal_1d, fps, filter=False, plot=False):

	# apply buttersworth if necessary
	if filter:
		normalized = signal.detrend(signal_1d)
		normalized = butter_bandpass_filter(normalized,0.8,3, fps ,order = 3)

	# normalize across the window to have zero-mean and unit variance
	else:
		mean = np.mean(signal_1d, axis=0)
		std = np.std(signal_1d, axis=0)
		normalized = (signal_1d - mean) / std

	hr = fft(normalized, fps, plot)

	return hr

def process_sliding_window(signal, fps, window, step, process, mode, filter=False, plot=False, best_n=5, pose_vectors=None):
	window_size = int(window * fps)
	step_size = int(step * fps)
	total_frames = signal.shape[0]
	hr_list = []
	stability = []

	if mode == 'best' and len(signal) != len(pose_vectors):
		diff = len(signal) - len(pose_vectors)
		signal = signal[diff:]

	for i in range(0, total_frames-window_size+1, step_size):
		window = signal[i:i+window_size]
		hr_list.append(process(window, fps, filter, plot))
		if mode == 'best':
			pose_window = pose_vectors[i:i+window_size]
			stability.append(window_variance(pose_window))

	if mode == 'median':
		hr_median = np.median(hr_list)
		return hr_median, hr_list
	if mode == 'mean':
		hr_mean = np.mean(hr_list)
		return hr_mean, hr_list
# 	if mode == 'min':
# 		hr_min = np.min(hr_list)
# 		return hr_min, hr_list
# 	if mode == 'max':
# 		hr_max = np.max(hr_list)
# 		return hr_max, hr_list

	if mode == 'best':
		rank = np.argsort(stability)[:best_n]
		return np.mean(np.take(hr_list,rank))


def calculate_rmse(gt, preds):
	return np.sqrt(np.mean(np.square(preds-gt)))

def window_variance(pose_vectors):
	concat = [np.concatenate([np.array(x[0]),np.array(x[1])]) for x in pose_vectors]
	variance = np.var(np.stack(concat),axis=0)
	variance = np.max(variance)
	# if we have [[0, 0, 0], [0, 0, 0]] in pose_vectors we are dealing with a window that has 
	# frames without faces detected.
	# TODO: find a better way to deal with frames without faces detected (since all of them are recorded
	# with pose as [[0, 0, 0], [0, 0, 0]], which would make variance low
	return variance



