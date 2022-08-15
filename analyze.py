import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
import panel as pn
import sounddevice as sd
import math
import IPython.display as ipd
import holoviews as hv
import soundfile as sf
import speech_recognition as sr

# import umap
import librosa
# import mplcursors
# from  scipy.signal import spectrogram
# from scipy.io.wavfile import write, read



def rms_calculate(signal, sampling_rate, rms_thresh=0.03):
    chunk = int(0.016 * sampling_rate) # 16 ms (700/sampling_rate)*1000 - need to be 100-200 samples

    # Normalization Audio:
    signal = signal/signal.max()

    # calculate rms:
    rms = []
    for i in range(0, 210000):
        rms.append(np.sqrt(np.mean(signal[i:i+chunk]**2)))
    return rms

def rms_plot(rms, thresh = 0.03):
    plt.figure(figsize=(20,4))
    plt.xticks(np.arange(0, len(rms), step=10000))
    plt.plot(rms)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title("RMS")
    #plt.show()


def audioSpliter(signal, sampling_rate, type, rms_thresh=0.03): # rms_thresh = 0.3? 0.1?


    silence_thresh = int(0.085 * sampling_rate) # 33 ms

    # Normalization Audio:
    signal = signal/signal.max()

    # calculate rms:
    rms = rms_calculate(signal, sampling_rate, rms_thresh)

    if type == "orig":
        # find the start&end indexs for each word:
        indexs  = []
        inWord = False
        silence = 0

        for i in range(len(rms)):
            if not inWord and rms[i] >= rms_thresh:
                inWord = True
                start = i

            if inWord:
                if rms[i] < rms_thresh:
                    silence += 1
                if silence > silence_thresh:
                    inWord = False
                    end = i
                    indexs.append((start, end))
                    silence = 0

    elif type == "hila":
        indexs = indexs_gen_hila
    elif type == "nofar":
        indexs = indexs_gen_nofar
    elif type == "hello":
        indexs = indexs_gen_hello

    return indexs



def STT(path):
    r = sr.Recognizer()
    text = ""
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        try:
            word = r.recognize_google(audio_listened)
            text += word
        except sr.UnknownValueError as e:
            text += error
    return text.split(' ')



def fft_calculate(signal, sampling_rate, start, new_len, color='r'):

    # calculte fft:
    N = len(signal)
    yf = abs(rfft(signal))
    xf = rfftfreq(N, 1 / sampling_rate)

    # interpulation:
    xf_new = np.linspace(0, xf[-1], num=new_len, endpoint=True)
    f = interp1d(xf, yf)  # kind='cubic'
    yf_new = f(xf_new)

    # normalize:
    end = len(xf_new)//8
    xf_new = xf_new[start:end]
    yf_new = yf_new[start:end]
    yf_new = yf_new/yf_new.sum()

    return xf_new, yf_new



def bolds_fft(words, signal_org, signal_gen, indexs_org, indexs_gen, new_len):
    bolds = {}
    ffts = {}
    cros_cor = {}
    for i in range(len(words)):

        # word:
        word_orig = signal_org[indexs_org[i][0]: indexs_org[i][1]]
        word_gen = signal_gen[indexs_gen[i][0]: indexs_gen[i][1]]

        if word_orig.shape[0] > new_len or word_gen.shape[0] > new_len:
            print("----- error----", word_orig.shape[0], word_gen.shape[0])

        # fft + interpulation:
        xf_orig, yf_orig = fft_calculate(word_orig, 16000, 10, new_len)
        xf_gen, yf_gen = fft_calculate(word_gen, 16000, 10, new_len)
        ffts[words[i]] = [xf_orig, yf_orig, xf_gen, yf_gen]

        # clculate distance:
        corr = scipy.signal.correlate(yf_orig, yf_gen)
        x = np.linspace(-250, 250, num=len(corr), endpoint=True)
        dist = x[np.argmax(corr)]
        cros_cor[words[i]] = [x, corr]

        if np.abs(dist) >= 20:
            bolds[words[i]] = True
        else:
            bolds[words[i]] = False


    return bolds, ffts, cros_cor



def fftAndCross_plot(xf, yf, title, start=20, color='r',x=[], corr=[]):
    N_normal = yf.shape[0]
    plt.subplot(1, 2, 1)
    plt.plot(xf[start:N_normal], np.abs(yf)[start:N_normal], color, label=title)
    plt.grid()
    plt.legend()
    plt.title("FFT")

    plt.subplot(1, 2, 2)
    plt.plot(x, corr)
    plt.xticks(np.arange(-250, 300, 50))
    plt.grid()
    plt.title("Cross-Correlation")


def get_val_H():
    path_yourBag = 'records/_yourBag.wav'
    signal_yourBag, sampling_rate = librosa.load(path_yourBag, sr=16000)
    path_gen = 'STT/_generated_yourBag.wav'
    signal_gen, sampling_rate= librosa.load(path_gen, sr=16000)
    words = STT(path_yourBag)
    rms=rms_calculate(signal_yourBag, 16000)
    #rms_plot(rms)
    indexs_yourBag = audioSpliter(signal_yourBag, 16000, 'orig')
    indexs_gen = audioSpliter(signal_gen, 16000, 'hila')
    return words, signal_yourBag, signal_gen, indexs_yourBag, indexs_gen

def get_val_N():
    path_steal = 'records/_steal.wav'
    signal_steal, sampling_rate = librosa.load(path_steal, sr=16000)
    path_gen_N = 'STT/_steal_generated.wav'
    signal_gen_N, sampling_rate= librosa.load(path_gen_N, sr=16000)
    words = STT(path_steal)
    rms=rms_calculate(signal_steal, 16000)
    #rms_plot(rms)
    indexs_steal = audioSpliter(signal_steal, 16000, 'orig')
    indexs_gen_N = audioSpliter(signal_gen_N, 16000, 'nofar')
    return words, signal_steal, signal_gen_N, indexs_steal, indexs_gen_N

def get_val_Hallo():
    path_hello= 'records/_hello.wav'
    signal_hello, sampling_rate = librosa.load(path_hello, sr=16000)
    path_gen_hello = 'STT/_hello_generated.wav'
    signal_gen_hello, sampling_rate= librosa.load(path_gen_hello, sr=16000)
    signal_gen_hello=signal_gen_hello[7000:len(signal_gen_hello)]
    words = STT(path_hello)
    rms=rms_calculate(signal_hello, 1600)
    #rms_plot(rms)
    indexs_hello = audioSpliter(signal_hello, 16000, 'orig')
    indexs_gen_hello = audioSpliter(signal_gen_hello, 16000, 'hello')
    return words, signal_hello, signal_gen_hello, indexs_hello, indexs_gen_hello

indexs_gen_hila = [
    (1500, 4000),   # 'I'
    (4000, 6000),   # 'did'
    (6000, 9000),   # 'not'
    (7000, 18000),  # 'steal'
    (18000, 21000), # your'
    (21000, 28000)  # 'bag'

]

indexs_gen_nofar = [
    (1500, 4600),   # 'I'
    (4300, 7000),   # 'did'
    (7000, 13000),  # 'not'
    (14000,18000),  #steal
    (18000, 22000), #your
    (22000, 31200) # yourBag

]

indexs_gen_hello = [
    (0, 8000),     # 'Hello'
    (8000, 11000),   # 'this'
    (12000, 13500),  # 'is'
    (13500, 17000),  # 'our'
    (17000, 23000),  # 'final'
    (23000, 30440) # 'project'

]
