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
import librosa
# import librosa
# from scipy.io.wavfile import write, read
# import mplcursors
# from librosa.display import specshow
# from  scipy.signal import spectrogram


indexs_gen = [
    (1500, 4000),   # 'I'
    (4000, 6000),   # 'did'
    (6000, 9000),   # 'not'
    (7000, 18000),  # 'steal'
    (18000, 21000), # your'
    (21000, 28000)  # 'bag'

]



def audioSpliter(signal, sampling_rate, type, rms_thresh=0.03): # rms_thresh = 0.3? 0.1?

    chunk = int(0.016 * sampling_rate) # 16 ms (700/sampling_rate)*1000 - need to be 100-200 samples
    silence_thresh = int(0.085 * sampling_rate) # 33 ms

    # Normalization Audio:
    signal = signal/signal.max()

    # calculate rms:
    rms = []
    for i in range(0, 210000):
        rms.append(np.sqrt(np.mean(signal[i:i+chunk]**2)))

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
    else:
        indexs = indexs_gen

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
            text += "error"
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
            print(word_orig.shape[0], word_gen.shape[0])

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



def fft_plot(xf, yf, title, start=20, color='r'):
    N_normal = yf.shape[0]
    plt.plot(xf[start:N_normal], np.abs(yf)[start:N_normal], color, label=title)
    plt.grid()
    plt.legend()



def cros_cor_plot(x, corr):
    plt.plot(x, corr)
    plt.xticks(np.arange(-250, 300, 50))
    plt.grid()
    plt.show()

def get_val():
    path_yourBag2 = 'STT/yourBag3.wav'
    signal_yourBag2, sampling_rate_yourBag2 = librosa.load(path_yourBag2, sr=16000)
    path_gen = 'STT/generated_.wav'
    signal_gen, sampling_rate_gen = librosa.load(path_gen, sr=16000)
    words = STT(path_yourBag2)
    indexs_yourBag2 = audioSpliter(signal_yourBag2, 16000, 'orig')
    indexs_gen2 = audioSpliter(signal_gen, 16000, 'gen')
    return words, signal_yourBag2, signal_gen, indexs_yourBag2, indexs_gen2
