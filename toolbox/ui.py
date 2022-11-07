import sys
import wave
from pathlib import Path
from time import sleep
from typing import List, Set
from warnings import filterwarnings, warn

from scipy.io.wavfile import write

import sounddevice
from encoder import inference as encoder
import speech_recognition as sr

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import umap
from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from encoder.inference import plot_embedding_as_heatmap
from toolbox.utterance import Utterance

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPlainTextEdit, QVBoxLayout
from STT.main import get_large_audio_transcription
from analyze import *
import simpleaudio as sa
import analyze
filterwarnings("ignore")


colormap = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255

default_text = ""

#path = 'STT\\preamble10.wav'
#default_text = get_large_audio_transcription(path)

class UI(QDialog):
    rms=[]
    min_umap_points = 4
    max_log_lines = 5
    max_saved_utterances = 20
    save_path_gen=""
    #bold_yourBag2={}
    #fft_yourBag2={}
    #cros_yourBag2={}
    words=["","","","","","","","","",""]
    thewords, signal_yourBag2, signal_gen, indexs_yourBag2, indexs_gen2, rms= analyze.get_val_H()
    main_path=""
    for i in range(len(thewords)):
        words.insert(i , thewords[i])
    print(words)
    bold_yourBag2, fft_yourBag2, cros_yourBag2 = bolds_fft(thewords, signal_yourBag2, signal_gen, indexs_yourBag2, indexs_gen2, 30000)
    flag=False


    def draw_utterance(self, utterance: Utterance, which):
        self.draw_spec(utterance.spec, which)
        self.draw_embed(utterance.embed, utterance.name, which)

    def draw_embed(self, embed, name, which):
        embed_ax, _ = self.current_ax
        embed_ax.figure.suptitle("" if embed is None else name)

        ## Embedding
        # Clear the plot
        if len(embed_ax.images) > 0:
            embed_ax.images[0].colorbar.remove()
        embed_ax.clear()

        # Draw the embed
        if embed is not None:
            plot_embedding_as_heatmap(embed, embed_ax)
            embed_ax.set_title("embedding")
        embed_ax.set_aspect("equal", "datalim")
        embed_ax.set_xticks([])
        embed_ax.set_yticks([])
        embed_ax.figure.canvas.draw()

    def draw_spec(self, spec, which):
        _, spec_ax = self.current_ax

        ## Spectrogram
        # Draw the spectrogram
        spec_ax.clear()
        #if spec is not None:
            #spec_ax.imshow(spec, aspect="auto", interpolation="none")
            #spec_ax.set_title("mel spectrogram")

        spec_ax.set_xticks([])
        spec_ax.set_yticks([])
        spec_ax.figure.canvas.draw()
        if which != "current":
             self.vocode_button.setDisabled(spec is None)

    def save_audio_file(self, wav, sample_rate):
        dialog = QFileDialog()
        dialog.setDefaultSuffix(".wav")
        fpath, _ = dialog.getSaveFileName(
            parent=self,
            caption="Select a path to save the audio file",
            filter="Audio Files (*.flac *.wav)"
        )
        if fpath:
            #Default format is wav
            if Path(fpath).suffix == "":
                fpath += ".wav"
            sf.write(fpath, wav, sample_rate)

    def setup_audio_devices(self, sample_rate):
        input_devices = []
        output_devices = []
        for device in sd.query_devices():
            # Check if valid input
            try:
                sd.check_input_settings(device=device["name"], samplerate=sample_rate)
                input_devices.append(device["name"])
            except:
                pass

            # Check if valid output
            try:
                sd.check_output_settings(device=device["name"], samplerate=sample_rate)
                output_devices.append(device["name"])
            except Exception as e:
                # Log a warning only if the device is not an input
                if not device["name"] in input_devices:
                    warn("Unsupported output device %s for the sample rate: %d \nError: %s" % (device["name"], sample_rate, str(e)))

        if len(input_devices) == 0:
            self.log("No audio input device detected. Recording may not work.")
            self.audio_in_device = None
        else:
            self.audio_in_device = input_devices[0]

        if len(output_devices) == 0:
            self.log("No supported output audio devices were found! Audio output may not work.")
            self.audio_out_devices_cb.addItems(["None"])
            self.audio_out_devices_cb.setDisabled(True)
        else:
            self.audio_out_devices_cb.clear()
            self.audio_out_devices_cb.addItems(output_devices)
            self.audio_out_devices_cb.currentTextChanged.connect(self.set_audio_device)

        self.set_audio_device()

    def set_audio_device(self):

        output_device = self.audio_out_devices_cb.currentText()
        if output_device == "None":
            output_device = None

        # If None, sounddevice queries portaudio
        sd.default.device = (self.audio_in_device, output_device)

    def play(self, wav, sample_rate):
        try:
            sd.stop()
            sd.play(wav, sample_rate)
        except Exception as e:
            print(e)
            self.log("Error in audio playback. Try selecting a different audio output device.")
            self.log("Your device must be connected before you start the toolbox.")

    def play1(self):
        # print(self.main_path)
        wave_obj = sa.WaveObject.from_wave_file(self.main_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    def stop(self):
        sd.stop()

    def record_one(self, sample_rate, duration):
        self.record_button.setText("Recording...")
        self.record_button.setDisabled(True)

        self.log("Recording %d seconds of audio" % duration)
        sd.stop()
        try:
            wav = sd.rec(duration * sample_rate, sample_rate, 1)
        except Exception as e:
            print(e)
            self.log("Could not record anything. Is your recording device enabled?")
            self.log("Your device must be connected before you start the toolbox.")
            return None

        for i in np.arange(0, duration, 0.1):
            self.set_loading(i, duration)
            sleep(0.1)
        self.set_loading(duration, duration)
        sd.wait()

        self.log("Done recording.")
        self.record_button.setText("Record")
        self.record_button.setDisabled(False)

        return wav.squeeze()

    def record_oneIn(self, sample_rate, duration):
        #self.record_buttonIn.setText("Recording...")
        #self.record_buttonIn.setDisabled(True)

        self.log("Recording %d seconds of audio" % duration)
        sd.stop()
        try:
            wav = sd.rec(duration * sample_rate, sample_rate, 1)
        except Exception as e:
            print(e)
            self.log("Could not record anything. Is your recording device enabled?")
            self.log("Your device must be connected before you start the toolbox.")
            return None

        sd.wait()

        self.log("Done recording.")
        #self.record_buttonIn.setText("Record")
        #self.record_buttonIn.setDisabled(False)

        return wav
    #@property
    def current_dataset_name(self):
        return self.dataset_box.currentText()

    #@property
    def current_speaker_name(self):
        return self.speaker_box.currentText()

    @property
    def current_utterance_name(self):
        return self.utterance_box.currentText()

    def browse_file(self):
        fpath = QFileDialog().getOpenFileName(
            parent=self,
            caption="Select an audio file",
            filter="Audio Files (*.mp3 *.flac *.wav *.m4a)"
        )
        self.flag= True
        self.save_path_gen=Path(fpath[0])
        return Path(fpath[0]) if fpath[0] != "" else ""

    def wav_graph(self,path):
        plt.ion()
        wav = wave.open(path, "r")
        raw = wav.readframes(-1)
        raw = np.frombuffer(raw, "int16")
        sampleRate = wav.getframerate()

        if wav.getnchannels() == 2:
            print("error")
            sys.exit(0)

        Time = np.linspace(0, len(raw) / sampleRate, num=len(raw))

        plt.title("Wavefrom")
        plt.plot(Time, raw, color="blue")
        plt.ylabel("Amplitude")
        plt.show()

    def browse_file1(self):
        fpath = QFileDialog().getOpenFileName(
            parent=self,
            caption="Select an audio file",
            filter="Audio Files (*.mp3 *.flac *.wav *.m4a)"
        )
        self.main_path = fpath[0]
        self.wav_graph(fpath[0])

        print("fpath        ", fpath[0])
        plt.ion()
        wav = wave.open(fpath[0], "r")
        raw = wav.readframes(-1)
        raw = np.frombuffer(raw, "int16")
        sampleRate = wav.getframerate()

        if wav.getnchannels() == 2:
            print("error")
            sys.exit(0)

        Time = np.linspace(0, len(raw) / sampleRate, num=len(raw))

        #plt.title("Wavefrom")
        #plt.plot(Time, raw, color="blue")
        #plt.ylabel("Amplitude")
        arr=analyze.STT(fpath[0])
        print(len(arr))
        if len(arr)>10:
            print("The sentence is too long, please insert a new sentence")
            return

        signal = []
        #print("fpath[0]    ",fpath[0])
        #print("fpath[0].contains(_yourBag)  ",fpath[0].contains("_yourBag"))
        #print("fpath[0].contains(_steal)  ",fpath[0].contains("_steal"))
        #print("fpath[0].contains(_hello)  ",fpath[0].contains("_hello"))
        if ("_yourBag" in fpath[0]):
            self.words = ["", "", "", "", "", "", "", "", "", ""]
            thewords, signal_yourBag, signal_gen, indexs_yourBag, indexs_gen,self.rms = analyze.get_val_H()
            print("thewords    ", thewords)
            for i in range(len(thewords)):
                self.words.insert(i, thewords[i])
            print(self.words)
            self.bold_yourBag2, self.fft_yourBag2, self.cros_yourBag2 = analyze.bolds_fft(thewords, signal_yourBag, signal_gen,
                                                                                          indexs_yourBag, indexs_gen, 30000)
            signal = signal_yourBag

        if ("_steal" in fpath[0]):
            self.words = ["", "", "", "", "", "", "", "", "", ""]
            thewords, signal_steal, signal_gen_N, indexs_steal, indexs_gen_N ,self.rms= analyze.get_val_N()
            thewords = ["I", "did", "not", "steal", "your", "bag"]
            for i in range(len(thewords)):
                self.words.insert(i, thewords[i])
            # self.words=["I","did","not","steal","your","bag","","","",""]
            print("thewords    ", thewords)
            self.bold_yourBag2, self.fft_yourBag2, self.cros_yourBag2 = analyze.bolds_fft(thewords, signal_steal, signal_gen_N,
                                                                                          indexs_steal, indexs_gen_N, 30000)
            print(self.bold_yourBag2)
            signal = signal_steal

        if ("_hello" in fpath[0]):
            self.words = ["", "", "", "", "", "", "", "", "", ""]
            thewords, signal_hello, signal_gen_hello, indexs_hello, indexs_gen_hello ,self.rms = analyze.get_val_Hallo()
            for i in range(len(thewords)):
                self.words.insert(i, thewords[i])
            print(self.words)
            self.bold_yourBag2, self.fft_yourBag2, self.cros_yourBag2 = analyze.bolds_fft(thewords, signal_hello,
                                                                                          signal_gen_hello, indexs_hello,
                                                                                          indexs_gen_hello, 72000)
            signal = signal_hello
        else:
            self.words = ["", "", "", "", "", "", "", "", "", ""]
            print(fpath[0])
            print(self.save_path_gen)
            thewords, signal_hello, signal_gen_hello, indexs_hello, indexs_gen_hello ,self.rms = analyze.get_val(fpath[0],str(self.save_path_gen))
            for i in range(len(thewords)):
                self.words.insert(i, thewords[i])
            print(self.words)
            self.bold_yourBag2, self.fft_yourBag2, self.cros_yourBag2 = analyze.bolds_fft(thewords, signal_hello,
                                                                                          signal_gen_hello, indexs_hello,
                                                                                          indexs_gen_hello, 30000)
            signal = signal_hello
        #self.rms = rms_calculate(signal, 1600)
        #rms = rms_calculate(signal, 1600)
        #plt.figure(figsize=(10,4))
        #plt.xticks(np.arange(0, len(rms), step=10000))
        #plt.plot(rms)
        #plt.axhline(y=0.03, color='r', linestyle='-')
        #plt.title("RMS")
        #analyze.rms_plot(rms)
        # self.wav_graph()
        return fpath[0]

    @staticmethod
    def repopulate_box(box, items, random=False):
        """
        Resets a box and adds a list of items. Pass a list of (item, data) pairs instead to join
        data to the items
        """
        box.blockSignals(True)
        box.clear()
        for item in items:
            item = list(item) if isinstance(item, tuple) else [item]
            box.addItem(str(item[0]), *item[1:])
        if len(items) > 0:
            box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
        box.setDisabled(len(items) == 0)
        box.blockSignals(False)

    def populate_browser(self, datasets_root: Path, recognized_datasets: List, level: int,
                         random=True):
        # Select a random dataset
        if level <= 0:
            if datasets_root is not None:
                datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
                datasets = [d.relative_to(datasets_root) for d in datasets if d.exists()]
                #self.browser_load_button.setDisabled(len(datasets) == 0)
            if datasets_root is None or len(datasets) == 0:
                msg = "Warning: you d" + ("id not pass a root directory for datasets as argument" \
                                              if datasets_root is None else "o not have any of the recognized datasets" \
                                                                            " in %s" % datasets_root)
                self.log(msg)
                msg += ".\nThe recognized datasets are:\n\t%s\nFeel free to add your own. You " \
                       "can still use the toolbox by recording samples yourself." % \
                       ("\n\t".join(recognized_datasets))
                print(msg, file=sys.stderr)

                #self.random_utterance_button.setDisabled(True)
                #self.random_speaker_button.setDisabled(True)
                #self.random_dataset_button.setDisabled(True)
                #self.utterance_box.setDisabled(True)
                self.speaker_box.setDisabled(True)
                #self.dataset_box.setDisabled(True)
                #self.browser_load_button.setDisabled(True)
                #self.auto_next_checkbox.setDisabled(True)
                return
            #self.repopulate_box(self.dataset_box, datasets, random)

        # Select a random speaker
        if level <= 1:
            speakers_root = datasets_root.joinpath(self.current_dataset_name)
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            self.repopulate_box(self.speaker_box, speaker_names, random)

        # Select a random utterance
        if level <= 2:
            utterances_root = datasets_root.joinpath(
                self.current_dataset_name,
                self.current_speaker_name
            )
            utterances = []
            for extension in ['mp3', 'flac', 'wav', 'm4a']:
                utterances.extend(Path(utterances_root).glob("**/*.%s" % extension))
            utterances = [fpath.relative_to(utterances_root) for fpath in utterances]
            #self.repopulate_box(self.utterance_box, utterances, random)

    #def browser_select_next(self):
    #index = (self.utterance_box.currentIndex() + 1) % len(self.utterance_box)
    #self.utterance_box.setCurrentIndex(index)

    @property
    def current_encoder_fpath(self):
        return self.encoder_box.itemData(self.encoder_box.currentIndex())

    @property
    def current_synthesizer_fpath(self):
        return self.synthesizer_box.itemData(self.synthesizer_box.currentIndex())

    @property
    def current_vocoder_fpath(self):
        return self.vocoder_box.itemData(self.vocoder_box.currentIndex())

    def populate_models(self, models_dir: Path):
        # Encoder
        encoder_fpaths = list(models_dir.glob("*/encoder.pt"))
        if len(encoder_fpaths) == 0:
            raise Exception("No encoder models found in %s" % models_dir)
        self.repopulate_box(self.encoder_box, [(f.parent.name, f) for f in encoder_fpaths])

        # Synthesizer
        synthesizer_fpaths = list(models_dir.glob("*/synthesizer.pt"))
        if len(synthesizer_fpaths) == 0:
            raise Exception("No synthesizer models found in %s" % models_dir)
        self.repopulate_box(self.synthesizer_box, [(f.parent.name, f) for f in synthesizer_fpaths])

        # Vocoder
        vocoder_fpaths = list(models_dir.glob("*/vocoder.pt"))
        vocoder_items = [(f.parent.name, f) for f in vocoder_fpaths] + [("Griffin-Lim", None)]
        self.repopulate_box(self.vocoder_box, vocoder_items)

        #choose_fpaths = []
        #choose_items = [(f.parent.name, f) for f in choose_fpaths] + [("Tramp", None)]
        #self.repopulate_box(self.choose_box, choose_items)
    @property
    def selected_utterance(self):
        return self.utterance_history.itemData(self.utterance_history.currentIndex())

    def register_utterance(self, utterance: Utterance):
        self.utterance_history.blockSignals(True)
        self.utterance_history.insertItem(0, utterance.name, utterance)
        self.utterance_history.setCurrentIndex(0)
        self.utterance_history.blockSignals(False)

        if len(self.utterance_history) > self.max_saved_utterances:
            self.utterance_history.removeItem(self.max_saved_utterances)

        #self.play_button.setDisabled(False)
        self.generate_button.setDisabled(False)
        self.synthesize_button.setDisabled(False)

    def log(self, line, mode="newline"):
        if mode == "newline":
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        log_text = '\n'.join(self.logs)

        #self.log_window.setText(log_text)
        self.app.processEvents()

    def set_loading(self, value, maximum=1):
        # self.loading_bar.setValue(value * 100)
        # self.loading_bar.setMaximum(maximum * 100)
        # self.loading_bar.setTextVisible(value != 0)
        self.app.processEvents()

    def populate_gen_options(self, seed, trim_silences):
        if seed is not None:
            # self.random_seed_checkbox.setChecked(True)
            self.seed_textbox.setText(str(seed))
            self.seed_textbox.setEnabled(True)
        else:
            # self.random_seed_checkbox.setChecked(False)
            self.seed_textbox.setText(str(0))
            self.seed_textbox.setEnabled(False)

        #if not trim_silences:
        #    self.trim_silences_checkbox.setChecked(False)
        #    self.trim_silences_checkbox.setDisabled(True)

    # def update_seed_textbox(self):
    #     if self.random_seed_checkbox.isChecked():
    #         self.seed_textbox.setEnabled(True)
    #     else:
    #         self.seed_textbox.setEnabled(False)

    def reset_interface(self):
        self.draw_embed(None, None, "current")
        self.draw_embed(None, None, "generated")
        self.draw_spec(None, "current")
        self.draw_spec(None, "generated")
        # self.draw_umap_projections(set())
        self.set_loading(0)
        #self.play_button.setDisabled(True)
        self.generate_button.setDisabled(True)
        self.synthesize_button.setDisabled(True)
        self.vocode_button.setDisabled(True)
        # self.replay_wav_button.setDisabled(True)
        # self.export_wav_button.setDisabled(True)
        [self.log("") for _ in range(self.max_log_lines)]

    def recordIn(self):
        wav = sounddevice.rec(5 * encoder.sampling_rate, encoder.sampling_rate, 1)
        # wav = self.record_oneIn(encoder.sampling_rate, 5)

        if wav is None:
            return

        sounddevice.wait()
        filename = write("out.wav", encoder.sampling_rate, wav)
        r = sr.Recognizer()

        with sr.AudioFile(filename) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)
            # print(text)

    def __init__(self):
        ## Initialize the application
        self.app = QApplication(sys.argv)
        super().__init__(None)
        self.setWindowTitle("Intonation-Project")


        ## Main layouts
        # Root
        root_layout = QGridLayout()
        self.setLayout(root_layout)

        # Browser
        browser_layout = QGridLayout()
        root_layout.addLayout(browser_layout, 0, 0, 1, 2)

        # Generation
        gen_layout = QVBoxLayout()
        root_layout.addLayout(gen_layout, 0, 2, 1, 2)

        # Projections
        #self.projections_layout = QVBoxLayout()
        #root_layout.addLayout(self.projections_layout, 1, 0, 1, 1)

        # Visualizations
        vis_layout = QVBoxLayout()
        root_layout.addLayout(vis_layout, 1, 0, 1, 2)


        ## Projections
        # UMap
        # fig, self.umap_ax = plt.subplots(figsize=(3, 3), facecolor="#F0F0F0")
        # fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
        # self.projections_layout.addWidget(FigureCanvas(fig))
        # self.umap_hot = False
        # self.clear_button = QPushButton("Clear")
        # self.projections_layout.addWidget(self.clear_button)


        ## Browser
        # Dataset, speaker and utterance selection
        i = 0
        self.dataset_box = QComboBox()
        #browser_layout.addWidget(QLabel("<b>Dataset</b>"), i, 0)
        #browser_layout.addWidget(self.dataset_box, i + 1, 0)
        self.speaker_box = QComboBox()
        #browser_layout.addWidget(QLabel("<b>Speaker</b>"), i, 1)
        #browser_layout.addWidget(self.speaker_box, i + 1, 1)
        #self.utterance_box = QComboBox()
        #browser_layout.addWidget(QLabel("<b>Utterance</b>"), i, 2)
        #browser_layout.addWidget(self.utterance_box, i + 1, 2)
        #self.browser_load_button = QPushButton("Load")
        #browser_layout.addWidget(self.browser_load_button, i + 1, 3)

        #i += 2

        # Random buttons
        #self.random_dataset_button = QPushButton("Random")
        #browser_layout.addWidget(self.random_dataset_button, i, 0)
        #self.random_speaker_button = QPushButton("Random")
        #browser_layout.addWidget(self.random_speaker_button, i, 1)
        #self.random_utterance_button = QPushButton("Random")
        #browser_layout.addWidget(self.random_utterance_button, i, 2)
        #self.auto_next_checkbox = QCheckBox("Auto select next")
        #self.auto_next_checkbox.setChecked(True)
        #browser_layout.addWidget(self.auto_next_checkbox, i, 3)
        i += 1

        # Utterance box
        self.titel_part_one = QLabel("<b>Part 1- Choose the voice : </b>")
        browser_layout.addWidget(self.titel_part_one , i, 0)
        self.titel_part_one.setFont(QFont('Times', 25))
        self.titel_part_two = QLabel("<b>Part 2- Recording for sentence selection :           </b>")
        browser_layout.addWidget(self.titel_part_two  , i, 1)
        self.titel_part_two.setFont(QFont('Times', 25))
        i += 1

        self.titel_explaine = QLabel("    - Choose one of the three options")
        browser_layout.addWidget(self.titel_explaine , i, 0)
        self.titel_explaine.setFont(QFont('Times', 17))
        i += 1

        self.titel_one = QLabel("<b>Option 1: Add New User</b>")
        browser_layout.addWidget(self.titel_one , i, 0)
        #self.titel_one.setFont(QFont( 15))

        self.titel_two = QLabel("<b>Record your own sentence with intonation</b>")
        browser_layout.addWidget(self.titel_two , i, 1)
        #self.titel_two.setFont(QFont( 15)) #Arial
        i += 1

        #self.record_buttonIn = QPushButton("Record")
        #browser_layout.addWidget(self.record_buttonIn, i, 1)
        self.browser_browse_button1 = QPushButton("Browse")
        browser_layout.addWidget(self.browser_browse_button1, i, 1)
        #self.play_button = QPushButton("Play")
        #browser_layout.addWidget(self.play_button, i, 2)
        #self.stop_button = QPushButton("Stop")
        #browser_layout.addWidget(self.stop_button, i, 3)

        ## Generation
        self.play_button1 = QPushButton("Play")
        browser_layout.addWidget(self.play_button1, i + 1, 1)
        self.RMS_button1 = QPushButton("RMS - GRAPH")
        browser_layout.addWidget(self.RMS_button1, i + 2, 1)

        self.utterance_historyText = QLineEdit("Your Name", self)
        browser_layout.addWidget(self.utterance_historyText, i,0)
        i += 1
        self.button = QPushButton("Add", self)
        browser_layout.addWidget(self.button, i,0)
        self.button.move(20, 80)
        self.button.clicked.connect(self.on_click1)
        self.show()

        # intonation text
        # text = ""
        # self.utterance_sentenceText = QLineEdit(text, self)
        # browser_layout.addWidget(self.utterance_sentenceText, i,1)
        i += 1

        browser_layout.addWidget(QLabel("<b>In order for us to recognize your voice please record a sentence:</b>"), i, 0)
        i += 1
        #browser_layout.addWidget(QLabel("<b>We the people of the united states in order to form a more perfect union</b>"), i, 0)
        #i += 1

        # Random & next utterance buttons
        self.record_button = QPushButton("Record")
        browser_layout.addWidget(self.record_button, i, 0)
        # self.play_buttonR = QPushButton("Play")
        # i += 1
        # browser_layout.addWidget(self.play_buttonR, i, 0)
        # i += 1
        # self.stop_button = QPushButton("Stop")
        # browser_layout.addWidget(self.stop_button, i, 0)
        i += 1

        self.titel_one = QLabel("<b>Option 2: Browse</b>")
        browser_layout.addWidget(self.titel_one , i, 0)
        i +=1
        self.browser_browse_button = QPushButton("Browse")
        browser_layout.addWidget(self.browser_browse_button, i, 0)
        i +=1

        # Model and audio output selection
        self.utterance_history = QComboBox()
        browser_layout.addWidget(QLabel("<b>Option 3: Voice selection</b>"), i, 0)
        browser_layout.addWidget(self.utterance_history, i + 1, 0)
        i += 1
        self.play_button2 = QPushButton("Play")
        i += 1
        browser_layout.addWidget(self.play_button2, i, 0)
        i += 1
        self.stop_button = QPushButton("Stop")
        browser_layout.addWidget(self.stop_button, i, 0)
        i += 1
        # self.encoder_box = QComboBox()
        # browser_layout.addWidget(QLabel("<b>Encoder</b>"), i, 1)
        # browser_layout.addWidget(self.encoder_box, i + 1, 1)
        # self.synthesizer_box = QComboBox()
        # browser_layout.addWidget(QLabel("<b>Synthesizer</b>"), i, 2)
        # browser_layout.addWidget(self.synthesizer_box, i + 1, 2)
        # self.vocoder_box = QComboBox()
        # browser_layout.addWidget(QLabel("<b>Vocoder</b>"), i, 3)
        # browser_layout.addWidget(self.vocoder_box, i + 1, 3)
        #
        # self.audio_out_devices_cb=QComboBox()
        # browser_layout.addWidget(QLabel("<b>Audio Output</b>"), i, 4)
        # browser_layout.addWidget(self.audio_out_devices_cb, i + 1, 4)
        # i += 2

        #Replay & Save Audio
        # # browser_layout.addWidget(QLabel("<b>Toolbox Output:</b>"), i, 0)
        self.waves_cb = QComboBox()
        self.waves_cb_model = QStringListModel()
        self.waves_cb.setModel(self.waves_cb_model)
        self.waves_cb.setToolTip("Select one of the last generated waves in this section for replaying or exporting")
        #browser_layout.addWidget(self.waves_cb, j, 4)
        # self.replay_wav_button = QPushButton("Replay")
        # self.replay_wav_button.setToolTip("Replay last generated vocoder")
        # browser_layout.addWidget(self.replay_wav_button, i, 2)
        #self.export_wav_button = QPushButton("Export")
        #self.export_wav_button.setToolTip("Save last generated vocoder audio in filesystem as a wav file")
        #browser_layout.addWidget(self.export_wav_button, i, 3)
        i += 1

        i += 1
        j=0
        j+= 1
        self.titel_part_oneII = QLabel("Analyze :")
        browser_layout.addWidget(self.titel_part_oneII , j, 4)
        self.titel_part_oneII.setFont(QFont('Times', 25))
        j+= 1
        self.word1 = QPushButton("")
        browser_layout.addWidget(self.word1, j, 4)
        j+= 1
        self.word2 = QPushButton("")
        browser_layout.addWidget(self.word2, j, 4)
        j+= 1
        self.word3 = QPushButton("")
        browser_layout.addWidget(self.word3, j, 4)
        j+= 1
        self.word4 = QPushButton("")
        browser_layout.addWidget(self.word4, j, 4)
        j+= 1
        self.word5 = QPushButton("")
        browser_layout.addWidget(self.word5, j, 4)
        j+= 1
        self.word6 = QPushButton("")
        browser_layout.addWidget(self.word6, j, 4)
        j+= 1
        self.word7 = QPushButton("")
        browser_layout.addWidget(self.word7, j, 4)
        j+= 1
        self.word8 = QPushButton("")
        browser_layout.addWidget(self.word8, j, 4)
        j+= 1
        self.word9 = QPushButton("")
        browser_layout.addWidget(self.word9, j, 4)
        j+= 1
        self.word10 = QPushButton("")
        browser_layout.addWidget(self.word10, j, 4)
        ## Embed & spectrograms
        #vis_layout.addStretch()

        ## Embed & spectrograms
        #vis_layout.addStretch()

        gridspec_kw = {"width_ratios": [1, 1]}
        fig, self.current_ax = plt.subplots(1, 2, figsize=(5, 2.25), facecolor="#F0F0F0",
                                            gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))
        '''
        fig, self.gen_ax = plt.subplots(1, 2, figsize=(5, 2.25), facecolor="#F0F0F0",
                                        gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))
        '''
        for ax in self.current_ax.tolist():
            ax.set_facecolor("#F0F0F0")
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(False)


        ## Generation
        self.text_prompt = QPlainTextEdit(default_text)
        gen_layout.addWidget(self.text_prompt, stretch=1)

        layout = QHBoxLayout()
        self.generate_button = QPushButton("Synthesize and vocode")
        layout.addWidget(self.generate_button)
        gen_layout.addLayout(layout)

        layout = QHBoxLayout()
        self.synthesize_button = QPushButton("Synthesize only")
        layout.addWidget(self.synthesize_button)
        gen_layout.addLayout(layout)


        layout = QHBoxLayout()
        self.vocode_button = QPushButton("Vocode only")
        layout.addWidget(self.vocode_button)
        gen_layout.addLayout(layout)
        #j+= 1
        #browser_layout.addWidget(self.waves_cb, j, 4)


        layout = QHBoxLayout()
        self.encoder_box = QComboBox()
        self.encoder_box.addItem("Encoder")
        #layout.addWidget(self.encoder_box)
        self.vocoder_box = QComboBox()
        self.vocoder_box.addItem("Vocoder")
        #layout.addWidget(self.vocoder_box)
        #gen_layout.addLayout(layout)

        layout = QHBoxLayout()
        self.synthesizer_box = QComboBox()
        self.synthesizer_box.addItem("Synthesizer")
        #layout.addWidget(self.synthesizer_box)
        self.audio_out_devices_cb = QComboBox()
        self.audio_out_devices_cb.addItem("Audio Output")
        #layout.addWidget(self.audio_out_devices_cb)
        #gen_layout.addLayout(layout)
        #layout_seed = QGridLayout()
        self.random_seed_checkbox = QCheckBox("Random seed:")
        self.random_seed_checkbox.setToolTip("When checked, makes the synthesizer and vocoder deterministic.")
        #layout_seed.addWidget(self.random_seed_checkbox, 0, 0)

        self.seed_textbox = QLineEdit()
        self.seed_textbox.setMaximumWidth(30)
        #layout_seed.addWidget(self.seed_textbox, 0, 1)
        #self.trim_silences_checkbox = QCheckBox("Enhance vocoder output")
        #self.trim_silences_checkbox.setToolTip("When checked, trims excess silence in vocoder output."
        #                                        " This feature requires `webrtcvad` to be installed.")
        # layout_seed.addWidget(self.trim_silences_checkbox, 0, 2, 1, 2)
        #gen_layout.addLayout(layout_seed)

        # self.loading_bar = QProgressBar()
        # gen_layout.addWidget(self.loading_bar)



        # self.log_window = QLabel()
        # self.log_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        # gen_layout.addWidget(self.log_window)

        # self.txt_window = QLabel()
        # self.txt_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        # browser_layout.addWidget(self.txt_window, i, 1)

        self.logs = []
        #gen_layout.addStretch()


        ## Set the size of the window and of the elements
        max_size = QDesktopWidget().availableGeometry(self).size() * 0.8
        self.resize(max_size)

        ## Finalize the display
        self.reset_interface()
        self.show()

    '''
    def wav_graph(self):
            wav = wave.open(self.main_path,"r")
        # read audio samples
        input_data = read(self.main_path)
        audio = input_data[1]
        # plot the first 1024 samples
        plt.plot(audio[0:1024])
        # label the axes
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        # set the title
        plt.title("Sample Wav")
        # display the plot
        #plt.show()
    '''
    def on_click1(self):
        self.textboxValue = self.utterance_historyText.text()

    def rms1(self):
        if self.flag:
            analyze.rms_plot(self.rms)
            plt.show()

    def start(self):
        self.app.exec_()

    def plot(self,level1):
        if self.flag:
            #plt.ion()
            #words, signal_yourBag2, signal_gen, indexs_yourBag2, indexs_gen2= analyze.get_val()
            #print(words)
            #bold_yourBag2, fft_yourBag2, cros_yourBag2 = bolds_fft(words, signal_yourBag2, signal_gen, indexs_yourBag2, indexs_gen2, 30000)
            #fig, axs = plt.subplots(2, 1)
            #print(self.words[ind])
            #print(ind)
            #print(type(ind))
            print("self.words   :",self.words)
            if level1<len(self.words):
                level=self.words[level1]
                if level!= "":
                    print(self.fft_yourBag2)
                    ind=1
                    print(self.words[ind])
                    fft_yourBag3=list(self.fft_yourBag2.items())
                    xf_orig = fft_yourBag3[level1][1][0]
                    yf_orig = fft_yourBag3[level1][1][1]
                    xf_gen = fft_yourBag3[level1][1][2]
                    yf_gen = fft_yourBag3[level1][1][3]

                    self.cros_yourBag3=list(self.cros_yourBag2.items())
                    x = self.cros_yourBag3[level1][1][0]
                    #print(x)
                    corr = self.cros_yourBag3[level1][1][1]
                    #print(corr)
                    analyze.fftAndCross_plot(xf_orig, yf_orig, "original", start=20, color='b',x= self.cros_yourBag3[level1][1][0],corr= self.cros_yourBag3[level1][1][1])
                    analyze.fftAndCross_plot(xf_gen, yf_gen, "generated", start=20, color='g',x= self.cros_yourBag3[level1][1][0],corr= self.cros_yourBag3[level1][1][1])

                    #plt.show()


    def textfun(self,text):

        #self.txt_window.setText(text)
        arr=text.split(" ")
        values = self.bold_yourBag2.values()
        values_list = list(values)
        print(values_list)

        for i in range (1,(len(arr))):
            if(i==1):
                self.word1.setStyleSheet('background-color: white;')
                self.word1.setText(arr[i-1])
                if values_list[i-1]:
                    self.word1.setStyleSheet('background-color: yellow;')
            if(i==2):
                self.word2.setStyleSheet('background-color: white;')
                self.word2.setText(arr[i-1])
                if values_list[i-1]:
                    self.word2.setStyleSheet('background-color: yellow;')
            if(i==3):
                self.word3.setStyleSheet('background-color: white;')
                self.word3.setText(arr[i-1])
                if values_list[i-1]:
                    self.word3.setStyleSheet('background-color: yellow;')
            if(i==4):
                self.word4.setStyleSheet('background-color: white;')
                self.word4.setText(arr[i-1])
                if values_list[i-1]:
                    self.word4.setStyleSheet('background-color: yellow;')
            if(i==5):
                self.word5.setStyleSheet('background-color: white;')
                self.word5.setText(arr[i-1])
                if values_list[i-1]:
                    self.word5.setStyleSheet('background-color: yellow;')
            if(i==6):
                self.word6.setStyleSheet('background-color: white;')
                self.word6.setText(arr[i-1])
                if values_list[i-1]:
                    self.word6.setStyleSheet('background-color: yellow;')
            if(i==7):
                self.word7.setStyleSheet('background-color: white;')
                self.word7.setText(arr[i-1])
                if values_list[i-1]:
                    self.word7.setStyleSheet('background-color: yellow;')
            if(i==8):
                self.word8.setStyleSheet('background-color: white;')
                self.word8.setText(arr[i-1])
                if values_list[i-1]:
                    self.word8.setStyleSheet('background-color: yellow;')
            if(i==9):
                self.word9.setStyleSheet('background-color: white;')
                self.word9.setText(arr[i-1])
                if values_list[i-1]:
                    self.word9.setStyleSheet('background-color: yellow;')
            if(i==10):
                self.word10.setStyleSheet('background-color: white;')
                self.word10.setText(arr[i-1])
                if values_list[i-1]:
                    self.word10.setStyleSheet('background-color: yellow;')


