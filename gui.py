import sounddevice as sd
import soundfile as sf
from tkinter import *
import tkinter.font as font
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import tkinter.messagebox 
import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory
def Voice_rec():
    fs = 48000
      
    # seconds
    duration = 5
    myrecording = sd.rec(int(duration * fs), 
                         samplerate=fs, channels=2)
    sd.wait()
    # Save as FLAC file at correct sampling rate
    return sf.write('my_Audio_file.wav', myrecording, fs)


def open_file():
   file = filedialog.askopenfilename(mode='r', filetypes=[("wav files", "*.wav")])
   if file:
      #fs = 48000
      #content = file.read()
      #file = open('myData.wav', 'w')
      #saveHere = askdirectory(initialdir='/', title='Select File')
      #sf.write('111.wav', saveHere, fs)
      #file.write(os.path.join(saveHere, 'myData.wav'))
      file = open('myData.wav', 'w')
      saveHere = askdirectory(initialdir='/', title='Select File')

      file.write('111.wav', saveHere)
      file.close()
      #print("%d characters in this file" % len(content))

def Export_File():
    file = open('myData.wav', 'w')
    saveHere = askdirectory(initialdir='/', title='Select File')

    file.write(os.path.join(saveHere, 'myData.wav'))

def callfun():
    if tkinter.messagebox.askyesno('Python Guides', 'Really quit'):
        tkinter.messagebox.showwarning('Python Guides','Close this window')
    else:
        tkinter.messagebox.showinfo('Python Guides', 'Quit has been cancelled')
# creating window & setting attribute 
root = Tk()
root.attributes('-fullscreen', True)
root.title("Intonation-Project")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)

mainframe = Frame(root)
mainframe.grid(column=100, row=100, sticky=(N, W, E, S))
root.columnconfigure(0, weight=4)
root.rowconfigure(0, weight=11)


# define font
myFont = font.Font(family='Helvetica',size=30)

# creating text label to display on window screen
Label(root, text=" Intonation-Project",width=40,height=1).grid(sticky= tk.N ,row=0,column=0,padx=5, pady=5)


#stt model-----------------------------------------------------------
Label(root, text=" input STT model :  choose the option to add Audio",width=40,height=1).grid(sticky= tk.NW ,row=1,column=0)

Label(root, text=" Voice Recoder : ",width=30,height=2).grid(sticky= tk.N,row=2,column=0)

Label(root, text="Click the Button to browse a .wav audio",width=30,height=2).grid(sticky= tk.N, row=3,column=0)

Button(root, text="Start", command=Voice_rec,width=10,height=2).grid(sticky= tk.NS,row=2 ,column=1)

Button(root, text="Browse", command=open_file,width=10,height=2).grid(sticky= tk.NS,row=3, column=1)


#tts model-----------------------------------------------------------
Label(root, text=" input TTS model :  choose the option to add Audio ",width=40,height=1).grid(sticky= tk.NW ,row=8,column=0)

Label(root, text=" Voice Recoder : ",width=30,height=2).grid(sticky= tk.N,row=9,column=0)

Label(root, text="Click the Button to browse a .wav audio",width=30,height=2).grid(sticky= tk.N, row=10,column=0)

Button(root, text="Start", command=Voice_rec,width=10,height=2).grid(sticky= tk.NS,row=9 ,column=1)

Button(root, text="Browse", command=open_file,width=10,height=2).grid(sticky= tk.NS,row=10, column=1)

Button(root, text='Close', command=root.destroy).grid(sticky= tk.SE,row=11, column=4)


#Button(root, text='Close', command=callfun).grid(column=100, row=100, columnspan=2, rowspan=2, padx=5, pady=5)

root.mainloop()
