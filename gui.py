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
from PIL import ImageTk, Image
from tkinter import *
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
   file = filedialog.askopenfilename(mode='r', filetypes=[("python files", "*.wav")])
   if file:
      #fs = 48000
      #content = file.read()
      #file = open('myData.wav', 'w')
      #saveHere = askdirectory(initialdir='/', title='Select File')
      #sf.write('111.wav', saveHere, fs)
      #file.write(os.path.join(saveHere, 'myData.wav'))
      file = open('myData.wav', 'rb')
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

def rgb_hack(rgb):
    return "#%02x%02x%02x" % rgb

# creating window & setting attribute 
gui = Tk()
gui.attributes('-fullscreen', True)
gui.title("Intonation-Project")

gui.columnconfigure(0, weight=1)
gui.columnconfigure(1, weight=3)

mainframe = Frame(gui)
mainframe.grid(column=100, row=100, sticky=(N, W, E, S))
#gui.columnconfigure(0, weight=4)
#gui.rowconfigure(0, weight=11)


# define font
myFont = font.Font(family='Helvetica',size=30)

# creating text label to display on window screen
Label(gui, text=" Intonation-Project",width=20,height=1,font=('Times', 50),bg='#116562',
    fg='#4a7abc').grid(sticky= tk.N ,row=0,column=0)

Button(gui, text=" Start",width=40,height=4,bg='#116562',fg='blue').grid(sticky= tk.S,row=14, rowspan=4)

Button(gui, text='Close', command=gui.destroy, bg='green', fg='blue',font=('Times', 18)).grid(sticky= tk.E,row=20, column=3)

'''
# adding image (remember image should be PNG and not JPG)
img = Image.open(r"/Users/dinamaizlis/Downloads/ODPEN_large.jpg")
img=img.resize((20, 20))
img=ImageTk.PhotoImage(img)
'''

#stt model-----------------------------------------------------------
Label(gui, text=" input STT model :  choose the option to add Audio",width=50,height=1,bg='#116562',font=('Times', 30)).grid(sticky= tk.NW ,row=1,column=0)

Label(gui, text=" Voice Recoder : ",width=10,height=2,bg='#116562').grid(sticky= tk.N,row=2,column=0)

Label(gui, text="Click the Button to browse a .wav audio :",width=30,height=2,bg='#116562').grid(sticky= tk.N, row=3,column=0)

Button(gui, text="Start", command=Voice_rec,width=10,height=2,bg="green",fg='blue').grid(sticky= tk.NS,row=2 ,column=1)

Button(gui, text="Browse", command=open_file,width=10,height=2,fg='blue').grid(sticky= tk.NS,row=3, column=1)


#tts model-----------------------------------------------------------
Label(gui, text=" input TTS model :  choose the option to add Audio ",width=50,height=1,bg='#116562',font=('Times', 30)).grid(sticky= tk.NW ,row=8,column=0)

Label(gui, text=" Voice Recoder : ",width=10,height=2,bg='#116562').grid(sticky= tk.N,row=9,column=0)

Label(gui, text="Click the Button to browse a .wav audio :",width=30,height=2,bg='#116562').grid(sticky= tk.N, row=10,column=0)

Button(gui, text="Start", command=Voice_rec,width=10,height=2,fg='blue').grid(sticky= tk.NS,row=9 ,column=1)

Button(gui, text="Browse", command=open_file,width=10,height=2,fg='blue').grid(sticky= tk.NS,row=10, column=1)




#Button(root, text='Close', command=callfun).grid(column=100, row=100, columnspan=2, rowspan=2, padx=5, pady=5)
gui.config(bg='#116562') 
gui.mainloop()
