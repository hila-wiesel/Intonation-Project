# Understanding the meaning of a sentence according to its intonation

There are sentences that have several meanings, depending on how the sentence is said, Which word is stressed.

For example:
The sentence "I did not steal your bag" can be interpreted in several ways, depending on the word that the speaker emphasizes (underlined):

**1.** "I did not  '*steal*' your bag" - I did not steal the bag, I may have taken or asked.
The emphasis is on the act of stealing, which the speaker wants to deny.

**2.** "I did not steal your '*bag*'" - I did not steal the bag, it is possible that I stole something else from you. The emphasis is on the stolen object, the speaker wants to deny that the stolen object is a bag.

**3.** "I did not steal '*your*' bag" - I did not steal from you. The negation is about who the bag was stolen from. The speaker wants to rule out that the bag was stolen from you (it may have been stolen by me, from someone else).

Humans distinguish between the change in meaning naturally, but it is difficult to point to a certain legality that determines the difference.

This project deals with building a model based on deep learning, which provides an interpretation of the meaning of a sentence according to the intonation of the speaker - the way in which the sentence is said, such as emphasis on certain words, voice pitch and the emotion in which the sentence is said (sarcasm, anger, etc.).

Language processing research is a field that has gained momentum in recent years, the goal of which is to help computers "understand" things that are said or typed in human natural language.

Understanding the meaning of the sentence is a milestone in many areas, such as identifying a person in distress by their voice, maintaining the quality of customer service by processing the recordings with them, converting audio to text with punctuation marks, developing a bot system and more.

### Description of the solution:

input: speech - on it we would like to carry out the process of understanding its meaning,
        And several other recordings of the same speaker.
Output: Analysis of the sentence meaning of the first audio

### The process:

![image](https://user-images.githubusercontent.com/61710157/184504179-edf732a4-3e61-4ced-84b4-5f13c1be3b3f.png)


### Intonation process:

![image](https://user-images.githubusercontent.com/61710157/184395287-20afd24b-6ea6-4a00-a791-30cea05787ce.png)


## Setup

### 1. Install Requirements
1. Python 3.7 is recommended.
2. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
3. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
4. run this pip install (pip==22.2.2) on directory "Intonation-Project" on cmd: `python.exe -m pip install --upgrade pip` 
5. Install the remaining requirements with `pip install -r requirements.txt`


### 2. (Optional) Download Pretrained Models
Pretrained models are now downloaded automatically. If this doesn't work for you, you can manually download them [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).


### 3. Launch the Toolbox
You can then try the toolbox:

`python demo_toolbox.py` 

### ToolBox explanation - youtube video
[<img src="https://github.com/hila-wiesel/Intonation-Project/blob/cc40b4cef2157d56648dbfa53d390de61b96c715/pictures/Image%20for%20video.png" width="50%">](https://youtu.be/gYPRfR5BTds "Now in Android: 55")

### Examples of the following sentences - youtube video

#### 1) I did not steal '*your*' bag - https://youtu.be/MfHIzr1g_1c
#### 2) I did not '*steal*' your bag - https://youtu.be/VfVKxUXDhN0
#### 3) '*Hello*' this is our finel project - https://youtu.be/cqkYHR0h8us

