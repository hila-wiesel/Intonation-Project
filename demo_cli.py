import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import librosa.display

from encoder import inference as encoder
from encoder.audio import wav_to_mel_spectrogram
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
from STT.main import get_large_audio_transcription
from encoder.audio import wav_to_mel_spectrogram

if __name__ == '__main__':

    # Original audio wave spectrogram:
    # path_original_wav = 'stt\\preamble10.wav'
    # original_wav, sampling_rate = librosa.load(str(path_original_wav))  # change name inside while loop !!!!!!!
    # original_spectrogram = wav_to_mel_spectrogram(original_wav)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help= \
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help= \
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help= \
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" %
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    #  Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)

    #  Run a test
    print("Testing your configuration with small inputs.")
    # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # sampling rate, which may differ.
    # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # (or sometimes integers, but mostly floats in this project) ranging from -1 to 1.
    # The sampling rate is the number of values (samples) recorded per second, it is set to
    # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # to an audio of 1 second.
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))

    # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
    # returns, but here we're going to make one ourselves just for the sake of showing that it's
    # possible.
    embed = np.random.rand(speaker_embedding_size)
    # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
    # embeddings it will be).
    embed /= np.linalg.norm(embed)
    # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
    # illustrate that
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds)

    # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
    # can concatenate the mel spectrograms to a single one.
    mel = np.concatenate(mels, axis=1)
    # The vocoder can take a callback function to display the generation. More on that later. For
    # now we'll simply hide it like this:
    no_action = lambda *args: None
    print("\tTesting the vocoder...")
    # For the sake of making this test short, we'll pass a short target length. The target length
    # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
    # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
    # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
    # that has a detrimental effect on the quality of the audio. The default parameters are
    # recommended in general.
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

    print("All test passed! You can now synthesize speech.\n\n")

    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")

    print("Interactive generation loop")
    num_generated = 0
    while True:
        try:
            # Get the reference audio filepath
            record_to_syn_withoutIntonation = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                                              "wav, m4a, flac, ...):\n"
            tts_path = Path(input(record_to_syn_withoutIntonation).replace("\"", "").replace("\'", ""))

            ## Computing the embedding
            # First, we load the wav using the function that the speaker encoder provides. This is
            # important: there is preprocessing that must be applied.

            # The following two methods are equivalent:
            # - Directly load from the filepath:
            preprocessed_wav = encoder.preprocess_wav(tts_path)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(str(tts_path))
            # cur_original_spec = wav_to_mel_spectrogram(cur_original_wav) //Adding by hila
            # print("cur_original_spec", original_wav)
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file succesfully")

            # Then we derive the embedding. There are many functions and parameters that the
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")

            ## Generating the spectrogram

            #########
            # Adding by Nofar - Call to STT model

            # stt model- this model gets a record and return two outputs:
            # 1. text that will be input for tts model
            # 2. spectrogram of the record (with intonation)

            # file path to record
            record_to_syn_withIntonation = "STT enter an audio filepath (mp3, " \
                                           "wav, m4a, flac, ...):\n"
            stt_path = Path(input(record_to_syn_withIntonation).replace("\"", "").replace("\'", ""))

            # create waveform
            preprocessed_wav_stt = encoder.preprocess_wav(stt_path)
            # - If the wav is already loaded:
            original_wav_stt, sampling_rate_stt = librosa.load(str(stt_path))
            preprocessed_wav_stt = encoder.preprocess_wav(original_wav_stt, sampling_rate_stt)

            print("preprocessed_wav_stt", preprocessed_wav_stt)
            print("preprocessed_wav_stt, shape", preprocessed_wav_stt.shape)

            # create spectrogram - output 2
            spec_of_stt = wav_to_mel_spectrogram(preprocessed_wav_stt)
            # print("spec_of_stt", spec_of_stt)

            # the text of the record- output 1
            text = get_large_audio_transcription(stt_path)
            print("text", text)
            #########

            # If seed is specified, reset torch seed and force synthesizer reload
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")

            ## Generating the waveform
            print("Synthesizing the waveform:")

            # If seed is specified, reset torch seed and reload vocoder
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)
            print("generated_wav", generated_wav)
            print("generated_wav-shape", generated_wav.shape)
            generated_spec = wav_to_mel_spectrogram(generated_wav)
            # print("\ngenerated_spec", generated_spec)

            #############
            # Adding by Nofar- plot the waves of the audio with intonation and without intonation(in the time line)

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

            fig.tight_layout()
            librosa.display.waveplot(preprocessed_wav_stt, sampling_rate_stt, ax=ax[0])
            ax[0].title.set_text('with intonation')
            librosa.display.waveplot(generated_wav, synthesizer.sample_rate, ax=ax[1])
            ax[1].title.set_text('without intonation')
            plt.plot(range(0,len(preprocessed_wav_stt)), preprocessed_wav_stt)

            plt.show()

            ##############

            #############
            # Adding by Nofar - try to Subtract between the two spectograms- with intonation and without intonation

            # Subtraction between the two spectrograms - with intuition(spec_of_stt) and
            # without intuition(generated_spec)

            # subtractSpec = spec_of_stt - generated_spec

            # print("subtractSpec", subtractSpec)

            #############

            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.preprocess_wav(generated_wav)


            # Play the generated audio (non-blocking)
            # if not args.no_sound:
            #     import sounddevice as sd
            #
            #     try:
            #         sd.stop()
            #         sd.play(generated_wav, synthesizer.sample_rate)
            #     except sd.PortAudioError as e:
            #         print("\nCaught exception: %s" % repr(e))
            #         print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
            #     except:
            #         raise

            # Save it on the disk
            filename = "demo_output_%02d.wav" % num_generated
            print(generated_wav.dtype)
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % filename)

            ###########
            # try to Retrieving the first word from the waveforms

            first_word_generated = generated_wav[1000:3500]
            filename = "first_word_generated.wav"
            sf.write(filename, first_word_generated .astype(np.float32), synthesizer.sample_rate)
            print("first_word_generated type", type(first_word_generated))

            first_word_original = preprocessed_wav_stt[1000:3000]
            filename = "first_word_original.wav"
            print("first_word_original type", type(first_word_original))
            sf.write(filename, first_word_original(np.float32),synthesizer.sample_rate)

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
