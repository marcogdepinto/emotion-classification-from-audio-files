import keras
import numpy as np
import librosa
import pyaudio
import wave

class livePredictions:

    def __init__(self, path, file):

        self.path = path
        self.file = file

    def load_model(self):
        '''
        I am here to load you model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        '''
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model

    def makepredictions(self):
        '''
        I am here to process the files and create your features.
        '''
        data, sampling_rate = librosa.load(self.file)
        print(data)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print( "Prediction is", " ", self.convertclasstoemotion(predictions))

    def convertclasstoemotion(self, pred):
        '''
        I am here to convert the predictions (int) into human readable strings.
        '''
        self.pred  = pred

        if pred == 0:
            pred = "neutral"
            return pred
        elif pred == 1:
            pred = "calm"
            return pred
        elif pred == 2:
            pred = "happy"
            return pred
        elif pred == 3:
            pred = "sad"
            return pred
        elif pred == 4:
            pred = "angry"
            return pred
        elif pred == 5:
            pred = "fearful"
            return pred
        elif pred == 6:
            pred = "disgust"
            return pred
        elif pred == 7:
            pred = "surprised"
            return pred

# Here you can replace path and file with the path of your model and of the file from the RAVDESS dataset you want to use for the prediction,
# Below, I have used a neutral file: the prediction made is neutral.
while True:
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 22050
    CHUNK = 1024
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "file.wav"
    audio = pyaudio.PyAudio()
 
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    #print ("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    #print ("finished recording")
    
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
 
    pred = livePredictions(path=r'C:\Users\apoochelvan\Downloads\Emotion_Voice_Detection_Model (3).h5',
                       file=r'C:\Users\apoochelvan\audio\file.wav')
    pred.load_model()
    pred.makepredictions()
