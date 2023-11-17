import socket
import time
import os
from datetime import datetime

import tensorflow as tf
import librosa
import io
import csv

from dotenv import load_dotenv
import os 

# load .env
load_dotenv()

rasbIp = os.environ.get('RasbIp')
videoSocketPort = int(os.environ.get('audioSocketPort'))


class YamNet:
    def __init__(self, model_path, csv_path):
        self.model_path = model_path
        self.csv_path = csv_path

        self.interpreter = tf.lite.Interpreter(model_path)
        self.waveform_input_index = None
        self.scores_output_index = None
        self.embeddings_output_index = None
        self.spectrogram_output_index = None
        self.class_names = self.get_class_names()
        self.set_interpreter()

    def __call__(self, waveform) -> bool:
        return self.is_baby_cry(waveform)

    def set_interpreter(self):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.waveform_input_index = input_details[0]['index']
        self.scores_output_index = output_details[0]['index']
        self.embeddings_output_index = output_details[1]['index']
        self.spectrogram_output_index = output_details[2]['index']

    def get_class_names(self):
        csv_text = open(self.csv_path).read()
        class_map_csv = io.StringIO(csv_text)
        class_names = [display_name for (
            class_index, mid, display_name) in csv.reader(class_map_csv)]
        class_names = class_names[1:]  # Skip CSV header
        return class_names

    def get_prediction(self, waveform, top_n=None):
        if top_n != None and top_n >= len(self.class_names):
            raise ValueError('top_n is bigger than classes length')

        self.interpreter.resize_tensor_input(
            self.waveform_input_index, [len(waveform)], strict=True)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.waveform_input_index, waveform)
        self.interpreter.invoke()
        scores = (
            self.interpreter.get_tensor(self.scores_output_index),
            self.interpreter.get_tensor(self.embeddings_output_index),
            self.interpreter.get_tensor(self.spectrogram_output_index))[0]
        # print(scores.shape, embeddings.shape, spectrogram.shape)

        index_sort_list = scores.mean(axis=0).argsort()

        if top_n != None:
            index_sort_list = scores.mean(axis=0).argsort()[-top_n:][::-1]

        return [self.class_names[index] for index in index_sort_list]

    def is_baby_cry(self, waveform) -> bool:
        prediction = self.get_prediction(waveform, top_n=5)
        return any([target in prediction for target in ['Crying, sobbing', 'Baby cry, infant cry']])



cur_path = os.getcwd()
csv_path = os.path.join(cur_path, 'yamnet_class_map.csv')
yam_model_path = os.path.join(cur_path, 'lite-model_yamnet_tflite_1.tflite')

def is_Crying(yam_model_path,csv_path,wavfile_path):
    yamNet = YamNet(yam_model_path, csv_path)
    waveform = librosa.load(wavfile_path, sr=16000)[0]
    return yamNet(waveform)
    
    


def create_json_entry(date, reason):
    return {"date": date, "reason": reason}

def append_to_json(json_file, key, entry):
    data = {}

    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        pass

    if key not in data:
        data[key] = []

    data[key].append(entry)

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

def prediction(file_name):
    import numpy as np
    import tensorflow as tf


    model_path = '/Users/kim/IOTT/resnet.h5'
    model = tf.keras.models.load_model(model_path)
    #model.summary() 

    import librosa
    from skimage.transform import resize

    classes = ['sad', 'hug', 'diaper', 'hungry',
            'sleepy', 'awake', 'uncomfortable']




    def get_input_vector_from_file(file_path: str) -> np.ndarray:
        #import pdb;pdb.set_trace()
        y, sr = librosa.load(file_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=501)
        mel_spec_dB = librosa.power_to_db(mel_spec, ref=np.max)
        RATIO = 862 / 128
        mel_spec_dB_resized = resize(mel_spec_dB, (mel_spec_dB.shape[0], mel_spec_dB.shape[1] * RATIO),
                                    anti_aliasing=True, mode='reflect')
        mel_spec_dB_stacked = np.stack([mel_spec_dB_resized] * 3, axis=-1)
        return mel_spec_dB_stacked[np.newaxis, ]




    test_vector = get_input_vector_from_file(file_name)

    #print(test_vector.shape)

    tf.config.set_visible_devices([], 'GPU')

    predictions = model.predict(test_vector)[0]

    return(classes[np.argmax(predictions)])

def receive_file():
    #prepare client socket - TCP 
    serverName = rasbIp # 연결하는 대상 ip
    serverPort = videoSocketPort

    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.connect((serverName, serverPort))
    current_dir = os.getcwd() 

    filesize = int(clientSocket.recv(1024).decode())
    clientSocket.sendall(b'ACK')
    
    # Open output file
    filename = f"received_{int(time.time())}.wav"
    with open(filename, 'wb') as f:
        bytes_received = 0
        while bytes_received < filesize:
            data = clientSocket.recv(1024)
            if not data:
                break
            f.write(data) 
            bytes_received += len(data)
        f.close()
    clientSocket.close()
    print("[received] === ", time.time())
    return filename


while True:
    result_list=[]
    new_file=receive_file()
    if is_Crying(yam_model_path,csv_path,new_file)==1:
        result=prediction(new_file)
    else:
        result='not_crying'
    
    import json
    
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = create_json_entry(current_date, result)
    
    json_file_path = "/Users/kim/IOTT/output.json"
    append_to_json(json_file_path, 'cry', entry)

    

#print(M)
    
    
    