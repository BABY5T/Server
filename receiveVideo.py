# -*- coding: utf8 -*-
import socket
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

from dotenv import load_dotenv
import os 

# load .env
load_dotenv()

serverIp = os.environ.get('serverIp')
videoSocketPort = int(os.environ.get('videoSocketPort'))

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

HOST = serverIp
PORT = videoSocketPort

# TCP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

# 서버의 아이피와 포트번호 지정
s.bind((HOST, PORT))
print('Socket bind complete')

# 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
s.listen(10)
print('Socket now listening')

# 연결
conn, addr = s.accept()

while True:
    length = recvall(conn, 16)
    if length is None:
        print("Link decline.")
        break

    stringData = recvall(conn, int(length))
    if stringData is None:
        print("Data get error")
        break

    data = np.frombuffer(stringData, dtype='uint8')

    # data를 디코딩한다.
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    cv2.imshow('ImageWindow', frame)
    
    # 모델에 입력하기 위한 이미지 전처리
    image = Image.fromarray(frame).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # 모델 예측
    prediction = model.predict(np.expand_dims(normalized_image_array, axis=0))
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # 결과 출력
    print("Class:", class_name, "Confidence Score:", confidence_score)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
cv2.destroyAllWindows()
