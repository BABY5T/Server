## IOT 요람 서버 파트

### Overview
1. 라즈베리 파이에서 음원과 영상을 받아온다.
2. 각각 음성 분류 모델과 자세 분류모델을 통해 결과 값을 JSON 으로 반환
3. 웹서버에서 GET 명령 통해 JSON 파일 읽어와서 디스플레이

### receiveAudio.py
주기적으로 새로 녹음된 음성파일을 받아옴
받아온 음성파일을 Yam_Net 에 적용시켜 울음소리가 담긴 음성파일인지 확인
만약 울음소리 음성파일이라고 판단된경우 울음소리 원인 판별하는 prediction 함수 적용
prediction 함수 적용 결과를 해당 시간과 함께 JSON 파일 업데이트

prediction -> 음성파일을 받아와 전처리 진행 후 학습시킨 ResNet 이용해서 울음소리 원인 파악

receive_file -> 소켓통신 통해 라즈베리 파이로부터 주기적으로 녹음된 음성파일을 받아옴

Yam_Net -> 아이의 울음소리를 감지하는 모델 , 사전학습 모델  521개의 오디오 이벤트 클래스를 예측


### receiveVideo.py
소켓 통신 통해 받아온 데이터를 이미지로 디코딩해서 디스플레이 하고 

해당 이미지를 모델의 입력으로 넣어 실시간으로 자세 분류 

분류 결과 값은 JSON 에 저장 

----

### 실제 데모 웹서비스 
![image](https://github.com/766O/Sejong_SW_academic_festical/assets/121467486/41b9ba92-b891-47d2-89b7-fcbc7923007a)


### 음성 분류 
![image](https://github.com/766O/Sejong_SW_academic_festical/assets/121467486/c7bc83f0-42c8-4816-b52e-d4fe3e9317ce)


### 자세 분류
<img width="352" alt="image" src="https://github.com/766O/Sejong_SW_academic_festical/assets/121467486/dcabbc8e-fa8d-4587-a273-9efc2517288c">

