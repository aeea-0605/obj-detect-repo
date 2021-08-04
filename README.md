# Object Detection And Text OCR 프로젝트

## 1. 개요

### 1-1. 프로젝트 목적
현업에서 사용되는 의료 영상의 특정 섹션을 감지하고, 해당 섹션에 존재하는 텍스트를 추출하여 의료기기 사용자에게 제공하는 프로젝트 입니다.

### 1-2. 프로젝트 목표
Yolov5와 OpenCV를 사용한 특성 섹션에 대한 Object Detecting 및 Tesseract와 regex를 통해 텍스트를 추출하는 일련의 과정을 결과 동영상으로 저장하여 제공합니다.

### 1-3. 기술적 Summary
- numpy를 사용한 data demension 조정 및 데이터 전처리
- OpenCV를 사용한 이미지 전처리 및 형변환, 프레임 추출, Object Detection, 영상 저장
- roboflow를 통한 라벨링 후 Yolov5를 사용한 Object Detection
- Tesseract, regex를 사용한 텍스트 추출

### 1-4. 데이터셋 및 설명
데이터 : 특정 기업이 실제로 사용하는 의료 영상을 제공받아 진행하였습니다.
- 총 20개의 의료 영상이 제공되었습니다.

Object Detecting을 진행할 섹션

<img width="171" alt="스크린샷 2021-08-04 오후 6 32 05" src="https://user-images.githubusercontent.com/80459520/128158134-cf6f925d-e996-4a3b-9c5a-78aa76c737ac.png">

- Text 추출을 진행할 섹션 : small

---
---

## 2. 결론

### **2-1. Object Detection**

<CASE 1> Yolov5

<img width="608" alt="스크린샷 2021-08-04 오후 5 25 03" src="https://user-images.githubusercontent.com/80459520/128187721-c929e056-00cf-44c7-bd69-601b51a224bd.png">

- 과정
    1. OpenCV를 사용한 의료 영상에서 sample frame들을 추출한 뒤 roboflow를 통해 small, big labeling을 진행
    2. 라벨링된 데이터를 기반으로 Yolov5를 사용한 Object Detection Model 생성

- **big 섹션에 대해서는 좋은 성능을 가지지만 small은 감지하지 못하거나 낮은 확률로 감지됨. 따라서 small에 대해서는 OpenCV의 matchTemplate 메서드로 detecting 진행**

<br/>

<CASE 2> OpenCV의 matchTemplate

<img width="600" alt="스크린샷 2021-08-04 오후 5 26 11" src="https://user-images.githubusercontent.com/80459520/128188566-04989d90-2a40-45f6-a673-c62b07e8c85f.png">

- Object Detection을 위한 총 15개의 target images 생성을 통해 95% 이상의 small에 대해 detecting.

<br/>

### **2-2. OCR Engine을 사용한 Text 추출**
<img width="680" alt="스크린샷 2021-08-04 오후 5 21 16" src="https://user-images.githubusercontent.com/80459520/128189784-75c066c6-6c45-4e2f-85c5-0abb16237055.png">

- 2-1의 CASE 2에서 detected small 섹션에 대해 Tesseract와 regex를 통해 Text 추출을 진행.
- 과정
    1. 섹션 픽셀값의 백분위 중 93.5의 값을 임계값으로 정해 0, 255로 이진화
    2. 섹션의 좌측 및 상단의 일정부분에 대해 255값 지정 (noise 제거)
    3. Median Blur를 사용해 전체적인 noise 제거
    4. pytesseract를 사용한 TEXT 추출
    5. 정규표현식을 사용한 최종 TEXT 도출을 위한 전처리
        - TEXT를 추출 못했을 경우 'detect failed' 라는 문구 출력

<br/>

### **2-3. 결과영상에 대한 Sample Video**
![화면-기록-2021-08-04-오후-10 45 08](https://user-images.githubusercontent.com/80459520/128192615-0ce69e87-5542-4cfe-acdd-8710d49cdcce.gif)

- **RED SECTION** : OpenCV matchTemplate를 통해 감지한 small 섹션
- **BLUE TEXT** : tesseract, regex를 통해 추출한 최종 text
- small, big 섹션에 대한 Object Detection 성능은 95%, 그 이상 감지하는 결과를 보였고 small 섹션에 대한 TEXT 추출은 약 80%의 성능을 보였음.

---
<br/>

# 💡 제언
- Pre-trained Tesseract 모델에 감지된 small 섹션 데이터에 대한 학습을 진행한다면 좀 더 좋은 성능을 보유할 것으로 예상됩니다.
- 감지된 섹션에 대해 noise 제거, 컨투어 도출, background 분리 등의 추가적인 방법을 적용한다면 좀 더 좋은 Text 추출 성능을 가질 수 있다고 생각합니다.
- Big 섹션에 대해서는 small보다 배경과 Text부분의 차이가 확연한 부분도 존재하고, 그 외의 부분은 small 섹션과 비슷하기 때문에 small의 text 도출 성능이 좋아진다면 big 섹션에 대한 text 도출 성능 또한 좋을 것이라 생각합니다.
- 아직 Section 감지 및 Text 도출에 있어 미진한 부분이 많기에 추후에 다양한 방법을 사용해 보완이 필요하다고 생각합니다.

---
<br/>

# Code Explanation
- [module.py](https://github.com/aeea-0605/obj-detect-repo/blob/main/module/module.py)
    - Input Video의 Class 및 detect section, detect text, make target image Function 등이 있는 모듈파일
- [detect.py](https://github.com/aeea-0605/obj-detect-repo/blob/main/detect.py)
    - OpenCV의 matchTemplate으로 Object Detection한 뒤 tesseract로 text 추출하고 input video에 대한 결과 영상을 만드는 code
- [Yolov5.ipynb](https://github.com/aeea-0605/obj-detect-repo/blob/main/Yolov5.ipynb)
    - labeling dataset을 불러와 Yolov5를 통한 Object Detection하는 Notebook (Colab에서 진행)
- [test_and_extract_target_img.py](https://github.com/aeea-0605/obj-detect-repo/blob/main/test_and_extract_target_img.py)
    - 결과 영상 생성 전 Input Video에 대한 test 및 key event를 통한 정지된 frame에서의 target image를 생성하는 code
- [make_frame.py](https://github.com/aeea-0605/obj-detect-repo/blob/main/make_frame.py) : Input Video에 대한 15개의 frame을 생성해주는 code
- [extract_target.py](https://github.com/aeea-0605/obj-detect-repo/blob/main/extract_target.py) : 특정 frame에서 SelectROI를 통한 target image 생성하는 code
