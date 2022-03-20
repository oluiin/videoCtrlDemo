from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import os

import socket
import time

host, port = "127.0.0.1", 9090
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))
sendNum = 0

# cascade classifier (다단계 분류)를 이용한 객체 검출(Object Detection)
# Haar cascade classifier 이란?
# 다수의 객체 이미지(이를 positive 이미지라 함)와 객체가 아닌 이미지(이를 negative 이미지라 함)를 cascade 함수로 트레이닝 시켜 객체 검출을 달성하는 머신러닝 기반의 접근 방법
# cascade로 사람 얼굴을 검출할 것임
# 얼굴 검출을 위해 많은 수으 ㅣ얼굴 이미지와 얼굴이 없는 이미지를 classifier에 트레이닝 시켜 얼굴에 대한 특징들을 추출해서 데이터로 저장
# 얼굴 검출을 위한 Haar-Cascade 트레이닝 데이터를 읽어 CascadeClassifier 객체를 생성
cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))  # 사람 얼굴 정면에 대한 Haar-Cascade 학습 데이터

model = load_model('hand_detect_model2.h5')  # 손가락을 식별하는 모델 호출


cap = cv.VideoCapture(0)    # VideoCapture 객체 생성

# 라이브로 들어오는 비디오를 frame 별로 캡쳐하고 이를 화면에 display
# 특정 키를 누를 때까지 무한 루프
while True:

    # 재생되는 비디오의 한 frame씩 읽기
    ret, img = cap.read()
    # 비디오 프레임을 제대로 읽었다면 ret 값이 True가 되고 실패하면 False가 된다
    if ret == False:
        break

    img_result = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지 흑백처리
    gray = cv.equalizeHist(gray)    # 히스토그램 평활화(Histogram Equalization)를 적용하여 이미지의 콘트라스트를 향상시킴

    # 얼굴 위치를 리스트로 리턴 (x, y, w, h) / (x, y ):얼굴의 좌상단 위치, (w, h): 가로 세로 크기
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)

    # 얼굴 영역에 검정 사각형 만들기
    height, width = img.shape[:2]
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1 - 10, 0), (x1+x2+10, height), (0, 0, 0), -1)

    # bgr -> hsv 로 변환
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # HSV : Hue(색상), Saturation(채도), Value(명도)의 요소를 갖는 색상공간
    # Hue(색상) : [0,179]
    # Saturation(채도) : [0,255]
    # Value(명도) : [0,255]
    # 범위의 값을 갖는다.
    # HSV 중 H에 해당하는 색상값을 이용해 이미지에서 특정 색상을 띠는 물체를 추출할 수 있다.

    # Skin HSV 범위 지정
    low = (0, 30, 0)
    high = (15, 255, 255)

    # 이미지를 binary 이미지로 전환
    img_binary = cv.inRange(img_hsv, low, high)
    # 검사할 이미지(img_hsv)에서 픽셀별로 검사하여 하한값(low)과 상한값(high) 사이에 들어오면 흰색, 그렇지 않은 픽셀은 검은색으로 표시한 이미지를 반환

    # 경계선 찾기
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))     # 타원 모양으로 매트리스 생성,
    # getStructuringElement 함수를 통해 커널과 같은 역할을 하는 구조화 요소를 구한다
    # cv.getStructuringElement(shape, ksize[, anchor]) -> retval
    # shape: 구조화 요소 커널의 모양
    # - cv.MORPH_CROSS: 십자가형
    # - cv.MORPH_ELLIPSE: 타원형
    # - cv.MORPH_RECT: 직사각형
    # kszie: 구조화 요소 커널의 크기
    # anchor: 구조화 요소 커널의 기준점. default 값 (-1, -1)은 기준점을 중심으로 잡는다.
    #         해당 값은 오직 MORPH_CROSS 모양을 사용할 때만 영향을 준다.
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
    # Dilation(팽창)과 Erosion(수축)의 형태학적 변환을 결합하여 연산한다
    # cv.morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
    # src: 입력 이미지. 채널 수는 상관 없으나 다음과 같은 이미지 데이터만 사용할 수 있다.
    #      CV_8U, CV_16S, CV_32F, CV_64F
    # op: 형태학적 연산의 종류
    # - cv.MORPH_OPEN: cv.dilation(cv.erode(src))
    # - cv.MORPH_CLOSE: cv.erode(cv.dilation(src))
    # - cv.MORPH_GRADIENT: cv.dilate(src)-cv.erode(cv)
    # - cv.MORPH_TOPHAT: src-opening
    # - cv.MORPH_BLACKHAT: closing-src
    # kernel: 구조화 요소 커널, 앞서 cv.getStructuringElement로 생성할 수 있다
    # anchor: 구조화 요소 커널에서의 기준점. default로 (-1, -1)이 설정되어 있으며 이는 커널의 중앙을 뜻한다
    # iterations: 형태학적 변환 반복 횟수
    # 여기서 말하는 Opening 과 closing이란 ??
    # Opening: 이미지에 Erosion(침식) 적용 후 Dilation(팽창) 적용하는 것으로 영역이 점점 둥글게 된다.
    #          따라서 점 잡음이나 작은 물체, 돌기 등을 제거하는데 적합
    # Closing: 이미지에 Dilation(팽창) 적용 후 Erosion(침식) 적용하는 것으로 영역과 영역이 서로 붙기 때문에
    #          이미지의 전체적인 윤곽을 파악하기에 적합하다

    # binary 이미지에서 윤곽선을 검색
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.findContours(이진화 이미지, 검색방법, 근사화방) -> 윤곽선, 계층 구조 반환
    # 검색 방법:
    # - cv.RETR_EXTERNAL: 외곽 윤곽선만 검출, 계층 구조를 구성하지 않는다
    # - cv.RETR_LIST: 모든 윤곽선을 검출, 계층 구조를 구성하지 않는다
    # - cv.RETR_CCOMP: 모든 윤곽선을 검출, 계층 구조는 2단계로 구성
    # - cv.RETR_TREE: 모든 윤곽선을 검출, 계층 구조를 모두 형 (Tree 구조)
    # 윤곽선: Numpy 구조의 배열로 검출된 윤곽선의 지점
    # 계층 구조: 윤곽선 계층 구조를 의미, 각 윤곽선에 해당하는 속성 정보들이 담겨 있음


    max_contour = None
    max_area = -1

    # 영익이 가장 큰 윤곽선을 선택 : 손 검출
    for contour in contours:
        area = cv.contourArea(contour)  # 폐곡선인 contour의 면적
        x, y, w, h = cv.boundingRect(contour)
        if (w * h) * 0.4 > area:
            continue
        if w > h:
            continue
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 10000:
        max_area = -1

    # 검출된 윤곽선을 그린다
    cv.drawContours(img_result, [max_contour], 0, (255, 0, 0), 3)
    # cv.drawContours(이미지, [윤곽선], 윤곽선 인덱스, (B,G,R), 두께, 선형 타입)
    # 윤곽선: 검출된 윤곽선들이 저장된 Numpy 배열
    # 윤곽선 인덱스: 검출된 윤곽선 배열에서 몇 번째 인덱스의 윤곽선을 그릴지를 의미

    # 손 영역의 위치 값을 찾는다
    contours_xy = np.array(max_contour)
    # x의 min과 max 찾기
    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        value.append(contours_xy[i][0][0])  # 네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
    # y의 min과 max 찾기
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        value.append(contours_xy[i][0][1])  # 네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)

    # frame에서 손 영역만 자른다
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min
    img_trim = img[y_min-10:y_max+10, x_min-10:x_max+10]

    # 손 영역 이미지에서 손가락 검출 모델을 이용하여 손가락 모양을 예측한다
    try:
        hand_input = cv.resize(img_trim, (128, 128))
        hand_input = np.expand_dims(hand_input, axis=0)
        hand_input = np.array(hand_input)
        cv.imshow("Result", img_result)
        predictions = model.predict(hand_input)

        orderNum = np.argmax(predictions)
        typeNum = int(orderNum)

        print("predict : ", np.argmax(predictions), ) # frame에서 손 영역에 윤곽선을 그린 이미지를 반환

        time.sleep(0.3)
        if (orderNum >= 0):
            sendNum = str(typeNum)  # 유니티로 보내기 위해 유니티에서 받는 값을 맞춰 string으로 type 변경
            sock.sendall(sendNum.encode("UTF-8"))  # 보내고자 하는 data 보냄
            receivedData = sock.recv(1024).decode("UTF-8")  # 유니티에서 보낸 데이터를 받음


        #cv.imshow("Result", img_trim)  # frame에서 손 영역을 자른 이미지를 반환
        cv.waitKey(100)
    except:
        print("손을 인식하지 못했습니다.")
        continue