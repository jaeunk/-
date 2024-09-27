# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
## 과거 tesseract와 text detection을 통하여 개발한 프로젝트가 있었으나, 과제 특성상 pytorch
## 혹은 tensor 모델을 사용해야하는 걸로 인식 하여, 최적화와 추론속도 향상에 중점을 두었기에 easyocr모델을 사용하여
## 최적화 및 추론 속도 향상을 보여드리려 합니다.
## 해당 과제에서는 다른 text detection model을 사용하지 않고 easyocr만을 활용하여 정확도 및 추론속도 향상하여 작업하도록 하겠습니다.
## !! 과제 내용에 있던 네트워크 레이어는 건드리지 않겠습니다.(레이어를 건드리면 학습을 시켜 확인하고 정확도 및 추론속도 확인 및 해야 할것이 많기 때문(시간부족, 1주일로 부족))
## 잘 사용은 못하지만 LLM을 활용해보도록 하겠습니다.
## 코드는 객체 지향적으로 짤예정입니다.
## Readme에 자세하게 설명드리겠습니다.


## 단일스레드에서 순차처리를 통한 개발
## 폴더에있는 이미지를 불러옴 -> 해당이미지의 글자 탐지와,  인식을 함 ->결과값 도출
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # thread 에러로 추가하였습니다. 없어도 무방하시면 지우시면 됩니다.
import easyocr # 저희가 최적화할 easyocr
import cv2 # 유명한 무료 비전 라이브러리
import matplotlib.pyplot as plt # text detection 결과 학인을 위한 라이브러리입니다.
import time # 처리 속도 측정을위함

class processor:
    def __init__(self):
        self.easymodel = easyocr.Reader(['ko','en']) # easyocr 한국어,영어 지원하는 객체 생성(model call)
        self.image_path = "C:/Users/user/Desktop/project/ocrdata/AIEngineer.png"
        self.img = None
        
    def ocr_process(self): 
        self.img = cv2.imread(self.image_path)
        #output = self.easymodel.readtext(cv_Data)
        # text detection and recognition
        start_time = time.process_time()
        result = self.easymodel.readtext(self.img)
        end_time = time.process_time()
        print("inference speed : ",(end_time - start_time)*1000,"ms")
        # 이미지상 바운딩 박스 옆 글자 출력, 결과값 확인하기 위한 코드입니다.
        for (bbox, text, prob) in result:
            # 바운딩 박스 좌표
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
    
            # 이미지에 바운딩 박스 그리기
            cv2.rectangle(self.img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(self.img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(text, " confidence : ", round(prob,3))
    # 결과 이미지 출력
    def displayimg(self):
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    ocr_run = processor()
    ocr_run.ocr_process()
    ocr_run.displayimg()
    
