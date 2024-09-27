# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:37:50 2024

@author: user
"""

## Optimization
## step 1. 모델 변환을 통한 detection 및 recognition inference 속도 향상 ..(recognition도 변경하려 하였으나 시간이 부족하여 detection만 변경)
## step 2. preprocessing을 통한 정확도 향상
## step 3. 병렬처리를 통한 속도 향상

## 여기서는 이미지가 한장인 관계로 취득은 바로하고 취득하여 텍스트 detection-> crop 영상 큐에 전달 -> output 뽑아내는 형식으로 개발하겠습니다..
## 전처리과정도 처리속도에 포함하여 진행하도록 하겠습니다.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import collections
import time
import cv2
import onnxruntime as ort
import easyocr
import torch
import threading
import numpy as np
from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
from easyocr.utils import reformat_input

class tobeprocessing:
    def __init__(self):
        self.image_path = "C:/Users/user/Desktop/project/ocrdata/AIEngineer.png" #단일 이미지 경로
        self.providers = ['CPUExecutionProvider']
        # easyocr text detection을 craft 모델을 사용하기 때문에 craft 모델을 변환하였습니다.
        self.session = ort.InferenceSession("C:/Users/user/Desktop/project/onnx/craft.onnx", providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.easymodel = easyocr.Reader(['ko','en'])
        #self.val_X = None
        self.thread_safe_detect_queue = collections.deque() # 읽어온 영상을 담기위한 큐
        self.thread_safe_recog_queue = collections.deque() # recognition을 위한 큐
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.recoginference = 0
        
        self.b_stop = False
    ## 사실 thread 로 빼서 계속 취득해야하는 함수
    def get_data(self):
        test_data = cv2.imread(self.image_path)
        #print(test_data)
        self.thread_safe_detect_queue.append(test_data)
        
# =============================================================================
    def preprocesssing(self,cvMat):
        # gray scale
        gray_image = cv2.cvtColor(cvMat, cv2.COLOR_BGR2GRAY)
        # contrast
        #constra_img = self.clahe.apply(gray_image)
        # noise filter
        denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # binary
        _, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contrast
        #constra_img = self.clahe.apply(binary_image)
        return binary_image
# 
# =============================================================================
    # 약간더 보완해야 하나 text recognition을 위한 thread 함수
    def processingRecog(self):
        while not self.b_stop:
            if self.thread_safe_recog_queue:
                try:
                    start_time = time.process_time()
                    result = self.easymodel.readtext(self.thread_safe_recog_queue.popleft())
                    end_time = time.process_time()
                    #print("inference speed : ",(end_time - start_time)*1000,"ms")
                    self.recoginference += (end_time - start_time)*1000
                    # 이미지상 바운딩 박스 옆 글자 출력, 결과값 확인하기 위한 코드입니다.
                    for (bbox, text, prob) in result:
                        print(text, " confidence : ", round(prob,3))
                    print(self.recoginference, "ms")
                except:
                    continue
            else:
                print("recog done")
                time.sleep(1)
    # 영상이 계속들어 올 경우에도 활용하기 위한 thread 함수
    def processingdetect(self):
        while not self.b_stop:
            if self.thread_safe_detect_queue:
                img = self.thread_safe_detect_queue.popleft()
                #preprocess_img = self.preprocesssing(img)
                start_time = time.process_time()
                img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, 512, interpolation=cv2.INTER_LINEAR, mag_ratio=1.)
                ratio_h = ratio_w = 1 / target_ratio
                val_X = normalizeMeanVariance(img_resized)
                val_X = torch.from_numpy(val_X).permute(2, 0, 1).unsqueeze(0)
                inp = {self.input_name: val_X.numpy()}
                y, _ = self.session.run(None, inp)
                end_time = time.process_time()
                print("inference speed : ",(end_time - start_time)*1000,"ms")
                boxes, polys, mapper = getDetBoxes(y[0, :, :, 0], y[0, :, :, 1], 0.2, 0.8, 0.05)
# =============================================================================
                boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
                preprocess_img = self.preprocesssing(img)
                for box in boxes:
                    crop_cvMat = preprocess_img[int(box[0][0])-1:int(box[2][0])-1,int(box[0][1])+1:int(box[2][1])+1]
                    self.thread_safe_recog_queue.append(crop_cvMat)
                    cv2.rectangle(img, (int(box[0][0]),int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 255, 0), 2)
        
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

                
            else:
                print("done")
                time.sleep(1)
#    def detection(self):
        
if __name__ == '__main__':
    ocr_run = tobeprocessing()
    ocr_run.get_data()
    threading.Thread(target = ocr_run.processingdetect, args = ()).start()
    threading.Thread(target = ocr_run.processingRecog, args = ()).start()
    #ocr_run.displayimg()
    
