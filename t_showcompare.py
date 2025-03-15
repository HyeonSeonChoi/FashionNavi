#입력 이미지 특징, 색상 히스토그램 시각화       입력이미지 특징, 색상정보 필요


#출력 이미지 특징, 색상 히스토그램 시각화       출력이미지 특징, 색상정보 필요
#입력, 출력 비교    

#실제 사용자에게 서비스할 프로젝트
from keras.models import load_model

import matplotlib.pyplot as plt

import argparse
import o_FeatureExtraction       #특징 추출
import o_compareFeature          #특징 비교
import o_selectRecomandImage     #추천 이미지 선택
import cloth_segmentation.process
import o_showImage                #이미지 출력
import o_Image_cut
import cv2
import numpy as np

#이미지 RGB 계산, HSV 분포 계산
#H값 평균계산, 플러스 마이너스 
def fefeColor(input_image_path, mask):
    input_image = cv2.imread(input_image_path)
    #input_image = np.array(input_image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    
    upper_mask = np.array(mask[1])
    lower_mask = np.array(mask[2])
    
    upper_hist = cv2.calcHist([input_image], [0], upper_mask, [180], [0, 180])
    upper_hist = cv2.normalize(upper_hist, upper_hist).flatten()
    lower_hist = cv2.calcHist([input_image], [0], lower_mask, [180], [0, 180])
    lower_hist = cv2.normalize(lower_hist, lower_hist).flatten()
    
    print(np.argmax(lower_hist))    #가장 많이 나온 H값
    
    plt.plot(lower_hist)
    plt.show()
    
    return upper_hist, lower_hist

def color_delta(input_image_path1, mask1, input_image_path2, mask2):
    upper_mask1 = np.array(mask1[1])
    lower_mask1 = np.array(mask1[2])
    
    upper_mask2 = np.array(mask2[1])
    lower_mask2 = np.array(mask2[2])
    
    input_image1 = cv2.imread(input_image_path1)
    lab_image1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2Lab)
    
    input_image2 = cv2.imread(input_image_path2)
    lab_image2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2Lab)
    
    upper_masked_image1 = cv2.bitwise_and(lab_image1, lab_image1, mask = upper_mask1)
    lower_masked_image1 = cv2.bitwise_and(lab_image1, lab_image1, mask = lower_mask1)
    upper_masked_image2 = cv2.bitwise_and(lab_image2, lab_image2, mask = upper_mask2)
    lower_masked_image2 = cv2.bitwise_and(lab_image2, lab_image2, mask = lower_mask2)
    
    #이미지 크기 맞춰주기
    upper_masked_image1 = cv2.resize(upper_masked_image1, (upper_masked_image2.shape[1], upper_masked_image2.shape[0]))
    lower_masked_image1 = cv2.resize(lower_masked_image1, (lower_masked_image2.shape[1], lower_masked_image2.shape[0]))
    
    plt.imshow(lower_masked_image1)
    plt.show()
    plt.imshow(lower_masked_image2)
    plt.show()
    
    upper_delta_e = np.sqrt(np.sum((upper_masked_image1 - upper_masked_image2) ** 2))
    lower_delta_e = np.sqrt(np.sum((lower_masked_image1 - lower_masked_image2) ** 2))
    
    return upper_delta_e, lower_delta_e

def color_diff(input_image_path1, mask1, input_image_path2, mask2):
    upper_mask1 = np.array(mask1[1])
    lower_mask1 = np.array(mask1[2])
    
    upper_mask2 = np.array(mask2[1])
    lower_mask2 = np.array(mask2[2])
    
    input_image1 = cv2.imread(input_image_path1)
    lab_image1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
    
    input_image2 = cv2.imread(input_image_path2)
    lab_image2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2GRAY)
    
    upper_masked_image1 = cv2.bitwise_and(lab_image1, lab_image1, mask = upper_mask1)
    lower_masked_image1 = cv2.bitwise_and(lab_image1, lab_image1, mask = lower_mask1)
    upper_masked_image2 = cv2.bitwise_and(lab_image2, lab_image2, mask = upper_mask2)
    lower_masked_image2 = cv2.bitwise_and(lab_image2, lab_image2, mask = lower_mask2)
    
    #이미지 크기 맞춰주기
    upper_masked_image1 = cv2.resize(upper_masked_image1, (upper_masked_image2.shape[1], upper_masked_image2.shape[0])).flatten()
    lower_masked_image1 = cv2.resize(lower_masked_image1, (lower_masked_image2.shape[1], lower_masked_image2.shape[0])).flatten()
    
    print(upper_masked_image1)
    
    #0요소 제거
    upper_masked_image1 = list(filter(lambda x: x != 0, upper_masked_image1))
    lower_masked_image1 = list(filter(lambda x: x != 0, lower_masked_image1))
    
    print(upper_masked_image1)
    
    upper_abs_diff = cv2.absdiff(upper_masked_image1, upper_masked_image2)
    lower_abs_diff = cv2.absdiff(lower_masked_image1, lower_masked_image2)
    
    mean_uad = np.mean(upper_abs_diff)
    mean_lad = np.mean(lower_abs_diff)
    
    print(mean_uad, mean_lad)
    
    return mean_uad, mean_lad
    
    
def compareHist(hist1, hist2):
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return score

setting = {
    "input_image_path" : "./input/2-47.jpg",
    "model_path" : "./model/fashion.model5",
    "feature_path" : "./feature/fashionnet/"
}

#초기 변수 세팅
input_image_path = setting['input_image_path']

#상하의 분리(입력: 사용자 입력 이미지,  출력: 상의 이미지, 하의 이미지)
parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
parser.add_argument('--image', default=input_image_path, type=str, help='Path to the input image')
parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
parser.add_argument('--checkpoint_path', type=str, default='cloth_segmentation/model/cloth_segm.pth', help='Path to the checkpoint file')
args = parser.parse_args()

seg_image, every_alpha_mask = cloth_segmentation.process.process_seg(args)

#parser.add_argument('--image', default="./input/2-47.jpg", type=str, help='Path to the input image')
#args = parser.parse_args
args.image = "./input/KakaoTalk_20230824_154114324.jpg"

seg_image1, every_alpha_mask1 = cloth_segmentation.process.process_seg(args)

#type = PLI Image
upper_image = seg_image[1]  #상의 부분
lower_image = seg_image[2]  #하의 부분

#불필요한 부분 줄이기
upper_image = o_Image_cut.cut(upper_image)
lower_image = o_Image_cut.cut(lower_image)

#옷 특징 추출(입력: 상하의 이미지, 출력: 상하의 특징)
model = load_model(setting['model_path']) #사전에 학습된 모델 load(옷 특징 추출 모델)
""" 
upper_del, lower_del = color_delta(input_image_path, every_alpha_mask, "./input/KakaoTalk_20230824_154114324.jpg", every_alpha_mask1)
print(upper_del, lower_del)
 """

#==============================================================

upper_hist, lower_hist = fefeColor(input_image_path, every_alpha_mask)
upper_hist1, lower_hist1 = fefeColor("./input/KakaoTalk_20230824_154114324.jpg", every_alpha_mask1)

sc = compareHist(upper_hist, upper_hist1)
sc1 = compareHist(lower_hist, lower_hist1)

print(sc, sc1)

#==============================================================

""" 
plt.figure()
plt.imshow(upper_image)
plt.show()
plt.plot(upper_hist)
plt.xlim([1, 256])
plt.show()

plt.imshow(lower_image)
plt.show()
plt.plot(lower_hist)
plt.xlim([1, 256])
plt.show()

 """
