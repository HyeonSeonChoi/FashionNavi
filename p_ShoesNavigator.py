#실제 사용자에게 서비스할 프로젝트
import os
from keras.models import load_model
import cv2

import matplotlib.pyplot as plt

import argparse
import o_FeatureExtraction       #특징 추출
import o_compareFeature_json          #특징 비교
#import o_compareFeature_oracle          #특징 비교
import o_selectRecomandImage     #추천 이미지 선택
import cloth_segmentation.process
import o_showImage                #이미지 출력
import o_Image_cut
import o_showImage_dis

setting = {
    "input_image_path" : "./input/KakaoTalk_20230824_154114324.jpg",
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

upper_image = seg_image[1]  #상의 부분 이미지
lower_image = seg_image[2]  #하의 부분 이미지
upper_mask = every_alpha_mask[1]
lower_mask = every_alpha_mask[2]

#불필요한 부분 줄이기
upper_image = o_Image_cut.cut(upper_image)
lower_image = o_Image_cut.cut(lower_image)

#옷 특징 추출(입력: 상하의 이미지, 출력: 상하의 특징)
model = load_model(setting['model_path']) #사전에 학습된 모델 load(옷 특징 추출 모델)

upper_feature, upper_cate = o_FeatureExtraction.FeatureExtraction(upper_image, model)
lower_feature, lower_cate = o_FeatureExtraction.FeatureExtraction(lower_image, model)

#옷 색상 추출
upper_color = o_FeatureExtraction.calcolor(input_image_path, upper_mask)
lower_color = o_FeatureExtraction.calcolor(input_image_path, lower_mask)

#옷 특징 비교
#upper_path, lower_path, upper_color_path, lower_color_path, upper_feature_dis, lower_feature_dis, upper_color_dis, lower_color_dis = o_compareFeature_oracle.compareFeature(upper_feature, lower_feature, upper_color, lower_color, setting["feature_path"])     #특징, 색이 비슷한 100개의 DB_path
upper_path, lower_path, upper_color_path, lower_color_path, upper_feature_dis, lower_feature_dis, upper_color_dis, lower_color_dis = o_compareFeature_json.compareFeature(upper_feature, lower_feature, upper_color, lower_color, setting["feature_path"])     #특징, 색이 비슷한 100개의 DB_path

#추천 이미지 선택
rec_image_path =  o_selectRecomandImage.selectImage(upper_path, lower_path, upper_color_path, lower_color_path)

#이미지 출력
#o_showImage.showRecImage(rec_image_path[:30], input_image_path)   #최대 30개 image 보여줌

#거리를 포함한 이미지 출력
o_showImage_dis.showRecImage(upper_path, lower_path, upper_color_path, lower_color_path, upper_feature_dis, lower_feature_dis, upper_color_dis, lower_color_dis)
