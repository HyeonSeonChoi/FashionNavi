#실제 비교할 이미지에 대한 상하의 특징과 색을 추출하여 json 파일로 저장해주는 프로젝트
from keras.models import load_model
from collections import OrderedDict
from imutils import paths

import argparse
import o_FeatureExtraction       #특징 추출
import cloth_segmentation.process
import o_Image_cut
import glob
import json

param = { "dataroot":".\DB\\",		#dataset 경로
        "errorlogpath":".\\error_log.txt",
		"savepath":".\\feature\\fashionnet\\",
        "model":".\\model\\fashion.model5"}		#feature 저장경로

#DB_image_path 가져오기(데이터가 많아지면 style 별로 load)
styles = ["americancasual", "casual", "chic", "dandy", "girllish", "romantic", "street"]
genders = ["men", "women"]
image_paths = sorted(list(paths.list_images(param["dataroot"])))

americancasual_json = OrderedDict()
casual_json = OrderedDict()
chic_json = OrderedDict()
dandy_json = OrderedDict()
girllish_json = OrderedDict()
romantic_json = OrderedDict()
street_json = OrderedDict()

error_log = ""   #특징추출이 안된 DB 확인용 log

""" 
for gender in genders:
    for style in styles: 
        image_subpath = list(glob.glob(param["dataroot"] + "{}/{}/".format(style, gender) + "*-*.jpg"))
        image_paths += image_subpath
 """
 
for i in image_paths:
    try:
        #상하의 분리(입력: 사용자 입력 이미지,  출력: 상의 이미지, 하의 이미지)
        #초기 변수 세팅
        image_path = i
        
        parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
        parser.add_argument('--image', default=i, type=str, help='Path to the input image')
        parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
        parser.add_argument('--checkpoint_path', type=str, default='cloth_segmentation/model/cloth_segm.pth', help='Path to the checkpoint file')
        args = parser.parse_args()

        seg_image, every_mask = cloth_segmentation.process.process_seg(args)

        upper_image = seg_image[1]  #상의
        lower_image = seg_image[2]  #하의   배경은 index:0
        upper_mask = every_mask[1]
        lower_mask = every_mask[2]
        
        #불필요한 부분 줄이기
        upper_image = o_Image_cut.cut(upper_image)
        lower_image = o_Image_cut.cut(lower_image)

        #옷 특징 추출(입력: 상하의 이미지, 출력: 상하의 특징)
        model = load_model(param["model"]) #사전에 학습된 모델 load

        upper_feature, upper_cate = o_FeatureExtraction.FeatureExtraction(upper_image, model)
        lower_feature, lower_cate = o_FeatureExtraction.FeatureExtraction(lower_image, model)
        
        #옷 색상 추출
        upper_color = o_FeatureExtraction.calcolor(image_path, upper_mask)
        lower_color = o_FeatureExtraction.calcolor(image_path, lower_mask)
        
        #json 저장을 위해 numpy to list 변환
        upper_feature_list = upper_feature.tolist()
        lower_feature_list = lower_feature.tolist()
        upper_color_list = upper_color.tolist()
        lower_color_list = lower_color.tolist()
        
        #DB에 저장(여기서는 스타일 별로 json 파일로 관리)
        element = list(image_path.split('\\'))
        style = element[2]
        
        print(style)
        print("image_path : " + image_path)
        
        if style == "americancasual" :
            americancasual_json[image_path] = (upper_feature_list, lower_feature_list, upper_color_list, lower_color_list)
        elif style == "casual" : 
            casual_json[image_path] = (upper_feature_list, lower_feature_list, upper_color_list, lower_color_list)
        elif style == "chic" : 
            chic_json[image_path] = (upper_feature_list, lower_feature_list, upper_color_list, lower_color_list)
        elif style == "dandy" : 
            dandy_json[image_path] = (upper_feature_list, lower_feature_list, upper_color_list, lower_color_list)
        elif style == "girllish" : 
            girllish_json[image_path] = (upper_feature_list, lower_feature_list, upper_color_list, lower_color_list)
        elif style == "romantic" : 
            romantic_json[image_path] = (upper_feature_list, lower_feature_list, upper_color_list, lower_color_list)
        elif style == "street" : 
            street_json[image_path] = (upper_feature_list, lower_feature_list, upper_color_list, lower_color_list)
    except:
        error_log += i + " error\n"
        
with open(param["errorlogpath"], 'w') as f:
            f.write(error_log)

#json으로 local에 저장
orderlist = [americancasual_json, casual_json, chic_json, dandy_json, girllish_json, romantic_json, street_json]

for ind, obj in enumerate(orderlist):
    file_data = obj
    style = styles[ind]
    
    with open(param["savepath"] + "{}.json".format(style), "w", encoding="utf-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent='\t')
