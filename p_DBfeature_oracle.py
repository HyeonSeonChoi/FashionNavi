#실제 비교할 이미지에 대한 상하의 특징과 색을 추출하여 json 파일로 저장해주는 프로젝트
from keras.models import load_model
from collections import OrderedDict

import argparse
import o_FeatureExtraction       #특징 추출
import cloth_segmentation.process
import o_Image_cut
import glob
import json

import oracledb
import io

con = oracledb.connect(user="system", password="admin", dsn="203.250.123.49/xe")
cursor = con.cursor()

param = {"dataroot":"./DB/everyDB/", "model":"./model/fashion.model2/"}		#feature 저장경로

#DB_image_path 가져오기(데이터가 많아지면 style 별로 load)
styles = ["americancasual", "casual", "chic", "dandy", "girllish", "romantic", "street"]
genders = ["men", "women"]
image_paths = []

for gender in genders:
    for style in styles: 
        image_subpath = list(glob.glob(param["dataroot"] + "{}/{}/".format(style, gender) + "*-*.jpg"))
        image_paths += image_subpath
    

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
        
        #배열을 text로 변환(list to str)
        ufl_text = ', '.join(map(str, upper_feature_list))
        lfl_text= ', '.join(map(str, lower_feature_list))
        ucl_text = ', '.join(map(str, upper_color_list))
        lcl_text = ', '.join(map(str, lower_color_list))

        #DB에 저장(여기서는 스타일 별로 json 파일로 관리)
        element = list(image_path.split('/'))
        style = element[3] 
        
        print(style)
        print("image_path : " + image_path)
        
        cursor.execute("insert into SN_DB (style, image_path, upper_feature, lower_feature, upper_color, lower_color, upper_cate, lower_cate) values (:1, :2, :3, :4, :5, :6, :7, :8)", [style, image_path, ufl_text, lfl_text, ucl_text, lcl_text, upper_cate, lower_cate])

        con.commit()

    except:
        print("error")
        
con.close()
