import cv2
import json
import numpy as np
import glob
from collections import OrderedDict

#입력은 입력 이미지의 특징(oracle DB와 비교), 출력은 DB상의 경로
#카테고리 정보 비교는 추후에 어떤식으로 비교할지 회의 후 결정

def compareFeature(input_upper_feature, input_lower_feature, input_upper_color, input_lower_color, feature_path):
  #oracle DB에 있는 특징을 load {image_path, upper_feature, lower_feature, upper_color, lower_color, upper_cate, lower_cate}
  styles = ["americancasual", "casual", "chic", "dandy", "girllish", "romantic", "street"]
  #data_root = "./feature/" + ""
  every_json = []
  
  #load json
  for style in styles:
    file_paths = list(glob.glob(feature_path + "*"))

  for style, file_path in zip(styles, file_paths):
    val = style + "_json"
    
    with open(file_path, 'r') as file:
      globals()[val] = json.load(file)

      every_json.append(globals()[val].keys())  #dictionary 형태

  #json 파일 형식
  #file_data = {("image_path" : "/content/drive/MyDrive/Colab Notebooks/00000005.jpg"
  #                "upper_feature" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "lower_feature" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "upper_color" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "lower_color" : [0, 0, 0, 0, 0, 0, 0, 0,] ),
  #                ("image_path" : "/content/drive/MyDrive/Colab Notebooks/00000005.jpg"
  #                "upper_feature" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "lower_feature" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "upper_color" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "lower_color" : [0, 0, 0, 0, 0, 0, 0, 0,] ),
  #                ("image_path" : "/content/drive/MyDrive/Colab Notebooks/00000005.jpg"
  #                "upper_feature" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "lower_feature" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "upper_color" : [0, 0, 0, 0, 0, 0, 0, 0,],
  #                "lower_color" : [0, 0, 0, 0, 0, 0, 0, 0,] ),}

  #색깔 비교
  similaritys_upper_hist = [] #색 특징 값
  similaritys_upper_path = [] #이미지 경로
  similaritys_lower_hist = []
  similaritys_lower_path = []
  query_hist_upper = [] #시각화를 위한 hist
  query_hist_lower = []

  #옷 특징 비교
  distances_upper = []    #옷 특징 값
  distances_lower = []
  distances_upper_path = [] #이미지 경로
  distances_lower_path = []

  i = 0

  for keys in every_json:
    for key in keys:
      style = styles[i]
      val = style + "_json"
      style_json = globals()[val]   #(styl_json)으로 선언된 json 변수

      #load feature and color
      load_upper_feature_list = style_json[key][0]
      load_lower_feature_list = style_json[key][1]
      load_upper_color_list = style_json[key][2]
      load_lower_color_list = style_json[key][3]

      #list to array
      load_upper_feature = np.array(load_upper_feature_list)
      load_lower_feature = np.array(load_lower_feature_list)
      load_upper_color = np.array(load_upper_color_list).astype(np.float32)
      load_lower_color = np.array(load_lower_color_list).astype(np.float32)

      #색깔 비교, 색 특징 비교값    
      similarity_upper = np.linalg.norm(input_upper_color - load_upper_color)
      similarity_lower = np.linalg.norm(input_lower_color - load_lower_color)
      #similarity_upper = cv2.compareHist(input_upper_color, load_upper_color, cv2.HISTCMP_BHATTACHARYYA)  #float
      #similarity_lower = cv2.compareHist(input_lower_color, load_lower_color, cv2.HISTCMP_BHATTACHARYYA)
      
      #+- 10 안에 있는거 path에 추가
      #similarity_upper = abs(input_upper_color - load_upper_color)
      #similarity_lower = abs(input_lower_color - load_lower_color)
        

      #옷 특징 비교(특징 정규화 > 차이를 정규화), 
      upf = input_upper_feature / np.linalg.norm(input_upper_feature)
      lof = input_lower_feature/ np.linalg.norm(input_lower_feature)
      load_upf = load_upper_feature / np.linalg.norm(load_upper_feature)
      load_lof = load_lower_feature / np.linalg.norm(load_lower_feature)
      
      #하나의 값이 나옴(거리), 옷 특징 비교값
      distance_upper = np.linalg.norm(upf - load_upf, axis = 1) #numpy
      distance_lower = np.linalg.norm(lof - load_lof, axis = 1) 

      #아래 부분은 DB 구조가 정해지면 더욱 간단하게 정리가능 할 것 같음, 메모리 사용 최소화를 위해 알고리즘 수정해야함
      similaritys_upper_hist.append(similarity_upper)
      similaritys_lower_hist.append(similarity_lower)
      similaritys_upper_path.append(key)
      similaritys_lower_path.append(key)
      
      query_hist_upper.append(input_upper_color)  #시각화를 위한 hist
      query_hist_lower.append(input_lower_color)

      distances_upper.append(distance_upper[0])
      distances_lower.append(distance_lower[0])
      distances_upper_path.append(key)
      distances_lower_path.append(key)
      
    i += 1

  sorted_similarity_upper = np.argsort(similaritys_upper_hist)[:100]  #upper 색깔이 비슷한 이미지 100개 index
  sorted_similarity_lower = np.argsort(similaritys_lower_hist)[:100]  #lower 색깔이 비슷한 이미지 100개 index

  #sorted_similarity를 similaritys_path의 index로 사용해서 image_path 접근

  sorted_distance_upper = np.argsort(distances_upper)[:100]  #추천하려는 상의이미지 100개(거리가 짧은 순서)
  sorted_distance_lower = np.argsort(distances_lower)[:100]  #추천하려는 하의이미지 100개(거리가 짧은 순서)

  upper_path = []
  lower_path = []
  upper_color_path = []
  lower_color_path = []
  
  #유사도가 높은 image_path
  upper_path = [similaritys_upper_path[k] for k in sorted_distance_upper]
  lower_path = [similaritys_lower_path[k] for k in sorted_distance_lower]
  upper_color_path = [distances_upper_path[k] for k in sorted_similarity_upper]
  lower_color_path = [distances_lower_path[k] for k in sorted_similarity_lower]
  
  #유사도가 높은 distance
  end_upper_feature_dis = [distances_upper[k] for k in sorted_distance_upper]
  end_lower_feature_dis = [distances_lower[k] for k in sorted_distance_lower]
  end_upper_color_dis = [similaritys_upper_hist[k] for k in sorted_similarity_upper]
  end_lower_color_dis = [similaritys_lower_hist[k] for k in sorted_similarity_lower]
  
  return upper_path, lower_path, upper_color_path, lower_color_path, end_upper_feature_dis, end_lower_feature_dis, end_upper_color_dis, end_lower_color_dis

  #return upper_path, lower_path, upper_color_path, lower_color_path, sorted_query_hist_upper, sorted_query_hist_lower     #특징, 색이 비슷한 100개의 DB_path
