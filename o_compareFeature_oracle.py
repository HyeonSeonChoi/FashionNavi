import cv2
import json
import numpy as np
import glob
import oracledb

from collections import OrderedDict

#입력은 입력 이미지의 특징(oracle DB와 비교), 출력은 DB상의 경로
#카테고리 정보 비교는 추후에 어떤식으로 비교할지 회의 후 결정

def compareFeature(input_upper_feature, input_lower_feature, input_upper_color, input_lower_color):
  #oracle DB open
  #oracle DB에 있는 특징을 load {image_path, upper_feature, lower_feature, upper_color, lower_color, upper_cate, lower_cate}
  con = oracledb.connect(user="system", password="admin", dsn="203.250.123.49/xe")
  cursor = con.cursor()

  #추천할 이미지의 유사도, 경로
  rec_upper_feature_dis = list()
  rec_upper_feature_path = list()
  rec_lower_feature_dis = list()
  rec_lower_feature_path = list()
  rec_upper_color_dis = list()
  rec_upper_color_path = list()
  rec_lower_color_dis = list()
  rec_lower_color_path = list()

  #tables = ["americancasual", "casual", "chic", "dandy", "girllish", "romantic", "street"]
  tables = ["SN_DB"]

  #table(style) 별로 DB 조회
  for table in tables:
    sql = "select * from {}".format(table)  #모든 record 조회
    records = cursor.execute(sql)

    #record 하나와 입력 특징 비교
    for record in records:
      style = record[0].read()              #str로 저장
      load_image_path = record[1].read()    #str로 저장
      load_upper_feature = record[2].read() #str로 저장
      load_lower_feature = record[3].read() #str로 저장
      load_upper_color = record[4].read()   #str로 저장
      load_lower_color = record[5].read()   #str로 저장
      load_upper_cate = record[6].read()    #str로 저장
      load_lower_cate = record[7].read()    #str로 저장

      #txt to list(각각의 feature, color), image_path와 category는 str로 사용
      load_upper_feature = [float(item) for item in load_upper_feature.strip("[]").split(', ')]
      load_lower_feature = [float(item) for item in load_lower_feature.strip("[]").split(', ')]
      load_upper_color = [float(item.strip()) for item in load_upper_color.split(',')]
      load_lower_color = [float(item.strip()) for item in load_lower_color.split(',')]

      #list to numpy
      load_upper_feature = np.array(load_upper_feature)
      load_lower_feature = np.array(load_lower_feature)
      load_upper_color = np.array(load_upper_color).astype(np.float32)
      load_lower_color = np.array(load_lower_color).astype(np.float32)

      #input 이미지와 비교
      #특징 비교
      #특징 정규화
      load_upper_feature = load_upper_feature / np.linalg.norm(load_upper_feature)
      load_lower_feature = load_lower_feature / np.linalg.norm(load_lower_feature)
      input_upper_feature = input_upper_feature / np.linalg.norm(input_upper_feature)
      input_lower_feature = input_lower_feature / np.linalg.norm(input_lower_feature)

      #특징 유사도 계산(유클리드 거리)
      dis_upper_feature = np.linalg.norm(load_upper_feature - input_upper_feature, axis = 1)
      dis_lower_feature = np.linalg.norm(load_lower_feature - input_lower_feature, axis = 1)

      #색상 비교
      dis_upper_color = np.linalg.norm(load_upper_color - input_upper_color)
      dis_lower_color = np.linalg.norm(load_lower_color - input_lower_color)

      #특정 수치 이하인 거리를 저장
      if dis_upper_feature < 12:
        rec_upper_feature_dis.append(dis_upper_feature[0])
        rec_upper_feature_path.append(load_image_path)

      if dis_lower_feature < 12:
        rec_lower_feature_dis.append(dis_lower_feature[0])
        rec_lower_feature_path.append(load_image_path)

      if dis_upper_color < 200:
        rec_upper_color_dis.append(dis_upper_color)
        rec_upper_color_path.append(load_image_path)

      if dis_lower_color < 200:
        rec_lower_color_dis.append(dis_lower_color)
        rec_lower_color_path.append(load_image_path)


  #유사도가 높은 순서로 각각의 image_path 정렬
  sorted_upper_feature = np.argsort(rec_upper_feature_dis)[:100]
  sorted_lower_feature = np.argsort(rec_lower_feature_dis)[:100]
  sorted_upper_color = np.argsort(rec_upper_color_dis)[:100]
  sorted_lower_color = np.argsort(rec_lower_color_dis)[:100]

  #유사도가 높은 image path
  end_upper_feature_path = [rec_upper_feature_path[k] for k in sorted_upper_feature]
  end_lower_feature_path = [rec_lower_feature_path[k] for k in sorted_lower_feature]
  end_upper_color_path = [rec_upper_color_path[k] for k in sorted_upper_color]
  end_lower_color_path = [rec_lower_color_path[k] for k in sorted_lower_color]
  
  #유사도가 높은 distance
  end_upper_feature_dis = [rec_upper_feature_dis[k] for k in sorted_upper_feature]
  end_lower_feature_dis = [rec_lower_feature_dis[k] for k in sorted_lower_feature]
  end_upper_color_dis = [rec_upper_color_dis[k] for k in sorted_upper_color]
  end_lower_color_dis = [rec_lower_color_dis[k] for k in sorted_lower_color]

  return end_upper_feature_path, end_lower_feature_path,  end_upper_color_path, end_lower_color_path, end_upper_feature_dis, end_lower_feature_dis, end_upper_color_dis, end_lower_color_dis
