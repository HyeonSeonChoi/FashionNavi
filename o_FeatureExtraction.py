# set the matplotlib backend so figures can be saved in the background
# import the necessary packages
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
import cv2

#가장 많이 나온 색상을 확인하는 palette
def palette(clusters):
  width=300
  palette = np.zeros((50, width, 3), np.uint8)
  steps = width/clusters.cluster_centers_.shape[0]
  
  for idx, centers in enumerate(clusters.cluster_centers_):
      palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
      
  plt.imshow(palette)
  plt.show()

def FeatureExtraction(input_image, model):    #옷 형태 식별 모델을 사용하여 특징 추출
  input_image = np.array(input_image)
  
  #특징 추출을 위한 출력층 변환
  modelFE = Model(inputs= model.input, outputs = model.get_layer('batch_normalization_5').output)  #layer 확인 잘하기, fashionnet
  #modelFE = Model(inputs= model.input, outputs = model.get_layer('dense').output)  #layer 확인 잘하기, VGG16

  #이미지 불러오기, 추후에 DB image 특징은 따로 추출해서 저장해놓기
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

  #이미지 전처리
  input_image = cv2.resize(input_image, (96, 96))
  input_image = input_image.astype("float") / 255.0
  input_image = img_to_array(input_image)
  input_image = np.expand_dims(input_image, axis=0)

  #특징 추출, 카테고리 일치 확인
  feature = modelFE.predict(input_image) #입력 이미지 특징
  cate = model.predict(input_image) #이미지 카테고리

  cat = {0 : "long_blouse",
         1 : "long_hoodie",
         2 : "long_pants",
         3 : "long_sleeve",
         4 : "short_blouse",
         5 : "short_pants",
         6 : "short_tshirts"}

  index = np.argsort(cate).flatten().tolist()
  rec_cate = cat[index[-1]]

  return feature, rec_cate

def calcolor(input_image_path, mask):
  #이미지 읽기
  input_image = cv2.imread(input_image_path)
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
  mask = np.array(mask)
  
  #마스킹처리
  masked_img = cv2.bitwise_and(input_image, input_image, mask=mask)
  masked_img = masked_img.reshape(-1, 3)
  
  #[0,0,0] 부분 제거
  condition = ~np.all(masked_img == 0, axis=1)
  masked_img = masked_img[condition]
  
  #클러스터 생성
  cluster = KMeans(n_clusters=1)
  
  res = cluster.fit(masked_img)
  color = res.cluster_centers_[0] #가장 많이 나온 색상 추출
  
  return color


""" 
def calhistogram(input_image_path, mask):
  input_image = cv2.imread(input_image_path)
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
  
  upper_mask = np.array(mask[1])
  lower_mask = np.array(mask[2])
  
  upper_hist = cv2.calcHist([input_image], [0,1], upper_mask, [180, 256], [0, 180, 0, 256])
  lower_hist = cv2.calcHist([input_image], [0,1], lower_mask, [180, 256], [0, 180, 0, 256])
  upper_hist = cv2.normalize(upper_hist, upper_hist).flatten()
  lower_hist = cv2.normalize(lower_hist, lower_hist).flatten()
  
  #upper_hist = np.argmax(upper_hist)
  #lower_hist = np.argmax(lower_hist)
  
  return upper_hist, lower_hist
 """

""" 
#현재 하나의 체널만 계산하는거 같음.... -> HSV 방식으로 비교(안돼... )
def calhistogram(input_image_path, mask):
    input_image = cv2.imread(input_image_path)
    #input_image = np.array(input_image)
    #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    upper_hist = []
    lower_hist = []
    
    upper_mask = np.array(mask[1])
    lower_mask = np.array(mask[2])
    
    chs = cv2.split(input_image)
    colors = ['b', 'g', 'r']
    
    for ch, color in zip(chs, colors):
      hist = cv2.calcHist([ch], [0], upper_mask, [256], [0, 256])
      hist = cv2.normalize(hist, hist)
      upper_hist.append(hist)
      
    for ch, color in zip(chs, colors):
      hist = cv2.calcHist([ch], [0], lower_mask, [256], [0, 256])
      hist = cv2.normalize(hist, hist)
      lower_hist.append(hist)
    
    
    
    upper_hist = cv2.calcHist([input_image], [0], upper_mask, [256], [0, 256])
    upper_hist = cv2.normalize(upper_hist, upper_hist).flatten()
    lower_hist = cv2.calcHist([input_image], [0], lower_mask, [256], [0, 256])
    lower_hist = cv2.normalize(lower_hist, lower_hist).flatten()
    
    
    return upper_hist, lower_hist #BGR 값
   """
