# FashionNavi
빅데이터 기반 패션 추천 도우미, FashionNavi

## 🖥️ 프로젝트 소개
패션 매칭의 어려움을 해결해 주기 위하여 빅데이터 기반 패션 매칭 시스템인 패션 도우미, FashionNavi이다. 이는 이들의 어려움을 해결하기 위해 컴퓨터 비전 및 딥 러닝 기술을 활용하여 의류 이미지에서 의류 특성 항목을 자동으로 감지하고, 스타일, 모양, 색상과 같은 패션 특성을 추출한다. 또한 사용자의 개인적인 선호도를 반영하기 위해서 효용이론으로 많이 활용되고 있는 MAUT (Multi
Attribute Utility Theory) 기법을 사용한다. FashionNavi 는 ‘무신사’ 쇼핑몰을 크롤링하고 이를 정제한 Dataset을 이용하여 패션 스타일의 핵심 요소인 상(하)의에 초점을 맞춘 패션추천 시스템이다.
<br>

### 🧑‍🤝‍🧑 맴버구성
#### 부경대학교 Capstone Design '최조장'팀
 - 팀장  : 최현선
 - 팀원1 : 조현우
 - 팀원2 : 장지완
 - 지도교수 : 부경대학교 정목동 교수님

### ⚙️ 개발 환경
- Python 3.8.6
- tensorflow 2.13.0
- opencv-python 4.8.0.74
- keras 2.13.1
  
### ⚙️ 개발 환경 설치
```sh
pip install -r requirements.txt
```

## 📌 프로젝트 구조
```shell
┌── cloth_segmentation
│   ├── command.py
│   ├── app.py
│   ├── network.py
│   ├── options.py
│   ├── process.py
│   └── requirements.txt
│
├── dataset
│
├── feature
│   ├── fashonnet
│   └── VGG16
│
├── model
│
├── Input
│
├── Output
│   └── segmentation
│
├── p_fashonnetModel.py
│
├── p_vgg16Model.py
│
├── p_seg_musinsa_dataset.py
│
├── p_DBfeature.py
│   ├── o_Image_cut.py
│   └── o_FeatureExtraction.py
│
├── p_ShoesNavigator.py
│   ├── o_Image_cut.py
│   ├── o_compareFeature.py
│   ├── o_selectRecommandImage.py
│   ├── o_showImage.py
│   └── o_FeatureExtraction.py
│
├── error_log.txt
│
├── DB_TEST.py
│
├── requirements.txt
│
└── README.md
```

## 📌 p_ShoesNavigator.py (Main 파일) 소개
- 구성 파일 : Input 이미지, 학습된 특징 추출 모델, Output으로 출력할 코디 DataBase
- 순서
  - Input 폴더 속 이미지 불러오기
  - 상·하의 분리 모델(U2Net)을 통해 Input 이미지의 상·하의 분리 (cloth_segmentation 속 process.py)
  - 분리된 상·하의를 upper_image, lower_image 변수에 저장
  - upper_image와 lower_image의 불필요한 배경 제거 (o_Image_cut.py)
  - model 변수에 학습된 특징 추출 모델 불러오기
  - 상·하의를 특징 추출 모델에 입력으로 넣고 특징과 색상 추출
  - DB에 존재하는 코디들과 입력의 코디를 특징과 색상 비교
  - 비교한 이미지 중 유사도가 높은 이미지를 차례로 정렬
  - 정렬한 이미지 중 30개 출력

