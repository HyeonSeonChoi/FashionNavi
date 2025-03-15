#옷 특징 추출 모델을 위한 dataset 생성하는 프로젝트

import argparse
import cloth_segmentation.process
from imutils import paths

args = { "dataroot":".\DB\\",		#dataset 경로
		"savepath":"C:\\labs\\capstone\\DB\\"}		#model 저장 경로

imagePaths = sorted(list(paths.list_images(args['dataroot'])))

for i in imagePaths:
    element = list(i.split('\\'))
    seg_cat = element[2]
    clot_cat = element[3]
    name = element[4]

    #segment 모델 불러오기
    parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
    parser.add_argument('--image', default=i, type=str, help='Path to the input image')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='cloth_segmentation/model/cloth_segm.pth', help='Path to the checkpoint file')
    args = parser.parse_args()

    #segment image
    seg_image = cloth_segmentation.process.process_seg(args)    
    
    try:
        if seg_cat == "upper":
            image = seg_image[1]
        elif seg_cat == "lower":
            image = seg_image[2]        

        save_path = args['savepath'] + "{}\\{}_remove\\".format(seg_cat, clot_cat) + name #연구실 공유 폴더에 바로 저장
        image.save(save_path)
        print("save success!!! path: '{}'".format(save_path))
    except:
        print("cant read generate_mask at '{}'".format(i)) 
