import matplotlib.pyplot as plt
from PIL import Image

#이미지, upper_feature 거리, lower_feature 거리, upper_color 거리, lower_color 거리, category 일치 여부
def showRecImage(upper_path, lower_path, upper_color_path, lower_color_path, upper_feature_dis, lower_feature_dis, upper_color_dis, lower_color_dis):
    rec_color = list()
    rec_seg = list()
    rec_image_path = list()
    rec_upper_color_dis = list()
    rec_lower_color_dis = list()
    rec_upper_dis = list()
    rec_lower_dis = list()
    
    #최종 return 값
    rec_image_path = list()
    end_upper_feature = list()
    end_lower_feature = list()
    end_upper_color = list()
    end_lower_color = list()
    
    #색 비교
    for lp, ld in zip(lower_color_path, lower_color_dis):
        for up, ud in zip(upper_color_path, upper_color_dis):
            if lp == up:
                rec_color.append(lp)
                rec_upper_color_dis.append(ud)
                rec_lower_color_dis.append(ld)
                break

    #옷 형태 비교
    for lp, ld in zip(lower_path, lower_feature_dis):
        for up, ud in zip(upper_path, upper_feature_dis):
            if lp == up:
                rec_seg.append(lp)
                rec_upper_dis.append(ud)
                rec_lower_dis.append(ld)
                break    

    #추천할 이미지 path 선택
    for rc, ucd, lcd in zip(rec_color, rec_upper_color_dis, rec_lower_color_dis):
        for rs, ud, ld in zip(rec_seg, rec_upper_dis, rec_lower_dis):
            if rc == rs:
                rec_image_path.append(rc)
                end_upper_feature.append(ud)
                end_lower_feature.append(ld)
                end_upper_color.append(ucd)
                end_lower_color.append(lcd)
                break
            
    rec_image_path = rec_image_path[:30]
    end_upper_feature = end_upper_feature[:30]
    end_lower_feature = end_lower_feature[:30]
    end_upper_color = end_upper_color[:30]
    end_lower_color = end_lower_color[:30]
    
    
    fig = plt.figure(figsize=(15, 15))
    fig.tight_layout()
    rows = 6
    cols = 5
    i = 0

    for path, uf, lf, uc, lc in zip(rec_image_path, end_upper_feature, end_lower_feature, end_upper_color, end_lower_color):
        image = Image.open(path)
        
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(image)
        ax.set_title("up_dis={}, lo_dis={}".format(int(uf), int(lf)))
        ax.set_xlabel("up_col={}, lo_col={}".format(int(uc), int(lc)))
        
        i += 1

    plt.show()
