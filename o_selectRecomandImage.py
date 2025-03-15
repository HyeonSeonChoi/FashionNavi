#입력은 정확도가 계산된 image_path, 출력은 추천 이미지 여러개

def selectImage(upper_path, lower_path, upper_color_path, lower_color_path):
    rec_color = list()
    rec_seg = list()
    rec_image_path = list()
    
    #색 비교
    for i in lower_color_path:
        for j in upper_color_path:
            if i == j:
                rec_color.append(i)
                break

    #옷 형태 비교
    for k in lower_path:
        for t in upper_path:
            if k == t:
                rec_seg.append(k)
                break    

    #추천할 이미지 path 선택
    for x in rec_color:
        for y in rec_seg:
            if x == y:
                rec_image_path.append(x)
                break
      
    return rec_image_path
