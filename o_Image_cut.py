import numpy as np
from PIL import Image

# input = Image.open("Segmentation_Final\cloth_segmentation\output\segmentation\musinsa2_2.png")

def cut(image):
    image = np.array(image)
    all_zeros_mask = np.all(image == 0, axis = 2)
    filtered_rows = image[~np.all(all_zeros_mask, axis=1)]
    filtered_image_data = filtered_rows[:, ~np.all(all_zeros_mask, axis=0)]
    # print(filtered_image_data.shape)
    filtered_image = Image.fromarray(filtered_image_data)
    
    return filtered_image