import matplotlib.pyplot as plt
from PIL import Image

def showRecImage(rec_image_path, input_image_path):
    fig = plt.figure(figsize=(15, 15))
    fig.tight_layout()
    rows = 6
    cols = 5

    input_image = Image.open(input_image_path)
    plt.imshow(input_image)

    for i, path in enumerate(rec_image_path):
        image = Image.open(path)
        
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(image)

    plt.show()
