#vgg16 모델에 우리가 생성한 dataset을 학습시키는 프로젝트
from keras.optimizers import Adam
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
from keras.models import Model
from keras.layers import Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

import numpy as np
import random
import cv2
import os

args = { "dataset":"./dataset/musinsa_dataset_4",		#dataset 경로
		"model":"./model/fashion.model_VGG16"}		#model 저장 경로

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)


# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

#이미지 읽어서 정답레이블 저장
data = []
categoryLabels = []

# loop over the input images
for imagePath in imagePaths:
	try:
		# load the image, pre-process it, and store it in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = img_to_array(image)
		data.append(image)

		# extract the clothing color and category from the path and
		# update the respective lists
		cate = imagePath.split(os.path.sep)[-2]
		categoryLabels.append(cate)
	except:
		print("error : {}".format(imagePath))

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# convert the label lists to NumPy arrays prior to binarization
categoryLabels = np.array(categoryLabels)

# binarize both sets of labels
print("[INFO] binarizing labels...")
categoryLB = LabelBinarizer()
categoryLabels = categoryLB.fit_transform(categoryLabels)

print(categoryLB.classes_)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, categoryLabels, test_size=0.2, random_state=42)
(trainX, testX, trainCategoryY, testCategoryY) = split

#============================================================================================================================================
#VGG16 loading
model = VGG16(input_shape=IMAGE_DIMS, include_top=False, weights='imagenet')
output = model.output

x = GlobalAveragePooling2D()(output)
x = Dense(50, activation='relu')(x)
output = Dense(17, activation='softmax', name='category_output')(x)

model = Model(inputs=model.input, outputs=output)

losses = {
	"category_output": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(trainX, trainCategoryY, validation_data= (testX, testCategoryY), epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])
