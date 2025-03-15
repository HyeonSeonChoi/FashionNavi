#fashionnet 모델에 우리가 생성한 dataset을 학습시키는 프로젝트
from keras.optimizers import Adam
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from datetime import datetime

# import the necessary packages
from keras.models import Model
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf

#============================================================================================================================================

class FashionNet:
	@staticmethod
	def build_category_branch(inputs, numCategories,
		finalAct="softmax", chanDim=-1):
		# utilize a lambda layer to convert the 3 channel input to a
		# grayscale representation
		x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

		# CONV => RELU => POOL
		x = Conv2D(32, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(3, 3))(x)
		x = Dropout(0.25)(x)

		# (CONV => RELU) * 2 => POOL
		x = Conv2D(64, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Conv2D(64, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

		# (CONV => RELU) * 2 => POOL
		x = Conv2D(128, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Conv2D(128, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

		# define a branch of output layers for the number of different
		# clothing categories (i.e., shirts, jeans, dresses, etc.)
		x = Flatten()(x)
		x = Dense(256)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(numCategories)(x)
		x = Activation(finalAct, name="category_output")(x)

		# return the category prediction sub-network
		return x

	@staticmethod
	def build(width, height, numCategories, finalAct="softmax"):
		# initialize the input shape and channel dimension (this code
		# assumes you are using TensorFlow which utilizes channels
		# last ordering)
		inputShape = (height, width, 3)
		chanDim = -1

		# construct both the "category" and "color" sub-networks
		inputs = Input(shape=inputShape)
		categoryBranch = FashionNet.build_category_branch(inputs, numCategories, finalAct=finalAct, chanDim=chanDim)

		# create the model using our input (the batch of images) and
		# two separate outputs -- one for the clothing category
		# branch and another for the color branch, respectively
		model = Model(inputs=inputs, outputs=categoryBranch, name="fashionnet")

		# return the constructed network architecture
		return model

#============================================================================================================================================
#전체적인 data 전처리 과정
args = { "dataset":"./dataset/musinsa_dataset_2",
		"model":"./model/fashion.model2"}

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

log_file = "./metrics_log.txt"

start_time = datetime.now()

# grab the image paths and randomly shuffle them
with open(log_file, "a") as file:
	print("[INFO] loading images...")
	imagePaths = sorted(list(paths.list_images(args["dataset"])))
	random.seed(42)
	random.shuffle(imagePaths)

	# initialize the data, clothing category labels (i.e., shirts, jeans,
	# dresses, etc.) along with the color labels (i.e., red, blue, etc.)
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

	# partition the data into training and testing splits using 80% of
	# the data for training and the remaining 20% for testing
	split = train_test_split(data, categoryLabels, test_size=0.2, random_state=42)
	(trainX, testX, trainCategoryY, testCategoryY) = split

	#============================================================================================================================================

	# initialize our FashionNet multi-output network
	model = FashionNet.build(96, 96, numCategories=len(categoryLB.classes_), finalAct="softmax")

	# define two dictionaries: one that specifies the loss method for
	# each output of the network along with a second dictionary that
	# specifies the weight per loss
	losses = {
		"category_output": "categorical_crossentropy"
	}
	lossWeights = {"category_output": 1.0}

	# initialize the optimizer and compile the model
	print("[INFO] compiling model...")
	opt = Adam(learning_rate=INIT_LR)
	model.compile(optimizer=opt, loss=losses, metrics=["accuracy", "mse", "mae"])

	# train the network to perform multi-output classification
	H = model.fit(trainX, trainCategoryY, validation_data=(testX, testCategoryY), epochs=EPOCHS, verbose=1, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: file.write(f"Epoch {epoch+1}/{EPOCHS} : "f"loss - {logs['loss']:.4f}, "f"accuracy - {logs['accuracy']:.4f}, "f"val_loss - {logs['val_loss']:.4f}, "f"val_accuracy - {logs['val_accuracy']:.4f}\n"))])

#============================================================================================================================================

# 실행 종료 시간 기록
end_time = datetime.now()

# 실행 시간 계산
execution_time = end_time - start_time

# 실행 날짜와 시간, 실행 시간을 로그 파일 끝에 추가
with open(log_file, "a") as file:
	now = datetime.now()
	current_time = now.strftime("%Y-%m-%d %H:%M:%S")
	start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
	end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
	execution_time_str = str(execution_time)
	file.write("\n시작 시간 : " + start_time_str + ", 종료 시간 : " + current_time + ", 동작 시간 : " + execution_time_str + "\n\n")

file.close()

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
