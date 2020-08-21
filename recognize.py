# USAGE
# python recognize.py --training ../images/training --testing ../images/testing
# python recognize.py --training ../images/test2/training --testing ../images/test2/testing

# import the necessary packages
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
import sys

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", default='../images/test2/training',
	help="path to the training images")
ap.add_argument("-e", "--testing", default='../images/test2/testing',
	help="path to the testing images")

args = vars(ap.parse_args('')) # get default arguments

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.lbp(gray)

	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split(os.path.sep)[-2])
	data.append(hist)

# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.lbp(gray)
	prediction = model.predict(hist.reshape(1, -1))

	# display the image and the prediction
	cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 0, 100), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)