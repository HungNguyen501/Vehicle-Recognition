# import the necessary packages
from skimage import feature
import cv2
import numpy as np

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        
    def lbp(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist

    def hu_moments(self, image, eps=1e-7):
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        #feature = feature.astype("float")
        #feature /= (feature.sum() + eps)

        return feature
    
