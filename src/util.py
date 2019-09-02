import numpy as np
import matplotlib.image as mpimg
import cv2
import pandas as pd


def getData(path, size, value="mel"):
	DF = pd.read_pickle(path)
	assert(len(DF["image"]) == len(DF["id"]))

	X = []
	for i in range(len(DF["image"])):

		if DF["id"][i] == value:
			tmp = cv2.resize(DF["image"][i], (int(size), int(size)), interpolation=cv2.INTER_CUBIC)
			result = (tmp - 127.5) / 127.5
			X.append(result)

	return np.array(X, dtype=np.float32)


def saveImages(filename, images):    
    for i in range(len(images)):
        mpimg.imsave(filename + "-" + str(i) + ".png",  ( (images[i] * 127.5) + 127.5 ).astype(np.uint8) )