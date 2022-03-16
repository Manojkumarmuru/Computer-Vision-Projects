# This python program is used to finely segment blue colored objects belonging to a blue 
# cluster parametrized by specific mean and covariance.

# The image is first roughly clustered into a background and foreground class.

# The foreground class is then finely refined to include only the pixels belonging to the 
# desired cluster with a higher probability.

# The pixels belonging to the desired cluster with much lower probability is ignored and 
# is considered to be in background class.

# Import necessary modules
import numpy as np
import cv2 as cv
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# Read input image
raw_img = cv.imread("sample.jpg")

# Resize image and display
resized_raw_img = cv.resize(raw_img, (int(1200), int(800)), interpolation = cv.INTER_AREA)
cv.imshow("Raw Image", resized_raw_img)

img_shape = resized_raw_img.shape

# Compute Gaussian mixture model with 2 clusters
gm = GaussianMixture(n_components=2, random_state=0).fit(resized_raw_img.reshape(-1, 3))

predictions = np.uint8(gm.predict(resized_raw_img.reshape(-1, 3)).reshape(img_shape[0], img_shape[1]))

# Get pixels belonging to blue cluster
mask = np.uint8(predictions==1)

# Get the cluster parameters of blue cluster
blue_mean = gm.means_[1]
blue_cov = gm.covariances_[1]

new_mask = np.zeros_like(resized_raw_img)
new_mask[:,:,0] = mask
new_mask[:,:,1] = mask
new_mask[:,:,2] = mask

masked = resized_raw_img*new_mask
cv.imshow("Blue cluster [Roughly Segmented Foreground Class]", masked)

# Get probability map [i.e probability of pixel belonging to blue cluster]
prob_map = multivariate_normal.pdf(masked.reshape(-1, 3), mean=blue_mean, cov=blue_cov)

probability_map = prob_map.reshape(img_shape[0], img_shape[1])

# Threshold the proability map
prob_thresh = 0.5e-8
fg_mask = np.zeros_like(resized_raw_img)
fg_mask[:,:,0] = np.uint8(probability_map>prob_thresh)
fg_mask[:,:,1] = np.uint8(probability_map>prob_thresh)
fg_mask[:,:,2] = np.uint8(probability_map>prob_thresh)

# Display probability map
probability_map_norm = cv.normalize(probability_map, None, 255,0, cv.NORM_MINMAX, cv.CV_8UC1)
cv.imshow("Probability Map", probability_map_norm)

# Get the refined blue cluster
fg = resized_raw_img*fg_mask
cv.imshow("Blue cluster [Finely Segmented Foreground Class]", fg)
cv.waitKey(0)

# cv.imwrite("resized_raw_img.jpg", resized_raw_img)
# cv.imwrite("roughly_segmented_foreground.jpg", masked)
# cv.imwrite("probability_map.jpg", probability_map_norm)
# cv.imwrite("finely_segmented_foreground.jpg", fg)
