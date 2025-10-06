import numpy as np
import cv2


image = cv2.imread('UiA front1.png')
image_2 = cv2.imread('UiA front3.jpg')

# Create ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(image, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)

# Draw keypoints on the image
output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Create BFMatcher and match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors, descriptors_2)

# Sort matches based on distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 50 matches
matched_image = cv2.drawMatches(image, keypoints, image_2, keypoints_2, matches[:50], None, flags=2)

cv2.imwrite("matched_result.jpg", matched_image)
print("Result saved as matched_result.jpg")
