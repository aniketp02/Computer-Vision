import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils
import sys


trainImg = cv.imread(sys.argv[1])
trainImg_gray = cv.cvtColor(trainImg, cv.COLOR_RGB2BGR)

querryImg = cv.imread(sys.argv[2])
querryImg_gray = cv.cvtColor(querryImg, cv.COLOR_RGB2BGR)

kpsA , featuresA = cv.ORB_create().detectAndCompute(trainImg_gray, None)
kpsB , featuresB = cv.ORB_create().detectAndCompute(querryImg_gray, None)

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

best_matches = bf.match(featuresA, featuresB)
rawmatches = sorted(best_matches, key = lambda x: x.distance)

# converting key points to numpu arrays
kpsA = np.float32([kp.pt for kp in kpsA])
kpsB = np.float32([kp.pt for kp in kpsB])

if len(rawmatches) > 4:
    ptsA = np.float32([kpsA[m.queryIdx] for m in rawmatches])
    ptsB = np.float32([kpsB[m.trainIdx] for m in rawmatches])

    # estimate the homography b/w sets of points
    (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, 4)

#print(H)

# Apply panorama correction
width = trainImg.shape[1] + querryImg.shape[1]
height = trainImg.shape[0] + querryImg.shape[0]

result = cv.warpPerspective(trainImg, H, (width, height))
result[0:querryImg.shape[0], 0:querryImg.shape[1]] = querryImg

# plt.figure(figsize=(20,10))
# plt.imshow(result)

# plt.axis('off')
# plt.savefig('abs.jpg')



# transform the panorama image to grayscale and threshold it 
gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

# Finds contours from the binary image
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# get the maximum contour area
c = max(cnts, key=cv.contourArea)

# get a bbox from the contour area
(x, y, w, h) = cv.boundingRect(c)

# crop the image to the bbox coordinates
result = result[y:y + h, x:x + w]

# show the cropped image
plt.figure(figsize=(20,10))
plt.imshow(result)
plt.savefig('final.jpg')
