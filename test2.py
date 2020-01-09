
import numpy as np

import matplotlib.pyplot as plt

import cv2

import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


f = open("path.txt", "r")
path = f.read()

im1 = cv2.imread('316.png',-1)
im2 = cv2.imread('316_grad.png',-1) # Create our own gradient image?
im3 = cv2.imread('316_grad.png',0)


# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(im3,(1,1),0) # (1,1) = blur becomes insignificant
th,im4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

im1a = im1[50:250, 300:500]
im4a = im4[50:250, 300:500]

im1b = im1[150:350, 800:1000]
im4b = im4[150:350, 800:1000]

im1c = im1[350:550, 550:750]
im4c = im4[350:550, 550:750]

im1d = im1[0:200, 525:725]
im4d = im4[0:200, 525:725]


circles = cv2.HoughCircles(im4,cv2.HOUGH_GRADIENT,1.05,35,
                            param1=50,param2=23,minRadius=5,maxRadius=90)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(im1,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(im1,(i[0],i[1]),2,(0,0,255),3)


# TODO: More elegant plot code

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im1, cmap=plt.cm.gray)
plt.show()

#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(im2, cmap=plt.cm.gray)
#plt.show()
#
#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(im3, cmap=plt.cm.gray)
#plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im4, cmap=plt.cm.gray)
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1a)
ax[1].imshow(im4a, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1b)
ax[1].imshow(im4b, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1c)
ax[1].imshow(im4c, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1d)
ax[1].imshow(im4d, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()
