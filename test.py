
import numpy as np

import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

from skimage import io

import os
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


f = open("path.txt", "r")
path = f.read()

im1 = io.imread('316.png')
im2 = io.imread('316_grad.png')
im3 = rgb2gray(im2)

grays = rgb2gray(np.uint8(im3 * 255))

thresh = threshold_otsu(grays)
im4 = grays > thresh

im1a = im1[50:250, 300:500]
im4a = im4[50:250, 300:500]

im1b = im1[150:350, 800:1000]
im4b = im4[150:350, 800:1000]

im1c = im1[350:550, 550:750]
im4c = im4[350:550, 550:750]

im1d = im1[0:200, 525:725]
im4d = im4[0:200, 525:725]


hough_radii = np.arange(10, 100, 2)
hough_res = hough_circle(im4a, hough_radii)

# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=5)
# Draw them
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=im4a.shape)
    im1a[circy, circx] = (220, 20, 20, 255)



# TODO: More elegant plot code

#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(im1, cmap=plt.cm.gray)
#plt.show()
#
#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(im2, cmap=plt.cm.gray)
#plt.show()
#
#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(im3, cmap=plt.cm.gray)
#plt.show()
#
#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(im4, cmap=plt.cm.gray)
#plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1a)
ax[1].imshow(im4a, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()


#fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#ax = axes.ravel()
#
#ax[0].imshow(im1b)
#ax[1].imshow(im4b, cmap=plt.cm.gray)
#
#fig.tight_layout()
#plt.show()
#
#
#fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#ax = axes.ravel()
#
#ax[0].imshow(im1c)
#ax[1].imshow(im4c, cmap=plt.cm.gray)
#
#fig.tight_layout()
#plt.show()
#
#
#fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#ax = axes.ravel()
#
#ax[0].imshow(im1d)
#ax[1].imshow(im4d, cmap=plt.cm.gray)
#
#fig.tight_layout()
#plt.show()
