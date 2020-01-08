
import matplotlib.pyplot as plt

#from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

from skimage import io


f = open("path.txt", "r")
path = f.read()

im1 = io.imread(path+'316.png')
im2 = io.imread(path+'316_grad.png')
im3 = rgb2gray(im2)

#im2t = threshold_otsu(rgb2gray(im2))
#im2bin = im2 > im2t

im1a = im1[50:250, 300:500]
im3a = im3[50:250, 300:500]

im1b = im1[150:350, 800:1000]
im3b = im3[150:350, 800:1000]

im1c = im1[350:550, 550:750]
im3c = im3[350:550, 550:750]

im1d = im1[0:200, 525:725]
im3d = im3[0:200, 525:725]

# TODO: More elegant plot code

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im1, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im2, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im3, cmap=plt.cm.gray)
plt.show()

#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(im2bin, cmap=plt.cm.gray)
#plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1a)
ax[1].imshow(im3a, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1b)
ax[1].imshow(im3b, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1c)
ax[1].imshow(im3c, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(im1d)
ax[1].imshow(im3d, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()
