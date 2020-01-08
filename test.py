
import matplotlib.pyplot as plt

from skimage import io


f = open("path.txt", "r")
path = f.read()

im1 = io.imread(path+'316.png')
im2 = io.imread(path+'316_grad.png')

im1a = im1[50:250, 300:500]
im2a = im2[50:250, 300:500]

im1b = im1[150:350, 800:1000]
im2b = im2[150:350, 800:1000]

im1c = im1[350:550, 550:750]
im2c = im2[350:550, 550:750]

im1d = im1[0:200, 525:725]
im2d = im2[0:200, 525:725]

# TODO: More elegant plot code

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im1, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im2, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
ax.imshow(im1a, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
ax.imshow(im2a, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
ax.imshow(im1b, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
ax.imshow(im2b, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
ax.imshow(im1c, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
ax.imshow(im2c, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
ax.imshow(im1d, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
ax.imshow(im2d, cmap=plt.cm.gray)
plt.show()