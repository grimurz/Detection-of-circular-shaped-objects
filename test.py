
import matplotlib.pyplot as plt

from skimage import io



f = open("path.txt", "r")
path = f.read()

image = io.imread(path+'316.png')

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.imshow(image, cmap=plt.cm.gray)
plt.show()

