import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

from skimage.io import imread
# importing matplotlib modules 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import pdb
import cv2
import math

# Load image
#image_rgb = data.coffee()[0:220, 160:420]
#image_gray = color.rgb2gray(image_rgb)

img_color = imread('test2.png') 
img_color = cv2.cvtColor(img_color,cv2.COLOR_RGBA2RGB)
img = color.rgb2gray(img_color)


#plt.imshow(img)
#plt.show()

# Canny
edge_minth = 0.03
edge_maxth = 0.3
edge_sigma = 1.0



edges = canny(img, edge_sigma,edge_maxth,edge_minth)

#plt.imshow(edges)
#plt.show()

# Ellipse detection
minMajorAxis = 35
maxMinorAxis = 110

result = hough_ellipse(edges,accuracy = 10, threshold = 100,
                        min_size=minMajorAxis, max_size = maxMinorAxis)

result.sort(order='accumulator')
result = result[::-1]


#best = list(result[-1])
#yc, xc, a, b = [int(round(x)) for x in best[1:5]]
#orientation = best[5]
#cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)

#pdb.set_trace()
#img_color[cy, cx] = (0, 0, 255)
#plt.show()

# Draw ellipses
#for r in result:
#    print(r)
outputList = []
for c_tmp in result:
    shouldAdd = True
    c = list(c_tmp)

    if c[3] == 0 or c[4] == 0:
        shouldAdd = False
        continue

    for tst in outputList:
        dist = math.sqrt((c[1]-tst[1])**2+(c[2]-tst[2])**2)

        if dist < 3:
            shouldAdd = False
            break

    if shouldAdd:
        outputList.append(c)



print(len(outputList))

for i in outputList:
    e = i
    #e = list(i)
    # pdb.set_trace()
    # Estimated parameters for the ellipse
    yc, xc, a, b = [int(round(x)) for x in e[1:5]]
    orientation = e[5]

    j = max(a,b); k = min(a,b)

    if (j > 0.01 and k/j > 0.5):
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)

        if max(cx) >= len(img_color[0]) or max(cy) >= len(img_color):
            print("x,y outside")
            print(yc,xc)
        else:
            img_color[cy, cx] = (0, 0, 255)    
    else:
        print("Skipping: ", a, b)
    
    
plt.imshow(img_color)
plt.show()
