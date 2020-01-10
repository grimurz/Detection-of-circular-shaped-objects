
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Fetch them images and find edges
im1 = cv2.imread('316.png')
im1gray = cv2.imread('316.png',0) # 0 = grayscale
edges = cv2.Canny(im1gray,240,255)


# Find connected components and get rid of all smaller than 30
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, None, None, None, 8, cv2.CV_32S)
areas = stats[1:,cv2.CC_STAT_AREA]
result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 30:   #keep
        result[labels == i + 1] = 255

binim = result # binary image


# Get coordinates and radii of all circles using Hough
circles = cv2.HoughCircles(binim,cv2.HOUGH_GRADIENT,1.05,35,
                            param1=50,param2=23,minRadius=5,maxRadius=90)

# Get rid of 10% worst peaks, located at the end of array
circles = circles[:,:-int(len(circles[0])*0.1)]


# Draw them circles
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(im1,(i[0],i[1]),i[2],(0,255,0),2) # outer circle
    cv2.circle(im1,(i[0],i[1]),2,(0,0,255),3) # center of the circle



# x and y: coordinates of circle
# r: radius of circle
# rp: relative padding for cropped image
# im: image
# returns: a cropped image of a circle
def get_cropped_circle(x,y,r,rp,im):

    x1 = int(x-r*rp)
    x2 = int(x+r*rp)
    y1 = int(y-r*rp)
    y2 = int(y+r*rp)
    
    x1 = 0 if x1 < 0 else x1
    x2 = im.shape[:2][1] if x2 > im.shape[:2][1] else x2
    y1 = 0 if y1 < 0 else y1
    y2 = im.shape[:2][0] if y2 > im.shape[:2][0] else y2

    print('x: ', x)
    print('y: ', y)
    
    return im[y1:y2, x1:x2]


##### experimental stuff below #####


# edge: 
# overlapping top: 
# overlapping bottom:  
# overlapping unclear:  
# clean:  
# deformed: 35,40,79
# double trouble: 
# junk inside: 3,17
i = 35
crop_im = get_cropped_circle(circles[0][i][0],circles[0][i][1],circles[0][i][2],1.1,binim)


# Do the wacky polar transformation!
#--- ensure image is of the type float ---
img = crop_im.astype(np.float32)

#--- the following holds the square root of the sum of squares of the image dimensions ---
#--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))

polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
polar_image = polar_image.astype(np.uint8)




# TODO: More elegant plot code

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im1)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(edges, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(binim, cmap=plt.cm.gray)
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

ax[0].imshow(crop_im, cmap=plt.cm.gray)
ax[1].imshow(polar_image, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()





## Hand picked circles:

#im1a = im1[50:250, 300:500]
#im4a = binim[50:250, 300:500]
#
#im1b = im1[150:350, 800:1000]
#im4b = binim[150:350, 800:1000]
#
#im1c = im1[350:550, 550:750]
#im4c = binim[350:550, 550:750]
#
#im1d = im1[0:200, 525:725]
#im4d = binim[0:200, 525:725]
#
#
#fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#ax = axes.ravel()
#
#ax[0].imshow(im1a)
#ax[1].imshow(im4a, cmap=plt.cm.gray)
#
#fig.tight_layout()
#plt.show()
#
#
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
#