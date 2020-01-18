
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Fetch them images and find edges
im1 = cv2.imread('316.png')
im2 = im1.copy()
im1gray = cv2.imread('316.png',0) # 0 = grayscale
edges = cv2.Canny(im1gray,235,255,L2gradient=True)


# Find connected components and get rid of all smaller than 30
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, None, None, None, 8, cv2.CV_32S)
areas = stats[1:,cv2.CC_STAT_AREA]
result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 30:   #keep
        result[labels == i + 1] = 255

binim = result # binary image



shifted = cv2.pyrMeanShiftFiltering(im1, 21, 51)
#cv2.imshow("Input", im1)

# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.copyMakeBorder(thresh,1,1,1,1,cv2.BORDER_CONSTANT,0)
#cv2.imshow("Thresh", thresh)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))


circles = np.empty((0,3), int)


# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue

	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(thresh.shape, dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(im1, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(im1, "#{}".format(label), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	circles = np.append(circles, np.array([[int(x),int(y),int(r)]]), axis=0)



#binim = thresh
#binim[binim > 0] = 255
#binim = cv2.bitwise_not(binim)



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
    
    #avoid edges
    x1 = 0 if x1 < 0 else x1
    x2 = im.shape[:2][1] if x2 > im.shape[:2][1] else x2
    y1 = 0 if y1 < 0 else y1
    y2 = im.shape[:2][0] if y2 > im.shape[:2][0] else y2

#    print('x: ', x)
#    print('y: ', y)

    return im[y1:y2, x1:x2]



# c: contour image
# rgb: color of contour [R,G,B]
# im: image
def draw_contour(c,rgb,im):
    
    for r in range(c.shape[0]):
        for s in range(c.shape[1]):

            if c[r][s] > 0:
                im[r][s][0] = rgb[0]
                im[r][s][1] = rgb[1]
                im[r][s][2] = rgb[2]



##### experimental stuff below ##### 7,9,13,20


i = 20
crop_im = get_cropped_circle(circles[i][0],circles[i][1],circles[i][2],1.1,binim)
crop_im2 = get_cropped_circle(circles[i][0],circles[i][1],circles[i][2],1.1,im2)



# Do the wacky polar transformation!
#--- ensure image is of the type float ---
img = crop_im.astype(np.float32)

#--- the following holds the square root of the sum of squares of the image dimensions ---
#--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
value = np.sqrt(((img.shape[1]/2.0)**2.0)+((img.shape[0]/2.0)**2.0))

polar_image = cv2.linearPolar(img,(img.shape[1]/2, img.shape[0]/2), value, cv2.WARP_FILL_OUTLIERS)
polar_image = polar_image.astype(np.uint8)



# Rotate image!
polar_image90 = np.rot90(polar_image, k=3, axes=(0, 1))



# Get top pixel from edge!
top_pixels = np.zeros(polar_image90.shape)
crop_len = polar_image90.shape[1]
pxl_loc = np.zeros(crop_len) # gaps/zeros potential problem?

for i in range(crop_len):
    for j,pxl in enumerate(polar_image90[:,i]):
        if pxl == 255:
            top_pixels[j][i] = 255
            pxl_loc[i] = j
            break




# try because of messed up edge cases
try:
    
    # Get rid of zeros, use median of x non-zero neighbour values
    for i in range(crop_len):
    
        if pxl_loc[i] == 0:
            
            stack = []
            l_stack = []
            r_stack = []
            x = 4
            
            left_itr = i-1
            while left_itr >= 0 and len(l_stack) < x:
                if pxl_loc[left_itr] != 0:
                    l_stack.append(pxl_loc[left_itr])
                left_itr -= 1
            
            right_itr = i+1
            while right_itr <= crop_len-1 and len(r_stack) < x:
                if pxl_loc[right_itr] != 0:
                    r_stack.append(pxl_loc[right_itr])
                right_itr += 1
    
            while len(stack) < x and (l_stack or r_stack):
                if len(l_stack) > 0:
                    stack.append(l_stack.pop(0)) 
                if len(r_stack) > 0:
                    stack.append(r_stack.pop(0))
    
            pxl_loc[i] = int(np.median(stack))
            top_pixels[int(pxl_loc[i])][i] = 255

except:
    print("MIXUP!")
    pass




# align ends a little better together
n = np.ones(crop_len)
n[0] = 100
n[-1] = 100
nu_pt = int((pxl_loc[0]+pxl_loc[-1])/2)
pxl_loc[0] = nu_pt
pxl_loc[-1] = nu_pt



# Do the poly fit!
z = np.polyfit(range(crop_len), pxl_loc, 4, w=np.sqrt(n)) # 3rd or 4th order?
f = np.poly1d(z)



# Add polyfit to top_pixels
f_pts = np.zeros(len(pxl_loc))
f_im = np.zeros(polar_image90.shape, np.uint8)
tp2 = top_pixels.copy()
for i,loc in enumerate(pxl_loc):
    top_pixels[int(f(i))][i] = 120
    f_pts[i] = int(f(i))
    f_im[int(f(i))][i] = 255



# Rotate image!
f_im_90 = np.rot90(f_im, k=1, axes=(0, 1))



# Dilate that shiz
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#kernel = np.concatenate((np.zeros((2,7),np.uint8), np.ones((3,7),np.uint8), (np.zeros((2,7),np.uint8))))
kernel = np.concatenate((np.zeros((1,5),np.uint8), np.ones((3,5),np.uint8), (np.zeros((1,5),np.uint8))))
f_im_90 = cv2.dilate(f_im_90,kernel,iterations = 1)



# Polar to cartisian
#img = f_im_90.astype(np.float32)
img = crop_im.astype(np.float32)
Mvalue = np.sqrt(((img.shape[1]/2.0)**2.0)+((img.shape[0]/2.0)**2.0))
cartisian_image = cv2.linearPolar(f_im_90, (img.shape[1]/2, img.shape[0]/2),Mvalue, cv2.WARP_INVERSE_MAP)



# Draw the fit
r_rgb = [rnd.randint(50, 150),rnd.randint(0, 100),rnd.randint(250, 250)]
draw_contour(cartisian_image,r_rgb,crop_im2)





# TODO: More elegant plot code

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(im1)
plt.show()

#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
#ax.imshow(im1gray, cmap=plt.cm.gray)
#plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
ax.imshow(binim, cmap=plt.cm.gray)
plt.show()



fig, axes = plt.subplots(2, 2, figsize=(16, 14))
ax = axes.ravel()

ax[0].imshow(crop_im, cmap=plt.cm.gray)
ax[1].imshow(polar_image, cmap=plt.cm.gray)
ax[2].imshow(polar_image90, cmap=plt.cm.gray)
ax[3].imshow(tp2, cmap=plt.cm.gray)

fig.tight_layout()
plt.show()



fig, axes = plt.subplots(2, 2, figsize=(16, 14))
ax = axes.ravel()

ax[0].imshow(top_pixels)
ax[1].imshow(f_im_90, cmap=plt.cm.gray)
ax[2].imshow(cartisian_image, cmap=plt.cm.gray)
ax[3].imshow(crop_im2)

fig.tight_layout()
plt.show()
