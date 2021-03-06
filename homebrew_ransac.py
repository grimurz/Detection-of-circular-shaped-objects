
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



def ransac_ellipse(iter, srcimg, x, y):

    x_size = np.size(x)
    best_count = x_size
    for i in range(iter):

        base = srcimg.copy()

        # get 5 random points
        r1 = int(rnd.random() * x_size)
        r2 = int(rnd.random() * x_size)  
        r3 = int(rnd.random() * x_size)
        r4 = int(rnd.random() * x_size)
        r5 = int(rnd.random() * x_size)  

        p1 = (x[r1],y[r1])
        p2 = (x[r2],y[r2])
        p3 = (x[r3],y[r3])
        p4 = (x[r4],y[r4])
        p5 = (x[r5],y[r5])

        p_set = np.array((p1,p2,p3,p4,p5))

        # fit ellipse
        ellipse = cv2.fitEllipse(p_set)

        # remove intersected ellipse
        cv2.ellipse(base,ellipse,(0),1)

        # count remain
        local_count = cv2.countNonZero(base)

        # if count is smaller than best, update
        if local_count < best_count:
            best_count = local_count
            best_ellipse = ellipse


    return best_ellipse




##### experimental stuff below #####



i = 3
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



# Get leftmost pixel from edge!
top_pixels = np.zeros(polar_image.shape)
crop_hgt = polar_image.shape[0]
for i in range(crop_hgt):
    for j,pxl in enumerate(polar_image[i,:]):
        if pxl == 255:
            top_pixels[i][j] = 255
            break



# Polar to cartesian
img = crop_im.astype(np.float32)
Mvalue = np.sqrt(((img.shape[1]/2.0)**2.0)+((img.shape[0]/2.0)**2.0))
cartesian_image = cv2.linearPolar(top_pixels, (img.shape[1]/2, img.shape[0]/2),Mvalue, cv2.WARP_INVERSE_MAP)



# Run the RANSAC
y, x = np.where(cartesian_image > 0)
ellipse = ransac_ellipse(10000,cartesian_image,x,y)

nu_crop = np.zeros(crop_im.shape)
el_contour = cv2.ellipse(nu_crop,ellipse,(255,255,255),2)



# Draw the fit
r_rgb = [rnd.randint(50, 150),rnd.randint(0, 100),rnd.randint(250, 250)]
draw_contour(el_contour,r_rgb,crop_im2)





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



fig, axes = plt.subplots(2, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(crop_im, cmap=plt.cm.gray)
ax[1].imshow(polar_image, cmap=plt.cm.gray)
ax[2].imshow(top_pixels, cmap=plt.cm.gray)
ax[3].imshow(cartesian_image, cmap=plt.cm.gray)
#ax[3].imshow(cartesian_image)

fig.tight_layout()
plt.show()


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
ax.imshow(crop_im2)
plt.show()

