
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import cv2


# Fetch them images and find edges
im1 = cv2.imread('316.png')
im1_final = cv2.imread('316.png')
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
# returns: a cropped image of a circle and coordinates of upper left corner
def get_cropped_circle(x,y,r,rp,im):

    x1 = int(x-r*rp)
    x2 = int(x+r*rp)
    y1 = int(y-r*rp)
    y2 = int(y+r*rp)
    
    x1 = 0 if x1 < 0 else x1
    x2 = im.shape[:2][1] if x2 > im.shape[:2][1] else x2
    y1 = 0 if y1 < 0 else y1
    y2 = im.shape[:2][0] if y2 > im.shape[:2][0] else y2

#    print('x: ', x)
#    print('y: ', y)
#    
    return im[y1:y2, x1:x2], x1, y1




# x and y: coordinates of upper left corner of contour image
# c: contour image
# rgb: color of contour [R,G,B]
# im: image
def draw_contour(x,y,c,rgb,im):
    
    for r in range(c.shape[0]):
        for s in range(c.shape[1]):

            if c[r][s] > 0:
                im[r+y][s+x][0] = rgb[0]
                im[r+y][s+x][1] = rgb[1]
                im[r+y][s+x][2] = rgb[2]


    
# F-ed up particles: 3,17,18,23,24,27,31,44,45!
#   51,53,55,56,63,70,71,76,80,82,88,89,90,93,
#   94,98,100,101,102,104,106,108,109,110,114,
#   116,117     +1?


itr = 0
#itr = 110

#for c in circles[0][itr:]:
for c in circles[0]:
    
    c_crop, x1, y1 = get_cropped_circle(c[0], c[1], c[2], 1.1, binim)


    # Do the wacky polar transformation!
    #--- ensure image is of the type float ---
    img = c_crop.astype(np.float32)
    
    #--- the following holds the square root of the sum of squares of the image dimensions ---
    #--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
    value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
    
    polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)

    
    # Rotate image!
    (h, w) = polar_image.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    
    scale = 1.0
    
    M = cv2.getRotationMatrix2D(center, -90, scale)
    polar_image90 = cv2.warpAffine(polar_image, M, (h, w))
    
    
    
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
        print("##### ATTENTION ######")
        print("circle number", itr, "needs to be looked at")
        print("along with probably a lot of other crap")


    # Do the poly fit!
    z = np.polyfit(range(crop_len), pxl_loc, 4) # 3rd or 4th order?
    f = np.poly1d(z)
    
    
    
    # Add polyfit to top_pixels
    f_pts = np.zeros(len(pxl_loc))
    f_im = np.zeros(polar_image90.shape, np.uint8)
    for i,loc in enumerate(pxl_loc):
        top_pixels[int(f(i))][i] = 120
        f_pts[i] = int(f(i))
        f_im[int(f(i))][i] = 255
    
    
    
    # Rotate image!
    (h, w) = f_im.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 90, scale)
    f_im_90 = cv2.warpAffine(f_im, M, (h, w))
    
    
    
    # Dilate that shiz
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #kernel = np.concatenate((np.zeros((2,7),np.uint8), np.ones((3,7),np.uint8), (np.zeros((2,7),np.uint8))))
    kernel = np.concatenate((np.zeros((1,5),np.uint8), np.ones((3,5),np.uint8), (np.zeros((1,5),np.uint8))))
    #kernel = np.concatenate((np.zeros((1,3),np.uint8), np.ones((1,3),np.uint8), (np.zeros((1,3),np.uint8))))
    f_im_90 = cv2.dilate(f_im_90,kernel,iterations = 1)
    
    
    
    
    # Polar to cartisian
    #img = f_im_90.astype(np.float32)
    img = c_crop.astype(np.float32)
    Mvalue = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
    cartisian_image = cv2.linearPolar(f_im_90, (img.shape[0]/2, img.shape[1]/2),Mvalue, cv2.WARP_INVERSE_MAP)
    


    r_rgb = [rnd.randint(0, 256),rnd.randint(0, 256),rnd.randint(0, 256)]

    draw_contour(x1,y1,cartisian_image,[0,0,255],im1_final)
    
#    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
#    ax.imshow(cartisian_image)
#    plt.show()
#
#    
#    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
#    ax = axes.ravel()
#    
#    ax[0].imshow(c_crop, cmap=plt.cm.gray)
#    ax[1].imshow(polar_image, cmap=plt.cm.gray)
#    ax[2].imshow(polar_image90, cmap=plt.cm.gray)
#    #ax[3].imshow(top_pixels, cmap=plt.cm.gray)
#    ax[3].imshow(top_pixels)
#    ax[4].imshow(f_im_90, cmap=plt.cm.gray)
#    ax[5].imshow(cartisian_image, cmap=plt.cm.gray)
#    
#    fig.tight_layout()
#    plt.show()

    
#    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
#    ax = axes.ravel()
#    
#    ax[0].imshow(c_crop, cmap=plt.cm.gray)
#    ax[1].imshow(c_crop + cartisian_image*0.6)
#    
#    fig.tight_layout()
#    plt.show()
#    
#    print("circle no.",itr)
#    
    itr += 1



fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 16))
ax.imshow(im1_final)
plt.show()
