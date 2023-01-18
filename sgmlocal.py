import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def local_thresholding(img,img_new, C):    
    x,y= img.shape[:2]
    count = 0
    for i in range(1,x-1):
        for j in range(1,y-1):
            neighbour_sum = int(img[i-1,j-1])+int(img[i-1,j])+int(img[i-1,j+1])+int(img[i,j-1])+int(img[i,j+1])+int(img[i+1,j-1])+int(img[i+1,j])+int(img[i+1,j+1])
            neighbour_mean = int(neighbour_sum / 8)
            t = neighbour_mean - C
            if t<0 :
                t = 0
            if img[i,j]<= t :
                img_new[i,j] = 0
            else :
                img_new[i,j] = 255
            

img = cv2.imread("effil.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
x,y= img.shape[:2]
img = cv2.resize(gray,(y,x))
orignal_img = img.copy()
local_thresholding(orignal_img,img,5)

plot1 = plt.figure("Original")
plt.imshow(orignal_img,cmap=cm.gray)
plot2 = plt.figure("Segmented image : ")
plt.imshow(img,cmap=cm.gray)
plt.show()

