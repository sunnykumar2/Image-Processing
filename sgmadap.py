import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def global_thresholding(img):    
    x,y= img.shape[:2]
    sum = 0
    count = 0
    for i in range(x):
        for j in range(y):
            sum = sum + img[i,j]
            count = count + 1

    Mean = int(sum/count)        
    return Mean

def local_thresholding(img, C): 
    img_new = img.copy()   
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
    img = img_new

img = cv2.imread("effil.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
orignal_img = img
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.resize(gray,(850,1179))

x,y=img.shape 

th1= global_thresholding(img[0:int(x/2), 0:int(y/2)]) # Global Thresholdin on R1 //depends upon the region as well.
th2 = global_thresholding(img[0:int(x/2), int(y/2):y]) # Global Thresholdin on R2
th3= global_thresholding(img[int(x/2):x, 0:int(y/2)]) # Global Thresholdin on R3
local_thresholding(img[int(x/2):x, int(y/2):y],10) # Local Thresholdin on R4


ret1, img[0:int(x/2), 0:int(y/2)] = cv2.threshold(img[0:int(x/2), 0:int(y/2)], th1, 255, cv2.THRESH_BINARY)
ret1, img[0:int(x/2), int(y/2):y] = cv2.threshold(img[0:int(x/2), int(y/2):y], th2, 255, cv2.THRESH_BINARY)
ret1, img[int(x/2):x, 0:int(y/2)] = cv2.threshold(img[int(x/2):x, 0:int(y/2)], th3, 255, cv2.THRESH_BINARY)


plot1 = plt.figure("Original")
plt.imshow(orignal_img)
plot2 = plt.figure("Segmented image : ")
plt.imshow(img,cmap=cm.gray)
plt.show()

