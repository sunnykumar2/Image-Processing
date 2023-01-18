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
        
    

img = cv2.imread("effil.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
orignal_img = img
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.resize(gray,(2000,1967))

final_threshold_value = global_thresholding(img)
print("Global Threshold value : " + str(final_threshold_value))
ret1, th1 = cv2.threshold(img, final_threshold_value, 255, cv2.THRESH_BINARY)
plot1 = plt.figure("Original")
plt.imshow(orignal_img)
plot2 = plt.figure("Segmented image : ")
plt.imshow(th1,cmap=cm.gray)
plt.show()


