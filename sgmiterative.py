import numpy as np
def thres_finder(img, thres=20,delta_T=1.0):
   # Step-2: Divide the images in two parts
  x_low, y_low = np.where(img<=thres)
  x_high, y_high = np.where(img>thres)
   # Step-3: Find the mean of two parts
  mean_low = np.mean(img[x_low,y_low])
  mean_high = np.mean(img[x_high,y_high])
   # Step-4: Calculate the new threshold
  new_thres = (mean_low + mean_high)/2
   # Step-5: Stopping criteria, otherwise iterate
  if abs(new_thres-thres)< delta_T:
      return new_thres
  else:
      return thres_finder(img, thres=new_thres,delta_T=1.0)
import cv2
# Load an image in the greyscale
img = cv2.imread('effil.jpg',cv2.IMREAD_GRAYSCALE)
# apply threshold finder
vv1 = thres_finder(img, thres=30,delta_T=1.0)
# threshold the image
ret, thresh = cv2.threshold(img,vv1,255,cv2.THRESH_BINARY)
# Display the image side by side
out = cv2.hconcat([img,thresh])
cv2.imshow('threshold',out)
cv2.waitKey(0)
 

