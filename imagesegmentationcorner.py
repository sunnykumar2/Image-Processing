import cv2
import numpy as np
from matplotlib import cm, pyplot as plt


square=np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,10,10,10,10,0,0],[0,0,10,10,10,10,0,0],[0,0,10,10,10,10,0,0],[0,0,10,10,10,10,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],np.uint8)
# square=cv2.imread('chess1.jpg',cv2.IMREAD_GRAYSCALE)

kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
dst = cv2.filter2D(square,-1,kernel)


print(dst)
thresh1=dst.copy()

for i in range(thresh1.shape[1]):
  for j in range(thresh1.shape[0]):
    if thresh1[i,j] !=50:
       thresh1[i,j]=0
    

print(thresh1)
plt.subplot(131),plt.imshow(square),plt.title('square')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(dst),plt.title('mask')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(thresh1),plt.title('AFTER threshold')
plt.xticks([]), plt.yticks([])
plt.show()