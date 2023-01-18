import cv2
import numpy as np
from matplotlib import cm, pyplot as plt


square=cv2.imread('line.jpeg')

v = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
h = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
sf = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
sb = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
dst1 = cv2.filter2D(square,-1,v)
dst2 = cv2.filter2D(square,-1,h)
dst3 = cv2.filter2D(square,-1,sf)
dst4 = cv2.filter2D(square,-1,sb)


plt.subplot(151),plt.imshow(square),plt.title('lines')
plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow(dst1),plt.title('vertical')
plt.xticks([]), plt.yticks([])
plt.subplot(153),plt.imshow(dst2),plt.title('horizontal')
plt.xticks([]), plt.yticks([])
plt.subplot(154),plt.imshow(dst3),plt.title('slant forward')
plt.xticks([]), plt.yticks([])
plt.subplot(155),plt.imshow(dst4),plt.title('slant backward')
plt.xticks([]), plt.yticks([])
plt.show()