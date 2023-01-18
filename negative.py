import cv2
from matplotlib import pyplot as plt

img = cv2.imread("neg1.jpg", 0)
cv2.imshow('original', img)
im = img.copy()
x, y = img.shape[:2]
for i in range(x):
    for j in range(y):
        im[i, j] = 255-img[i, j]
cv2.imshow("Negative", im)

plt.figure("Histogram")
plt.hist(img.flatten(), 254, [1, 254], color='r')
plt.plot(cv2.calcHist([im], [0], None, [254], [1, 254]), color='b')
plt.legend(('Changed', 'Original'), loc='upper left')
plt.xlabel('Pixel values')
plt.ylabel('No. of pixels')
plt.show()
