import cv2 as cv
import numpy as np

img = cv.imread('chess.png',0)
img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
img = cv.threshold(img,175,255,cv.THRESH_BINARY)[1]

img = np.int16(img)
filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
corners = cv.filter2D(img, -1, filter)
mval = np.max(corners)

corners=cv.threshold(corners,mval-1,255,cv.THRESH_BINARY)[1]
corners = np.uint8(corners)

contours,hierarchy = cv.findContours(corners, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    print(x,y,w,h)
    cv.rectangle(img_bgr, (x-10, y-10), (x + 10, y + 10), (0, 250, 0), 2)

cv.imshow('Hello',img_bgr)

cv.waitKey(0)
cv.destroyAllWindows()