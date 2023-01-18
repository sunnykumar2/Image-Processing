
import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
import cv2
# import os
# getting the mask from the rgb images


def preprocessing(img):
    # resizing using aspect ratio intact and finding the circle
    # reduce size retain aspect ratio intact
    # invert BGR 2 RGB
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Ig = RGB[:, :, 2]
    [w, h] = np.shape(Ig)
    r = 1200.0/Ig.shape[1]
    dim = (1200, int(Ig.shape[0]*r))
    rz = cv2.resize(Ig, dim, interpolation=cv2.INTER_AREA)
    #  convert in to float and get log trasform for contrast streching
    g = 0.2 * (np.log(1 + np.float32(rz)))
    # change into uint8
    cvuint = cv2.convertScaleAbs(g)
    # cvuint8.dtype
    ret, th = cv2.threshold(cvuint, 0, 255, cv2.THRESH_OTSU)
    ret1, th1 = cv2.threshold(Ig, 0, 255, cv2.THRESH_OTSU)
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    cls = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    Im = cls*rz  # the mask with resize image
    # cv2.imwrite('mynew.jpg', mask)
    return (Im, th, th1, cls, g, RGB)


path_dir = glob.glob("die.jpeg")
for k in path_dir:
    print('running code....', k)
    img = cv2.imread(k)
    (Im, th, th1, cls, g, RGB) = preprocessing(img)
    from matplotlib import pyplot as plt
   
    titles = ['Original Image', 'log_transform',
              'mask using logT', 'mask without log_T ']
    images = [RGB, g, cls, th]
    for i in range(0, np.size(images)):
        print(i)
        plt.subplot(2, 3, i + 1)
        plt.imshow((images[i]), 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
