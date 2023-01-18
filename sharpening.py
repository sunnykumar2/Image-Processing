import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np

# read input image
# image = cv2.imread("Downloads/krosharp.png")
image = cv2.imread("die.jpeg")

# check is input image exists
if image is None:
    print("can not find image")
    sys.exit()

# define sharpening kernel
sharpeningKernel = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")


output = cv2.filter2D(image, -1, sharpeningKernel)
plt.imshow(image)
plt.show()
plt.imshow(output)
plt.show()
