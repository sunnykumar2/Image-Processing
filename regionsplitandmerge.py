import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.cm as cm
import numpy as np
from operator import add
from functools import reduce
import cv2


def find_max_min(img,xi,xf,yi,yf):    
    max_value_r = 0
    min_value_r = 255
    max_value_g = 0
    min_value_g = 255
    max_value_b = 0
    min_value_b = 255
    for i in range(xi,xf+1):
        for j in range(yi,yf+1):
            max_value_r = max(max_value_r,img[i,j][0])
            min_value_r = min(min_value_r,img[i,j][0])
            max_value_g = max(max_value_g,img[i,j][1])
            min_value_g = min(min_value_g,img[i,j][1])
            max_value_b = max(max_value_b,img[i,j][2])
            min_value_b = min(min_value_b,img[i,j][2])
       
    return max_value_r,min_value_r,max_value_g,min_value_g,max_value_b,min_value_b

def split4(img):
    half_split = np.array_split(img, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)

def concatenate4(north_west, north_east, south_west, south_east):
    top = np.concatenate((north_west, north_east), axis=1)
    bottom = np.concatenate((south_west, south_east), axis=1)
    return np.concatenate((top, bottom), axis=0)

def calculate_mean(img):
    return np.mean(img, axis=(0, 1))

def checkEqual(myList):
    first=myList[0]
    return all((x==first).all() for x in myList)


class QuadTree:

    def insert(self, img, level=0):
        self.level = level
        self.mean = calculate_mean(img).astype(int)
        self.resolution = (img.shape[0], img.shape[1])
        self.final = True

        if not checkEqual(img):
            split_img = split4(img)

            self.final = False
            self.north_west = QuadTree().insert(split_img[0], level + 1)
            self.north_east = QuadTree().insert(split_img[1], level + 1)
            self.south_west = QuadTree().insert(split_img[2], level + 1)
            self.south_east = QuadTree().insert(split_img[3], level + 1)

        return self

    def get_image(self, level):
        if (self.final or self.level == level):
            return np.tile(self.mean, (self.resolution[0], self.resolution[1], 1))

        return concatenate4(
            self.north_west.get_image(level),
            self.north_east.get_image(level),
            self.south_west.get_image(level),
            self.south_east.get_image(level))

    

def split(img, T, xi, xf, yi, yf):

    if xf>=xi and yf>=yi:

        value_max_r,value_min_r,value_max_g,value_min_g,value_max_b,value_min_b = find_max_min(img,xi,xf,yi,yf)

        if abs(value_max_r-value_min_r)<= T and abs(value_max_g-value_min_g)<= T and abs(value_max_b-value_min_b)<= T :
            # Don't need to split further.
            # Painting this region 
            paint_value_r = int((int(value_max_r)+int(value_min_r))/2)
            paint_value_g = int((int(value_max_g)+int(value_min_g))/2)
            paint_value_b = int((int(value_max_b)+int(value_min_b))/2)
           
            if paint_value_r > 127:
               paint_value_r = 255
            else:
                paint_value_r = 0
            
            if paint_value_g > 127:
               paint_value_g = 255
            else:
                paint_value_g = 0
            
            if paint_value_b > 127:
               paint_value_b = 255
            else:
                paint_value_b = 0

            for i in range(xi,xf+1):
                for j in range(yi,yf+1):
                    img[i,j] = [paint_value_r,paint_value_g,paint_value_b]
            
        else:
            Mid_xi = int((xi+xf)/2)
            Mid_yi = int((yi+yf)/2)
            split(img,T,xi,Mid_xi,yi,Mid_yi)
            split(img,T,Mid_xi+1,xf,yi,Mid_yi)
            split(img,T,xi,Mid_xi,Mid_yi+1,yf)
            split(img,T,Mid_xi+1,xf,Mid_yi+1,yf)

def post_processing(img):
    x,y= img.shape[:2]
    for i in range(x):
        for j in range(y):
            for k in range(3):
                if img[i,j][k] > 127:
                    img[i,j][k] = 255
                else:
                    img[i,j][k] = 0
                    

img = cv2.imread("effil.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
split_img = img.copy()
x,y= split_img.shape[:2]
split(split_img,30,0,x-1,0,y-1)

plot1 = plt.figure("Original Image")
plt.imshow(img)
plot2 = plt.figure("After splitting : ")
plt.imshow(split_img)

quadtree = QuadTree().insert(img)
merge_img = quadtree.get_image(7)
post_processing(merge_img)
plot3 = plt.figure("After applying  Merging : ")
plt.imshow(merge_img)

plt.show()