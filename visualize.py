import cv2
import numpy as np

img = cv2.imread('data_zhoucheng/3.png',-1)
# img[img<50]=50
# img[img>150]=150
img = cv2.applyColorMap(img, cv2.COLORMAP_PARULA)
cv2.imwrite('data_zhoucheng/color_3.png',img)