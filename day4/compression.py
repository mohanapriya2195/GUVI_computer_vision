
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("river.jpg")
cv2.imwrite("compressed.jpg", img, [cv2.IMWRITE_AVIF_QUALITY, 50])
cv2.imwrite("com[ressed.png]", img, [cv2.IMWRITE_EXR_COMPRESSION,9 ])
cv2.imwrite("compressed.webp", img, [cv2.IMWRITE_WEBP_QUALITY,50])