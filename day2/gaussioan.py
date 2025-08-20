import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("image.jpg")
gaussian = cv2.GaussianBlur(img,(5,5),0)
plt.imshow(cv2.cvtColor(gaussian,cv2.COLOR_BGR2RGB))
plt.show()