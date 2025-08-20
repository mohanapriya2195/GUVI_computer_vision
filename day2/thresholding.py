import cv2
import matplotlib.pyplot as plt
img = cv2.imread("image.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threshold,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
plt.imshow(cv2.cvtColor(thresh,cv2.COLOR_BGR2RGB))
plt.show()